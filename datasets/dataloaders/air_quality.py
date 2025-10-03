import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import haversine_distances
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj

from datasets.dataloaders.graphloader import GraphLoader
from graphs_transformations.proximity_graphs import from_geo_nn, from_knn, from_radius

EARTH_RADIUS = 6371.0088


class AirQualityLoader(GraphLoader):
    def __init__(
        self,
        dataset_path: str = "./datasets/data/air_quality/",
        small: bool = False,
        normalization_type: str = "min_max",
        replace_nan=True,
        nan_method="mean",
    ):
        self.dataset_path = dataset_path
        self.validation_mask = torch.tensor([])
        self.original_data, self.missing_mask, self.distances = self.load(
            small=small, normalization_type=normalization_type
        )
        self.missing_data = torch.empty_like(self.original_data)
        if replace_nan:
            self._replace_nan(nan_method)
        self.corrupt_data = torch.empty_like(self.original_data)
        self.corrupt_mask = torch.empty_like(self.original_data)
        self.use_corrupted_data = False

    def __len__(self) -> int:
        return self.original_data.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_corrupted_data:
            missing_data = self.corrupt_data[index, :]
            ori_data = self.original_data[index, :]
            missing_mask = self.corrupt_mask[index, :]
            test_mask = self.test_mask[index, :]
        else:
            missing_data = self.current_data[index, :]
            ori_data = self.original_data[index, :]
            missing_mask = self.missing_mask[index, :]
            test_mask = self.test_mask[index, :]
        return missing_data, missing_mask, ori_data.nan_to_num_(0.0), test_mask

    def load_raw(
        self, small: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
            eval_mask = pd.DataFrame(pd.read_hdf(path, "eval_mask"))
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
            eval_mask = pd.DataFrame({})
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations, eval_mask

    def load(
        self, small: bool = False, normalization_type: str = "min_max"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, stations, eval_mask = self.load_raw(small=small)
        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)
        data = torch.from_numpy(data.to_numpy()).float()
        data = data.unsqueeze(-1)
        if eval_mask is not None:
            eval_mask = torch.from_numpy(eval_mask.to_numpy()).bool()
            eval_mask = eval_mask.unsqueeze(-1)
            self.validation_mask = eval_mask
        mask = torch.where(data.isnan(), False, True)
        assert torch.isnan(data[~mask]).all(), (
            "non-missing values found under missing values mask"
        )
        data = self._normalize(data, mask, normalization_type)
        return data, mask, dist

    def split(
        self,
        mask_pattern: str = "default",
        test_percent: float = 0.2,
        total_missing_percent: float = 0.2,
        cols: List = [],
        time_blocks: int = 5,
    ):
        """
        Split the data into training and validation sets using random blocks.

        Args:
            mask_pattern: The pattern of missing values to be created. Choice: [default, blackout]
            test_percent: Percentage of non-missing values held-out for training (loss)
            total_missing_percent: Percentage of non-missing values held-out for both training and validation.
                                    If equal to training hold-out both are the same
        """
        total_points = self.original_data.numel()
        if len(cols) > 0:
            mask_cols = torch.zeros(self.original_data.shape[1], dtype=torch.bool)
            mask_cols[cols] = True
            working_mask = self.missing_mask & mask_cols.unsqueeze(0).expand_as(
                self.missing_mask
            )
        else:
            working_mask = self.missing_mask
        working_points = torch.sum(working_mask).item()

        if mask_pattern == "default":
            if self.validation_mask is not None:
                print("Using predefined validation mask")
                working_mask = working_mask & ~self.validation_mask

                orig_valid_points = torch.sum(self.missing_mask).item()
                post_val_points = torch.sum(working_mask).item()
                test_percent = test_percent * (orig_valid_points / post_val_points)

                self.test_mask = self.sample_mask_by_time(
                    data=self.original_data,
                    valid_mask=working_mask,
                    percent=test_percent,
                    time_blocks=time_blocks,
                )

                test_points = torch.sum(self.test_mask).item()
                val_points = torch.sum(self.validation_mask).item()
                missing = (
                    torch.sum(~self.missing_mask).item() + test_points + val_points
                )

                print(
                    f"Test is {test_points / total_points:.2f} of total. {test_points / working_points:.2f} of non-missing"
                )
                print(
                    f"Validation is {val_points / total_points:.2f} of total. {val_points / working_points:.2f} of non-missing"
                )
                print(
                    f"Original missing values: {torch.sum(~self.missing_mask) / total_points:.2f}; Actual missing values: {missing / total_points:.2f}"
                )

                assert not torch.isnan(
                    self.original_data[self.validation_mask]
                ).any(), "Missing values found under evaluation mask (second pass)"
                assert not torch.isnan(self.original_data[self.test_mask]).any(), (
                    "Missing values found under evaluation mask (second pass)"
                )
        elif mask_pattern == "blackout":
            print(
                f"Creating a blackout mask covering {total_missing_percent * 100}% of the rows"
            )
            self.test_mask, self.validation_mask = self.split_blackout(
                valid_mask=working_mask,
                test_frac=test_percent,
                total_missing=total_missing_percent,
            )

        elif mask_pattern == "mcar":
            print(
                f"Creating a MCAR mask covering {total_missing_percent * 100}% of non-missing data"
            )
            self.test_mask, self.validation_mask = self.split_mcar(
                valid_mask=working_mask,
                test_frac=test_percent,
                total_missing=total_missing_percent,
            )
        else:
            raise ValueError("Provide a valid mask pattern")

        if self.test_mask is None or self.validation_mask is None:
            raise ValueError("Test mask or validation mask shouldn't be None")

        test_points = torch.sum(self.test_mask).item()
        val_points = torch.sum(self.validation_mask).item()
        missing = torch.sum(~self.missing_mask).item() + test_points + val_points

        print(
            f"Test is {test_points / total_points:.2f} of total. {test_points / working_points:.2f} of non-missing"
        )
        print(
            f"Validation is {val_points / total_points:.2f} of total. {val_points / working_points:.2f} of non-missing"
        )
        print(
            f"Original missing values: {torch.sum(~self.missing_mask) / total_points:.2f}; Actual missing values: {missing / total_points:.2f}"
        )

        assert not torch.isnan(self.original_data[self.validation_mask]).any(), (
            "Missing values found under evaluation mask (second pass)"
        )
        assert not torch.isnan(self.original_data[self.test_mask]).any(), (
            "Missing values found under evaluation mask (second pass)"
        )
        self.validation_mask = self.validation_mask.bool()
        self.test_mask = self.test_mask.bool()

    def sample_mask_by_time(
        self,
        data: torch.Tensor,
        valid_mask: torch.Tensor,
        percent: float,
        time_blocks: int,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample a boolean mask of `percent` of the True entries in `valid_mask`,
        stratified across `time_blocks` along dim 0.
        """
        assert 0 <= percent <= 1, "percent must be in [0,1]"
        rows, cols, feats = data.shape
        mask_out = torch.zeros_like(data, dtype=torch.bool)

        total_valid = torch.sum(valid_mask).item()
        target_points = int(total_valid * percent)
        if target_points == 0 or total_valid == 0:
            return mask_out

        time_segments = torch.linspace(0, rows, time_blocks + 1).long()
        current_points = 0

        for block in range(time_blocks):
            start_row = time_segments[block].item()
            end_row = time_segments[block + 1].item()

            seg_mask = valid_mask[start_row:end_row]
            valid_count = torch.sum(seg_mask).item()
            if valid_count == 0:
                continue

            block_target = int(valid_count / total_valid * target_points)
            if block_target == 0:
                continue

            valid_idx = torch.nonzero(seg_mask)
            perm = torch.randperm(len(valid_idx), generator=rng)
            chosen = valid_idx[perm[:block_target]]
            chosen[:, 0] += start_row  # shift back to global row coords
            mask_out[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
            current_points += block_target

        # second pass if rounding undershot
        if current_points < target_points:
            remaining = target_points - current_points
            remaining_idx = torch.nonzero(valid_mask & ~mask_out)
            if len(remaining_idx) > 0:
                perm = torch.randperm(len(remaining_idx), generator=rng)
                chosen = remaining_idx[perm[:remaining]]
                mask_out[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True

        # Optional sanity check
        assert not torch.isnan(data[mask_out]).any(), "Selected NaNs"

        return mask_out

    def split_mcar(
        self,
        valid_mask: torch.Tensor,
        test_frac: float = 0.2,
        total_missing: float = 0.4,
        time_blocks: int = 5,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split data into train and eval MCAR masks (random missing points, stratified by time).

        Args:
            data: Tensor of shape [T, N, F]
            valid_mask: Boolean mask, same shape as data (True if observed).
            test_frac: Fraction of total valid entries to mask for training.
            total_missing: Total fraction of valid entries to mask (train + eval).
            time_blocks: Number of time segments to stratify sampling.
            rng: Optional torch.Generator for reproducibility.

        Returns:
            train_mask, eval_mask (boolean tensors)
        """
        assert 0 <= test_frac <= total_missing <= 1.0, (
            "Must have 0 <= train_frac <= total_missing <= 1"
        )

        rows, cols, feats = self.original_data.shape
        total_valid = torch.sum(valid_mask).item()
        target_total = int(total_valid * total_missing)
        target_train = int(total_valid * test_frac)
        target_eval = target_total - target_train

        train_mask = torch.zeros_like(self.original_data, dtype=torch.bool)
        eval_mask = torch.zeros_like(self.original_data, dtype=torch.bool)

        # Edge case: if train == total → eval = train
        if target_eval == 0:
            if target_train == 0 or total_valid == 0:
                return train_mask, train_mask
            # Fall back to single sampling
            chosen = torch.nonzero(valid_mask)
            perm = torch.randperm(len(chosen), generator=rng)
            selected = chosen[perm[:target_train]]
            train_mask[selected[:, 0], selected[:, 1], selected[:, 2]] = True
            return train_mask, train_mask.clone()

        # Split timeline into blocks
        time_segments = torch.linspace(0, rows, time_blocks + 1).long()

        current_train = 0
        current_eval = 0

        for block in range(time_blocks):
            start_row = time_segments[block].item()
            end_row = time_segments[block + 1].item()
            seg_mask = valid_mask[start_row:end_row]
            valid_count = torch.sum(seg_mask).item()
            if valid_count == 0:
                continue

            # Block quotas
            block_train = int(valid_count / total_valid * target_train)
            block_eval = int(valid_count / total_valid * target_eval)

            valid_idx = torch.nonzero(seg_mask)
            if len(valid_idx) == 0:
                continue
            perm = torch.randperm(len(valid_idx), generator=rng)

            # Train samples
            if block_train > 0:
                chosen = valid_idx[perm[:block_train]]
                chosen[:, 0] += start_row
                train_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                current_train += block_train

            # Eval samples
            if block_eval > 0:
                chosen = valid_idx[perm[block_train : block_train + block_eval]]
                chosen[:, 0] += start_row
                eval_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                current_eval += block_eval

        # Second pass if rounding undershot
        if current_train < target_train:
            remaining = target_train - current_train
            remaining_idx = torch.nonzero(valid_mask & ~(train_mask | eval_mask))
            if len(remaining_idx) > 0:
                perm = torch.randperm(len(remaining_idx), generator=rng)
                chosen = remaining_idx[perm[:remaining]]
                train_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True

        if current_eval < target_eval:
            remaining = target_eval - current_eval
            remaining_idx = torch.nonzero(valid_mask & ~(train_mask | eval_mask))
            if len(remaining_idx) > 0:
                perm = torch.randperm(len(remaining_idx), generator=rng)
                chosen = remaining_idx[perm[:remaining]]
                eval_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True

        # Sanity checks
        assert not torch.isnan(self.original_data[train_mask | eval_mask]).any(), (
            "Selected NaNs"
        )
        assert not (train_mask & eval_mask).any(), "Train and eval masks overlap"

        return train_mask, eval_mask

    def split_blackout(
        self,
        valid_mask: torch.Tensor,
        test_frac: float = 0.2,
        total_missing: float = 0.4,
        start_percent: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split data into train and eval blackout masks.
        Each is a contiguous block of rows (time steps).

        Args:
            data: Tensor of shape [T, N, F]
            valid_mask: Boolean mask, same shape as data (True if observed).
            test_frac: Fraction of total rows to blackout for testing during training.
            total_missing: Total fraction (train + eval) to blackout.
            start_percent: Earliest possible blackout start fraction.

        Returns:
            train_mask, eval_mask
        """
        assert 0 <= test_frac <= total_missing <= 1.0, (
            "Must have 0 <= test_frac <= total_missing <= 1"
        )

        T = self.original_data.shape[0]
        blackout_len = int(T * total_missing)
        train_len = int(T * test_frac)
        val_len = blackout_len - train_len

        start_min = int(T * start_percent)
        start_max = max(start_min, T - blackout_len)
        start_idx = torch.randint(start_min, start_max + 1, (1,)).item()
        end_idx = start_idx + blackout_len

        # Initialize masks
        train_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
        eval_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

        # Assign blackout segments
        if train_len > 0:
            train_mask[start_idx : start_idx + train_len, :, :] = True
        if val_len > 0:
            eval_mask[start_idx + train_len : end_idx, :, :] = True
        else:
            # Special case: train == total → eval = train
            eval_mask = train_mask.clone()

        # Ensure blackout only applies where data was originally valid
        train_mask = train_mask & valid_mask
        eval_mask = eval_mask & valid_mask

        return train_mask, eval_mask

    def get_geolocation_graph(
        self,
        threshold: float = 0.1,
        threshold_on_input: bool = False,
        include_self: bool = False,
        force_symmetric: bool = False,
        weighted: bool = True,
    ) -> torch.Tensor:
        theta = self.distances.std()
        # adj = np.exp(-(self.distances**2) / (2 * theta**2))
        adj = torch.exp(-torch.square(self.distances / theta))
        dist = (self.distances - self.distances.min()) / (
            self.distances.max() - self.distances.min()
        )
        mask = dist > threshold if threshold_on_input else adj < threshold
        adj[mask] = 0
        if not weighted:
            adj[adj > 0] = 1.0
        if not include_self:
            adj.fill_diagonal_(0.0)
        if force_symmetric:
            adj = torch.from_numpy(np.maximum.reduce([adj.numpy(), adj.T.numpy()]))
        return adj

    def get_knn_graph(
        self,
        k: float,
        use_corrupted_data: bool = False,
        loop: bool = False,
        cosine: bool = False,
        full_dataset: bool = False,
    ) -> torch.Tensor:
        train_mask = self.test_mask.any((1, 2))

        if use_corrupted_data:
            data_tensor = (
                self.corrupt_data
                if full_dataset
                else self.corrupt_data[train_mask, :, :]
            )
            mask_tensor = (
                self.corrupt_mask
                if full_dataset
                else self.corrupt_mask[train_mask, :, :]
            )
        else:
            data_tensor = (
                self.original_data
                if full_dataset
                else self.original_data[train_mask, :, :]
            )
            mask_tensor = (
                self.missing_mask
                if full_dataset
                else self.missing_mask[train_mask, :, :]
            )

        data = data_tensor.permute(1, 0, 2).reshape(self.original_data.shape[1], -1)
        mask = mask_tensor.permute(1, 0, 2).reshape(self.original_data.shape[1], -1)

        edge_index = from_knn(data=data, mask=mask, k=k, loop=loop, cosine=cosine)
        adj = to_dense_adj(edge_index).squeeze()
        return adj

    def get_geo_nn_graph(
        self,
        k: int | float,
        include_self: bool = False,
        force_symmetric: bool = False,
        weighted: bool = True,
    ) -> torch.Tensor:
        return from_geo_nn(
            self.distances,
            k=k,
            include_self=include_self,
            force_symmetric=force_symmetric,
            weighted=weighted,
        )

    def get_radius_graph(
        self,
        radius: float,
        use_corrupted_data: bool = False,
        loop: bool = False,
        cosine: bool = False,
        full_dataset: bool = False,
    ) -> torch.Tensor:
        train_mask = self.test_mask.any((1, 2))

        if use_corrupted_data:
            data_tensor = (
                self.corrupt_data
                if full_dataset
                else self.corrupt_data[train_mask, :, :]
            )
            mask_tensor = (
                self.corrupt_mask
                if full_dataset
                else self.corrupt_mask[train_mask, :, :]
            )
        else:
            data_tensor = (
                self.original_data
                if full_dataset
                else self.original_data[train_mask, :, :]
            )
            mask_tensor = (
                self.missing_mask
                if full_dataset
                else self.missing_mask[train_mask, :, :]
            )

        data = data_tensor.permute(1, 0, 2).reshape(self.original_data.shape[1], -1)
        mask = mask_tensor.permute(1, 0, 2).reshape(self.original_data.shape[1], -1)
        edge_index = from_radius(
            data=data, mask=mask, radius=radius, loop=loop, cosine=cosine
        )
        adj = to_dense_adj(edge_index).squeeze()
        return adj

    def _geographical_distance(
        self, coords: pd.DataFrame, to_rad: bool = True
    ) -> torch.Tensor:
        """
        Compute the geographical distance between coordinates points
        """

        _AVG_EARTH_RADIUS_KM = 6371.0088

        coords_array = coords.to_numpy()
        if to_rad:
            coords_array = np.radians(coords_array)
        dist = torch.from_numpy(
            haversine_distances(coords_array) * _AVG_EARTH_RADIUS_KM
        ).float()
        return dist

    def get_dataloader(
        self,
        test_percent: float = 0.2,
        total_missing_percent: float = 0.4,
        mask_pattern: str = "default",
        shuffle: bool = False,
        batch_size: int = 128,
    ) -> DataLoader:
        self.split(
            mask_pattern=mask_pattern,
            test_percent=test_percent,
            total_missing_percent=total_missing_percent,
        )
        if self.validation_mask is None:
            raise ValueError("Validation mask should not be None after split")
        self.missing_data = torch.where(self.test_mask, 0.0, self.missing_data)
        self.missing_data = torch.where(self.validation_mask, 0.0, self.missing_data)
        self.current_data = self.missing_data.clone()
        self.missing_mask = self.missing_mask & ~self.validation_mask & ~self.test_mask
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)

    def shape(self):
        return self.original_data.shape

    def corrupt(self, missing_type="perc", missing_size=20):
        """
        Add missing data to the dataset according to the specified missing pattern
        Args:
            missing_type (str): Denotes the type of missing pattern to apply. Currently only
                missing percentage is available
        """
        print(
            "WARNING: Air Quality dataset already contains missing data. This may change"
            "the results compared to other studies using this dataset "
        )
        if missing_type == "perc":
            self.missing_percentage(missing_size)
        else:
            pass

    def missing_percentage(self, missing_percent: int = 20):
        data_length, _ = self.original_data.shape
        missing_start = data_length * 5 // 100
        missing_length = data_length * missing_percent // 100
        missing_data = self.original_data
        missing_data[missing_start : missing_start + missing_length, 0] = torch.nan
        mask = torch.ones_like(missing_data)
        mask.masked_fill_(torch.isnan(self.original_data), 0.0)
        self.corrupt_mask = mask
        missing_data.nan_to_num_(nan=0.0)
        self.corrupt_data = missing_data

    def _normalize_corrupt(self):
        data = self.corrupt_data
        mask = self.corrupt_data
        # mean = data[mask].mean()
        # std = data[mask].std()
        min = data[mask].min()
        max = data[mask].max()

        self.corrupt_data = (data - min) / (max - min)

    def _normalize(self, data, mask, type: str = "min_max") -> torch.Tensor:
        if type == "min_max":
            min = data[mask].min()
            max = data[mask].max()

            return (data - min) / (max - min)
        elif type == "std":
            mean = data[mask].min()
            std = data[mask].std()

            return (data - mean) / (std + 1e-8)
        else:
            return data

    def _replace_nan(self, method="mean"):
        print("------------------ Replacing NaNs ------------------")
        if method == "mean":
            if torch.isnan(self.original_data).any():
                # data.nan_to_num_(nan=0.0)
                means = (
                    self.original_data.nanmean(dim=1)
                    .unsqueeze(1)
                    .expand_as(self.original_data)
                )
                # print(f"NaNs in means?: {torch.isnan(means).any()}")
                means = means.nan_to_num(0.0)
                self.missing_data = torch.where(
                    self.missing_mask, self.original_data, means
                )
        elif method == "zero":
            self.missing_data = self.original_data.nan_to_num(nan=0.0)
