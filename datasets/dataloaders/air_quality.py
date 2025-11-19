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
        impute_nans: bool = False,
        nan_method: str = "mean",
        freq: str = "60T",
        masked_sensors: list | None = None,
    ):
        self.dataset_path = dataset_path

        self.test_months = [3, 6, 9, 12]
        self.infer_eval_from = "next"

        data, missing_mask, distances = self.load(
            impute_nans=impute_nans,
            small=small,
            masked_sensors=masked_sensors,
        )
        self.distances = distances
        self.masked_sensors = (
            list(masked_sensors) if masked_sensors is not None else list()
        )
        super().__init__(
            dataframe=data, missing_mask=missing_mask, freq=freq, aggr="nearest"
        )

    @property
    def training_mask(self):
        return (
            self._mask
            if self.validation_mask is None
            else (self._mask & ~self.validation_mask)
        )

    # def __len__(self) -> int:
    #     return self.original_data.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        missing_data = self.current_data[index, :]
        ori_data = self.original_data[index, :]
        missing_mask = self.missing_mask[index, :]
        test_mask = self.test_mask[index, :]
        return missing_data, missing_mask, ori_data.nan_to_num_(0.0), test_mask

    def load_raw(
        self, small: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
            eval_mask = pd.DataFrame(pd.read_hdf(path, "eval_mask"))
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
            eval_mask = None
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations, eval_mask

    def load(
        self,
        impute_nans: bool = True,
        small: bool = False,
        masked_sensors: list | None = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        data, stations, eval_mask = self.load_raw(small=small)
        missing_mask = np.isnan(data.values).astype("bool")  # 1=missing, 0=observed
        if eval_mask is None:
            eval_mask = self._infer_mask(data)
        eval_mask = eval_mask.values.astype("bool")
        if masked_sensors is not None:
            eval_mask[:masked_sensors] = np.where(
                missing_mask[:, masked_sensors], True, False
            )
        self.validation_mask = eval_mask
        if impute_nans:
            data = data.fillna(self._compute_mean(data))

        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)

        return data, missing_mask, dist

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
            working_mask = ~self.missing_mask & mask_cols.unsqueeze(0).expand_as(
                self.missing_mask
            )
        else:
            working_mask = ~self.missing_mask
            assert not torch.isnan(self.original_data[working_mask]).any(), (
                "NaNs found in working mask"
            )
        working_points = torch.sum(working_mask).item()

        if mask_pattern == "default":
            if self.validation_mask is not None:
                print("Using predefined validation mask")
                working_mask = working_mask & ~self.validation_mask

                orig_valid_points = torch.sum(~self.missing_mask).item()
                post_val_points = torch.sum(working_mask).item()
                test_percent = test_percent * (orig_valid_points / post_val_points)

                self.test_mask = self.sample_mask_by_time(
                    data=self.original_data,
                    valid_mask=working_mask,
                    percent=test_percent,
                    time_blocks=time_blocks,
                )
            else:
                # if no predefined validation mask → create one with same logic
                self.test_mask = self.sample_mask_by_time(
                    data=self.original_data,
                    valid_mask=working_mask,
                    percent=test_percent,
                    time_blocks=time_blocks,
                )
                self.validation_mask = torch.zeros_like(
                    self.test_mask, dtype=torch.bool
                )
        elif mask_pattern == "blackout":
            print(
                f"Creating a blackout mask covering {total_missing_percent * 100}% of the rows"
            )
            working_mask = ~self.missing_mask
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
        missing = torch.sum(self.missing_mask).item() + test_points + val_points

        print(
            f"Test is {test_points / total_points:.2f} of total. {test_points / working_points:.2f} of non-missing"
        )
        print(
            f"Validation is {val_points / total_points:.2f} of total. {val_points / working_points:.2f} of non-missing"
        )
        print(
            f"Original missing values: {torch.sum(self.missing_mask) / total_points:.2f}; Actual missing values: {missing / total_points:.2f}"
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
        end_blackout: bool = True,
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

        if end_blackout:
            start_idx = T - blackout_len
        else:
            start_min = int(T * start_percent)
            start_max = max(start_min, T - blackout_len)
            start_idx = torch.randint(start_min, start_max + 1, (1,)).item()
        print(f"Blackout start at row {start_idx}")
        end_idx = min(start_idx + blackout_len, T)

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

        assert not (train_mask & ~valid_mask).any(), (
            "Train blackout overlaps invalid data"
        )
        assert not (eval_mask & ~valid_mask).any(), (
            "Eval blackout overlaps invalid data"
        )

        return train_mask, eval_mask

    def grin_splitter(
        self,
        val_len: float = 1.0,
        in_sample: bool = False,
        window: int = 0,
    ):
        nontest_idxs, test_idxs = self._disjoint_months(
            months=self.test_months, sync_mode="horizon"
        )
        if in_sample:
            train_idxs = np.arange(len(self))
            val_months = [(m - 1) % 12 for m in self.test_months]
            _, val_idxs = self._disjoint_months(months=val_months, sync_mode="horizon")
        else:
            val_len = (
                int(val_len * len(nontest_idxs)) if val_len < 1.0 else val_len
            ) // len(self.test_months)
            # get indices of first day of each testing month
            delta_idxs = np.diff(test_idxs)
            end_month_idxs = test_idxs[1:][
                np.flatnonzero(delta_idxs > delta_idxs.min())
            ]
            if len(end_month_idxs) < len(self.test_months):
                end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
            # expand month indices
            month_val_idxs = [
                np.arange(v_idx - val_len, v_idx) - window for v_idx in end_month_idxs
            ]
            val_idxs = np.concatenate(month_val_idxs) % len(self)
            val_idxs = val_idxs.astype(bool)
            print(f"{len(month_val_idxs)=}")
            assert isinstance(val_idxs, np.ndarray)
            assert val_idxs.dtype == bool
            assert val_idxs.shape == (len(self),)
            assert val_idxs.any(), (
                "val_idxs of all-False produces empty indices and breaks expand_indices"
            )
            # remove overlapping indices from training set
            ovl_idxs, _ = self.overlapping_indices(
                nontest_idxs, val_idxs, sync_mode="horizon", as_mask=True
            )
            train_idxs = nontest_idxs[~ovl_idxs]
        return train_idxs, val_idxs, test_idxs

    def _disjoint_months(
        self,
        months: List = [],
        sync_mode: str = "window",
    ):
        idxs = np.arange(len(self))
        if sync_mode == "window":
            start, end = 0, self.window - 1
        elif sync_mode == "horizon":
            horizon_offset = self.horizon_offset
            start, end = horizon_offset, horizon_offset + horizon_offset - 1
        else:
            raise ValueError(
                f"Invalid sync mode type: {sync_mode}. Expected 'window' or 'horizon'"
            )
        if self.index is not None:
            # after idxs
            start_in_months = np.isin(self.index[self._indices + start].month, months)
            end_in_months = np.isin(self.index[self._indices + end].month, months)
            idxs_in_months = start_in_months & end_in_months
            after_idxs = idxs[idxs_in_months]

            # before idxs
            months_before = np.setdiff1d(np.arange(1, 13), months)
            start_in_months = np.isin(
                self.index[self._indices + start].month, months_before
            )
            end_in_months = np.isin(
                self.index[self._indices + end].month, months_before
            )
            idxs_in_months = start_in_months & end_in_months
            prev_idxs = idxs[idxs_in_months]

            return prev_idxs, after_idxs
        else:
            raise ValueError

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
        loop: bool = False,
        cosine: bool = False,
        full_dataset: bool = False,
    ) -> torch.Tensor:
        total_missing_masK = self.missing_mask | self.test_mask | self.validation_mask

        available_rows = (~total_missing_masK).any(dim=(1, 2))

        data_tensor = (
            self.original_data
            if full_dataset
            else self.original_data[available_rows, :, :]
        )
        mask_tensor = (
            self.missing_mask
            if full_dataset
            else self.missing_mask[available_rows, :, :]
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
        loop: bool = False,
        cosine: bool = False,
        full_dataset: bool = False,
    ) -> torch.Tensor:
        total_missing_masK = self.missing_mask | self.test_mask | self.validation_mask

        available_rows = (~total_missing_masK).any(dim=(1, 2))

        data_tensor = (
            self.original_data
            if full_dataset
            else self.original_data[available_rows, :, :]
        )
        mask_tensor = (
            self.missing_mask
            if full_dataset
            else self.missing_mask[available_rows, :, :]
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
    ) -> pd.DataFrame:
        """
        Compute the geographical distance between coordinates points
        """

        _AVG_EARTH_RADIUS_KM = 6371.0088

        coords_pairs = coords.values
        if to_rad:
            coords_pairs = np.vectorize(np.radians)(coords_pairs)
        dist = haversine_distances(coords_pairs) * _AVG_EARTH_RADIUS_KM
        dist_df = pd.DataFrame(dist, coords.index, coords.index)
        return dist_df

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
        self.missing_mask = self.missing_mask | self.validation_mask | self.test_mask
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)

    def _normalize(self, data, mask, type: str = "min_max") -> torch.Tensor:
        observed = ~mask
        if type == "min_max":
            min_val = data[observed].min()
            max_val = data[observed].max()

            return (data - min_val) / ((max_val - min_val) + 1e-6)
        elif type == "std":
            mean_val = data[observed].min()
            std_val = data[observed].std()

            return (data - mean_val) / (std_val + 1e-8)
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
                    self.missing_mask, means, self.original_data
                )
        elif method == "zero":
            self.missing_data = self.original_data.nan_to_num(nan=0.0)

    def _infer_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        observed_mask = data.isna().astype("uint8")
        eval_mask = pd.DataFrame(index=data.index, columns=data.columns, data=0).astype(
            "uint8"
        )
        if self.infer_eval_from == "previous":
            offset = -1
        elif self.infer_eval_from == "next":
            offset = 1
        else:
            raise ValueError("infer_eval_mask can only be one of ['previous', 'next']")
        months = sorted(set(zip(data.index.year, data.index.month)))
        length = len(months)
        for i in range(length):
            j = (i + offset) % length
            year_i, month_i = months[i]
            year_j, month_j = months[j]
            mask_j = observed_mask[
                (data.index.year == year_j) & (data.index.month == month_j)
            ]
            mask_i = mask_j.shift(
                1, pd.DateOffset(months=12 * (year_i - year_j) + (month_i - month_j))
            )
            mask_i = mask_i[~mask_i.index.duplicated(keep="first")]
            mask_i = mask_i[np.isin(mask_i.index, data.index)]
            eval_mask.loc[mask_i.index] = (
                ~mask_i.loc[mask_i.index] & data.loc[mask_i.index]
            )
        return eval_mask

    def _compute_mean(self, data: pd.DataFrame) -> pd.DataFrame:
        data_mean = data.copy()
        condition0 = [
            data_mean.index.year,
            data_mean.index.isocalendar().week,
            data_mean.index.hour,
        ]
        condition1 = [
            data_mean.index.year,
            data_mean.index.month,
            data_mean.index.hour,
        ]
        conditions = [condition0, condition1, condition1[1:], condition1[2:]]
        while data_mean.isna().values.sum() and len(conditions):
            nan_mean = data_mean.groupby(conditions[0]).transform(np.nanmean)
            data_mean = data_mean.fillna(nan_mean)
            conditions = conditions[1:]
        if data_mean.isna().values.sum():
            data_mean = data_mean.fillna(method="ffill")
            data_mean = data_mean.fillna(method="bfill")
        return data_mean
