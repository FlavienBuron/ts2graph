import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import haversine_distances
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
        impute_nans: bool = True,
        nan_method: str = "mean",
        freq: str = "60min",
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
            dataframe=data,
            missing_mask=missing_mask,
            eval_mask=self.eval_mask,
            freq=freq,
            aggr="nearest",
        )

    #
    # @property
    # def training_mask(self):
    #     return self._mask if self.eval_mask is None else (self._mask & ~self.eval_mask)

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
        missing_mask = (~np.isnan(data.values)).astype("bool")  # 0=missing, 1=observed
        if eval_mask is None:
            print("Infering eval mask")
            eval_mask = self._infer_mask(data)
        eval_mask = eval_mask.values.astype("bool")
        if masked_sensors is not None:
            eval_mask[:masked_sensors] = np.where(
                missing_mask[:, masked_sensors], True, False
            )
        self.eval_mask = eval_mask
        if impute_nans:
            data = data.fillna(self._compute_mean(data))

        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)
        return data, missing_mask, dist

    def grin_split(
        self,
        val_len: float = 0.1,
        in_sample: bool = False,
        window: int = 36,
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
                int(val_len * len(nontest_idxs)) if val_len < 1 else val_len
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
            # remove overlapping indices from training set
            ovl_idxs, _ = self.overlapping_indices(
                nontest_idxs, val_idxs, sync_mode="horizon", as_mask=True
            )
            train_idxs = nontest_idxs[~ovl_idxs]
        return train_idxs, val_idxs, test_idxs

    def get_geolocation_graph(
        self,
        threshold: float = 0.1,
        threshold_on_input: bool = False,
        include_self: bool = False,
        force_symmetric: bool = False,
        weighted: bool = True,
    ) -> torch.Tensor:
        distances = torch.from_numpy(self.distances.to_numpy())
        theta = distances.std()
        # adj = np.exp(-(self.distances**2) / (2 * theta**2))
        adj = torch.exp(-torch.square(distances / theta))
        dist = (distances - distances.min()) / (distances.max() - distances.min())
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
        total_missing_masK = self.missing_mask | self.test_mask | self.eval_mask

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
        total_missing_masK = self.missing_mask | self.test_mask | self.eval_mask

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

    def _infer_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        observed_mask = data.isna().astype("bool")
        eval_mask = pd.DataFrame(index=data.index, columns=data.columns, data=0).astype(
            "bool"
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
                ~mask_i.loc[mask_i.index].astype(bool)
                & data.loc[mask_i.index].astype(bool)
            ).astype(eval_mask.dtypes.iloc[0])
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
            nan_mean = data_mean.groupby(conditions[0]).transform("mean")
            data_mean = data_mean.fillna(nan_mean)
            conditions = conditions[1:]
        if data_mean.isna().values.sum():
            data_mean = data_mean.fillna(method="ffill")
            data_mean = data_mean.fillna(method="bfill")
        return data_mean

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
            start, end = horizon_offset, horizon_offset + self.horizon - 1
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

    def overlapping_indices(
        self, idxs1, idxs2, sync_mode="window", as_mask=False
    ) -> tuple[np.ndarray, np.ndarray]:
        assert sync_mode in ["window", "horizon"], (
            "sync_mode can only be 'window' or 'horizon'"
        )
        timestamp1 = self.data_timestamps(idxs1, flatten=False)[sync_mode]
        timestamp2 = self.data_timestamps(idxs2, flatten=False)[sync_mode]
        common_timestamps = np.intersect1d(np.unique(timestamp1), np.unique(timestamp2))
        is_overlapping = lambda sample: np.any(np.isin(sample, common_timestamps))
        m1 = np.apply_along_axis(is_overlapping, 1, timestamp1)
        m2 = np.apply_along_axis(is_overlapping, 1, timestamp2)
        if as_mask:
            return m1, m2
        return np.sort(idxs1[m1]), np.sort(idxs2[m2])

    def expand_indices(self, indices=None, unique=False) -> Dict:
        ds_indices = dict.fromkeys(
            [time for time in ["window", "horizon"] if getattr(self, time) > 0]
        )
        indices = np.arange(len(self._indices)) if indices is None else indices
        if "window" in ds_indices:
            window_idxs = [
                np.arange(idx, idx + self.window) for idx in self._indices[indices]
            ]
            ds_indices["window"] = np.concatenate(window_idxs)
        if "horizon" in ds_indices:
            horizon_idxs = [
                np.arange(
                    idx + self.horizon_offset,
                    idx + self.horizon_offset + self.horizon,
                )
                for idx in self._indices[indices]
            ]
            ds_indices["horizon"] = np.concatenate(horizon_idxs)
        if unique:
            ds_indices = {
                k: np.unique(v) for k, v in ds_indices.items() if v is not None
            }
        return ds_indices

    def data_timestamps(self, indices=None, flatten=True) -> Dict:
        ds_indices = self.expand_indices(indices, unique=False)
        ds_timestamp = {k: self.index[v] for k, v in ds_indices.items()}
        if not flatten:
            ds_timestamp = {
                k: np.array(v).reshape(-1, getattr(self, k))
                for k, v in ds_timestamp.items()
            }
        return ds_timestamp
