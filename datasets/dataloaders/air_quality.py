import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import haversine_distances
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj

from datasets.dataloaders.graphloader import GraphLoader
from graphs_transformations.knn import from_knn

EARTH_RADIUS = 6371.0088


class AirQualityLoader(GraphLoader):
    def __init__(
        self,
        dataset_path: str = "./datasets/data/air_quality/",
        small: bool = False,
        normalization_type=None,
        replace_nan=True,
        nan_method="mean",
    ):
        self.dataset_path = dataset_path
        self.original_data, self.mask, self.distances = self.load(small=small)
        self.missing_data = torch.empty_like(self.original_data)
        if replace_nan:
            self._replace_nan(nan_method)
        self.validation_mask = torch.zeros_like(self.original_data).bool()
        self.corrupt_data = torch.empty_like(self.original_data)
        self.corrupt_mask = torch.empty_like(self.original_data)
        self.use_corrupted_data = False

    def __len__(self) -> int:
        return self.original_data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_corrupted_data:
            data = self.corrupt_data[index, :]
            mask = self.corrupt_mask[index, :]
        else:
            data = self.current_data[index, :]
            mask = self.mask[index, :]
        return data, mask

    def load_raw(self, small: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations

    def load(
        self, small: bool = False, normalization_type=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, stations = self.load_raw(small=small)
        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)
        data = torch.from_numpy(data.to_numpy()).float()
        mask = torch.where(data.isnan(), False, True)
        data = self._normalize(data, mask, normalization_type)
        return data, mask, dist

    def split(self, validation_len: int, contiguous: bool, cols: List = None):
        if contiguous:
            for row_start in range(self.original_data.shape[0] - validation_len + 1):
                window = self.original_data[row_start : row_start + validation_len, :]
                if torch.count_nonzero(window.isnan()).le(
                    validation_len * self.original_data.shape[1] * 0.5
                ):
                    val_mask = torch.zeros_like(self.original_data)
                    val_mask[row_start : row_start + validation_len, :] = True
                    val_mask[~self.mask] = False
                    self.validation_mask = val_mask.bool()
                    break

    def get_adjacency(
        self,
        threshold: float = 0.1,
        threshold_on_input: bool = False,
        include_self: bool = False,
        force_symmetric: bool = False,
    ) -> torch.Tensor:
        theta = self.distances.std()
        # adj = np.exp(-(self.distances**2) / (2 * theta**2))
        adj = torch.exp(-torch.square(self.distances / theta))
        mask = self.distances > threshold if threshold_on_input else adj < threshold
        adj[mask] = 0
        if not include_self:
            adj.fill_diagonal_(0.0)
        if force_symmetric:
            adj = torch.from_numpy(np.maximum.reduce([adj.numpy(), adj.T.numpy()]))
        return adj

    def get_similarity_knn(
        self,
        k: int,
        use_corrupted_data: bool = False,
        loop: bool = False,
        cosine: bool = False,
    ) -> torch.Tensor:
        if use_corrupted_data:
            data = self.corrupt_data.T
            mask = self.corrupt_mask.T
        else:
            data = self.original_data.T
            mask = self.mask.T
        edge_index = from_knn(data=data, mask=mask, k=k, loop=loop, cosine=cosine)
        adj = to_dense_adj(edge_index).squeeze()
        return adj

    def _geographical_distance(
        self, coords: pd.DataFrame, to_rad: bool = True
    ) -> torch.Tensor:
        """
        Compute the geographical distance between coordinates points
        """
        coords_array = coords.to_numpy()
        if to_rad:
            coords_array = np.radians(coords_array)
        dist = torch.from_numpy(haversine_distances(coords_array)).float()
        return dist

    def get_dataloader(
        self, use_corrupted_data: bool, shuffle: bool = False, batch_size: int = 8
    ) -> DataLoader:
        self.use_corrupted_data = use_corrupted_data
        self.split(
            validation_len=self.original_data.shape[0] * 50 // 100, contiguous=True
        )
        # print(self.validation_mask)
        self.missing_data = torch.where(self.validation_mask, 0.0, self.missing_data)
        self.current_data = self.missing_data.clone()
        self.mask = self.mask & ~self.validation_mask
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

    def _normalize(self, data, mask, type=None) -> torch.Tensor:
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
                self.missing_data = torch.where(self.mask, self.original_data, means)
        elif method == "zero":
            self.missing_data = self.original_data.nan_to_num(nan=0.0)
