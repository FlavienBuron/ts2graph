import os
from typing import Tuple

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
        self, dataset_path: str = "./datasets/data/air_quality/", small: bool = False
    ):
        self.dataset_path = dataset_path
        self.data, self.distances = self.load(small=small)
        self.missing_data = torch.empty_like(self.data)
        self.missing_mask = torch.empty_like(self.data)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_missing_data:
            data = self.missing_data[index, :]
            mask = self.missing_mask[index, :]
        else:
            data = self.data[index, :]
            mask = torch.ones_like(data)
        return data, mask

    def load_raw(self, small: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations

    def load(self, small: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        data, stations = self.load_raw(small=small)
        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self._geographical_distance(stations_coords)
        data = torch.from_numpy(data.to_numpy()).float()
        return data, dist

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

    def get_similarity_knn(self, k: int) -> torch.Tensor:
        data = self.missing_data.T
        edge_index = from_knn(data=data, k=k)
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
        self, use_missing_data: bool, shuffle: bool = False, batch_size: int = 8
    ) -> DataLoader:
        self.use_missing_data = use_missing_data
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)

    def shape(self):
        return self.data.shape

    def corrupt(self, missing_type="perc"):
        if missing_type == "perc":
            self.missing_percentage()
        else:
            pass

    def missing_percentage(self, missing_percent: int = 20):
        data_length, _ = self.data.shape
        missing_start = data_length // 5
        missing_length = data_length // missing_percent
        missing_data = self.data
        missing_data[missing_start : missing_start + missing_length, 0] = torch.nan
        mask = torch.ones_like(missing_data)
        mask.masked_fill_(torch.isnan(self.data), 0.0)
        self.missing_mask = mask
        missing_data.nan_to_num_(nan=0.0)
        self.missing_data = missing_data

    def _normalize(self):
        data = self.missing_data
        mask = self.missing_data
        # mean = data[mask].mean()
        # std = data[mask].std()
        min = data[mask].min()
        max = data[mask].max()

        self.missing_data = (data - min) / (max - min)
