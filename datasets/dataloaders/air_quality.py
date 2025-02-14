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

    def load_raw(self, small: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if small:
            path = os.path.join(self.dataset_path, "small36.h5")
        else:
            path = os.path.join(self.dataset_path, "full437.h5")
        data = pd.DataFrame(pd.read_hdf(path, key="pm25"))
        stations = pd.DataFrame(pd.read_hdf(path, key="stations"))
        return data, stations

    def load(self, small: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        data, stations = self.load_raw(small=small)
        stations_coords = stations.loc[:, ["latitude", "longitude"]]
        dist = self.geographical_distance(stations_coords)
        return data, dist

    def get_adjacency(
        self,
        threshold: float = 0.1,
        threshold_on_input: bool = False,
        include_self: bool = False,
        force_symmetric: bool = False,
    ) -> np.ndarray:
        theta = np.std(self.distances)
        # adj = np.exp(-(self.distances**2) / (2 * theta**2))
        adj = np.exp(-np.square(self.distances / theta))
        mask = self.distances > threshold if threshold_on_input else adj < threshold
        adj[mask] = 0
        if not include_self:
            np.fill_diagonal(adj, 0)
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        return adj

    def get_similarity_knn(self, k: int) -> torch.Tensor:
        data = self.data.to_numpy().T
        data = torch.tensor(data)
        edge_index = from_knn(data=data, k=k)
        adj = to_dense_adj(edge_index).squeeze()
        return adj

    @staticmethod
    def geographical_distance(coords: pd.DataFrame, to_rad: bool = True) -> np.ndarray:
        """
        Compute the geographical distance between coordinates points
        """
        coords_array = coords.to_numpy()
        if to_rad:
            coords_array = np.radians(coords_array)
        dist = haversine_distances(coords_array)
        return dist

    def get_dataloader(self, shuffle: bool = False, batch_size: int = 8) -> DataLoader:
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size)

    def shape(self):
        return self.data.shape
