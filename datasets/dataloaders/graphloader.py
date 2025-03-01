from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class GraphLoader(Dataset, ABC):
    def __init__(self) -> None:
        self.data: torch.Tensor
        self.missing_data: torch.Tensor
        self.mask: torch.Tensor
        self.validation_mask: torch.Tensor
        self.distances = None
        self.corrupt_data: torch.Tensor = torch.empty(
            0,
        )
        self.corrupt_mask: torch.Tensor = torch.empty(
            0,
        )

    @abstractmethod
    def corrupt(self, missing_type: str = "perc"):
        pass

    @abstractmethod
    def get_similarity_knn(self, k: int, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_adjacency(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_dataloader(
        self, use_missing_data: bool, shuffle: bool, batch_size: int
    ) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
