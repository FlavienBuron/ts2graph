from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
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

    def update_data(self, new_data: Tensor) -> None:
        """
        Safely updates the dataset data with new tensor values,
        ensuring proper memory management.

        Args:
            new_data: The new data to replace self.data with
        """
        # Ensure tensor is detached from computational graph
        if new_data.requires_grad:
            new_data = new_data.detach()

        # Move to CPU if on another device
        if new_data.device.type != "cpu":
            new_data = new_data.cpu()

        new_data = new_data.clone()
        # Store old data ref for cleanup
        old_data = self.data

        self.data = new_data

        del old_data

        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
