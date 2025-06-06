from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class GraphLoader(Dataset, ABC):
    def __init__(self) -> None:
        self.original_data: (
            torch.Tensor
        )  # The original data with/without missing values. Static
        self.missing_data: torch.Tensor  # The original data with the missing values
        self.current_data: torch.Tensor  # The working copy, used during training
        self.missing_mask: torch.Tensor
        self.train_mask: torch.Tensor
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

        # Only update missing values
        self.current_data[~self.missing_mask] = new_data[~self.missing_mask]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_current_data(self) -> None:
        """Reset the current data to the initial missing data state"""
        self.current_data = self.missing_data.clone()

    @abstractmethod
    def corrupt(self, missing_type: str = "perc"):
        pass

    @abstractmethod
    def get_knn_graph(self, k: int, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_radius_graph(self, radius: float, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_geolocation_graph(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def get_dataloader(
        self, use_corrupted_data: bool, shuffle: bool, batch_size: int
    ) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        pass
