from abc import ABC, abstractmethod

import torch


class DistanceFunction(ABC):
    name: str
    symmetric: bool = True
    non_negative: bool = True
    supports_mask: bool = True

    @abstractmethod
    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the distance/similarity matrix D"""
