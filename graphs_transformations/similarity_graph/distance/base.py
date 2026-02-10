from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch


class DistanceFunction(ABC):
    name: str
    symmetric: bool = True
    non_negative: bool = True
    supports_mask: bool = True
    input_kind = Literal["series", "coordinates"]

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Return the distance/similarity matrix D"""
