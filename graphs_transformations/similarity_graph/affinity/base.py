from abc import ABC, abstractmethod

import torch


class AffinityFunction(ABC):
    name: str
    requires_non_negative: bool = True
    preserves_order: bool = True
    output_range: str = "[0, 1]"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, D: torch.Tensor) -> torch.Tensor:
        """Convert a distance or similarity matrix D
        to an affinity matrix
        """
