from abc import ABC, abstractmethod

import torch


class AffinityFunction(ABC):
    name: str
    requires_non_negative: bool = True
    preserves_order: bool = True
    output_range: str = "[0, 1]"
    epsilon = 1e-6

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, D: torch.Tensor) -> torch.Tensor:
        """Convert a distance or similarity matrix D
        to an affinity matrix
        """

    def _normalize(self, D: torch.Tensor, valid: torch.Tensor):
        scale = D[valid].median()
        return D / scale.clamp_min(self.epsilon)
