import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("identity")
class Identity(AffinityFunction):
    name = "identity"
    requires_non_negative = False
    epsilon = 1e-6

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, D: torch.Tensor) -> torch.Tensor:
        return D
