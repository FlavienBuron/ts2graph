import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("identity")
class Identity(DistanceFunction):
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x.detach().clone()
