import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("identity")
class Identity(DistanceFunction):
    def __init__(self, _):
        pass

    def __call__(self, x: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return x.detach().clone()
