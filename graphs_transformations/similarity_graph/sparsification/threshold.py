import torch

from ..specs.registry import (
    register_sparsification,
)
from .base import (
    SparsificationFunction,
)


@register_sparsification("threshold")
class Threshold(SparsificationFunction):
    name = "threshold"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        mask = A >= self.threshold
        adj = A * mask
        adj.fill_diagonal_(0.0)
        return adj
