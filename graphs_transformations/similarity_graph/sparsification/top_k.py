import torch

from ..specs.registry import (
    register_sparsification,
)
from .base import (
    SparsificationFunction,
)


@register_sparsification("topk")
class TopK(SparsificationFunction):
    name = "top k"

    def __init__(self, k):
        self.k = k

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        values, idx = torch.topk(A, self.k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, idx, 1)
        return A * mask
