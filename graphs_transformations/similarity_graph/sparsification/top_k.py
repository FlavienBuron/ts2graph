import torch

from graphs_transformations.similarity_graph.sparsification.base import (
    SparsificationFunction,
)
from graphs_transformations.similarity_graph.specs.registry import (
    register_sparsification,
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
