import torch

from ..specs.registry import (
    register_sparsification,
)
from .base import (
    SparsificationFunction,
)


@register_sparsification("fc")
class FullyConnected(SparsificationFunction):
    name = "fully connected"

    def __init__(
        self,
        binary: bool = True,
        self_loop_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.binary = binary
        self.self_loop_weight = self_loop_weight

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        print(f"Sparsifier: {self.name}")
        adj = A
        if self.binary:
            adj = torch.ones_like(A)
        adj.fill_diagonal_(self.self_loop_weight)
        return adj
