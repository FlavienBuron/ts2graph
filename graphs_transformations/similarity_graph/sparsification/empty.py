import torch

from ..specs.registry import (
    register_sparsification,
)
from .base import (
    SparsificationFunction,
)


@register_sparsification("empty")
class Empty(SparsificationFunction):
    name = "empty"

    def __init__(
        self,
        binary: bool = True,
        keep_self_loop: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.binary = binary
        self.keep_self_loop = keep_self_loop

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        adj = torch.zeros_like(A)
        if self.keep_self_loop:
            adj.fill_diagonal_(1.0)
        return adj
