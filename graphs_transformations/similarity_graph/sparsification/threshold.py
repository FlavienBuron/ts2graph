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

    def __init__(
        self,
        threshold: float,
        binary: bool = False,
        keep_self_loop: bool = False,
        make_symmetric: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.binary = binary
        self.keep_self_loop = keep_self_loop
        self.make_symmetric = make_symmetric

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        mask = A >= self.threshold
        adj = A * mask
        if self.binary:
            adj = (adj > 0).to(A.dtype)
        if not self.keep_self_loop:
            adj.fill_diagonal_(0.0)
        if self.make_symmetric:
            adj = torch.maximum(adj, adj.T)

        return adj
