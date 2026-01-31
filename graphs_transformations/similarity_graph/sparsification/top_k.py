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

    def __init__(
        self,
        k: int,
        binary: bool = False,
        keep_self_loop: bool = False,
        make_symmetric: bool = True,
    ):
        self.k = k
        self.binary = binary
        self.keep_self_loop = keep_self_loop
        self.make_symmetric = make_symmetric

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        N = A.size(0)

        k = min(self.k, N - 1)

        A_ori = A.clone()

        A.fill_diagonal_(-torch.inf)

        _, idx = torch.topk(A, k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, idx, 1)

        A_sparse = A * mask

        if self.keep_self_loop:
            A_sparse.diagonal().copy_(A_ori.diagonal())

        if self.make_symmetric:
            A_sparse = torch.maximum(A_sparse, A_sparse.T)

        if self.binary:
            A_sparse = (A_sparse > 0).to(A.dtype)

        return A_sparse
