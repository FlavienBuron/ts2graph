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
        param: dict,
        binary: bool = False,
        keep_self_loop: bool = False,
        make_symmetric: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.param_cfg = param
        self.binary = binary
        self.keep_self_loop = keep_self_loop
        self.make_symmetric = make_symmetric

    def _resolve_k(self, num_nodes: int) -> int:
        mode = self.param_cfg["mode"]
        value = self.param_cfg["value"]

        if mode == "absolute":
            return int(value)

        elif mode == "fraction":
            if not (0.0 < value <= 1.0):
                raise ValueError("Fractional k must be in ")
            max_k = num_nodes - 1
            k = round(value * max_k)
            print(f"{value=} {num_nodes=} {max_k=} {k=}")
            return max(0, min(k, max_k))
        else:
            raise ValueError(f"Unknown mode for k value resolutoin: {mode}")

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        N = A.size(0)

        k = self._resolve_k(N)

        diag = A.diagonal().clone()

        A.fill_diagonal_(-torch.inf)

        _, idx = torch.topk(A, k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(1, idx, 1)

        A_sparse = A * mask

        if self.keep_self_loop:
            A_sparse.diagonal().copy_(diag)
        else:
            A_sparse.fill_diagonal_(0.0)

        if self.make_symmetric:
            A_sparse = torch.maximum(A_sparse, A_sparse.T)

        if self.binary:
            A_sparse = (A_sparse > 0).to(A.dtype)

        return A_sparse
