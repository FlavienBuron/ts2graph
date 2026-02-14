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

    def _resolve_tau(self, A: torch.Tensor):
        mode = self.param_cfg["mode"]
        value = self.param_cfg["value"]

        if mode == "absolute":
            return value

        elif mode == "percentile":
            # global percentile threshold
            tau = torch.quantile(A.flatten(), value)
            return tau

        elif mode == "adaptive":
            # per-node mean + alpha * std
            mean = A.mean(dim=-1, keepdim=True)
            std = A.std(dim=-1, keepdim=True)
            tau = mean + value * std
            return tau

        else:
            raise ValueError(f"Unknown tau mode: {mode}")

    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        tau = self._resolve_tau(A)

        mask = A >= tau
        adj = A * mask
        if self.binary:
            adj = (adj > 0).to(A.dtype)
        if not self.keep_self_loop:
            adj.fill_diagonal_(0.0)
        if self.make_symmetric:
            adj = torch.maximum(adj, adj.T)

        return adj
