from typing import Literal, Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction

GapMode = Literal["zero", "mean"]


@register_distance("erp")
class ERP(DistanceFunction):
    name = "erp"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(
        self,
        gap_value: GapMode = "zero",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gap_value = gap_value

    def _compute_gap(self, X: torch.Tensor) -> float:
        if self.gap_value == "zero":
            return 0.0
        elif self.gap_value == "mean":
            return X[X.isfinite()].mean().item()
        else:
            raise ValueError(f"Gap value must be 'zero' or 'mean', got {self.gap_value}")

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:

        T, N, F = X.shape
        gap_value = self._compute_gap(X)

        D = torch.full(
            (N, N),
            float("inf"),
        )

        for i in range(N):
            Xi = X[:, i, :].clone()

            for j in range(i + 1, N):
                Xj = X[:, j, :].clone()

                if mask is not None:
                    Mi = mask[:, i, :].bool()
                    Mj = mask[:, j, :].bool()

                    Xi = Xi.clone()
                    Xj = Xj.clone()

                    Xi[~Mi] = gap_value
                    Xj[~Mj] = gap_value

                if F == 1:
                    Xi = Xi.squeeze(-1)
                    Xj = Xj.squeeze(-1)

                n, m = Xi.shape[0], Xj.shape[1]

                costs = torch.full(
                    (n + 1, m + 1),
                    float("inf"),
                )

                costs[0, 0] = 0.0

                for p in range(1, n + 1):
                    costs[p, 0] = costs[p - 1, 0] + (Xi[p - 1] - gap_value).abs()

                for q in range(1, m + 1):
                    costs[0, q] = costs[0, q - 1] + (Xj[q - 1] - gap_value).abs()

                for p in range(1, n + 1):
                    for q in range(1, m + 1):
                        cost = (Xi[p - 1] - Xj[q - 1]).abs()

                        costs[p, q] = (
                            torch.stack(
                                [
                                    costs[p - 1, q - 1] + cost,
                                    costs[p - 1, q] + (Xi[p - 1] - gap_value).abs(),
                                    costs[p, q - 1] + (Xj[q - 1] - gap_value).abs(),
                                ]
                            )
                        ).min()

                    D[i, j] = D[j, i] = costs[n, m]

        D.fill_diagonal_(0.0)

        return D
