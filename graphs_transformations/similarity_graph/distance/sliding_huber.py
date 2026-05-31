from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("sliding_huber")
class SlidingHuber(DistanceFunction):
    name = "sliding huber"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, lag_fraction: int, min_overlap: int = 1, delta: float = 1.0, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lag_fraction = lag_fraction
        self.min_overlap = min_overlap
        self.delta = delta
        self.normalize = normalize
        self.eps = 1e-6

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        if mask is None:
            raise ValueError("Sliding Huber Distance requires a masked to be passed")
        T, N, _ = X.shape
        max_lag = int(round(self.lag_fraction * T))
        D = torch.full((N, N), float("inf"))

        for i in range(N):
            Xi = X[:, i, :]
            Mi = mask[:, i, :]

            for j in range(i + 1, N):
                Xj = X[:, j, :]
                Mj = mask[:, j, :]

                best = float("inf")

                for tau in range(-max_lag, max_lag + 1):
                    if tau >= 0:
                        Xi_tau = Xi[: T - tau]
                        Xj_tau = Xj[tau:]
                        Mi_tau = Mi[: T - tau]
                        Mj_tau = Mj[tau:]
                    else:
                        lag = -tau
                        Xi_tau = Xi[lag:]
                        Xj_tau = Xj[: T - lag]
                        Mi_tau = Mi[lag:]
                        Mj_tau = Mj[: T - lag]

                    Mij = Mi_tau & Mj_tau
                    K = Mij.sum()

                    if K < self.min_overlap:
                        continue

                    xi = Xi_tau[Mij]
                    xj = Xj_tau[Mij]

                    r = xi - xj
                    abs_r = r.abs()

                    loss = torch.where(
                        abs_r <= self.delta,
                        0.5 * r.square(),
                        self.delta * (abs_r - 0.5 * self.delta),
                    )

                    if self.normalize:
                        Dij = loss.mean()
                    else:
                        Dij = loss.sum()
                    best = min(best, float(Dij))
                D[i, j] = D[j, i] = best
        D.fill_diagonal_(0.0)
        return D
