from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("sliding_euclidean")
class SlidingEuclidean(DistanceFunction):
    name = "sliding euclidean"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, lag_fraction: int, min_overlap: int = 1, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lag_fraction = lag_fraction
        self.min_overlap = min_overlap
        self.normalize = normalize

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        if mask is None:
            raise ValueError("Masked Euclidean Distance requires a masked to be passed")
        T, N, F = X.shape
        max_lag = int(round(self.lag_fraction * T))
        lags = torch.arange(-max_lag, max_lag + 1)
        D = torch.full((N, N), float("inf"))

        # precompute squared values for efficiency
        X = X.squeeze(-1) if F == 1 else X
        mask = mask.squeeze(-1).bool()

        for i in range(N):
            Xi = X[:, i]
            Mi = mask[:, i]

            for j in range(i + 1, N):
                Xj = X[:, j]
                Mj = mask[:, j]

                best = float("inf")

                for tau in lags.tolist():
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

                    diff = Xi_tau[Mij] - Xj_tau[Mij]
                    square = (diff * diff).sum().item()
                    if self.normalize:
                        Dij = (square / K) ** 0.5
                    else:
                        Dij = square**0.5

                    best = min(best, float(Dij))

                D[i, j] = D[j, i] = best
        D.fill_diagonal_(0.0)
        return D
