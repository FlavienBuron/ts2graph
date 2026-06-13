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
                best_K = 1

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

                    diff = Xi_tau[Mij] - Xj_tau[Mij]
                    square = diff.square().flatten()
                    running = 0.0
                    if self.normalize:
                        threshold = (best**2) * max(best_K, 1)
                    else:
                        threshold = best**2

                    for v in square:
                        running += float(v)
                        if running >= threshold:
                            break
                    else:
                        # full evaluation only if not pruned
                        if self.normalize:
                            Dij = (running / K) ** 0.5
                        else:
                            Dij = running**0.5

                        if Dij < best:
                            best = Dij
                            best_K = K
                D[i, j] = D[j, i] = best
        D.fill_diagonal_(0.0)
        return D
