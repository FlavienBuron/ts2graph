from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("masked_euclidean")
class MaskedEuclidean(DistanceFunction):
    name = "masked euclidean"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.normalize = normalize

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            raise ValueError("Masked Euclidean Distance requires a masked to be passed")
        _, N, _ = X.shape
        D = torch.full((N, N), float("inf"))

        for i in range(N):
            Xi = X[:, i, :]
            Mi = mask[:, i, :]
            for j in range(i + 1, N):
                Xj = X[:, j, :]
                Mj = mask[:, j, :]

                Mij = Mi & Mj
                K = Mij.sum()
                if K == 0:
                    continue

                diff = Xi[Mij] - Xj[Mij]
                if self.normalize:
                    Dij = torch.linalg.norm(diff) / torch.sqrt(K)
                else:
                    Dij = torch.linalg.norm(diff)
                D[i, j] = D[j, i] = Dij
        D.fill_diagonal_(0.0)
        return D
