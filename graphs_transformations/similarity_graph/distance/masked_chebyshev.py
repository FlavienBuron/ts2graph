from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("masked_chebyshev")
class MaskedChebyshev(DistanceFunction):
    name = "masked chebyshev"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        if mask is None:
            raise ValueError("Masked Chebyshev Distance requires a masked to be passed")
        _, N, _ = X.shape
        D = torch.full((N, N), float("inf"))

        for i in range(N):
            Xi = X[:, i, :]
            Mi = mask[:, i, :]

            for j in range(i + 1, N):
                Xj = X[:, j, :]
                Mj = mask[:, j, :]

                Mij = Mi & Mj

                if Mij.sum() == 0:
                    continue

                diff = (Xi[Mij] - Xj[Mij]).abs()
                Dij = diff.max()
                D[i, j] = D[j, i] = Dij
        D.fill_diagonal_(0.0)
        return D
