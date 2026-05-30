from typing import Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("masked_canberra")
class MaskedCanberra(DistanceFunction):
    name = "masked canberra"
    input_kind = "series"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, normalize: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.normalize = normalize
        self.eps = 1e-6

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        if mask is None:
            raise ValueError("Masked Canberra Distance requires a masked to be passed")
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

                xi = Xi[Mij]
                xj = Xj[Mij]

                numerator = (xi - xj).abs()
                denominator = xi.abs() + xj.abs()

                terms = numerator / (denominator + self.eps)

                if self.normalize:
                    Dij = terms.mean()
                else:
                    Dij = terms.sum()
                D[i, j] = D[j, i] = Dij
        D.fill_diagonal_(0.0)
        return D
