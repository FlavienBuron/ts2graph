from typing import Literal, Optional

import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("pearson_similarity")
class PearsonSimilarity(DistanceFunction):
    name = "pearson similarity"
    input_kind = "series"
    symmetric = True
    non_negative = False
    supports_mask = True
    bounded = True

    def __init__(
        self,
        min_overlap: int = 2,
        as_a: Literal["distance", "similarity"] = "distance",
        keep: Literal["raw", "positive", "absolute"] = "positive",
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_overlap = min_overlap
        self.as_a = as_a
        self.keep = keep
        self.eps = eps

    def __call__(self, X: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(f"Distance: {self.name}")
        if mask is None:
            raise ValueError("Pearson Similarity requires a masked to be passed")

        _, N, F = X.shape

        X = X.squeeze(-1) if F == 1 else X
        mask = mask.squeeze(-1).bool()

        S = torch.zeros((N, N), device=X.device, dtype=X.dtype)

        for i in range(N):
            Xi = X[:, i]
            Mi = mask[:, i]

            S[i, i] = 1.0

            for j in range(i + 1, N):
                Xj = X[:, j]
                Mj = mask[:, j]

                Mij = Mi & Mj
                K = int(Mij.sum())

                if K < self.min_overlap:
                    sij = 0.0
                else:
                    xi = Xi[Mij]
                    xj = Xj[Mij]

                    xi = xi - xi.mean()
                    xj = xj - xj.mean()

                    numerator = (xi * xj).sum()

                    denominator = xi.square().sum().sqrt() * xj.square().sum().sqrt()

                    sij = numerator / (denominator + self.eps)

                S[i, j] = S[j, i] = sij

        # clamp for stability issues
        S.clamp_(-1.0, 1.0)

        if self.keep == "positive":
            S.clamp_(min=0.0)
        elif self.keep == "absolute":
            S.abs_()
        else:
            # Keep S raw
            pass

        if self.as_a == "distance":
            S = 1 - S

        S.fill_diagonal_(0.0)
        return S
