import torch

from ..specs.registry import register_distance
from .base import DistanceFunction


@register_distance("masked euclidean")
class MaskedEuclidean(DistanceFunction):
    name = "masked euclidean"
    symmetric = True
    non_negative = True
    supports_mask = True
    bounded = False

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def __call__(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
