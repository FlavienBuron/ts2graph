import torch

from .affinity.base import AffinityFunction
from .distance.base import DistanceFunction
from .sparsification.base import (
    SparsificationFunction,
)


class SimilarityGraph:
    def __init__(
        self,
        distance: DistanceFunction,
        affinity: AffinityFunction,
        sparsifier: SparsificationFunction,
    ):
        self.distance = distance
        self.affinity = affinity
        self.sparsifier = sparsifier
        self._validate()

    def _validate(self):
        if self.affinity.requires_non_negative and not self.distance.non_negative:
            raise ValueError(
                f"{self.affinity.name} incompatiable with {self.distance.name}"
            )

    def __call__(self, x, mask=None) -> torch.Tensor:
        D = self.distance(x, mask=mask)
        A = self.affinity(D)
        A = self.sparsifier(A)
        return A
