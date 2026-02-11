import torch
from omegaconf import DictConfig

from datasets.dataloaders.graphloader import GraphLoader
from graphs_transformations.similarity_graph.specs.specs import (
    AffinitySpec,
    DistanceSpec,
    SimilarityGraphSpec,
    SparsifierSpec,
)

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

    def __call__(self, dataset: GraphLoader) -> torch.Tensor:
        if self.distance.input_kind == "series":
            x, mask = dataset.training_data
        elif self.distance.input_kind == "coordinates":
            x = dataset.distances.to_numpy()
            mask = None
        else:
            raise ValueError("Unknown input kind")
        D = self.distance(x, mask=mask)
        A = self.affinity(D)
        A = self.sparsifier(A)
        return A


def build(cfg: DictConfig) -> SimilarityGraph:
    return SimilarityGraphSpec(
        distance=DistanceSpec(**cfg.graph.distance),
        affinity=AffinitySpec(**cfg.graph.affinity),
        sparsifier=SparsifierSpec(**cfg.graph.sparsifier),
    ).build()
