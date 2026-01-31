from typing import Literal

from .pipeline import SimilarityGraph
from .specs.specs import (
    AffinitySpec,
    DistanceSpec,
    SimilarityGraphSpec,
    SparsifierSpec,
)


def knn_graph(
    k: int,
    distance: Literal["masked euclidean", "identity"] = "masked euclidean",
    affinity: Literal["gaussian kernel"] = "gaussian kernel",
    **kwargs,
) -> SimilarityGraphSpec:
    return SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, **kwargs),
        affinity=AffinitySpec(name=affinity, **kwargs),
        sparsifier=SparsifierSpec(name="topk", k=k, **kwargs),
    )


def radius_graph(
    threshold: float,
    distance: Literal["masked euclidean", "identity"] = "masked euclidean",
    affinity: Literal["gaussian kernel"] = "gaussian kernel",
    **kwargs,
) -> SimilarityGraph:
    graph = SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, **kwargs),
        affinity=AffinitySpec(name=affinity, **kwargs),
        sparsifier=SparsifierSpec(name="threshold", threshold=threshold, **kwargs),
    )
    return graph.build()
