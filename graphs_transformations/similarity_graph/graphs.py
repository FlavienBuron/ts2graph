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
) -> SimilarityGraph:
    graph = SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, params=kwargs),
        affinity=AffinitySpec(name=affinity, params=kwargs),
        sparsifier=SparsifierSpec(name="topk", params={"k": k, **kwargs}),
    )
    return graph.build()


def radius_graph(
    threshold: float,
    distance: Literal["masked euclidean", "identity"] = "masked euclidean",
    affinity: Literal["gaussian kernel"] = "gaussian kernel",
    **kwargs,
) -> SimilarityGraph:
    graph = SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, params=kwargs),
        affinity=AffinitySpec(name=affinity, params=kwargs),
        sparsifier=SparsifierSpec(
            name="threshold", params={"threshold": threshold, **kwargs}
        ),
    )
    return graph.build()
