from .specs.specs import (
    AffinitySpec,
    DistanceSpec,
    SimilarityGraphSpec,
    SparsifierSpec,
)


def knn_graph(
    k: int,
    distance: str = "masked euclidean",
    affinity: str = "gaussian kernel",
    gamma: float = 1.0,
    normalize: bool = True,
) -> SimilarityGraphSpec:
    return SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, normalize=normalize),
        affinity=AffinitySpec(name=affinity),
        sparsifier=SparsifierSpec(name="topk", k=k),
    )


def radius_graph(
    threshold: float,
    distance: str = "masked euclidean",
    affinity: str = "gaussian kernel",
    gamma: float = 1.0,
    normalize: bool = True,
) -> SimilarityGraphSpec:
    return SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, normalize=normalize),
        affinity=AffinitySpec(name=affinity),
        sparsifier=SparsifierSpec(name="threshold", threshold=threshold),
    )
