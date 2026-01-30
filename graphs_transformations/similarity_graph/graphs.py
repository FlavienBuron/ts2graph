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
        affinity=AffinitySpec(name=affinity, gamma=gamma),
        sparsifier=SparsifierSpec(name="topk", k=k),
    )
