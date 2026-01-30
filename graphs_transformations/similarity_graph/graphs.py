from graphs_transformations.similarity_graph.specs.specs import (
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
    print(f"Specs for KNN graph with {distance=} {affinity=}")
    return SimilarityGraphSpec(
        distance=DistanceSpec(name=distance, normalize=normalize),
        affinity=AffinitySpec(name=affinity, gamma=gamma),
        sparsifier=SparsifierSpec(name="topk", k=k),
    )
