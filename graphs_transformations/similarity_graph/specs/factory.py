from ..pipeline import SimilarityGraph
from .specs import SimilarityGraphSpec


def build_graph_constructor(specs: SimilarityGraphSpec) -> SimilarityGraph:
    distance = specs.distance.build()
    affinity = specs.affinity.build()
    sparsifier = specs.sparsifier.build()

    return SimilarityGraph(distance=distance, affinity=affinity, sparsifier=sparsifier)
