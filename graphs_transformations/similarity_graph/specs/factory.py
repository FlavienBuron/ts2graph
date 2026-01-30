from pipeline import SimilarityGraphContructor
from specs import SimilarityGraphSpec


def build_graph_constructor(specs: SimilarityGraphSpec) -> SimilarityGraphContructor:
    distance = specs.distance.build()
    affinity = specs.affinity.build()
    sparsifier = specs.sparsifier.build()

    return SimilarityGraphContructor(
        distance=distance, affinity=affinity, sparsifier=sparsifier
    )
