import json
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class DistanceSpec:
    def __init__(self, name: Literal["masked euclidean", "identity"], **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def build(self):
        from .registry import DISTANCE_REGISTRY

        cls = DISTANCE_REGISTRY[self.name]
        return cls(**self.kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DistanceSpec":
        return cls(**d)


@dataclass
class AffinitySpec:
    def __init__(self, name: Literal["gaussian kernel"], **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def build(self):
        from .registry import AFFINITY_REGISTRY

        cls = AFFINITY_REGISTRY[self.name]
        return cls(**self.kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AffinitySpec":
        return cls(**d)


@dataclass
class SparsifierSpec:
    def __init__(self, name: Literal["topk", "threshold"], **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    def build(self):
        from .registry import SPARCITY_REGISTRY

        cls = SPARCITY_REGISTRY[self.name]
        return cls(**self.kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SparsifierSpec":
        return cls(**d)


@dataclass
class SimilarityGraphSpec:
    """Complete pipeline configuration."""

    distance: DistanceSpec
    affinity: AffinitySpec
    sparsifier: SparsifierSpec

    def build(self):
        from .factory import build_graph_constructor

        return build_graph_constructor(self)

    def to_dict(self) -> dict:
        """Serialize to nested dictionary."""
        return {
            "distance": self.distance.to_dict(),
            "affinity": self.affinity.to_dict(),
            "sparsifier": self.sparsifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimilarityGraphSpec":
        """Deserialize from nested dictionary."""
        return cls(
            distance=DistanceSpec.from_dict(d["distance"]),
            affinity=AffinitySpec.from_dict(d["affinity"]),
            sparsifier=SparsifierSpec.from_dict(d["sparsifier"]),
        )

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimilarityGraphSpec":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
