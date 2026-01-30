import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional


@dataclass
class DistanceSpec:
    name: Literal["masked euclidean", "normalized masked euclidean"]
    normalize: bool = True

    def build(self):
        from .registry import DISTANCE_REGISTRY

        cls = DISTANCE_REGISTRY[self.name]
        kwargs = {k: v for k, v in vars(self).items() if k != "name" and v is not None}
        print(f"DEBUG: {kwargs=}")
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DistanceSpec":
        return cls(**d)


@dataclass
class AffinitySpec:
    name: Literal["gaussian kernel"]
    gamma: Optional[float] = None
    epsilon: float = 1e-6

    def build(self):
        from .registry import AFFINITY_REGISTRY

        cls = AFFINITY_REGISTRY[self.name]
        kwargs = {k: v for k, v in vars(self).items() if k != "name" and v is not None}
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AffinitySpec":
        return cls(**d)


@dataclass
class SparsifierSpec:
    name: Literal["topk"]
    k: Optional[int] = None
    radius: Optional[float] = None
    threshold: Optional[float] = None
    symmetric: bool = True

    def build(self):
        from .registry import SPARCITY_REGISTRY

        cls = SPARCITY_REGISTRY[self.name]
        kwargs = {k: v for k, v in vars(self).items() if k != "name" and v is not None}
        return cls(**kwargs)

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
