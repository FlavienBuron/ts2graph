from typing import Dict, Type

from ..affinity.base import AffinityFunction
from ..distance.base import DistanceFunction
from ..sparsification.base import SparsificationFunction

DISTANCE_REGISTRY: Dict[str, Type[DistanceFunction]] = {}
AFFINITY_REGISTRY: Dict[str, Type[AffinityFunction]] = {}
SPARCITY_REGISTRY: Dict[str, Type[SparsificationFunction]] = {}


def register_distance(name: str):
    def decorator(cls: Type[DistanceFunction]):
        DISTANCE_REGISTRY[name] = cls
        return cls

    return decorator


def register_affinity(name: str):
    def decorator(cls: Type[AffinityFunction]):
        AFFINITY_REGISTRY[name] = cls
        return cls

    return decorator


def register_sparsification(name: str):
    def decorator(cls: Type[SparsificationFunction]):
        SPARCITY_REGISTRY[name] = cls
        return cls

    return decorator
