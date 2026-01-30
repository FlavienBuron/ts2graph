from typing import Dict, Type

# from ..affinity.base import AffinityFunction
# from ..distance.base import DistanceFunction
# from ..sparsification.base import SparsificationFunction

DISTANCE_REGISTRY: Dict[str, Type] = {}
AFFINITY_REGISTRY: Dict[str, Type] = {}
SPARCITY_REGISTRY: Dict[str, Type] = {}


def register_distance(name: str):
    def decorator(cls):
        DISTANCE_REGISTRY[name] = cls
        return cls

    return decorator


def register_affinity(name: str):
    def decorator(cls):
        AFFINITY_REGISTRY[name] = cls
        return cls

    return decorator


def register_sparsification(name: str):
    def decorator(cls):
        SPARCITY_REGISTRY[name] = cls
        return cls

    return decorator
