from typing import Dict, Type

DISTANCE_REGISTRY: Dict[str, Type] = {}
AFFINITY_REGISTRY: Dict[str, Type] = {}
SPARSITY_REGISTRY: Dict[str, Type] = {}


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
        SPARSITY_REGISTRY[name] = cls
        return cls

    return decorator
