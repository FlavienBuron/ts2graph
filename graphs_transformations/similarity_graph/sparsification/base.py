from abc import ABC, abstractmethod

import torch


class SparsificationFunction(ABC):
    name: str
    preserves_symmetry: bool = True
    preserves_weights: bool = True

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, A: torch.Tensor) -> torch.Tensor:
        """Return the sparse adjacency matrix from the given affinity matrix A"""
