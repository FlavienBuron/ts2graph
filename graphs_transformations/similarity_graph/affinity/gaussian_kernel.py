import torch

from graphs_transformations.similarity_graph.affinity.base import AffinityFunction
from graphs_transformations.similarity_graph.specs.registry import register_affinity


@register_affinity("gaussian kernel")
class GaussianKernel(AffinityFunction):
    name = "gaussian kernel (RBF)"
    requires_non_negative = True

    def __init__(self, theta: str = "madian") -> None:
        self.theta = theta

    def __call__(self, D: torch.Tensor):
        theta = D.median() if self.theta == "median" else D.std()
        return torch.exp(-((D / theta) ** 2))
