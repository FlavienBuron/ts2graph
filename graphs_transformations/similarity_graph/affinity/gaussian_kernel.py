import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("gaussian kernel")
class GaussianKernel(AffinityFunction):
    name = "gaussian kernel (RBF)"
    requires_non_negative = True
    epsilon = 1e-6

    def __init__(self, theta: str = "std") -> None:
        self.theta = theta

    def __call__(self, D: torch.Tensor):
        theta = D.median() if self.theta == "median" else D.std()
        theta = theta.clamp_min(self.epsilon)
        return torch.exp(-((D / theta) ** 2))
