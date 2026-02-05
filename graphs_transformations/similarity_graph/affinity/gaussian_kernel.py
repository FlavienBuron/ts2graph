import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("gaussian kernel")
class GaussianKernel(AffinityFunction):
    name = "gaussian kernel (RBF)"
    requires_non_negative = True
    epsilon = 1e-6

    def __init__(self, theta: str = "std", **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta = theta

    def __call__(self, D: torch.Tensor):
        valid = torch.isfinite(D) & (D > 0)
        if self.theta == "median":
            theta = D[valid].median()
        else:
            theta = D[valid].std()
        theta = theta.clamp_min(self.epsilon)
        A = torch.exp(-((D / theta) ** 2))
        A[~valid] = 0.0
        return A
