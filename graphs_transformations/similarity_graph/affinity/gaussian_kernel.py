import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("gaussian_kernel")
class GaussianKernel(AffinityFunction):
    name = "gaussian kernel (RBF)"
    requires_non_negative = True
    epsilon = 1e-6

    def __init__(self, gamma: float = 1.0, theta: str = "std", **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = gamma
        self.theta = theta

    def _normalize(self, D: torch.Tensor, valid: torch.Tensor):
        scale = D[valid].median()
        return D / scale.clamp_min(self.epsilon)

    def __call__(self, D: torch.Tensor):
        print(f"Affinity: {self.name}")
        valid = torch.isfinite(D) & (D > 0)

        D = self._normalize(D, valid)

        if self.theta == "median":
            theta = D[valid].median()
        else:
            theta = D[valid].std()
        theta = theta.clamp_min(self.epsilon)
        A = torch.exp(-self.gamma * ((D / theta) ** 2))
        A[~valid] = 0.0
        return A
