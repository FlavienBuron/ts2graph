import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("gaussian_kernel")
class GaussianKernel(AffinityFunction):
    name = "gaussian kernel (RBF)"
    requires_non_negative = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, D: torch.Tensor):
        print(f"Affinity: {self.name}")

        valid = torch.isfinite(D) & (D > 0)

        D = self._normalize(D, valid)

        A = torch.exp(-(D**2))

        A[~valid] = 0.0

        return A
