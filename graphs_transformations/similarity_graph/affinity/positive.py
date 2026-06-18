import torch

from ..specs.registry import register_affinity
from .base import AffinityFunction


@register_affinity("positive")
class Positive(AffinityFunction):
    name = "positive"
    requires_non_negative = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, D: torch.Tensor) -> torch.Tensor:
        print(f"Affinity: {self.name}")
        valid = torch.isfinite(D) & (D > 0)

        D = self._normalize(D, valid)

        D.clamp_(min=0.0)

        return D
