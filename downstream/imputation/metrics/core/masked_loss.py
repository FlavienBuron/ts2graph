from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class MaskedLoss(nn.Module, ABC):
    def __init__(
        self,
        mask_nans: bool = False,
        mask_inf: bool = False,
        reduction: str = "mean",
        eps: float = 1e-6,
        at: Optional[int] = None,
    ) -> None:
        super(MaskedLoss, self).__init__()
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        self.reduction = reduction
        self.eps = eps
        self.at = slice(None) if at is None else slice(at, at + 1)

        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Invalid reduction: {reduction}")

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Slice if needed
        prediction = prediction[:, self.at]
        target = target[:, self.at]

        # Compute elementwise loss
        value = self._elementwise(prediction, target)  # [B, H, W, C]

        # Build mask
        mask = self._build_mask(value, mask)  # same shape as value

        # Apply mask
        value = torch.where(mask, value, torch.zeros_like(value))

        # Sum reduction over all elements
        if self.reduction == "sum":
            return value.sum()

        # Per-sample reduction: flatten spatial dims, keep batch
        value = value.view(value.size(0), -1).sum(dim=1)

        # Denominator: number of valid pixels per sample
        denom = mask.view(mask.size(0), -1).sum(dim=1).clamp_min(self.eps)

        # Mean over batch
        return (value / denom).mean()

    def _build_mask(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(values, dtype=torch.bool)
        else:
            if mask.shape != values.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match values {values.shape}"
                )
            mask = mask.bool()

        if self.mask_nans:
            mask &= ~torch.isnan(values)
        if self.mask_inf:
            mask &= ~torch.isinf(values)

        return mask

    @abstractmethod
    def _elementwise(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pass
