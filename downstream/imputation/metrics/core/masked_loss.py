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
        prediction = prediction[:, self.at]
        target = target[:, self.at]
        print(
            f"{prediction.min()=} {prediction.max()=} {prediction.mean()=} {prediction.std()=} {prediction.sum()=}"
        )
        print(
            f"{target.min()=} {target.max()=} {target.mean()=} {target.std()=} {target.sum()=}"
        )
        if mask is not None:
            mask = mask[:, self.at]

        # elementwise loss
        value = self._elementwise(prediction, target)

        # build mask
        mask = self._build_mask(value, mask)

        # apply mask
        value2 = torch.where(mask, value, torch.tensor(0)

        if self.reduction == "sum":
            return value2.sum()

        # mean
        denom = mask.sum().clamp_min(self.eps)
        print(
            f"{value.min()=} {value.max()=} {value.mean()=} {value.std()=} {value.sum()=}"
        )
        print(
            f"{value2.min()=} {value2.max()=} {value2.mean()=} {value2.std()=} {value2.sum()=}"
        )
        print(
            f"forward: {prediction.mean()=} {target.mean()=} {mask.float().mean()=} {value.mean()=} {value2.mean()=} {denom.mean()=}"
        )
        return value2.sum() / denom

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
