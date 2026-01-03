from functools import partial
from typing import Optional, Tuple

import torch
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


class MaskedMetric(Metric):
    def __init__(
        self,
        metric_fn,
        mask_nans: bool = False,
        mask_inf: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group=None,
        dist_sync_fn=None,
        metric_kwargs=None,
        at=None,
    ):
        super(MaskedMetric, self).__init__(
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        metric_kwargs = dict() if metric_kwargs is None else metric_kwargs
        self.metric_fn = partial(metric_fn, **metric_kwargs)
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        self.at = slice(None) if at is None else slice(at, at + 1)
        self.add_state("value", dist_reduce_fx="sum", default=torch.tensor(0.0).float())
        self.add_state("numel", dist_reduce_fx="sum", default=torch.tensor(0))

    def _check_mask(
        self, mask: Optional[torch.Tensor], values: torch.Tensor
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(values)
        else:
            _check_same_shape(mask, values)
        if self.mask_nans:
            mask = mask * ~torch.isnan(values)
        if self.mask_inf:
            mask = mask * ~torch.isinf(values)
        return mask.bool()

    def _compute_masked(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _check_same_shape(prediction, target)
        print(
            f"{prediction.min()=} {prediction.max()=} {prediction.mean()=} {prediction.std()=} {prediction.sum()=}"
        )
        print(
            f"{target.min()=} {target.max()=} {target.mean()=} {target.std()=} {target.sum()=}"
        )
        value = self.metric_fn(prediction, target)
        mask = self._check_mask(mask, value)
        print(f"1. compute masked {value.sum()=} {mask.sum()}")
        value = torch.where(mask, value, torch.tensor(0.0, device=value.device).float())
        print(f"2. compute masked {value.sum()=} {mask.sum()}")
        # value = value * mask.to(value.dtype)
        return value.sum(), mask.sum(), None

    def _compute_std(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        _check_same_shape(prediction, target)
        value = self.metric_fn(prediction, target)
        return value.sum(), value.numel(), None

    def is_masked(self, mask) -> bool:
        return self.mask_inf or self.mask_nans or (mask is not None)

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        prediction = prediction[:, self.at]
        target = target[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        else:
            raise ValueError("Mask is None")
        if self.is_masked(mask):
            value, numel, _ = self._compute_masked(prediction, target, mask)
        else:
            value, numel, _ = self._compute_std(prediction, target)
        self.value += value
        self.numel += numel
        # print(
        #     f"Update: {prediction.mean()=} {target.mean()=} {mask.float().mean()=} {value=} {numel=} {self.value=} {self.numel=}"
        # )

    def compute(self) -> torch.Tensor:
        if self.numel > 0:
            return self.value / self.numel
        return self.value
