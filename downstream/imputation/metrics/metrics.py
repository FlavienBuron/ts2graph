import torch
import torch.nn.functional as F
from torchmetrics.utilities.checks import _check_same_shape

from downstream.imputation.metrics.core.base import mae, mape, mre, mse
from downstream.imputation.metrics.core.masked_metric import MaskedMetric


class MaskedMAE(MaskedMetric):
    def __init__(
        self,
        mask_nans=False,
        mask_inf=False,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        at=None,
    ):
        super(MaskedMAE, self).__init__(
            metric_fn=mae,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            metric_kwargs={"reduction": "none"},
            at=at,
        )


class MaskedMAPE(MaskedMetric):
    def __init__(
        self,
        mask_nans=False,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        at=None,
    ):
        super(MaskedMAPE, self).__init__(
            metric_fn=mape,
            mask_nans=mask_nans,
            mask_inf=True,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            at=at,
        )


class MaskedMSE(MaskedMetric):
    def __init__(
        self,
        mask_nans=False,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        at=None,
    ):
        super(MaskedMSE, self).__init__(
            metric_fn=mse,
            mask_nans=mask_nans,
            mask_inf=True,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            metric_kwargs={"reduction": "none"},
            at=at,
        )


class MaskedMRE(MaskedMetric):
    def __init__(
        self,
        mask_nans=False,
        mask_inf=False,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        at=None,
    ):
        super(MaskedMRE, self).__init__(
            metric_fn=F.l1_loss,
            mask_nans=mask_nans,
            mask_inf=mask_inf,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            metric_kwargs={"reduction": "none"},
            at=at,
        )
        self.add_state(
            "tot", dist_reduce_fx="sum", default=torch.tensor(0.0, dtype=torch.float)
        )

    def _compute_masked(self, prediction, target, mask):
        _check_same_shape(prediction, target)
        val = self.metric_fn(prediction, target)
        mask = self._check_mask(mask, val)
        # print(f"DEBUG: MRE {val.sum()=}")
        val = torch.where(
            mask, val, torch.tensor(0.0, device=target.device, dtype=torch.float)
        )
        target_masked = torch.where(
            mask, target, torch.tensor(0.0, device=target.device, dtype=torch.float)
        )
        # print(f"MRE {val.sum()=} {mask.sum()=} {target_masked.sum()=}")
        return val.sum(), mask.sum(), target_masked.sum()

    def _compute_std(self, prediction, target):
        _check_same_shape(prediction, target)
        val = self.metric_fn(prediction, target)
        return val.sum(), val.numel(), target.sum()

    def compute(self):
        if self.tot > 1e-6:
            return self.value / self.tot
        return self.value

    def update(self, prediction, target, mask=None):
        prediction = prediction[:, self.at]
        target = target[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val, numel, tot = self._compute_masked(prediction, target, mask)
        else:
            val, numel, tot = self._compute_std(prediction, target)
        self.value += val
        self.numel += numel
        self.tot += tot


class MaskedMRE2(MaskedMetric):
    def __init__(
        self,
        mask_nans=False,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
        at=None,
    ):
        super(MaskedMRE2, self).__init__(
            metric_fn=mre,
            mask_nans=mask_nans,
            mask_inf=True,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            metric_kwargs={"reduction": "none"},
            at=at,
        )
