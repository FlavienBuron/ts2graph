import inspect
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from torchmetrics import MetricCollection

from downstream.imputation.metrics.core.masked_loss import MaskedLoss
from downstream.imputation.metrics.core.masked_metric import MaskedMetric

epsilon = 1e-6


class Imputer:
    def __init__(
        self,
        model_class: type[torch.nn.Module],
        model_kwargs,
        optim_class,
        optim_kwargs,
        loss_fn,
        scaled_target=False,
        whiten_prob=0.05,
        pred_loss_weigth=1.0,
        warm_up=0,
        metrics: Optional[Dict] = None,
        scheduler_class=None,
        scheduler_kwargs=None,
    ) -> None:
        super(Imputer, self).__init__()
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = (
            scheduler_kwargs if scheduler_kwargs is not None else dict()
        )
        self.loss_fn = self._check_metric(loss_fn, on_step=True)
        self.scaled_target = scaled_target

        assert 0.0 <= whiten_prob <= 1.0
        self.keep_prob = 1.0 - whiten_prob
        metrics = metrics if metrics is not None else dict()
        self._set_metrics(metrics)
        self.model = self.model_class(**self.model_kwargs)

        self.tradeoff = pred_loss_weigth
        self.trimming = (warm_up, warm_up)

    @staticmethod
    def _check_metric(metric, on_step=False):
        print(f"{type(metric)=} {isinstance(metric, MaskedLoss)=}")
        if not isinstance(metric, MaskedMetric) or not isinstance(metric, MaskedLoss):
            if "reduction" in inspect.getfullargspec(metric).args:
                metric_kwargs = {"reduction": "none"}
            else:
                metric_kwargs = dict()
            return MaskedMetric(
                metric, compute_on_step=on_step, metric_kwargs=metric_kwargs
            )
        return deepcopy(metric)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            {
                f"train_{k}": self._check_metric(metric, on_step=True)
                for k, metric in metrics.items()
            }
        )
        self.val_metrics = MetricCollection(
            {f"val_{k}": self._check_metric(m) for k, m in metrics.items()}
        )
        self.test_metrics = MetricCollection(
            {f"test_{k}": self._check_metric(m) for k, m in metrics.items()}
        )

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0] : s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def _preprocess(self, data: torch.Tensor, batch_preprocessing: Dict):
        trend = batch_preprocessing.get("trend", 0.0)
        bias = batch_preprocessing.get("bias", 0.0)
        scale = batch_preprocessing.get("scale", 1.0)
        return (data - trend - bias) / (scale + epsilon)

    def _postprocess(self, data: torch.Tensor, batch_preprocessing: Dict):
        trend = batch_preprocessing.get("trend", 0.0)
        bias = batch_preprocessing.get("bias", 0.0)
        scale = batch_preprocessing.get("scale", 1.0)
        return data * (scale + epsilon) + bias + trend

    def _unpack_batch(self, batch: Tuple[Dict, Dict]):
        batch_data, batch_preprocessing = batch
        return batch_data, batch_preprocessing

    def _predict_batch(
        self,
        batch: Tuple[Dict, Dict],
        preprocess: bool = False,
        postprocess: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop("x")
            x = self._preprocess(x, batch_preprocessing)
            imputation, prediction, _ = self.forward(x, **batch_data)
        else:
            imputation, prediction, _ = self.forward(**batch_data)

        if postprocess:
            imputation = self._postprocess(imputation, batch_preprocessing)
            prediction = self._postprocess(prediction, batch_preprocessing)
        return imputation, prediction

    def training_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        mask = batch_data["mask"].clone().detach()
        batch_data["mask"] = torch.bernoulli(
            mask.clone().detach().float() * self.keep_prob
        ).byte()
        eval_mask = batch_data.pop("eval_mask")
        eval_mask = (mask | eval_mask) & ~batch_data["mask"].bool()

        y = batch_data.pop("y")

        imputation, prediction = self._predict_batch(
            batch, preprocess=False, postprocess=False
        )

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            prediction = self._postprocess(prediction, batch_preprocessing)

        print(f"{imputation.shape=} {target.shape=} {mask.shape=} {self.loss_fn=}")
        loss = self.loss_fn(imputation, target, mask)
        for pred in prediction:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)
        # TODO: self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # TODO: self.log()
        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        imputation, _ = self._predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        # TODO: self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # TODO: self.log()

        return val_loss

    def test_step(self, batch, batch_idx) -> Dict:
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        imputation, _ = self._predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        # TODO: self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return {
            "loss": test_loss,
            "preds": imputation.detach().clone(),
            "target": y.detach().clone(),
            "mask": eval_mask.detach().clone(),
        }
