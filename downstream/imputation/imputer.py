import inspect
from copy import deepcopy
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection

from downstream.imputation.metrics.core.masked_loss import MaskedLoss
from downstream.imputation.metrics.core.masked_metric import MaskedMetric

epsilon = 1e-6


class Imputer(pl.LightningModule):
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

    def reset_model(self):
        self.model = self.model_cls(**self.model_kwargs)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @staticmethod
    def _check_metric(metric, on_step=False):
        if not isinstance(metric, MaskedMetric) and not isinstance(metric, MaskedLoss):
            if "reduction" in inspect.getfullargspec(metric).args:
                metric_kwargs = {"reduction": "none"}
            else:
                metric_kwargs = dict()
            return MaskedMetric(
                metric, compute_on_step=on_step, metric_kwargs=metric_kwargs
            )
        return deepcopy(metric)

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

    def reset_metrics(self) -> None:
        if hasattr(self, "train_metrics") and self.train_metrics is not None:
            self.train_metrics.reset()
        if hasattr(self, "val_metrics") and self.val_metrics is not None:
            self.val_metrics.reset()
        if hasattr(self, "test_metrics") and self.test_metrics is not None:
            self.test_metrics.reset()

    def compute_metrics(self, phase: str) -> Dict:
        if phase == "train":
            metrics = self.train_metrics
        elif phase == "val":
            metrics = self.val_metrics
        elif phase == "test":
            metrics = self.test_metrics
        else:
            raise ValueError(f"Unknown phase: {phase}")

        computed = metrics.compute()

        return {k: v.item() for k, v in computed.items()}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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

    def predict_loader(
        self,
        loader,
        preprocess: bool = False,
    ):
        targets, imputations, masks = [], [], []

        for batch in loader:
            batch_data, batch_preprocessing = self._unpack_batch(batch)

            target = batch_data.get("y")
            eval_mask = batch_data.get("eval_mask")

            imputation, _ = self._predict_batch(batch, preprocess=preprocess)

            imputation = self._postprocess(imputation, batch_preprocessing)

            targets.append(target.detach() if target is not None else None)
            imputations.append(imputation.detach())

            masks.append(eval_mask.detach() if eval_mask is not None else None)

        targets_tensor = torch.cat(targets, dim=0)
        imputations_tensor = torch.cat(imputations, dim=0)
        masks_tensor = torch.cat(masks, dim=0)

        return targets_tensor, imputations_tensor, masks_tensor

    def _predict_batch(
        self,
        batch: Tuple[Dict, Dict],
        preprocess: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        if preprocess:
            x = batch_data.pop("x")
            x = self._preprocess(x, batch_preprocessing)
            imputation, prediction, _ = self.forward(x, **batch_data)
        else:
            imputation, prediction, _ = self.forward(**batch_data)
        return imputation, prediction

    def training_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        mask = batch_data["mask"].clone().detach()
        batch_data["mask"] = torch.bernoulli(
            mask.clone().detach().float() * self.keep_prob
        ).bool()
        eval_mask = batch_data.pop("eval_mask").detach().clone()
        print(f"{eval_mask.sum()=}")
        debug_mask_relationship(mask, eval_mask, "Before ")
        eval_mask = (mask | eval_mask) & ~batch_data["mask"]
        eval_mask = eval_mask.bool()
        print(f"{eval_mask.sum()=} {mask.sum()=} {batch_data['mask'].sum()=}")
        debug_mask_relationship(mask, eval_mask, "train mask")
        debug_mask_relationship(batch_data["mask"], eval_mask, "batch mask")

        y = batch_data.pop("y")

        imputation, prediction = self._predict_batch(batch, preprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for h, _ in enumerate(prediction):
                prediction[h] = self._postprocess(prediction[h], batch_preprocessing)

        mad = (imputation - target).abs()
        mad = mad[eval_mask].mean()
        print(f"DEBUG: MAD train {mad=}")
        mad2 = (imputation - target).abs()
        mad2 = mad2[mask].mean()
        print(f"DEBUG: MAD mask train {mad2=}")

        loss = self.loss_fn(imputation, target, mask)
        print(f"imputation loss: {loss}")
        for h, _ in enumerate(prediction):
            pred_loss = self.tradeoff * self.loss_fn(prediction[h], target, mask)
            print(f"predictions loss: {pred_loss}")
            loss += pred_loss

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        mad = (imputation - y).abs()
        mad = mad[eval_mask].mean()
        print(f"DEBUG: MAD train scaled {mad=}")
        print(f"{imputation.mean()=} {y.mean()=}")

        self.train_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(
            self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )

        self.log(
            "train_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        imputation, _ = self._predict_batch(batch, preprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(
            self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "val_loss",
            val_loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )

        return val_loss

    def test_step(self, batch, batch_idx) -> Dict:
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        imputation, _ = self._predict_batch(batch, preprocess=False)
        imputation_post = self._postprocess(imputation, batch_preprocessing)
        if batch_idx == 0:
            print(f"{y.mean()=} {imputation.mean()=} {imputation_post.mean()=}")
        test_loss = self.loss_fn(imputation_post, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(
            self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            "test_loss",
            test_loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )
        return {
            "loss": test_loss,
            "preds": imputation.detach().clone(),
            "target": y.detach().clone(),
            "mask": eval_mask.detach().clone(),
        }

    def predict_step(self, batch, batch_idx) -> Dict:
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        eval_mask = batch_data.pop("eval_mask", None)
        y = batch_data.pop("y")

        imputation, _ = self._predict_batch(batch, preprocess=False)
        imputation_post = self._postprocess(imputation, batch_preprocessing)
        if batch_idx == 0:
            print(f"{y.mean()=} {imputation.mean()=} {imputation_post.mean()=}")
        pred_loss = self.loss_fn(imputation_post, y, eval_mask)

        return {
            "loss": pred_loss,
            "imputation": imputation_post.detach().clone(),
            "target": y.detach().clone(),
            "mask": eval_mask.detach().clone(),
        }

    def configure_optimizers(self):
        optimizer = self.optim_class(self.model.parameters(), **self.optim_kwargs)

        if self.scheduler_class is None:
            return optimizer

        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # required for ReduceLROnPlateau
        }

    def on_after_backward(self):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.log(
            "grad_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )


def dump_module_attrs(module, tag=""):
    print(f"\n{'=' * 20} MODULE ATTRS [{tag}] {'=' * 20}")
    for k in sorted(vars(module).keys()):
        v = getattr(module, k)
        try:
            if torch.is_tensor(v):
                print(
                    f"{k}: tensor "
                    f"shape={tuple(v.shape)} "
                    f"mean={v.mean().item():.4g} "
                    f"std={v.std().item():.4g}"
                )
            else:
                print(f"{k}: {type(v).__name__} = {v}")
        except Exception as e:
            print(f"{k}: <error: {e}>")


def debug_mask_relationship(mask, eval_mask, name=""):
    # ensure boolean
    mask = mask.bool()
    eval_mask = eval_mask.bool()

    total = mask.numel()

    overlap = (mask & eval_mask).sum()
    mask_only = (mask & ~eval_mask).sum()
    eval_only = (~mask & eval_mask).sum()
    neither = (~mask & ~eval_mask).sum()

    print(f"\n=== MASK DEBUG {name} ===")
    print(f"total elements         : {total}")
    print(f"mask true              : {mask.sum().item()} ({mask.float().mean():.4f})")
    print(
        f"eval_mask true         : {eval_mask.sum().item()} ({eval_mask.float().mean():.4f})"
    )
    print(f"overlap (mask & eval)  : {overlap.item()}")
    print(f"mask only              : {mask_only.item()}")
    print(f"eval_mask only         : {eval_only.item()}")
    print(f"neither                : {neither.item()}")

    # logical relations
    print("mask == eval_mask      :", torch.equal(mask, eval_mask))
    print("eval ⊆ mask            :", eval_only.item() == 0)
    print("mask ⊆ eval            :", mask_only.item() == 0)
