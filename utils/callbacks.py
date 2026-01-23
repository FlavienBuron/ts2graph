import pytorch_lightning as pl
import torch

from downstream.imputation.helpers import EpochReport


class ConsoleMetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_metrics = {k: v.item() for k, v in metrics.items()}
        ordered = sorted(val_metrics.items())

        line = (
            f"[Epoch {trainer.current_epoch}] "
            # f"train_mae={metrics['train_mae']:.3f} | "
            # f"val_mae={metrics['val_mae']:.3f}"
        )
        line += " | ".join(f"{k}={v:.3f}" for k, v in ordered)

        trainer.print(line)


class EpochReportCallback(pl.Callback):
    def __init__(self, report: EpochReport):
        super().__init__()
        self.report = report

    def _collect(self, trainer, phase: str):
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            if not k.startswith(phase):
                continue
            k = k.replace(phase + "_", "")
            if torch.is_tensor(v):
                v = v.detach().cpu().item()
            metrics[k] = float(v)
        return metrics

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = self._collect(trainer, "train")
        self.report.add(
            phase="train",
            epoch=trainer.current_epoch,
            **metrics,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = self._collect(trainer, "val")
        self.report.add(
            phase="val",
            epoch=trainer.current_epoch,
            **metrics,
        )

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = self._collect(trainer, "test")
        self.report.add(
            phase="test",
            epoch=trainer.current_epoch,
            **metrics,
        )
