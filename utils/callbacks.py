import pytorch_lightning as pl


class ConsoleMetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_metrics = {k: v.item() for k, v in metrics.items()}
        ordered = sorted(val_metrics.items())

        line = (
            f"[Epoch {trainer.current_epoch}]"
            # f"train_mae={metrics['train_mae']:.3f} | "
            # f"val_mae={metrics['val_mae']:.3f}"
        )
        line += " | ".join(f"{k}={v:.3f}" for k, v in ordered)
        from pytorch_lightning.utilities import rank_zero_info

        rank_zero_info(line)
