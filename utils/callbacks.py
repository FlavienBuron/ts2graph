import pytorch_lightning as pl


class ConsoleMetricsCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        line = (
            f"[Epoch {trainer.current_epoch}]"
            # f"train_mae={metrics['train_mae']:.3f} | "
            f"{metric}: {value}"
            for metric, value in metrics.items()
        )
        from pytorch_lightning.utilities import rank_zero_info

        rank_zero_info(line)
