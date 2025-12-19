from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressDisplay:
    def __init__(self, max_epochs: int) -> None:
        self.console = Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[status]}"),
            console=self.console,
        )
        self.max_epochs = max_epochs
        self.epoch_task = None

    def __enter__(self):
        self.progress.__enter__()
        self.epoch_task = self.progress.add_task(
            "[Epochs]", total=self.max_epochs, status="starting"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    def set_phase(self, phase: str):
        self.progress.update(self.epoch_task, status=phase)

    def log_train(self, loss: float, metrics: dict):
        self._update_status("train", loss, metrics)

    def log_val(self, loss: float, metrics: dict):
        self._update_status("val", loss, metrics)

    def log_test(self, metrics: dict):
        metric_str = self._format_metrics(metrics)
        self.progress.update(self.epoch_task, status=f"test | {metric_str}")

    def nex_epoch(self):
        self.progress.advance(self.epoch_task)

    def info(self, msg: str):
        self.console.print(msg)

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        return " ".join(f"{k}={v:.2f}" for k, v in metrics.items())

    def _update_status(self, phase: str, loss: float, metrics: dict) -> None:
        metrics_str = self._format_metrics(metrics)
        self.progress.update(
            self.epoch_task,
            status=f"{phase} loss={loss:.2f} | {metrics_str}",
        )

    @contextmanager
    def batch_bar(self, task: str, phase: str, total_batches: int):
        description = f"[{task}] {phase} batches"

        batch_task = self.progress.add_task(
            description,
            total=total_batches,
            status="running",
        )

        try:
            yield lambda n=1: self.progress.advance(batch_task, n)
        finally:
            self.progress.remove_task(batch_task)
