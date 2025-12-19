import os
from typing import List, Optional

import torch

from datasets.dataloaders.graphloader import GraphLoader
from downstream.imputation.imputer import Imputer
from utils.progess import ProgressDisplay


class Trainer:
    def __init__(
        self,
        imputer: Imputer,
        dataloader: GraphLoader,
        max_epochs: int = 100,
        patience: Optional[int] = None,
        accumulate_grad_batches: int = 1,
        grad_clip_val: Optional[float] = None,
        grad_clip_algorithm: str = "norm",
        default_root_dir: str = ".",
        save_name: str = "best_model.pt",
        verbose: bool = True,
    ) -> None:
        self.imputer = imputer
        self.torch_model = self.imputer.model
        self.dataloader = dataloader
        self.max_epochs = max_epochs
        self.patience = patience
        self.accumulate_grad_batches = accumulate_grad_batches
        self.grad_clip_val = grad_clip_val
        self.grad_clip_algorithm = grad_clip_algorithm
        self.default_root_dir = default_root_dir
        os.makedirs(self.default_root_dir, exist_ok=True)
        self.save_path = os.path.join(self.default_root_dir, save_name)
        self.verbose = verbose

        self.train_loader = self.dataloader._train_dataloaders(seed=1)
        self.val_loader = self.dataloader._val_dataloader()
        self.test_loader = self.dataloader._test_dataloader()

        optim_class = getattr(self.imputer, "optim_class", None)
        optim_kwargs = getattr(self.imputer, "optim_kwargs", None)
        if optim_class is None or optim_kwargs is None:
            raise ValueError(
                "Model must provide 'optim_class' and 'optim_kwargs' attributes"
            )
        self.optimizer = optim_class(self.torch_model.parameters(), **optim_kwargs)

        scheduler_class = getattr(self.torch_model, "scheduler_class", None)
        scheduler_kwargs = getattr(self.torch_model, "scheduler_kwargs", dict())
        self.scheduler = (
            scheduler_class(self.optimizer, **scheduler_kwargs)
            if scheduler_class is not None
            else None
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(self) -> List[float]:
        self.torch_model.train()
        torch.set_grad_enabled(True)
        batch_losses: List[float] = []

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.imputer.training_step(batch, batch_idx)
            if loss is None:
                raise RuntimeError("Training step returned None loss")

            # scale gradient for accumulation
            loss_scaled = loss / float(self.accumulate_grad_batches)

            # zero grads at accumulation start
            if (batch_idx % self.accumulate_grad_batches) == 0:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    raise self.optimizer.zero_grad()

            loss_scaled.backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                params = [p for p in self.torch_model.parameters() if p.requires_grad]
                if (self.grad_clip_val is not None) and params:
                    if self.grad_clip_algorithm == "norm":
                        torch.nn.utils.clip_grad_norm_(params, self.grad_clip_val)
                    elif self.grad_clip_algorithm == "value":
                        torch.nn.utils.clip_grad_value_(params, self.grad_clip_val)
                    else:
                        raise ValueError(f"Unknown {self.grad_clip_algorithm=}")

                self.optimizer.step()
                self.global_step += 1

            batch_losses.append(float(loss.detach().cpu().item()))

        return batch_losses

    def validate(self) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.torch_model.eval()
        torch.set_grad_enabled(False)

        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss = self.imputer.validation_step(batch, batch_idx)
            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else None

    def test(
        self,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.test_loader is None:
            return None, None, None
        self.torch_model.eval()
        torch.set_grad_enabled(False)

        batch_losses: List[float] = []
        preds_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        for batch_idx, batch in enumerate(self.test_loader):
            test_output = self.imputer.test_step(batch, batch_idx)

            test_loss = test_output["loss"]
            preds = test_output["preds"]
            target = test_output["target"]
            mask = test_output["mask"]

            batch_losses.append(test_loss)
            preds_list.append(preds)
            targets_list.append(target)
            mask_list.append(mask)

        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)
        all_masks = torch.cat(mask_list, dim=0)

        return all_preds, all_targets, all_masks

    def run(self):
        if self.train_loader is None:
            return None
        all_train_losses = []
        val_losses = []

        with ProgressDisplay(self.max_epochs) as p:
            for epoch in range(self.max_epochs):
                self.imputer.reset_metrics()
                p.set_phase("training")

                batch_losses = self.train_epoch()
                train_loss = sum(batch_losses) / len(batch_losses)
                train_metrics = self.imputer.compute_metrics(phase="train")

                all_train_losses.append(batch_losses)
                p.log_train(train_loss, train_metrics)

                self.imputer.reset_metrics()
                p.set_phase("validating")

                val_loss = self.validate()
                val_metrics = self.imputer.compute_metrics(phase="val")
                val_losses.append(val_loss)

                val_loss = val_loss if val_loss is not None else 0.0

                p.log_val(val_loss, val_metrics)

                if self.scheduler is not None:
                    try:
                        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                    except Exception:
                        try:
                            self.scheduler.step(val_loss)
                        except Exception:
                            print("Scheduler step failed; skipped")

                if val_loss is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        torch.save(self.torch_model.state_dict(), self.save_path)
                        print("Saved new best model")
                    else:
                        if self.patience is not None:
                            self.patience_counter += 1
                            if self.patience_counter >= self.patience:
                                print(
                                    f"Early stopping triggered (patience={self.patience})"
                                )
                                break
                p.nex_epoch()

            self.imputer.reset_metrics()
            p.set_phase("testing")

            preds, targets, masks = self.test()
            test_metrics = self.imputer.compute_metrics(phase="test")
            p.log_test(test_metrics)

        final_val = self.best_val_loss if self.best_val_loss != float("inf") else None
        return {
            "train_losses": all_train_losses,
            "val_losses": val_losses,
            "best_val": final_val,
            "test": {
                "target": targets,
                "prediction": preds,
                "mask": masks,
            },
        }
