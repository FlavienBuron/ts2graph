import copy
import datetime
import json
import math
import os
import random
from functools import partial
from time import perf_counter
from typing import Callable, Dict, Optional

import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from datasets.dataloaders.graphloader import GraphLoader
from datasets.datamodule import DataModule
from datasets.dataset_registry import DatasetRegistry
from downstream.imputation.helpers import EpochReport
from downstream.imputation.imputer import Imputer
from downstream.imputation.metrics.correlations import (
    MaskedCCC,
    MaskedCosineSimilarity,
    MaskedLagCorrelation,
    MaskedPearson,
)
from downstream.imputation.metrics.losses import MaskedMAELoss
from downstream.imputation.metrics.metrics import (
    MaskedMAE,
    MaskedMAPE,
    MaskedMRE,
    MaskedMRE2,
    MaskedMSE,
    MaskedRMSE,
    MaskedSMAPE,
)
from downstream.imputation.models.GRIN.grin import GRINet
from downstream.imputation.models.STGI.stgi import STGI
from graphs_transformations.similarity_graph import similarity_graph
from graphs_transformations.temporal_graphs import k_hop_graph, recurrence_graph_rs
from graphs_transformations.ts2net import Ts2Net
from graphs_transformations.utils import (
    save_graph_characteristics,
)
from utils.callbacks import ConsoleMetricsCallback, EpochReportCallback
from utils.helpers import (
    aggregate_predictions,
    prediction_dataframe,
)

torch.set_num_threads(16)
torch.set_num_interop_threads(1)
#


def get_decay_function(name: Optional[str]) -> Optional[Callable[[int, int], float]]:
    """
    Returns a decay function given a string identifier.

    Supported:
    - 'none'           : constant weight of 1.0
    - 'exponential'    : 0.9 ** hop
    - 'inverse'        : 1 / hop
    - 'inverse_square' : 1 / hop**2
    - 'logarithmic'    : 1 / log(1 + hop)
    - 'linear'         : max(0, 1 - hop / max_hop) — requires lambda binding externally

    Returns None if name is None or 'none'.
    Raises ValueError for unsupported strings.
    """
    if name is None or name.lower() == "none":
        return None

    name = name.lower()
    if "exp" in name:
        return lambda hop, _: 0.9**hop
    elif "inv" in name:
        return lambda hop, _: 1.0 / hop if hop != 0 else 1.0
    elif "squ" in name:
        return lambda hop, _: 1.0 / (hop**2) if hop != 0 else 1.0
    elif "log" in name:
        return lambda hop, _: 1.0 / math.log1p(hop) if hop > 0 else 1.0
    elif "linear" in name:  # requires a max_hop context
        return lambda hop, max_hop: 1 - (hop - 1) / (max_hop)
    else:
        raise ValueError(f"Unsupported decay function: '{name}'")


def get_spatial_graph(dataset: GraphLoader, cfg: DictConfig) -> tuple[torch.Tensor, float]:
    start = perf_counter()
    graph = similarity_graph.build(cfg)
    end = perf_counter()
    adj_matrix = graph(dataset)
    total_time = end - start
    return adj_matrix, total_time


def get_temporal_graph_function(technique: str, parameter: list[float]) -> Callable:
    if "naive" in technique:
        print("Using Naive Temporal Graph")
        param = int(parameter[0])
        decay = str(parameter[1]) if len(parameter) > 1 else "none"
        decay_fn = get_decay_function(decay)
        return partial(k_hop_graph, k=param, decay=decay_fn)
    if "chunked" in technique:
        ts2net = Ts2Net()
        print("Using Chuncked Visual Temporal Graph")
        method = "hvg" if parameter[0] == 1 else "nvg"
        limit = int(parameter[1])
        window_size = int(parameter[2])
        stride = int(parameter[3]) if len(parameter) > 3 else window_size
        return partial(
            ts2net.chunked_tsnet_vg,
            window_size=window_size,
            stride=stride,
            method=method,
            limit=limit,
        )
    if "vis" in technique:
        ts2net = Ts2Net()
        print("Using Visual Temporal Graph")
        method = "hvg" if parameter[0] == 1 else "nvg"
        limit = parameter[1] if len(parameter) > 1 else None
        return partial(ts2net.tsnet_vg, method=method, limit=limit)
    if "rec" in technique or "rn" in technique:
        ts2net = Ts2Net()
        alpha = float(parameter[0])
        time_lag = int(parameter[1]) if len(parameter) > 1 else 1
        # embedding_dim = int(parameter[2]) if len(parameter) > 2 else None
        print("Using Reccurrent Temporal Graph")
        return partial(
            # ts2net.tsnet_rn,
            recurrence_graph_rs,
            radius=alpha,
            time_lag=time_lag,
            # embedding_dim=embedding_dim,
        )
    if "qn" in technique or "quant" in technique:
        ts2net = Ts2Net()
        breaks = int(parameter[0])
        print("Using Transition/Quantile Temporal Graph")
        return partial(ts2net.tsnet_qn, breaks=breaks)

    def empty_temporal_graph():
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)

    return empty_temporal_graph


def make_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print("#" * 100)
    make_deterministic(cfg.seed)

    # Registers a conditional resolver: ${cond:condition, true_val, false_val}
    OmegaConf.register_new_resolver(
        "cond",
        lambda cond, t, f: t if cond else f,
        use_cache=False,
        replace=True,
    )

    sparsifier = cfg.graph.sparsifier

    param_name = next(iter(sparsifier), None)  # first key in sparsifier dict
    print(f"DEBUG: {sparsifier=}")
    param_val = sparsifier["param"]
    if isinstance(param_val, DictConfig):
        cfg.graph.label = str(param_val.get("value", "null"))
    else:
        cfg.graph.label = str(param_val)

    res_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(res_cfg)

    metrics_data = {}
    metrics_data["config"] = res_cfg
    # save_path_dir = cfg.paths.save_path
    save_path_dir = os.path.join(
        cfg.paths.save_path,
        cfg.graph.distance.name,
        cfg.graph.affinity.name,
        cfg.graph.sparsifier.name,
    )
    print(f"[INFO]: save directory path is '{save_path_dir}'")
    save_file_name = cfg.paths.file_name
    print(f"[INFO]: save file name is '{save_file_name}'")
    save_file_path = os.path.join(save_path_dir, save_file_name)
    os.makedirs(save_path_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.paths.save_path, "resolved_config.yaml"))
    model = cfg.model.name.lower()

    print(f"{cfg.use_spatial=} {cfg.use_temporal=}")

    # dataset = get_dataset(cfg.dataset.name)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if isinstance(dataset_cfg, Dict):
        dataset = DatasetRegistry.get(dataset_cfg)
    else:
        raise TypeError("Dataset config should resolve to a Dict, got ", type(dataset_cfg))

    # Log injection info if enabled
    if cfg.dataset.get("missingness", {}).get("enabled", False):
        print("   Missingness injection ENABLED")
        print(f"   Target rate: {cfg.dataset.missingness.target_rate:.0%}")
        print(f"   Eval mask mode: {cfg.dataset.missingness.eval_mask_mode}")
        print(f"   Pattern: {cfg.dataset.missingness.pattern}")
    else:
        print("   Baseline mode (no injection)")

    train, val, test = dataset.grin_split(in_sample=cfg.dataset.in_sample)
    dm = DataModule(
        copy.deepcopy(dataset),
        train_indices=train,
        test_indices=test,
        val_indices=val,
        samples_per_epoch=cfg.training.samples_per_epoch,
        scaling_type=cfg.dataset.scaling_type,
        batch_size=cfg.training.batch_size,
    )
    # if out of sample in air, add values removed for evaluation in train set
    if "air" in cfg.dataset.name and not cfg.dataset.in_sample:
        dm.dataset.mask[dm.train_slice] |= dm.dataset.eval_mask[dm.train_slice]
    dataset.training_slice = dm.train_slice

    spatial_graph_time = 0.0
    if cfg.use_spatial:
        if cfg.dataset.get("missingness", {}).get("enabled", False):
            scenario_key = dataset._scenario.config.get_cache_key()
            OmegaConf.update(cfg, "graph.distance.scenario_key", scenario_key, force_add=True)
        spatial_adj_matrix, spatial_graph_time = get_spatial_graph(dataset, cfg)
    else:
        spatial_adj_matrix = torch.tensor([[]])

    if cfg.use_temporal:
        temporal_graph_fn = get_temporal_graph_function(
            "",
            [0.1],
        )
        # temporal_graph_fn = get_temporal_graph_function(
        #     temporal_graph_technique,
        #     temporal_graph_params,
        # )
    else:
        temporal_graph_fn = get_temporal_graph_function(
            "",
            [0.1],
        )

    metrics_data.update({"spatial_graph_time": spatial_graph_time})

    if cfg.graph_stats:
        save_stats_path = save_path_dir
        is_binary = cfg.graph.sparsifier.binary
        if cfg.use_spatial:
            save_path = os.path.join(
                save_stats_path,
                f"{cfg.dataset.name}_{cfg.graph.name}_{cfg.graph.label}_stats",
            )
            save_graph_characteristics(spatial_adj_matrix, is_binary, save_path)

    # if args.downstream_task:
    gnn_model = None
    print(f"Running using model {cfg.model.name}")
    if model == "stgi":
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if not isinstance(model_cfg, Dict):
            raise TypeError(f"Model config should resolve to Dict, got {type(model_cfg)}")
        model_kwargs = {
            "adj": spatial_adj_matrix,
            "in_dim": dm.d_in,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.layer_num,
            "layer_type": cfg.model.layer_type,
            "use_spatial": cfg.use_spatial,
            "use_temporal": cfg.use_temporal,
            "temporal_graph_fn": temporal_graph_fn,
            "add_self_loops": False,
        }
        other_kwargs = {k: v for k, v in model_cfg.items() if k not in model_kwargs}
        model_kwargs = {**model_kwargs, **other_kwargs}
        gnn_model = STGI
    elif model == "grin":
        args = {}
        with open("./downstream/imputation/models/GRIN/config.yaml", "r") as f:
            args = yaml.safe_load(f)
        # for key, value in config_args.items():
        #     setattr(args, key, value)
        model_kwargs = {
            "adj": spatial_adj_matrix,
            "d_in": dm.d_in,
            "d_hidden": args.d_hidden,
            "d_ff": args.d_ff,
            "ff_dropout": args.ff_dropout,
            "n_layers": args.layer_num,
            "kernel_size": args.kernel_size,
            "decoder_order": args.decoder_order,
            "global_att": args.global_att,
            "d_u": args.d_u,
            "d_emb": args.d_emb,
            "layer_norm": args.layer_norm,
            "merge": args.merge,
            "impute_only_holes": args.impute_only_holes,
        }
        gnn_model = GRINet
    else:
        raise ValueError(f"Unsupported model {model}")

    assert gnn_model is not None, "Model instantiation failed"

    loss_fn = MaskedMAELoss()

    metrics = {
        "mae": MaskedMAE(compute_on_step=False),
        "mape": MaskedMAPE(compute_on_step=False),
        "mse": MaskedMSE(compute_on_step=False),
        "mre": MaskedMRE(compute_on_step=False),
        "mre2": MaskedMRE2(compute_on_step=False),
        "rmse": MaskedRMSE(compute_on_step=False),
        "smape": MaskedSMAPE(compute_on_step=False),
        "pearson": MaskedPearson(),
        "ccc": MaskedCCC(),
        "cosine": MaskedCosineSimilarity(),
        "lag": MaskedLagCorrelation(),
    }
    report = EpochReport()
    report_callback = EpochReportCallback(report=report)
    tb_logger = TensorBoardLogger(
        save_dir=save_path_dir,
        name="tensorboard",
    )
    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logdir = os.path.join(save_path_dir, cfg.dataset.name, cfg.model.name, exp_name)
    early_stop_callback = EarlyStopping(
        monitor=cfg.training.early_stopping.monitor,
        patience=cfg.training.early_stopping.patience,
        mode=cfg.training.early_stopping.mode,
    )
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor="val_mae", mode="min")
    task = Imputer(
        model_class=gnn_model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": cfg.training.optim.learning_rate, "weight_decay": 0.0},
        loss_fn=loss_fn,
        scaled_target=cfg.training.scaled_target,
        metrics=metrics,
        scheduler_class=CosineAnnealingLR,
        scheduler_kwargs=cfg.training.scheduler,
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=[tb_logger],
        default_root_dir=save_path_dir,
        gradient_clip_algorithm=cfg.training.optim.gradient_clip_algorithm,
        gradient_clip_val=cfg.training.optim.gradient_clip_val,
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=[
            RichProgressBar(),
            ConsoleMetricsCallback(),
            early_stop_callback,
            checkpoint_callback,
            report_callback,
        ],
        num_sanity_val_steps=2,
        log_every_n_steps=cfg.training.logging.log_every_n_steps,
    )

    trainer.fit(task, datamodule=dm)
    fit_report = report.as_dict()
    metrics_data.update(fit_report)

    # Load last best checkpoint
    task.load_state_dict(torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)["state_dict"])

    outputs = trainer.predict(task, datamodule=dm)
    if outputs is None:
        print("Trainer prediction return None results")
        return

    _, imputation, mask = aggregate_predictions(outputs)
    imputation = imputation.squeeze(-1).cpu().numpy()

    df_true = dataset.df.iloc[dm.test_slice]
    index = dataset.data_timestamps(dm.test_set.indices, flatten=False)["horizon"]

    # Create prediction dataframe
    aggr_methods = ["mean"]
    df_hats = prediction_dataframe(imputation, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    prediction_metrics = {"prediction_metrics": {}}

    eval_masks = {}

    # Check if dataset has multi-eval mask attributes
    if dataset._scenario is not None:
        # Multi-eval mask mode (from injection)
        scenario = dataset._scenario
        eval_masks["fixed"] = scenario.eval_mask_fixed[dm.test_slice].astype(int)
        eval_masks["newly"] = scenario.eval_mask_newly[dm.test_slice].astype(int)
        eval_masks["cumulative"] = scenario.eval_mask_cumulative[dm.test_slice].astype(int)
        print("✅ Using multi-eval masks from scenario")
    else:
        # Single eval mask mode (baseline or legacy)
        eval_masks["primary"] = dataset.eval_mask[dm.test_slice]
        print("✅ Using single eval mask (baseline mode)")

    for aggr_by, df_hat in df_hats.items():
        print(f"- AGGREGATE BY {aggr_by.upper()}")

        for mask_name, eval_mask in eval_masks.items():
            print(f"\nEval Mask: {mask_name.upper()}")
            print(f"  Test eval targets: {eval_mask.sum():,} ({eval_mask.mean():.2%} of test)")

            pred_tensor = torch.tensor(df_hat.values)
            true_tensor = torch.tensor(df_true.values)

            mask_tensor = torch.tensor(eval_mask)
            for metric_name, metric_fn in metrics.items():
                if hasattr(metric_fn, "reset"):
                    metric_fn.reset()

                metric_fn.update(pred_tensor, true_tensor, mask_tensor)

                error = metric_fn.compute().item()
                print(f"{mask_name} {metric_name}: {error:.4f}")
                prediction_metrics["prediction_metrics"].update({f"{mask_name}_{metric_name}": error})

    metrics_data.update(prediction_metrics)

    df_pred = df_hats["mean"]
    df_true = dataset.df.iloc[dm.test_slice]
    eval_mask = dataset.eval_mask[dm.test_slice]
    missing_mask = dataset.missing_mask[dm.test_slice]

    with open(save_file_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    imputation_path = os.path.join(save_path_dir, f"{save_file_name}_imputation.h5")
    with h5py.File(imputation_path, "w") as f:
        f.create_dataset(
            "prediction",
            data=df_pred.values,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "target",
            data=df_true.values,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "eval_mask_fixed",
            data=eval_masks["fixed"].astype(np.uint8),  # bool → uint8 is safer in HDF5
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "eval_mask_newly",
            data=eval_masks["newly"].astype(np.uint8),  # bool → uint8 is safer in HDF5
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "eval_mask_cumulative",
            data=eval_masks["cumulative"].astype(np.uint8),  # bool → uint8 is safer in HDF5
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "missing_mask",
            data=missing_mask.numpy().astype(np.uint8),  # bool → uint8 is safer in HDF5
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "time",
            data=df_pred.index.values.astype("datetime64[ns]").astype("int64"),
        )


if __name__ == "__main__":
    run()

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
