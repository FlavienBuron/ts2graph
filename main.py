import copy
import datetime
import json
import math
import os
import random
from functools import partial
from time import perf_counter
from typing import Callable, Optional

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

from datasets.dataloader import get_dataset
from datasets.dataloaders.graphloader import GraphLoader
from datasets.datamodule import DataModule
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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.environ["PYTHONHASHSEED"] = str(42)
torch.set_num_threads(8)
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


def get_spatial_graph(
    dataset: GraphLoader, cfg: DictConfig
) -> tuple[torch.Tensor, float]:
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
        return torch.empty((2, 0), dtype=torch.long), torch.empty(
            (0,), dtype=torch.float
        )

    return empty_temporal_graph


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print("#" * 100)
    sparsifier = cfg.graph.sparsifier

    param_name = next(iter(sparsifier), None)  # first key in sparsifier dict
    param_val = sparsifier[param_name]
    if isinstance(param_val, dict):
        cfg.graph.label = str(param_val.get("value", "null"))
    else:
        cfg.graph.label = str(param_val)

    print(cfg)
    save_path_dir = cfg.paths.save_path
    os.makedirs(save_path_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.paths.save_path, "resolved_config.yaml"))
    model = cfg.model.name.lower()

    print(f"{cfg.use_spatial=} {cfg.use_temporal=}")

    metrics_data = {}
    metrics_data["config"] = OmegaConf.to_container(cfg, resolve=True)

    dataset = get_dataset(cfg.dataset.name)

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

    # spatial_graph_technique, spatial_graph_param = args.spatial_graph_technique
    # temporal_graph_technique = args.temporal_graph_technique[0]
    # temporal_graph_params = args.temporal_graph_technique[1:]
    # spatial_graph_param = float(spatial_graph_param)

    # D = graph(torch.from_numpy(dataset.distances.to_numpy()))
    # print(f"DEBUG: {D=}")
    # A = dataset.get_geolocation_graph(
    #     threshold=0.3,
    #     include_self=args.self_loop,
    #     weighted=not args.unweighted_graph,
    # )
    # print(f"DEBUG: {A=}")

    spatial_graph_time = 0.0
    if cfg.use_spatial:
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
        save_stats_path = cfg.save_path
        if cfg.use_spatial:
            save_path = os.path.join(
                save_stats_path,
                f"{cfg.dataset.name}_{cfg.graph.name}_{cfg.graph.label}",
            )
            save_graph_characteristics(spatial_adj_matrix, save_path)

    # if args.downstream_task:
    gnn_model = None
    print(f"Running using model {cfg.model.name}")
    if model == "stgi":
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir, save_top_k=1, monitor="val_mae", mode="min"
    )
    task = Imputer(
        model_class=gnn_model,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": cfg.training.learning_rate, "weight_decay": 0.0},
        loss_fn=loss_fn,
        scaled_target=cfg.training.scaled_target,
        metrics=metrics,
        scheduler_class=CosineAnnealingLR,
        scheduler_kwargs={"eta_min": 0.0001, "T_max": cfg.training.max_epochs},
    )
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=[tb_logger],
        default_root_dir=save_path_dir,
        gradient_clip_algorithm="norm",
        gradient_clip_val=0.5,
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
        log_every_n_steps=16,
    )

    trainer.fit(task, datamodule=dm)
    fit_report = report.as_dict()
    metrics_data.update(fit_report)
    task.load_state_dict(
        torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)[
            "state_dict"
        ]
    )
    outputs = trainer.predict(task, datamodule=dm)
    if outputs is None:
        print("Trainer prediction return None results")
        return

    target, imputation, mask = aggregate_predictions(outputs)
    imputation = imputation.squeeze(-1).cpu().numpy()
    # pred_imp = pred_imp.squeeze(-1).cpu().numpy()

    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]

    index = dataset.data_timestamps(dm.test_set.indices, flatten=False)["horizon"]

    aggr_methods = ["mean"]

    df_hats = prediction_dataframe(
        imputation, index, dataset.df.columns, aggregate_by=aggr_methods
    )
    # df_imps = prediction_dataframe(
    #     pred_imp, index, dataset.df.columns, aggregate_by=aggr_methods
    # )
    df_hats = dict(zip(aggr_methods, df_hats))
    # df_imps = dict(zip(aggr_methods, df_imps))
    prediction_metrics = {"prediction_metrics": {}}
    # for aggr_by, df_hat in df_hats.items():
    #     # Compute error
    #     print(f"- AGGREGATE BY {aggr_by.upper()}")

    for aggr_by, df_hat in df_hats.items():
        print(f"- AGGREGATE BY {aggr_by.upper()}")

        # Convert predictions and targets to torch tensors
        pred_tensor = torch.tensor(df_hat.values)
        true_tensor = torch.tensor(df_true.values)

        # If your mask is 2D/3D, make sure its shape matches pred/true
        mask_tensor = eval_mask.detach().clone().squeeze()

        for metric_name, metric_fn in metrics.items():
            # Reset metric state before computing
            if hasattr(metric_fn, "reset"):
                metric_fn.reset()

            # Update metric with prediction, target, mask
            metric_fn.update(pred_tensor, true_tensor, mask_tensor)

            # Compute the metric
            error = metric_fn.compute().item()
            print(f" {metric_name}: {error:.4f}")
            prediction_metrics["prediction_metrics"].update({metric_name: error})

    metrics_data.update(prediction_metrics)

    df_pred = df_hats["mean"]
    df_true = dataset.df.iloc[dm.test_slice]
    eval_mask = dataset.eval_mask[dm.test_slice]
    missing_mask = dataset.missing_mask[dm.test_slice]

    with open(cfg.paths.save_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    imputation_path = os.path.join(save_path_dir, "imputation_results.h5")
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
            "eval_mask",
            data=eval_mask.numpy().astype(np.uint8),  # bool → uint8 is safer in HDF5
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
