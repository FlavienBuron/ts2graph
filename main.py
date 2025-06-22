import json
import math
import os
import random
from argparse import ArgumentParser, Namespace
from functools import partial
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse

from datasets.dataloader import get_dataset
from datasets.dataloaders.graphloader import GraphLoader
from downstream.imputation.STGI import STGI
from graphs_transformations.temporal_graphs import k_hop_graph
from graphs_transformations.ts2net import Ts2Net
from graphs_transformations.utils import (
    compute_edge_difference_smoothness,
    compute_laplacian_smoothness,
    save_graph_characteristics,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

os.environ["PYTHONHASHSEED"] = str(42)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        help="The device to use",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        help="The path to save the metrics to",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="which dataset to use e.g. 'synthetic'",
        required=True,
    )
    parser.add_argument(
        "--normalization_type",
        "-n",
        type=str,
        help="How should the data be normalized",
        default="min_max",
        choices=[None, "min_max", "std"],
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Which mode should STGI use e.g. s=spatial, t=temporal, st=spatio-temporal",
        choices=["s", "t", "st"],
        default="s",
    )
    parser.add_argument(
        "--spatial_graph_technique",
        "-sg",
        nargs=2,
        help="which algorithm to use for spatial graph completion, if used, e.g. 'knn 3'",
        default=["knn", "3"],
    )
    parser.add_argument(
        "--temporal_graph_technique",
        "-tg",
        nargs="+",
        help="which algorithm to use for temporal graph completion, if used, e.g. 'naive 1'",
        default=["naive", "1"],
    )
    parser.add_argument(
        "--self_loop",
        "-sl",
        type=int,
        help="whether the graphs allows for nodes to connect to themselves",
        default=False,
    )
    parser.add_argument(
        "--similarity_metric",
        "-sm",
        type=str,
        help="if used by the graph completion algorithm, which similarity metric to use in the completion",
        default="",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        help="The learning rate of the optimizer",
        default=1e-3,
    )
    parser.add_argument(
        "--iter_num",
        "-it",
        type=int,
        help="The number of iteration from the model pass",
        default=1,
    )
    parser.add_argument(
        "--layer_type",
        "-l",
        type=str,
        help="The GNN layer type to use e.g. GCNConv",
        default="GCNConv",
    )
    parser.add_argument(
        "--layer_num",
        "-ln",
        type=int,
        help="The GNN depth",
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        help="The batch size for the DataLoader",
        default=128,
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="The number of Epochs that the model shoud be trained on",
        default=10,
    )
    parser.add_argument(
        "--hidden_dim",
        "-hd",
        type=int,
        help="The size of the hidden dimension of the GNN",
        default=32,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        help="Should the training intermediate results be printed",
        default=1,
    )
    parser.add_argument(
        "--graph_stats",
        "-gs",
        action="store_true",
        help="whether to output the graph statistics",
    )
    parser.add_argument(
        "--downstream_task",
        "-dt",
        action="store_false",
        help="whether to execute the downstream task (imputation)",
    )
    args = parser.parse_args()
    return args


def log_results(metrics: dict, filename: str, mode: str = "a"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(metrics)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)


def train_imputer(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    spatial_edge_index: torch.Tensor,
    spatial_edge_weight: torch.Tensor,
    optimizer: Optimizer,
    metrics: dict,
    epochs: int = 5,
    num_iteration: int = 100,
    device: str = "cpu",
    verbose: bool = True,
):
    nb_batches = len(dataloader)
    batch_size = dataloader.batch_size if dataloader.batch_size else 1
    model.train()
    train_metrics = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        sum_ls_before = 0.0
        sum_ls_after = 0.0
        sum_eds_before = 0.0
        sum_eds_after = 0.0
        sum_ls_before_masked = 0.0
        sum_ls_after_masked = 0.0
        sum_eds_before_masked = 0.0
        sum_eds_after_masked = 0.0
        batch_temp_graph_times = []
        for iter in range(num_iteration):
            iteration_imputed_data = []
            batch_losses = []

            # create a collection to hold batch data references temporatily
            batch_references = []

            for i, (batch_data, batch_mask, batch_ori, batch_train_mask) in enumerate(
                dataloader
            ):
                batch_references.append((batch_data.clone(), batch_mask.clone()))

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # Get the Smoothess values before imputation
                    sum_ls_before += compute_laplacian_smoothness(
                        batch_data.detach(), spatial_edge_index, spatial_edge_weight
                    )
                    sum_eds_before += compute_edge_difference_smoothness(
                        batch_data.detach(), spatial_edge_index, spatial_edge_weight
                    )
                    # Mask, to compare
                    sum_ls_before_masked += compute_laplacian_smoothness(
                        batch_data.detach(),
                        spatial_edge_index,
                        spatial_edge_weight,
                        mask=batch_mask,
                    )
                    sum_eds_before_masked += compute_edge_difference_smoothness(
                        batch_data.detach(),
                        spatial_edge_index,
                        spatial_edge_weight,
                        mask=batch_mask,
                    )

                    # Imputation step
                    imputed_data, temp_graph_time = model(
                        # x=batch_data.unsqueeze(2).to(device),
                        x=batch_data.to(device),
                        mask=batch_mask.to(device),
                        spatial_edge_index=spatial_edge_index.to(device),
                        spatial_edge_weight=spatial_edge_weight.to(device),
                    )
                    batch_temp_graph_times.append(temp_graph_time)
                    # imputed_data = imputed_data.squeeze(-1)
                    train_mask_cpu = batch_train_mask.cpu().bool()
                    # print(
                    #     f"{torch.isnan(imputed_data).any()=} {torch.isnan(batch_ori).any()}"
                    # )
                    batch_loss = mse_loss(
                        imputed_data[train_mask_cpu],
                        batch_ori[train_mask_cpu],
                        reduction="mean",
                    )
                    batch_loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    # replace the missing data in the batch with the imputed data
                    # imputed_batch = batch_data.clone()
                    imputed_data = imputed_data.cpu()
                    missing_mask_cpu = batch_mask.cpu().bool()

                    # print(f"{imputed_batch[~missing_mask_cpu]}")
                    # print(f"{imputed_data[~missing_mask_cpu]}")
                    imputed_data[missing_mask_cpu] = (
                        batch_data[missing_mask_cpu].detach().clone()
                    )

                    iteration_imputed_data.append(imputed_data)

                    # Get the Smoothess AFTER imputation
                    sum_ls_after += compute_laplacian_smoothness(
                        imputed_data.detach(), spatial_edge_index, spatial_edge_weight
                    )
                    sum_eds_after += compute_edge_difference_smoothness(
                        imputed_data.detach(), spatial_edge_index, spatial_edge_weight
                    )
                    sum_ls_after_masked += compute_laplacian_smoothness(
                        imputed_data.detach(),
                        spatial_edge_index,
                        spatial_edge_weight,
                        mask=batch_mask,
                    )
                    sum_eds_after_masked += compute_edge_difference_smoothness(
                        imputed_data.detach(),
                        spatial_edge_index,
                        spatial_edge_weight,
                        mask=batch_mask,
                    )

                    batch_losses.append(batch_loss.item())
                if verbose:
                    print(
                        f"Batch {i}/{nb_batches} loss: {batch_loss.item():.4e}",
                        end="\r",
                    )
                del batch_data, batch_mask, imputed_data, missing_mask_cpu

            with torch.no_grad():
                iteration_imputed_data = torch.cat(iteration_imputed_data, dim=0)
            iter_loss = sum(batch_losses)
            epoch_loss += iter_loss
            import gc

            gc.collect()
            if verbose:
                print(
                    f"Iteration {iter + 1}/{num_iteration} loss {iter_loss:.4e} | Epoch {epoch + 1}/{epochs} loss: {epoch_loss:.4e}",
                )
            dataset.update_data(iteration_imputed_data)
            del iteration_imputed_data, batch_losses, batch_references
        mean_loss = epoch_loss / (nb_batches * num_iteration)
        dataset.reset_current_data()
        train_metrics.append(
            {
                "phase": "train",
                "epoch": epoch + 1,
                "lap_smooth_before": sum_ls_before,
                "lap_smooth_after": sum_ls_after,
                "eds_before": sum_eds_before,
                "eds_after": sum_eds_after,
                "masked_lap_smooth_before": sum_ls_before_masked,
                "masked_lap_smooth_after": sum_ls_after_masked,
                "masked_eds_before": sum_eds_before_masked,
                "masked_eds_after": sum_eds_after_masked,
                "temp_graph_total_time": sum(batch_temp_graph_times),
                "temp_graph_avg_time": sum(batch_temp_graph_times) / nb_batches,
            }
        )
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} mean loss: {mean_loss:.4e}")
        imput_metrics = {"epoch": epoch + 1}
        imputed_data = impute_missing_data(
            model,
            dataset,
            dataloader,
            spatial_edge_index,
            spatial_edge_weight,
            imput_metrics,
            num_iteration,
            device,
        )
        train_metrics.append(imput_metrics)
        eval_metrics = evaluate(
            imputed_data.numpy(),
            dataset.original_data.numpy(),
            dataset.validation_mask.numpy(),
        )
        eval_metrics.update({"phase": "eval", "epoch": epoch + 1})
        train_metrics.append(eval_metrics)
        metrics["train_metrics"] = train_metrics
    return model


def impute_missing_data(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    spatial_edge_index: torch.Tensor,
    spatial_edge_weight: torch.Tensor,
    metrics: dict,
    num_iteration: int,
    device: str,
):
    dataset.reset_current_data()
    model.eval()
    with torch.no_grad():
        sum_ls_before = 0.0
        sum_ls_after = 0.0
        sum_ls_after_masked = 0.0
        sum_eds_before = 0.0
        sum_eds_after = 0.0
        sum_ls_before_masked = 0.0
        sum_eds_before_masked = 0.0
        sum_eds_after_masked = 0.0
        sum_imputed_ls_before = 0.0
        sum_imputed_ls_after = 0.0
        sum_imputed_eds_before = 0.0
        sum_imputed_eds_after = 0.0
        temp_graph_times = []
        batch_size = dataloader.batch_size if dataloader.batch_size else 1
        nb_batches = len(dataloader)
        for _ in range(num_iteration):
            imputed_batches = []
            for batch_data, batch_mask, _, _ in dataloader:
                sum_ls_before += compute_laplacian_smoothness(
                    batch_data, spatial_edge_index, spatial_edge_weight
                )
                sum_eds_before += compute_edge_difference_smoothness(
                    batch_data, spatial_edge_index, spatial_edge_weight
                )
                # Mask, to compare
                sum_ls_before_masked += compute_laplacian_smoothness(
                    batch_data.detach(),
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=batch_mask,
                )
                sum_eds_before_masked += compute_edge_difference_smoothness(
                    batch_data.detach(),
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=batch_mask,
                )
                sum_imputed_ls_before += compute_laplacian_smoothness(
                    batch_data.detach(),
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=~batch_mask,
                )
                sum_imputed_eds_before += compute_edge_difference_smoothness(
                    batch_data.detach(),
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=~batch_mask,
                )

                imputed_data, temp_graph_time = model(
                    # batch_data.unsqueeze(2).to(device),
                    x=batch_data.to(device),
                    mask=batch_mask.to(device),
                    spatial_edge_index=spatial_edge_index.to(device),
                    spatial_edge_weight=spatial_edge_weight.to(device),
                )
                temp_graph_times.append(temp_graph_time)
                # imputed_data = imputed_data.squeeze(-1)
                # imputed_batch = batch_data.clone().detach().cpu()
                mask_cpu = batch_mask.cpu().bool()
                # print(f"{imputed_batch[~mask_cpu]}")
                # print(f"{imputed_data[~mask_cpu]}")

                imputed_data[mask_cpu] = batch_data[mask_cpu].detach().clone()

                imputed_batches.append(imputed_data)

                sum_ls_after_masked += compute_laplacian_smoothness(
                    imputed_data,
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=batch_mask,
                )
                sum_ls_after += compute_laplacian_smoothness(
                    imputed_data, spatial_edge_index, spatial_edge_weight
                )
                sum_eds_after += compute_edge_difference_smoothness(
                    imputed_data, spatial_edge_index, spatial_edge_weight
                )
                sum_eds_after_masked += compute_edge_difference_smoothness(
                    imputed_data,
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=batch_mask,
                )
                sum_imputed_ls_after += compute_laplacian_smoothness(
                    imputed_data,
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=~batch_mask,
                )
                sum_imputed_eds_after += compute_laplacian_smoothness(
                    imputed_data,
                    spatial_edge_index,
                    spatial_edge_weight,
                    mask=~batch_mask,
                )

            imputed_data = torch.cat(imputed_batches, dim=0)
            dataset.update_data(imputed_data)
            del imputed_data
        for param in model.parameters():
            if param.grad is None:
                print(f"Gradient is None for param: {param}")
        metrics.update(
            {
                "phase": "impute",
                "imputed_lap_smooth_before": sum_imputed_ls_before,
                "imputed_lap_smooth_after": sum_imputed_ls_after,
                "imputed_eds_before": sum_imputed_eds_before,
                "imputed_eds_after": sum_imputed_eds_after,
                "lap_smooth_before": sum_ls_before,
                "lap_smooth_after": sum_ls_after,
                "eds_before": sum_eds_before,
                "eds_after": sum_eds_after,
                "masked_lap_smooth_before": sum_ls_before_masked,
                "masked_lap_smooth_after": sum_ls_after_masked,
                "masked_eds_before": sum_eds_before_masked,
                "masked_eds_after": sum_eds_after_masked,
                "temp_graph_total_time": sum(temp_graph_times),
                "temp_graph_avg_time": sum(temp_graph_times) / nb_batches,
            }
        )
    return dataset.current_data


def evaluate(
    imputed_data: np.ndarray,
    target_data: np.ndarray,
    evaluation_mask: np.ndarray,
) -> dict:
    evaluation_points = evaluation_mask.astype(bool)
    print("Target NaNs:", np.isnan(target_data[evaluation_points]).sum())
    print("Imputed NaNs:", np.isnan(imputed_data[evaluation_points]).sum())
    mae = mean_absolute_error(
        target_data[evaluation_points],
        imputed_data[evaluation_points],
    )

    mse = mean_squared_error(
        target_data[evaluation_points],
        imputed_data[evaluation_points],
    )

    rmse = root_mean_squared_error(
        target_data[evaluation_points],
        imputed_data[evaluation_points],
    )

    metrics = {"mae": mae, "mse": mse, "rmse": rmse}

    print(f"Imputation MAE: {mae:.4e}, MSE: {mse:.4e}, RMSE: {rmse:.4e}")

    return metrics


def flatten_metrics(metrics: dict) -> list[dict]:
    run_config = {k: v for k, v in metrics.items() if k != "train_metrics"}
    return [
        {**run_config, **phase_metrics} for phase_metrics in metrics["train_metrics"]
    ]


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
    technique: str, parameter: float, dataset: GraphLoader, args: Namespace
) -> tuple[torch.Tensor, float]:
    total_time = 0.0
    if "loc" in technique:
        start = perf_counter()
        adj_matrix = dataset.get_geolocation_graph(
            threshold=parameter, include_self=args.self_loop
        )
        end = perf_counter()
    elif "zero" in technique:
        start = perf_counter()
        adj_matrix = dataset.get_geolocation_graph(threshold=parameter)
        adj_matrix = torch.zeros_like(adj_matrix)
        if args.self_loop:
            adj_matrix.fill_diagonal_(1.0)
        end = perf_counter()
    elif "one" in technique:
        start = perf_counter()
        adj_matrix = dataset.get_geolocation_graph(threshold=parameter)
        adj_matrix = torch.ones_like(adj_matrix)
        if not bool(args.self_loop):
            adj_matrix.fill_diagonal_(0.0)
        end = perf_counter()
    elif "rad" in technique:
        start = perf_counter()
        param = float(parameter)
        adj_matrix = dataset.get_radius_graph(
            radius=param, loop=args.self_loop, cosine=args.similarity_metric == "cosine"
        )
        end = perf_counter()
    else:
        start = perf_counter()
        param = int(parameter)
        adj_matrix = dataset.get_knn_graph(
            k=param,
            loop=args.self_loop,
            cosine=args.similarity_metric == "cosine",
        )
        end = perf_counter()
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
        if len(parameter) == 1:
            alpha = float(parameter[0])
            time_lag = 1
            embedding_dim = None
        else:
            alpha = float(parameter[0])
            time_lag = int(parameter[1]) if len(parameter) > 1 else 1
            embedding_dim = int(parameter[2]) if len(parameter) > 2 else None
        print("Using Reccurrent Temporal Graph")
        return partial(
            ts2net.tsnet_rn,
            radius=alpha,
            time_lag=time_lag,
            embedding_dim=embedding_dim,
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


def run(args: Namespace) -> None:
    # test = np.random.rand(10, 100)
    print("#" * 100)
    print(args)
    device = args.device
    stgi_mode = args.mode
    if stgi_mode.lower() in ["st"]:
        use_spatial = True
        use_temporal = True
    elif stgi_mode.lower() in ["t"]:
        use_spatial = False
        use_temporal = True
    else:
        use_spatial = True
        use_temporal = False

    print(f"{use_spatial=} {use_temporal=}")

    dataset = get_dataset(args.dataset)

    if args.batch_size == 0:
        args.batch_size = dataset.original_data.shape[0]

    # dataset.corrupt(missing_type="perc", missing_size=50)
    dataloader = dataset.get_dataloader(
        use_corrupted_data=False, shuffle=False, batch_size=args.batch_size
    )
    assert not torch.isnan(dataset.original_data[dataset.validation_mask]).any(), (
        "Missing values present under evaluation mask (run)"
    )
    spatial_graph_technique, spatial_graph_param = args.spatial_graph_technique
    temporal_graph_technique = args.temporal_graph_technique[0]
    temporal_graph_params = args.temporal_graph_technique[1:]
    spatial_graph_param = float(spatial_graph_param)

    metrics = {}
    metrics.update(vars(args))

    spatial_graph_time = 0.0
    if use_spatial:
        spatial_adj_matrix, spatial_graph_time = get_spatial_graph(
            spatial_graph_technique, spatial_graph_param, dataset, args
        )
    else:
        spatial_adj_matrix = torch.tensor([[]])

    if use_temporal:
        temporal_graph_fn = get_temporal_graph_function(
            temporal_graph_technique,
            temporal_graph_params,
        )
    else:
        temporal_graph_fn = get_temporal_graph_function(
            "",
            temporal_graph_params,
        )

    metrics.update({"spatial_graph_time": spatial_graph_time})

    if args.graph_stats:
        save_stats_path = args.save_path
        if use_spatial:
            save_path = os.path.join(
                save_stats_path,
                f"{args.dataset}_{spatial_graph_technique}_{spatial_graph_param}",
            )
            save_graph_characteristics(spatial_adj_matrix, save_path)

    if args.downstream_task:
        spatial_edge_index, spatial_edge_weight = dense_to_sparse(spatial_adj_matrix)

        stgi = STGI(
            in_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            layer_type=args.layer_type,
            use_spatial=use_spatial,
            use_temporal=use_temporal,
            temporal_graph_fn=temporal_graph_fn,
        )

        stgi.to(device)
        geo_optim = Adam(stgi.parameters(), lr=args.learning_rate)
        stgi = train_imputer(
            stgi,
            dataset,
            dataloader,
            spatial_edge_index,
            spatial_edge_weight,
            geo_optim,
            metrics,
            args.epochs,
            args.iter_num,
            device=device,
            verbose=args.verbose,
        )

        with open(args.save_path, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
