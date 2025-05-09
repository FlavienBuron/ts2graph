import random
from argparse import ArgumentParser, Namespace

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
from graphs_transformations.ts2net import Ts2Net
from graphs_transformations.utils import (
    compute_edge_difference_smoothness,
    compute_laplacian_smoothness,
    save_graph_characteristics,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
import os

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
        default=None,
        choices=[None, "min_max", "std"],
    )
    parser.add_argument(
        "--graph_technique",
        "-g",
        nargs=2,
        help="which algorithm to use for graph completion e.g. 'KNN'",
        default=["knn", 3],
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
        "--iter_num",
        "-i",
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


def train_imputer(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    optimizer: Optimizer,
    epochs: int = 5,
    num_iteration: int = 100,
    device: str = "cpu",
    verbose: bool = True,
):
    nb_batches = len(dataloader)
    batch_size = dataloader.batch_size if dataloader.batch_size else 1
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        sum_ls_before = 0.0
        sum_ls_after = 0.0
        sum_eds_before = 0.0
        sum_eds_after = 0.0
        sum_ls_before_masked = 0.0
        sum_eds_before_masked = 0.0
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
                        batch_data.detach(), edge_index, edge_weight
                    )
                    sum_eds_before += compute_edge_difference_smoothness(
                        batch_data.detach(), edge_index, edge_weight
                    )
                    # Mask, to compare
                    sum_ls_before_masked += compute_laplacian_smoothness(
                        batch_data.detach(), edge_index, edge_weight, mask=batch_mask
                    )
                    sum_eds_before_masked += compute_edge_difference_smoothness(
                        batch_data.detach(), edge_index, edge_weight, mask=batch_mask
                    )

                    # Imputation step
                    imputed_data, batch_loss = model(
                        # x=batch_data.unsqueeze(2).to(device),
                        x=batch_data.to(device),
                        edge_index=edge_index.to(device),
                        edge_weight=edge_weight.to(device),
                        # missing_mask=batch_mask.unsqueeze(2).to(device).bool(),
                        missing_mask=batch_mask.to(device).bool(),
                    )
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
                    imputed_batch = batch_data.clone()
                    imputed_data = imputed_data.cpu()
                    missing_mask_cpu = batch_mask.cpu().bool()

                    # print(f"{imputed_batch[~missing_mask_cpu]}")
                    # print(f"{imputed_data[~missing_mask_cpu]}")
                    imputed_batch[~missing_mask_cpu] = imputed_data[~missing_mask_cpu]

                    iteration_imputed_data.append(imputed_batch)

                    # Get the Smoothess AFTER imputation
                    sum_ls_after += compute_laplacian_smoothness(
                        imputed_data.detach(), edge_index, edge_weight
                    )
                    sum_eds_after += compute_edge_difference_smoothness(
                        imputed_data.detach(), edge_index, edge_weight
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
        # dataset.reset_current_data()
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} mean loss: {mean_loss:.4e}")
        print(
            f"\n\nAverage Masked Laplacian Smoothess: before {sum_ls_before_masked / (batch_size * nb_batches):.4e}, after {sum_ls_after / (batch_size * nb_batches):.4e}"
        )
        print(
            f"Average Laplacian Smoothess: before {sum_ls_before / (batch_size * nb_batches):.4e}, after {sum_ls_after / (batch_size * nb_batches):.4e}"
        )
        print(
            f"Average Edge Distance Smoothess: before {sum_eds_before / (batch_size * nb_batches):.4e}, after {sum_eds_after / (batch_size * nb_batches):.4e}"
        )
        print(
            f"Average Masked Edge Distance Smoothess: before {sum_eds_before_masked / (batch_size * nb_batches):.4e}, after {sum_eds_after / (batch_size * nb_batches):.4e}"
        )
    return model


def impute_missing_data(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_iteration: int,
    device: str,
):
    dataset.reset_current_data()
    model.eval()
    with torch.no_grad():
        sum_ls_before = 0.0
        sum_ls_after = 0.0
        sum_eds_before = 0.0
        sum_eds_after = 0.0
        sum_ls_before_masked = 0.0
        sum_eds_before_masked = 0.0
        batch_size = dataloader.batch_size if dataloader.batch_size else 1
        nb_batches = len(dataloader)
        for _ in range(num_iteration):
            imputed_batches = []
            for batch_data, batch_mask, _, _ in dataloader:
                sum_ls_before += compute_laplacian_smoothness(
                    batch_data, edge_index, edge_weight
                )
                sum_eds_before += compute_edge_difference_smoothness(
                    batch_data, edge_index, edge_weight
                )
                # Mask, to compare
                sum_ls_before_masked += compute_laplacian_smoothness(
                    batch_data.detach(), edge_index, edge_weight, mask=batch_mask
                )
                sum_eds_before_masked += compute_edge_difference_smoothness(
                    batch_data.detach(), edge_index, edge_weight, mask=batch_mask
                )

                imputed_data, _ = model(
                    # batch_data.unsqueeze(2).to(device),
                    batch_data.to(device),
                    edge_index.to(device),
                    edge_weight.to(device),
                    # batch_mask.unsqueeze(2).to(device),
                    batch_mask.to(device),
                )
                # imputed_data = imputed_data.squeeze(-1)
                imputed_batch = batch_data.clone().detach().cpu()
                mask_cpu = batch_mask.cpu().bool()
                # print(f"{imputed_batch[~mask_cpu]}")
                # print(f"{imputed_data[~mask_cpu]}")
                imputed_batch[~mask_cpu] = imputed_data[~mask_cpu]

                imputed_batches.append(imputed_data)

                sum_ls_after += compute_laplacian_smoothness(
                    imputed_batch, edge_index, edge_weight, mask=batch_mask
                )
                sum_eds_after += compute_edge_difference_smoothness(
                    imputed_batch, edge_index, edge_weight
                )
            imputed_data = torch.cat(imputed_batches, dim=0)
            dataset.update_data(imputed_data)
            del imputed_data
        print(
            f"\n\nAverage Imputation Laplacian Smoothess: before {sum_ls_before / (batch_size * nb_batches):.4e}, after {sum_ls_after / (batch_size * nb_batches):.4e}"
        )
        print(
            f"Average Imputation Edge Distance Smoothess: before {sum_eds_before / (batch_size * nb_batches):.4e}, after {sum_eds_after / (batch_size * nb_batches):.4e}"
        )
        for param in model.parameters():
            if param.grad is None:
                print(f"Gradient is None for param: {param}")

    return dataset.current_data


def evaluate(
    imputed_data: np.ndarray,
    target_data: np.ndarray,
    evaluation_mask: np.ndarray,
):
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

    print(f"Imputation MAE: {mae:.4e}, MSE: {mse:.4e}, RMSE: {rmse:.4e}")


def run(args: Namespace) -> None:
    # test = np.random.rand(10, 100)
    print(args)
    device = args.device
    dataset = get_dataset(args.dataset)
    # dataset.corrupt(missing_type="perc", missing_size=50)
    dataloader = dataset.get_dataloader(
        use_corrupted_data=False, shuffle=False, batch_size=128
    )
    assert not torch.isnan(dataset.original_data[dataset.validation_mask]).any(), (
        "Missing values present under evaluation mask (run)"
    )
    print(f"{dataset.validation_mask.sum()}")
    graph_technique, param = args.graph_technique
    param = float(param)
    ts2net = Ts2Net()
    if "loc" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param, include_self=args.self_loop)
    elif "zero" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param)
        adj_matrix = torch.zeros_like(adj_matrix)
        if args.self_loop:
            adj_matrix.fill_diagonal_(1.0)
    elif "one" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param)
        adj_matrix = torch.ones_like(adj_matrix)
        if not bool(args.self_loop):
            adj_matrix.fill_diagonal_(0.0)
    else:
        param = int(param)
        adj_matrix = dataset.get_similarity_knn(
            k=param, loop=args.self_loop, cosine=args.similarity_metric == "cosine"
        )
    if args.graph_stats:
        save_stats_path = "./experiments/results/graphs_stats/"
        save_path = os.path.join(save_stats_path, f"{graph_technique}_{param}")
        save_graph_characteristics(adj_matrix, save_path)

    if args.downstream_task:
        edge_index, edge_weight = dense_to_sparse(adj_matrix)

        stgi = STGI(
            in_dim=1,
            hidden_dim=args.hidden_dim,
            num_layers=args.layer_num,
            model_type=args.layer_type,
        )

        stgi.to(device)
        geo_optim = Adam(stgi.parameters(), lr=1e-2)
        stgi = train_imputer(
            stgi,
            dataset,
            dataloader,
            edge_index,
            edge_weight,
            geo_optim,
            args.epochs,
            args.iter_num,
            device=device,
            verbose=args.verbose,
        )
        imputed_data = impute_missing_data(
            stgi,
            dataset,
            dataloader,
            edge_index,
            edge_weight,
            args.iter_num,
            device,
        )
        evaluate(
            imputed_data.numpy(),
            dataset.original_data.numpy(),
            dataset.validation_mask.numpy(),
        )


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN base
