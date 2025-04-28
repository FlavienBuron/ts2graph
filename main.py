import random
from argparse import ArgumentParser, Namespace

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
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
        "--graph_technique",
        "-g",
        nargs=2,
        help="which algorithm to use for graph completion e.g. 'KNN'",
        default=["knn", 3],
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

    args = parser.parse_args()
    return args


def graph_characteristics(adj):
    G = nx.from_numpy_array(adj.numpy())
    degrees = [d for _, d in G.degree()]
    clustering_coeff = nx.average_clustering(G)
    n_component = nx.number_connected_components(G)
    largest_component = max(nx.connected_components(G), key=len)
    connectivity = len(largest_component) / G.number_of_nodes()
    print(
        f"Degrees: {np.mean(degrees):.4f} | Clustering coefficient: {clustering_coeff:.4f} | Number Components: {n_component} | Connectivity: {connectivity:.4f}"
    )


def train_imputer(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: Optimizer,
    epochs: int = 5,
    num_iteration: int = 100,
    device: str = "cpu",
    verbose: bool = True,
):
    nb_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        sum_ls_before = 0.0
        sum_ls_after = 0.0
        sum_eds_before = 0.0
        sum_eds_after = 0.0
        for iter in range(num_iteration):
            iteration_imputed_data = []
            batch_losses = []

            # create a collection to hold batch data references temporatily
            batch_references = []

            for i, (batch_data, batch_mask) in enumerate(dataloader):
                batch_references.append((batch_data.clone(), batch_mask.clone()))

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    sum_ls_before += compute_laplacian_smoothness(
                        batch_data.detach(), edge_index, debug=True
                    )
                    sum_eds_before += compute_edge_difference_smoothness(
                        batch_data.detach(), edge_index
                    )

                    imputed_data, batch_loss = model(
                        x=batch_data.unsqueeze(2).to(device),
                        edge_index=edge_index.to(device),
                        mask=batch_mask.unsqueeze(2).to(device),
                    )
                    imputed_data = imputed_data.squeeze(-1)

                    sum_ls_after += compute_laplacian_smoothness(
                        imputed_data.detach(), edge_index, debug=True
                    )
                    sum_eds_after += compute_edge_difference_smoothness(
                        imputed_data.detach(), edge_index
                    )

                    batch_loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    # replace the missing data in the batch with the imputed data
                    imputed_batch = batch_data.clone()
                    imputed_data = imputed_data.detach().cpu()
                    mask_cpu = batch_mask.cpu()
                    imputed_batch[~mask_cpu] = imputed_data[~mask_cpu]
                    iteration_imputed_data.append(imputed_batch)

                    batch_losses.append(batch_loss.item())
                if verbose:
                    print(
                        f"Batch {i}/{nb_batches} loss: {batch_loss.item():.4e}",
                        end="\r",
                    )
                del batch_data, batch_mask, imputed_data, mask_cpu

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
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} mean loss: {mean_loss:.4e}")
        print(
            f"Average Laplacian Smoothess: before {sum_ls_before / (batch_size * nb_batches)}, after {sum_ls_after / (batch_size * nb_batches)}"
        )
        print(
            f"Average Edge Distance Smoothess: before {sum_eds_before / (batch_size * nb_batches)}, after {sum_eds_after / (batch_size * nb_batches)}"
        )


def impute_missing_data(
    model: nn.Module,
    dataset: GraphLoader,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    num_iteration: int,
    device: str,
):
    model.eval()
    with torch.no_grad():
        for _ in range(num_iteration):
            imputed_batchs = []
            for batch_data, batch_mask in dataloader:
                imputed_batch, _ = model(
                    batch_data.unsqueeze(2).to(device),
                    edge_index.to(device),
                    batch_mask.unsqueeze(2).to(device),
                )
                imputed_batchs.append(imputed_batch.cpu().data)
            imputed_data = torch.cat(imputed_batchs, dim=0).squeeze(-1)
            dataset.update_data(imputed_data)
            del imputed_data
    return dataset.current_data


def evaluate(
    imputed_data: np.ndarray,
    target_data: np.ndarray,
    evaluation_mask: np.ndarray,
):
    print("Target NaNs:", np.isnan(target_data[evaluation_mask]).sum())
    print("Imputed NaNs:", np.isnan(imputed_data[evaluation_mask]).sum())
    mae = mean_absolute_error(
        target_data[evaluation_mask],
        imputed_data[evaluation_mask],
    )

    mse = mean_squared_error(
        target_data[evaluation_mask],
        imputed_data[evaluation_mask],
    )

    rmse = root_mean_squared_error(
        target_data[evaluation_mask],
        imputed_data[evaluation_mask],
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
    graph_technique, param = args.graph_technique
    param = float(param)
    ts2net = Ts2Net()
    if "loc" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param)
    elif "zero" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param)
        adj_matrix = torch.zeros_like(adj_matrix)
    elif "one" in graph_technique:
        adj_matrix = dataset.get_adjacency(threshold=param)
        adj_matrix = torch.ones_like(adj_matrix)
    else:
        param = int(param)
        adj_matrix = dataset.get_similarity_knn(k=param)
    edge_index, _ = dense_to_sparse(adj_matrix)

    graph_characteristics(adj_matrix)

    stgi = STGI(
        in_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.layer_num,
        model_type=args.layer_type,
    )

    stgi.to(device)
    geo_optim = Adam(stgi.parameters(), lr=5e-4)
    train_imputer(
        stgi,
        dataset,
        dataloader,
        edge_index,
        geo_optim,
        args.epochs,
        args.iter_num,
        device=device,
        verbose=args.verbose,
    )
    imputed_data_geo = impute_missing_data(
        stgi,
        dataset,
        dataloader,
        edge_index,
        args.iter_num,
        device,
    )
    evaluate(
        imputed_data_geo.numpy(),
        dataset.original_data.numpy(),
        dataset.validation_mask.numpy(),
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
