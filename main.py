from argparse import ArgumentParser, Namespace
from typing import Tuple

import networkx as nx
import torch
import torch.nn as nn
from numpy import mean
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
from downsteam.imputation.STGI.stgi import STGI


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="which dataset to use e.g. 'synthetic'",
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        help="which algorithm to use for graph completion e.g. 'KNN'",
        default=None,
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
        f"Degrees: {mean(degrees)} | Clustering coefficient: {clustering_coeff} | Number Components: {n_component} | Connectivity: {connectivity}"
    )


def train_imputer(
    model: nn.Module,
    dataloader: DataLoader,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
    optimizer: Optimizer,
    epochs: int = 50,
):
    nb_batches = len(dataloader)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        for i, (batch_data, batch_mask) in enumerate(dataloader):
            imputed_batch_data, batch_loss = model(
                batch_data.unsqueeze(2), edge_index, batch_mask.unsqueeze(2)
            )
            batch_loss.backward()
            optimizer.step()
            print(f"Batch {i}/{nb_batches} loss: {batch_loss:.4e}", end="\r")
            epoch_loss += batch_loss
        mean_loss = epoch_loss / nb_batches
        print(f"Epoch {epoch + 1}/{epochs} mean loss: {mean_loss}")


def evalutate_imputer(
    model: nn.Module,
    dataloader: DataLoader,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
):
    model.eval()
    with torch.no_grad():
        imputed_batchs = []
        for batch_data, batch_mask in dataloader:
            imputed_batch, loss = model(
                batch_data.unsqueeze(2), edge_index, batch_mask.unsqueeze(2)
            )
            imputed_batchs.append(imputed_batch)
    imputed_data = torch.cat(imputed_batchs, dim=0)
    return imputed_data


def run(args: Namespace) -> None:
    dataset = get_dataset(args.dataset)
    dataset.corrupt()
    dataloader = dataset.get_dataloader(
        use_missing_data=True, shuffle=False, batch_size=128
    )
    adj_matrix = dataset.get_adjacency()
    edge_index, _ = dense_to_sparse(adj_matrix)
    adj_matrix_knn = dataset.get_similarity_knn(k=5)
    # print(adj_matrix.shape)
    # print(adj_matrix_knn.shape)
    # print(dataset.shape())
    graph_characteristics(adj_matrix)
    graph_characteristics(adj_matrix_knn)

    stgi = STGI(
        in_dim=1,
        hidden_dim=32,
        gcn_out_dim=16,
        lstm_hidden_dim=64,
        num_layers=2,
    )

    # stgi = torch.compile(stgi)
    optimizer = Adam(stgi.parameters(), lr=1e-3)
    train_imputer(stgi, dataloader, edge_index, optimizer, 50)
    X_hat = evalutate_imputer(stgi, dataloader, edge_index)

    mae = mean_absolute_error(
        dataset.missing_data[dataset.missing_mask == 0].numpy(),
        X_hat[dataset.missing_mask == 0],
    )

    mse = mean_squared_error(
        dataset.missing_data[dataset.missing_mask == 0].numpy(),
        X_hat[dataset.missing_mask == 0],
    )

    rmse = root_mean_squared_error(
        dataset.missing_data[dataset.missing_mask == 0].numpy(),
        X_hat[dataset.missing_mask == 0],
    )

    print(f"Imputation MAE: {mae:.4e}, MSE: {mse:.4e}, RMSE: {rmse:.4e}")


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
