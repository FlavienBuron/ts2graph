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
from downstream.imputation.STGI import STGI
from graphs_transformations.ts2net import Ts2Net


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
        f"Degrees: {np.mean(degrees):.4f} | Clustering coefficient: {clustering_coeff:.4f} | Number Components: {n_component} | Connectivity: {connectivity:.4f}"
    )


def train_imputer(
    model: nn.Module,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: Optimizer,
    epochs: int = 5,
    num_iteration: int = 100,
    device: str = "cpu",
):
    nb_batches = len(dataloader)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        for iter in range(num_iteration):
            iteration_imputed_data = []
            batch_losses = []
            for i, (batch_data, batch_mask) in enumerate(dataloader):
                imputed_data, batch_loss = model(
                    x=batch_data.unsqueeze(2).to(device),
                    edge_index=edge_index.to(device),
                    mask=batch_mask.unsqueeze(2).to(device),
                )
                imputed_data = imputed_data.squeeze(-1)
                # replace the missing data in the batch with the imputed data
                imputed_batch = batch_data.clone()
                imputed_batch[~batch_mask] = imputed_data[~batch_mask]
                iteration_imputed_data.append(batch_data.cpu())
                batch_losses.append(batch_loss)
                batch_loss.backward()
                optimizer.step()
                print(f"Batch {i}/{nb_batches} loss: {batch_loss:.4e}", end="\r")
                epoch_loss += batch_loss
            iteration_imputed_data = torch.cat(iteration_imputed_data, dim=0)
            epoch_loss += torch.stack(batch_losses).sum()
            epoch_loss.backward()
            optimizer.step()
            print(
                f"Iteration {iter + 1}/{num_iteration} | Epoch {epoch + 1}/{epochs} loss: {epoch_loss.item():.4e}",
                end="\r",
            )
            dataloader.data = iteration_imputed_data
        mean_loss = epoch_loss / (nb_batches * num_iteration)
        print(f"Epoch {epoch + 1}/{epochs} mean loss: {mean_loss:.4e}")


def impute_missing_data(
    model: nn.Module,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: str,
):
    model.eval()
    with torch.no_grad():
        imputed_batchs = []
        for batch_data, batch_mask in dataloader:
            imputed_batch, _ = model(
                batch_data.unsqueeze(2).to(device),
                edge_index.to(device),
                batch_mask.unsqueeze(2).to(device),
            )
            imputed_batchs.append(imputed_batch.cpu().data)
    imputed_data = torch.cat(imputed_batchs, dim=0).squeeze(-1)
    return imputed_data


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
    device = args.device
    dataset = get_dataset(args.dataset)
    dataloader = dataset.get_dataloader(
        use_missing_data=False, shuffle=False, batch_size=128
    )

    ts2net = Ts2Net()
    adj_matrix = dataset.get_adjacency(threshold=0.3)
    geo_edge_index, _ = dense_to_sparse(adj_matrix)
    adj_matrix_knn = dataset.get_similarity_knn(k=3)
    knn_edge_index, _ = dense_to_sparse(adj_matrix_knn)
    # data_matx = ts2net.tsnet_vg(data, "nvg", num_cores=8)

    graph_characteristics(adj_matrix)
    graph_characteristics(adj_matrix_knn)
    # graph_characteristics(torch.from_numpy(data_matx))

    stgi_geo = STGI(
        in_dim=1,
        hidden_dim=32,
        out_dim=16,
        lstm_hidden_dim=64,
        num_layers=2,
    )
    stgi_knn = STGI(
        in_dim=1,
        hidden_dim=32,
        out_dim=16,
        lstm_hidden_dim=64,
        num_layers=2,
    )

    # grin_geo = GRINModel(input_size=1, hidden_size=32, merge_mode="mean")
    # grin_knn = GRINModel(input_size=1, hidden_size=32, merge_mode="mean")

    #
    stgi_geo.to(device)
    stgi_knn.to(device)
    geo_optim = Adam(stgi_geo.parameters(), lr=5e-4)
    knn_optim = Adam(stgi_knn.parameters(), lr=5e-4)
    train_imputer(stgi_geo, dataloader, geo_edge_index, geo_optim, 10, device=device)
    train_imputer(stgi_knn, dataloader, knn_edge_index, knn_optim, 10, device=device)
    imputed_data_geo = impute_missing_data(stgi_geo, dataloader, geo_edge_index, device)
    imputed_data_knn = impute_missing_data(stgi_knn, dataloader, knn_edge_index, device)
    # geo_optim = Adam(grin_geo.parameters(), lr=5e-4)
    # knn_optim = Adam(grin_knn.parameters(), lr=5e-4)
    # train_imputer(grin_geo, dataloader, geo_edge_index, geo_optim, 5)
    # train_imputer(grin_knn, dataloader, knn_edge_index, knn_optim, 5)
    # imputed_data_geo = impute_missing_data(grin_geo, dataloader, geo_edge_index)
    # imputed_data_knn = impute_missing_data(grin_knn, dataloader, knn_edge_index)

    evaluate(
        imputed_data_geo.numpy(),
        dataset.data.numpy(),
        dataset.validation_mask.numpy(),
    )
    evaluate(
        imputed_data_knn.numpy(), dataset.data.numpy(), dataset.validation_mask.numpy()
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)

    # TODO: Transform dataset into standardized format
    # TODO: Check for the presence of adjacency data, positional data etc, or have the user use arg
    # TODO: Add adjacency based method from GRIN
    # TODO: Add KNN based method from MPIN
