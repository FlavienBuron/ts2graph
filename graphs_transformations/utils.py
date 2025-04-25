import torch
from torch_geometric.utils import get_laplacian, to_dense_adj


def compute_laplacian_smoothness(x, edge_index):
    lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization="sym")
    print(f"{lap_edge_index.shape=}")
    laplacian = to_dense_adj(
        lap_edge_index, edge_attr=lap_edge_weight, max_num_nodes=x.shape[1]
    )[0]
    print(f"{laplacian.shape=} {x.shape=}")
    smoothness = torch.trace(x.T @ laplacian @ x)
    return smoothness


def compute_edge_difference_smoothness(x, edge_index, edge_weigth=None):
    row, col = edge_index
    diff = x[row] - x[col]
    sq_diff = (diff**2).sum(dim=1)
    if edge_weigth is not None:
        return (sq_diff * edge_weigth).sum()
    return sq_diff.sum()
