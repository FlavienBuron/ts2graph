import torch
from torch_geometric.utils import get_laplacian, to_dense_adj


def compute_laplacian_smoothness(x, edge_index):
    batch_size, nodes = x.shape
    lap_edge_index, lap_edge_weight = get_laplacian(edge_index, normalization="sym")
    laplacian = to_dense_adj(
        lap_edge_index, edge_attr=lap_edge_weight, max_num_nodes=nodes
    )[0]
    x_reshaped = x.unsqueeze(1)
    laplacian_expanded = laplacian.unsqueeze(0).expand(batch_size, -1, -1)
    smoothness = torch.bmm(
        torch.bmm(x_reshaped, laplacian_expanded), x_reshaped.transpose(1, 2)
    ).squeeze()
    return smoothness.mean()


def compute_edge_difference_smoothness(x, edge_index, edge_weigth=None):
    row, col = edge_index
    x_row = x[:, row]
    x_col = x[:, col]
    diff = x_row - x_col
    sq_diff = diff**2
    if edge_weigth is not None:
        edge_weigth_expanded = edge_weigth.unsqueeze(0)
        weighted_sq_diff = sq_diff * edge_weigth_expanded
        smoothness = weighted_sq_diff.sum(dim=1)
    else:
        smoothness = sq_diff.sum(dim=1)
    return smoothness
