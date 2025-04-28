import torch
from torch_geometric.utils import get_laplacian, to_dense_adj


def compute_laplacian_smoothness(x, edge_index, edge_weight=None, debug=False):
    batch_size, nodes = x.shape
    lap_edge_index, lap_edge_weight = get_laplacian(
        edge_index, edge_weight, normalization="sym"
    )
    laplacian = to_dense_adj(
        lap_edge_index, edge_attr=lap_edge_weight, max_num_nodes=nodes
    ).squeeze(0)

    # FIX: Force symmetry and positive semi-definiteness
    laplacian = 0.5 * (laplacian + laplacian.t())  # Ensure symmetry

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)  # Remove negative eigenvalues
    laplacian = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()

    if debug:
        # Check if Laplacian is symmetric
        is_symmetric = torch.allclose(laplacian, laplacian.t(), atol=1e-6)
        print(f"Laplacian is symmetric: {is_symmetric}")

        # Check eigenvalues to verify positive semi-definiteness
        try:
            eigenvalues = torch.linalg.eigvalsh(laplacian)
            min_eig = eigenvalues.min().item()
            max_eig = eigenvalues.max().item()
            print(f"Eigenvalue range: [{min_eig:.6f}, {max_eig:.6f}]")
            print(f"Any negative eigenvalues: {(eigenvalues < -1e-6).any().item()}")
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
    x_reshaped = x.unsqueeze(1)
    laplacian_expanded = laplacian.unsqueeze(0).expand(batch_size, -1, -1)
    smoothness = torch.bmm(
        torch.bmm(x_reshaped, laplacian_expanded), x_reshaped.transpose(1, 2)
    ).squeeze()
    return smoothness.sum().item()


def compute_edge_difference_smoothness(x, edge_index, edge_weight=None):
    row, col = edge_index
    x_row = x[:, row]
    x_col = x[:, col]
    diff = x_row - x_col
    sq_diff = diff**2
    if edge_weight is not None:
        edge_weight_expanded = edge_weight.unsqueeze(0)
        weighted_sq_diff = sq_diff * edge_weight_expanded
        smoothness = weighted_sq_diff.sum(dim=1)
    else:
        smoothness = sq_diff.sum(dim=1)
    return smoothness.sum().item()
