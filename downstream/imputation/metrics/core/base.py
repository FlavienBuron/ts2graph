import torch
import torch.nn.functional as F
from einops import reduce
from torch_geometric.utils import get_laplacian, to_dense_adj

epsilon = 1e-6


def mae(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return F.l1_loss(prediction, target, reduction=reduction)


def mse(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return F.mse_loss(prediction, target, reduction=reduction)


def mre(prediction: torch.Tensor, target: torch.Tensor, reduction="none"):
    return torch.abs(prediction - target) / (torch.abs(target) + epsilon)


def mape(prediction: torch.Tensor, target: torch.Tensor):
    return torch.abs((prediction - target) / target)


def wape(prediction: torch.Tensor, target: torch.Tensor):
    abs_diff = torch.abs(prediction - target)
    return abs_diff.sum() / (target.sum() + epsilon)


def smape(prediction: torch.Tensor, target: torch.Tensor):
    valid_mask = torch.abs(target) > epsilon
    abs_diff = torch.abs(prediction - target)
    abs_sum = torch.abs(prediction + target) + epsilon
    loss = 2 * abs_diff / abs_sum * valid_mask.float()
    return loss.sum() / valid_mask.sum()


def peak_prediction_loss(
    prediction: torch.Tensor, target: torch.Tensor, reduction: str = "none"
):
    true_obs_max = reduce(target, "b s n 1 -> b 1 n 1", "max")
    true_obs_min = reduce(target, "b s n 1 -> b 1 n 1", "min")
    target = torch.cat([true_obs_max, true_obs_min], dim=1)
    return F.mse_loss(prediction, target, reduction=reduction)


def wrap_loss_fn(base_loss):
    pass


def laplacian_smoothness(
    x, edge_index, edge_weight=None, mask=None, normalize=True, debug=False
):
    batch_size, nodes, features = x.shape
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

    if mask is not None:
        x = x.masked_fill(~mask, 0.0)

    # x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

    x_flat = (
        x.permute(0, 2, 1).reshape(batch_size * features, nodes).unsqueeze(1)
    )  # [B*F, 1, N]
    laplacian_expanded = laplacian.unsqueeze(0).expand(
        batch_size * features, -1, -1
    )  # [B*F, N, N]
    smoothness = torch.bmm(
        torch.bmm(x_flat, laplacian_expanded), x_flat.transpose(1, 2)
    ).squeeze()

    smoothness_total = smoothness.sum()

    if normalize:
        energy = torch.sum(x**2) + 1e-8
        return (smoothness_total / energy).item()
    else:
        return smoothness_total.item()


def edge_difference_smoothness(
    x, edge_index, edge_weight=None, mask=None, normalize=True
):
    B, N, F = x.shape
    row, col = edge_index  # [E]

    x_i = x[:, row, :]  # [B, E, F]
    x_j = x[:, col, :]
    diff = x_i - x_j
    sq_diff = diff**2  # [B, E, F]

    if mask is not None:
        m_i = mask[:, row, :]
        m_j = mask[:, col, :]
        edge_mask = m_i & m_j  # [B, E, F]
        sq_diff = sq_diff * edge_mask.float()

    if edge_weight is not None:
        w = edge_weight.view(1, -1, 1)  # [1, E, 1]
        sq_diff = sq_diff * w  # weighted squared diff

    smoothness = sq_diff.sum()

    if normalize:
        energy = (x**2).sum() + 1e-8
        return (smoothness / energy).item()
    return smoothness.item()
