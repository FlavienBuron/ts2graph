import torch
from torch.nn.functional import normalize
from torch_geometric.nn import knn_graph, radius_graph

from graphs_transformations.utils import get_percentile_k, get_percentile_radius


def from_knn(
    data: torch.Tensor,
    mask: torch.Tensor,
    k: int | float,
    loop=False,
    cosine=False,
) -> torch.Tensor:
    if torch.isnan(data).any():
        means = data.nanmean(dim=1, keepdim=True)
        data = torch.where(mask, data, means)

    # Step 2: Check for invalid values
    if torch.isnan(data).any() or torch.isinf(data).any():
        raise ValueError("Data tensor contains NaN or inf values after imputation.")

    # Step 3: Check for identical rows
    if torch.allclose(data, data[0]):
        raise ValueError("All rows in the data tensor are identical.")

    if cosine:
        data = normalize(input=data, p=2, dim=1)

    if type(k) is float:
        real_k = get_percentile_k(data, k, loop)
    elif type(k) is int:
        real_k = k
    else:
        real_k = int(k)

    edge_index = knn_graph(
        x=data, k=real_k, loop=(real_k == data.shape[1]), cosine=cosine
    )
    return edge_index


def from_radius(
    data: torch.Tensor,
    mask: torch.Tensor,
    radius: float,
    loop=False,
    cosine=False,
) -> torch.Tensor:
    if torch.isnan(data).any():
        means = data.nanmean(dim=1, keepdim=True)
        data = torch.where(mask, data, means)

    radius = get_percentile_radius(
        data=data, mask=mask, percentile=radius, cosine=cosine
    )

    # Step 2: Check for invalid values
    if torch.isnan(data).any() or torch.isinf(data).any():
        raise ValueError("Data tensor contains NaN or inf values after imputation.")

    # Step 3: Check for identical rows
    if torch.allclose(data, data[0]):
        raise ValueError("All rows in the data tensor are identical.")

    if cosine:
        data = normalize(input=data, p=2, dim=1)

    edge_index = radius_graph(
        x=data, r=radius, loop=loop, max_num_neighbors=data.shape[1] + 1
    )
    return edge_index
