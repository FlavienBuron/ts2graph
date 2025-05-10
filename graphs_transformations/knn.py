import torch
from torch import Tensor
from torch_geometric.nn import knn_graph


def from_knn(
    data: Tensor,
    mask: torch.Tensor,
    k: int,
    loop=False,
    cosine=False,
) -> torch.Tensor:
    # print(f"{data.var(dim=1)=} {torch.min(data.var(dim=1))=}")
    if torch.isnan(data).any():
        # data.nan_to_num_(nan=0.0)
        means = data.nanmean(dim=1)
        # means.nan_to_num_(0.0)
        data = torch.where(mask, data, means)
    # data = (data - data.mean()) / (data.std() + 1e-8)

    # Step 2: Check for invalid values
    if torch.isnan(data).any() or torch.isinf(data).any():
        raise ValueError("Data tensor contains NaN or inf values after imputation.")

    # Step 3: Check for identical rows
    if torch.allclose(data, data[0]):
        raise ValueError("All rows in the data tensor are identical.")
    edge_index = knn_graph(x=data, k=k, loop=loop, cosine=cosine)
    return edge_index
