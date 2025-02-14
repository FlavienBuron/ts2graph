import torch
from torch import Tensor
from torch_geometric.nn import knn_graph


def from_knn(data: Tensor, k: int, temportal: bool = False) -> Tensor:
    # print(f"{data.var(dim=1)=} {torch.min(data.var(dim=1))=}")
    if torch.isnan(data).any():
        data = torch.nan_to_num(data, nan=0.0)
    # data = (data - data.mean()) / (data.std() + 1e-8)
    edge_index = knn_graph(x=data, k=k)
    return edge_index
