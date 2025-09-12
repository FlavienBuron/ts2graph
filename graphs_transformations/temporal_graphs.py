from typing import Callable, Optional

import numpy as np
import torch
from ts2graph_rs import k_hop_graph as khgrs
from ts2graph_rs import recurrence_graph


def k_hop_graph_rs(
    x: torch.Tensor,
    num_nodes: int = 1,
    k: int = 1,
    bidirectional: bool = True,
    decay_name=None,
):
    time_steps, _ = x.shape

    edge_index, edge_weight = khgrs(
        time_steps=time_steps,
        num_nodes=num_nodes,
        k=k,
        bidirectional=bidirectional,
        decay_name=decay_name,
    )

    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()

    return edge_index, edge_weight


def recurrence_graph_rs(
    x: torch.Tensor,
    radius: float,
    embedding_dim: Optional[int] = None,
    time_lag: int = 1,
    self_loop: bool = False,
):
    x = x.contiguous()
    # Ensure CPU and 1D float64 NumPy array
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy().astype(np.float64)
    elif isinstance(x, np.ndarray):
        x_np = x.astype(np.float64)
    else:
        raise TypeError("x must be a torch.Tensor or np.ndarray")

    x_np = np.ravel(x_np).astype(np.float64)
    edge_index, edge_weight = recurrence_graph(
        x_np, radius, embedding_dim, time_lag, self_loop
    )

    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()

    return edge_index, edge_weight


def k_hop_graph(
    x: torch.Tensor,
    num_nodes: int = 1,
    k: int = 1,
    bidirectional: bool = True,
    decay: Optional[Callable[[int, int], float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a k-hop temporal graph: t -> {t+1, ..., t+k} over a dataset of shape [T, N, F],
    where each of the N nodes is a time series of length T
    """
    time_steps, _ = x.shape
    if k == 0 or time_steps < 2:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge_index
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)

        return edge_index, edge_weight

    edge_list = []
    weight_list = []

    for node in range(num_nodes):
        node_offset = node * time_steps

        max_valid_k = time_steps - 1
        for offset in range(1, min(k, max_valid_k) + 1):
            src = torch.arange(time_steps - offset)
            dst = src + offset

            src += node_offset
            dst += node_offset

            edge_list.append(torch.stack([src, dst], dim=0))

            weight = decay(offset, min(k, max_valid_k)) if decay is not None else 1
            weight_tensor = torch.full((src.shape[0],), weight, dtype=torch.float)
            weight_list.append(weight_tensor)

            if bidirectional:
                edge_list.append(torch.stack([dst, src], dim=0))
                weight_list.append(weight_tensor)

    edge_index = torch.cat(edge_list, dim=1)
    edge_weight = torch.cat(weight_list)

    return edge_index, edge_weight
