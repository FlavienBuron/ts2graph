from typing import Callable, Optional

import torch
from ts2graph_rs import k_hop_graph as khgrs


def k_hop_graph_rs(
    x: torch.Tensor,
    num_nodes: int = 1,
    k: int = 1,
    bidirectional: bool = True,
    decay_name=None,
):
    time_steps, _ = x.shape

    edge_index, edge_weight = khgrs(time_steps, num_nodes, k, bidirectional, decay_name)

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
