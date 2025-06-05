import torch


def k_hop_graph(
    x: torch.Tensor,
    num_nodes: int,
    k: int = 1,
    bidirectional: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a k-hop temporal graph: t -> {t+1, ..., t+k} over a dataset of shape [T, N, F],
    where each of the N nodes is a time series of length T
    """
    time_steps, _ = x.shape
    if k == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # Empty edge_index
        edge_weight = torch.ones(edge_index.shape[1])

        return edge_index, edge_weight

    edge_list = []

    for node in range(num_nodes):
        node_offset = node * time_steps

        for offset in range(1, k + 1):
            src = torch.arange(time_steps - offset)
            dst = src + offset

            src += node_offset
            dst += node_offset

            edge_list.append(torch.stack([src, dst], dim=0))

            if bidirectional:
                edge_list.append(torch.stack([dst, src], dim=0))

    edge_index = torch.cat(edge_list, dim=1)
    edge_weight = torch.ones(edge_index.shape[1])

    return edge_index, edge_weight
