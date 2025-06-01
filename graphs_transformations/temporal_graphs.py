import torch


def k_hop_graph(
    time_steps: int,
    num_nodes: int,
    k: int = 1,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Construct a k-hop temporal graph: t -> {t+1, ..., t+k} over a dataset of shape [T, N, F],
    where each of the N nodes is a time series of length T
    """
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

    return edge_index
