from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from pytorch_metric_learning.losses import NTXentLoss


class STGN_C(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        layer_type: str = "GCNConv",
        use_spatial=True,
        use_temporal: bool = False,
        temporal_graph_fn: Optional[Callable] = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> None:
        super(STGN_C, self).__init__()

        if not hasattr(pyg_nn, layer_type):
            raise ValueError(f"Layer type '{layer_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, layer_type)
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.temporal_graph_fn = temporal_graph_fn
        self.loss_fn = NTXentLoss(temperature=temperature)

        if not use_spatial and not use_temporal:
            print(
                "WARNING: neither spatial not temporal aspects are set to be used. Defaulting to useing spatial only"
            )
            use_spatial = True
        out_dim = in_dim

        self.spatial_block = (
            self._build_gnn_layers(
                ModelClass, in_dim, hidden_dim, out_dim, num_layers, **kwargs
            )
            if use_spatial
            else None
        )
        self.temporal_block = (
            self._build_gnn_layers(
                ModelClass, in_dim, hidden_dim, out_dim, num_layers, **kwargs
            )
            if use_temporal
            else None
        )

    def _build_gnn_layers(
        self,
        ModelClass,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        **kwargs,
    ) -> nn.ModuleList:
        layers = nn.ModuleList()

        if num_layers == 1:
            layers.append(ModelClass(in_dim, out_dim, add_self_loops=False, **kwargs))
        else:
            layers.append(
                ModelClass(in_dim, hidden_dim, add_self_loops=False, **kwargs)
            )
            for _ in range(num_layers - 2):
                layers.append(
                    ModelClass(hidden_dim, hidden_dim, add_self_loops=False, **kwargs)
                )
            layers.append(
                ModelClass(hidden_dim, out_dim, add_self_loops=False, **kwargs)
            )
        return layers

    def _encode(
        self,
        x,
        spatial_edge_index,
        spatial_edge_weight,
    ):
        time_steps, num_nodes, features = x.shape

        if self.spatial_block:
            spatial_output = []

            for t in range(time_steps):
                x_t = x[t]
                for i, gnn_layer in enumerate(self.spatial_block):
                    x_t = gnn_layer(x_t, spatial_edge_index, spatial_edge_weight)
                    if i < len(self.spatial_block) - 1:
                        x_t = F.relu(x_t)
                spatial_output.append(x_t)
            x = torch.stack(spatial_output, dim=0)

        if self.temporal_block:
            temporal_output = []

            for node_idx in range(num_nodes):
                x_node = x[:, node_idx, :]

                if self.temporal_graph_fn is not None:
                    temporal_edge_index, temporal_edge_weight = self.temporal_graph_fn(
                        x=x_node
                    )
                else:
                    temporal_edge_index = torch.empty((2, 0), dtype=torch.long)
                    temporal_edge_weight = torch.empty((0,), dtype=torch.float)

                for i, gnn_layer in enumerate(self.temporal_block):
                    x_node = gnn_layer(
                        x_node, temporal_edge_index, temporal_edge_weight
                    )
                    if i < len(self.temporal_block) - 1:
                        x_node = F.relu(x_node)
                temporal_output.append(x_node)

            x = torch.stack(temporal_output, dim=1)

        return x

    def _graph_readout(self, embedding):
        mean_pool = embedding.mean(dim=(0, 1))
        return F.normalize(mean_pool, dim=-1)

    def forward(
        self,
        view1,
        view2,
        spatial_edge_index,
        spatial_edge_weight,
    ):
        enc1 = self._encode(view1, spatial_edge_index, spatial_edge_weight)
        enc2 = self._encode(view2, spatial_edge_index, spatial_edge_weight)

        z1 = self._graph_readout(enc1)
        z2 = self._graph_readout(enc2)

        embedding = torch.cat([z1, z2], dim=0)

        N = z1.shape[0]
        a1 = torch.arange(0, N)
        p = torch.arange(N, 2 * N)
        indices_tuple = (a1, p, p, a1)

        loss = self.loss_fn(embedding, indices_tuple=indices_tuple)

        return loss
