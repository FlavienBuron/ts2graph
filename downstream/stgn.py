from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class STGN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        layer_type: str = "GCNConv",
        use_spatial: bool = True,
        use_temporal: bool = False,
        temporal_graph_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super(STGN).__init__()

        if not hasattr(pyg_nn, layer_type):
            raise ValueError(
                f"Given layer type '{layer_type} not found in torch_geometric"
            )

        ModelClass = getattr(pyg_nn, layer_type)
        self.temporal_graph_fn = temporal_graph_fn

        if not use_spatial and not use_temporal:
            print(
                "WARNING: neither spatial nor temporal aspects are set to be used. Defaulting to using spatial only"
            )
            use_spatial = True

        self.spatial_layers = (
            self._build_gnn_layers(
                ModelClass, in_dim, hidden_dim, out_dim, num_layers, **kwargs
            )
            if use_spatial
            else None
        )

        self.temporal_layers = (
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

    def apply_spatial_layers(
        self,
        x,
        spatial_edge_index,
        spatial_edge_weight,
    ):
        time_steps = x.shape[0]

        if self.spatial_layers:
            spatial_output = []

            for t in range(time_steps):
                x_t = x[t]
                for i, gnn_layer in enumerate(self.spatial_layers):
                    x_t = gnn_layer(x_t, spatial_edge_index, spatial_edge_weight)
                    if i < len(self.spatial_layers) - 1:
                        x_t = F.relu(x_t)
                spatial_output.append(x_t)
            x = torch.stack(spatial_output, dim=0)

        return x

    def apply_temporal_layers(
        self,
        x,
    ):
        num_nodes = x.shape[1]

        if self.temporal_layers:
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

                for i, gnn_layer in enumerate(self.temporal_layers):
                    x_node = gnn_layer(
                        x_node, temporal_edge_index, temporal_edge_weight
                    )
                    if i < len(self.temporal_layers) - 1:
                        x_node = F.relu(x_node)
                temporal_output.append(x_node)
            x = torch.stack(temporal_output, dim=1)

        return x

    def apply_gnn_layers(
        self,
        x,
        spatial_edge_index,
        spatial_edge_weight,
    ):
        x = self.apply_spatial_layers(
            x=x,
            spatial_edge_index=spatial_edge_index,
            spatial_edge_weight=spatial_edge_weight,
        )

        x = self.apply_temporal_layers(x=x)
