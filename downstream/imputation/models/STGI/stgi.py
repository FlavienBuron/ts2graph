import inspect
from time import perf_counter
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class STGI(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        layer_type: str = "GCNConv",
        use_spatial: bool = True,
        use_temporal: bool = False,
        temporal_graph_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, layer_type):
            raise ValueError(f"Model type '{layer_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, layer_type)
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.temporal_graph_fn = temporal_graph_fn

        if not use_spatial and not use_temporal:
            print(
                "WARNING: neither spatial not temporal aspects are set to be used. Defaulting to using spatial only"
            )
            use_spatial = True

        out_dim = in_dim

        if use_spatial:
            print("Building Spatial Block in STGI")
            self.gnn_layers = self._build_gnn_layers(
                ModelClass, in_dim, hidden_dim, out_dim, num_layers, **kwargs
            )

        if use_temporal:
            print("Building Temporal Block in STGI")
            self.temp_gnn_layers = self._build_gnn_layers(
                ModelClass, in_dim, hidden_dim, out_dim, num_layers, **kwargs
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
        """Helper method to build GNN layers"""

        layers = nn.ModuleList()
        init_signature = inspect.signature(ModelClass.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in init_signature.parameters}

        if num_layers == 1:
            layers.append(ModelClass(in_dim, out_dim, **kwargs))
        else:
            layers.append(ModelClass(in_dim, hidden_dim, **kwargs))
            for _ in range(num_layers - 2):
                layers.append(ModelClass(hidden_dim, hidden_dim, **kwargs))
            layers.append(ModelClass(hidden_dim, out_dim, **kwargs))

        return layers

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_weight: torch.Tensor,
        **kwargs,
    ):
        """
        x: (time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        edge_weight: Graph edges weights (from adjacency matrix)
        """
        time_steps, num_nodes, _ = x.shape
        ori_x = x.detach().clone()
        temporal_graph_time = 0.0

        # === Spatial GNN ===
        if self.use_spatial:
            spatial_outputs = []

            for t in range(time_steps):
                x_t = x[t]
                for i, gnn_layer in enumerate(self.gnn_layers):
                    if isinstance(gnn_layer, pyg_nn.GCNConv):
                        x_t = gnn_layer(x_t, spatial_edge_index, spatial_edge_weight)
                    else:
                        x_t = gnn_layer(x_t, spatial_edge_index)
                    if i < len(self.gnn_layers) - 1:
                        x_t = F.relu(x_t)
                spatial_outputs.append(x_t)

            # Stack to shape (time, nodes, out_dim)
            x = torch.stack(spatial_outputs, dim=0)

        if self.use_spatial and self.use_temporal:
            x[observed_mask] = ori_x[observed_mask]

        # === Temporal GNN ===
        if self.use_temporal:
            temporal_outputs = []

            for node_idx in range(num_nodes):
                # Get the time series for this node: shape (T, F)
                x_node = x[:, node_idx, :]
                temporal_graph_start = perf_counter()
                if self.temporal_graph_fn is not None:
                    temporal_edge_index, temporal_edge_weight = self.temporal_graph_fn(
                        x=x_node
                    )
                else:
                    temporal_edge_index = torch.empty((2, 0), dtype=torch.long)
                    temporal_edge_weight = torch.empty((0,), dtype=torch.float)
                temporal_graph_end = perf_counter()
                temporal_graph_time = temporal_graph_end - temporal_graph_start
                # Apply temporal GNN layers
                for i, temp_gnn_layers in enumerate(self.temp_gnn_layers):
                    x_node = temp_gnn_layers(
                        x_node, temporal_edge_index, temporal_edge_weight
                    )
                    if i < len(self.temp_gnn_layers) - 1:
                        x_node = F.relu(x_node)
                temporal_outputs.append(x_node)

            # Stack to shape (time, nodes, out_dim)
            x = torch.stack(temporal_outputs, dim=1)

        # x = self.layer_norm(x)
        # x = torch.tanh(x)

        if torch.isnan(x).any():
            print("NaNs detected in imputed_x")
            print("Stats:", x.min(), x.max(), x.mean())
            raise ValueError("NaNs in model output")

        return x, temporal_graph_time
