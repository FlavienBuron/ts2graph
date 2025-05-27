from typing import Tuple

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
        model_type: str = "GCNConv",
        use_temporal: bool = False,
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, model_type):
            raise ValueError(f"Model type '{model_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, model_type)
        self.use_temporal = use_temporal

        out_dim = in_dim

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

    def _create_temporal_edges(
        self, time_steps: int, num_nodes: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create temporal edges connecting nodes across consecutive time steps"""
        temporal_edges = []

        for t in range(time_steps - 1):
            src = torch.arange(num_nodes, device=device) + t * num_nodes
            dst = torch.arange(num_nodes, device=device) + (t + 1) * num_nodes
            temporal_edges.append(torch.stack([src, dst], dim=0))

        temporal_edge_index = torch.cat(temporal_edges, dim=1)

        # Make bidirectional connections
        temporal_edge_index = torch.cat(
            [temporal_edge_index, temporal_edge_index.flip(0)], dim=1
        )

        # Uniform edge weights (could be made learnable)
        temporal_edge_weight = torch.ones(temporal_edge_index.shape[1], device=device)

        return temporal_edge_index, temporal_edge_weight

    def forward(self, x, edge_index, edge_weight):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        edge_weight: Graph edges weights (from adjacency matrix)
        """
        time_steps, num_nodes, features = x.shape
        device = x.device
        # ori_x = x.detach().clone()

        spatial_outputs = []

        for t in range(time_steps):
            x_t = x[t]
            for i, gnn_layer in enumerate(self.gnn_layers):
                x_t = gnn_layer(x_t, edge_index, edge_weight)
                if i < len(self.gnn_layers) - 1:
                    x_t = F.relu(x_t)
            spatial_outputs.append(x_t)

        # Stack to shape (time, nodes, out_dim)
        x = torch.stack(spatial_outputs, dim=0)

        # === Temporal GNN ===
        if self.use_temporal:
            # Flatten nodes over time into a single batch dimension
            x = x.reshape(time_steps * num_nodes, features)

            # Create temporal connections
            temporal_edge_index, temporal_edge_weight = self._create_temporal_edges(
                time_steps, num_nodes, device
            )

            # Apply temporal GNN layers
            for i, temp_gnn_layer in enumerate(self.temp_gnn_layers):
                x = temp_gnn_layer(x, temporal_edge_index, temporal_edge_weight)

                # Apply activation and dropout (except for last layer)
                if i < len(self.temp_gnn_layers) - 1:
                    x = F.relu(x)

            x = x.reshape(time_steps, num_nodes, -1)

        # x = self.layer_norm(x)
        x = torch.tanh(x)

        if torch.isnan(x).any():
            print("NaNs detected in imputed_x")
            print("Stats:", x.min(), x.max(), x.mean())
            raise ValueError("NaNs in model output")

        return x
