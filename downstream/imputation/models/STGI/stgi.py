import inspect
from time import perf_counter
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from einops import rearrange
from torch_geometric.utils import dense_to_sparse


class STGI(nn.Module):
    def __init__(
        self,
        adj,
        in_dim,
        hidden_dim,
        num_layers,
        layer_type: str = "GCNConv",
        use_spatial: bool = True,
        use_temporal: bool = False,
        temporal_graph_fn: Optional[Callable] = None,
        impute_only_holes=True,
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, layer_type):
            raise ValueError(f"Model type '{layer_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, layer_type)
        self.spatial_edge_index, self.spatial_edge_weight = dense_to_sparse(adj)
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.temporal_graph_fn = temporal_graph_fn

        self.impute_only_holes = impute_only_holes

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
        **kwargs,
    ):
        """
        x: (time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        edge_weight: Graph edges weights (from adjacency matrix)
        """
        x = rearrange(x, "batches steps nodes channels -> batches channels nodes steps")
        mask = rearrange(
            mask, "batches steps nodes channels -> batches channels nodes steps"
        )
        ori_x = x.detach().clone()
        ori_mask = mask.detach().clone()

        B, C, N, S = x.shape
        temporal_graph_times = []

        channel_outputs = []
        for channel in range(C):
            x_c = x[:, channel, :, :]  # [B, N, S]
            m_c = mask[:, channel, :, :]  # [B, N, S]
            # === Spatial GNN ===
            if self.use_spatial:
                spatial_outputs = torch.zeros_like(x_c)

                for batch in range(B):
                    for step in range(S):
                        x_t = x_c[batch, :, step].unsqueeze(-1)
                        for i, gnn_layer in enumerate(self.gnn_layers):
                            if isinstance(gnn_layer, pyg_nn.GCNConv):
                                x_t = gnn_layer(
                                    x_t,
                                    self.spatial_edge_index,
                                    self.spatial_edge_weight,
                                )
                            else:
                                x_t = gnn_layer(x_t, self.spatial_edge_index)
                            if i < len(self.gnn_layers) - 1:
                                x_t = F.relu(x_t)
                        spatial_outputs[batch, :, step] = x_t.squeeze(-1)

                # x_c = torch.where(m_c, x_c, spatial_outputs)
                x_c = spatial_outputs

            # === Temporal GNN ===
            if self.use_temporal:
                temporal_outputs = torch.zeros_like(x_c)

                for batch in range(B):
                    for node in range(N):
                        x_node = x_c[batch, node, :]
                        temporal_graph_start = perf_counter()
                        if self.temporal_graph_fn is not None:
                            temporal_edge_index, temporal_edge_weight = (
                                self.temporal_graph_fn(x=x_node)
                            )
                        else:
                            temporal_edge_index = torch.empty((2, 0), dtype=torch.long)
                            temporal_edge_weight = torch.empty((0,), dtype=torch.float)
                        temporal_graph_end = perf_counter()
                        temporal_graph_times.append(
                            temporal_graph_end - temporal_graph_start
                        )
                        # Apply temporal GNN layers
                        for i, temp_gnn_layers in enumerate(self.temp_gnn_layers):
                            x_node = temp_gnn_layers(
                                x_node, temporal_edge_index, temporal_edge_weight
                            )
                            if i < len(self.temp_gnn_layers) - 1:
                                x_node = F.relu(x_node)
                        temporal_outputs[batch, node, :]

                # x_c = torch.where(m_c, x_c, temporal_outputs)
                x_c = temporal_outputs

            channel_outputs.append(x_c)
        x_out = torch.stack(channel_outputs, dim=1)
        if self.impute_only_holes and not self.training:
            x_out = torch.where(ori_mask, ori_x, x_out)
        x_out = rearrange(
            x_out, "batches channels nodes steps -> batches steps nodes channels"
        )

        # x = self.layer_norm(x)
        # x = torch.tanh(x)

        if torch.isnan(x_out).any():
            print("NaNs detected in imputed_x")
            print("Stats:", x.min(), x.max(), x.mean())
            raise ValueError("NaNs in model output")

        temporal_graph_times = torch.tensor(temporal_graph_times)
        _pred = torch.tensor([])
        return x_out, _pred, temporal_graph_times
