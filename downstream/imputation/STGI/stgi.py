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
        use_temporal=False,
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, model_type):
            raise ValueError(f"Model type '{model_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, model_type)
        self.use_temporal = use_temporal

        self.gnn_layers = nn.ModuleList()

        out_dim = in_dim

        if num_layers == 1:
            self.gnn_layers.append(
                ModelClass(in_dim, out_dim, add_self_loops=False, **kwargs)
            )
        else:
            self.gnn_layers.append(
                ModelClass(in_dim, hidden_dim, add_self_loops=False, **kwargs)
            )
            for _ in range(num_layers - 2):
                self.gnn_layers.append(
                    ModelClass(hidden_dim, hidden_dim, add_self_loops=False, **kwargs)
                )
            self.gnn_layers.append(
                ModelClass(hidden_dim, out_dim, add_self_loops=False, **kwargs)
            )

        if use_temporal:
            print("Building Temporal Block in STGI")
            self.temp_gnn_layers = nn.ModuleList()
            if num_layers == 1:
                self.temp_gnn_layers.append(
                    ModelClass(in_dim, out_dim, add_self_loops=False, **kwargs)
                )
            else:
                self.temp_gnn_layers.append(
                    ModelClass(in_dim, hidden_dim, add_self_loops=False, **kwargs)
                )
                for _ in range(num_layers - 2):
                    self.temp_gnn_layers.append(
                        ModelClass(
                            hidden_dim, hidden_dim, add_self_loops=False, **kwargs
                        )
                    )
                self.temp_gnn_layers.append(
                    ModelClass(hidden_dim, out_dim, add_self_loops=False, **kwargs)
                )

    def forward(self, x, edge_index, edge_weight, missing_mask):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        edge_weight: Graph edges weights (from adjacency matrix)
        mask: Binary mask (1 = observed, 0 = missing)
        """
        time, nodes, features = x.shape
        # ori_x = x.detach().clone()

        gnn_output = []

        for t in range(time):
            x_t = x[t]
            for i, gnn in enumerate(self.gnn_layers):
                x_t = gnn(x_t, edge_index, edge_weight)
                if i < len(self.gnn_layers) - 1:
                    x_t = F.relu(x_t)
            gnn_output.append(x_t)

        # Stack to shape (time, nodes, out_dim)
        x = torch.stack(gnn_output, dim=0)

        # === Temporal GNN ===
        if self.use_temporal:
            # Flatten nodes over time into a single batch dimension
            x = x.reshape(time * nodes, features)

            # Temporal edges: connect node i at t to itself at t+1
            temporal_edge_index = []
            for t in range(time - 1):
                src = torch.arange(nodes) + t * nodes
                dst = torch.arange(nodes) + (t + 1) * nodes
                temporal_edge_index.append(torch.stack([src, dst], dim=0))

            temporal_edge_index = torch.cat(temporal_edge_index, dim=1)  # shape [2, E]
            temporal_edge_index = torch.cat(
                [temporal_edge_index, temporal_edge_index[[1, 0]]], dim=1
            )  # Make bidirectional

            # Use uniform weights or learnable later
            temporal_edge_weight = torch.ones(
                temporal_edge_index.shape[1], device=x.device
            )

            for i, gnn in enumerate(self.temp_gnn_layers):
                x = gnn(x, temporal_edge_index, temporal_edge_weight)
                if i < len(self.temp_gnn_layers) - 1:
                    x = F.relu(x)

            x = x.reshape(time, nodes, -1)

        x = torch.tanh(x)

        if torch.isnan(x).any():
            print("NaNs detected in imputed_x")
            print("Stats:", x.min(), x.max(), x.mean())
            raise ValueError("NaNs in model output")

        # Compute the batch MSE only using non-missing data
        # x_loss = torch.sum(missing_mask * (imputed_x - ori_x) ** 2) / (
        #     torch.sum(missing_mask) + 1e-8
        # )
        # x_final = torch.where(missing_mask.bool(), ori_x, imputed_x)
        return x, 0
