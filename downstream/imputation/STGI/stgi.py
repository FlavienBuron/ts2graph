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
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, model_type):
            raise ValueError(f"Model type '{model_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, model_type)

        self.gnn_layers = nn.ModuleList()

        if num_layers == 1:
            self.gnn_layers.append(ModelClass(in_dim, in_dim, **kwargs))
        else:
            self.gnn_layers.append(ModelClass(in_dim, hidden_dim, **kwargs))
            for _ in range(num_layers - 2):
                self.gnn_layers.append(ModelClass(hidden_dim, hidden_dim, **kwargs))
            self.gnn_layers.append(ModelClass(hidden_dim, in_dim, **kwargs))

    def forward(self, x, edge_index, edge_weight, missing_mask):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        edge_weight: Graph edges weights (from adjacency matrix)
        mask: Binary mask (1 = observed, 0 = missing)
        """
        time_steps, nodes, _ = x.shape
        # ori_x = x.detach().clone()

        gnn_output = []

        for t in range(time_steps):
            x_t = x[t]
            for i, gnn in enumerate(self.gnn_layers):
                x_t = gnn(x_t, edge_index, edge_weight)
                if i < len(self.gnn_layers) - 1:
                    x_t = F.relu(x_t)
            gnn_output.append(x_t)

        # Stack to shape (time, nodes, out_dim)
        x = torch.stack(gnn_output, dim=0)
        imputed_x = torch.tanh(x)

        if torch.isnan(imputed_x).any():
            print("NaNs detected in imputed_x")
            print("Stats:", imputed_x.min(), imputed_x.max(), imputed_x.mean())
            raise ValueError("NaNs in model output")

        # Compute the batch MSE only using non-missing data
        # x_loss = torch.sum(missing_mask * (imputed_x - ori_x) ** 2) / (
        #     torch.sum(missing_mask) + 1e-8
        # )
        # x_final = torch.where(missing_mask.bool(), ori_x, imputed_x)
        return x, 0
