import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class STGI(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        lstm_hidden_dim,
        num_layers,
        model_type: str = "GCNConv",
        **kwargs,
    ):
        super(STGI, self).__init__()

        if not hasattr(pyg_nn, model_type):
            raise ValueError(f"Model type '{model_type}' not found in torch_geometric")

        ModelClass = getattr(pyg_nn, model_type)

        self.gnn1 = ModelClass(in_dim, hidden_dim, **kwargs)
        self.gnn2 = ModelClass(hidden_dim, hidden_dim * 2, **kwargs)
        self.gnn3 = ModelClass(hidden_dim * 2, in_dim, **kwargs)

        # Temporal Bi-GRU
        self.lstm = nn.LSTM(
            out_dim,
            lstm_hidden_dim,
            num_layers,
            # batch_first=True,
            bidirectional=True,
            proj_size=0,
        )

        # Decoder (MLP for imputation)
        self.decoder = nn.Linear(
            lstm_hidden_dim * 2, in_dim
        )  # *2 for bidirectional GRU

        self.gnn_decoder = nn.Linear(
            out_dim,
            in_dim,
        )

    def forward(self, x, edge_index, mask):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        mask: Binary mask (1 = observed, 0 = missing)
        """
        time_steps, nodes, features = x.shape
        ori_x = x.detach().clone()

        gnn_output = []

        for t in range(time_steps):
            x_t = x[t]
            x_res = x_t
            x_t = self.gnn1(x_t, edge_index)
            x_t = self.gnn2(x_t, edge_index)
            x_t = self.gnn3(x_t, edge_index)
            gnn_output.append(x_t)

        # Stack to shape (time, nodes, out_dim)
        x = torch.stack(gnn_output, dim=0)
        # Reshape for LSTM: (nodes, time, features)
        # x = x.reshape(1, 0, 2)

        # Apply Bi-GRU for temporal modeling
        # Output shape: (num_nodes, time, lstm_hidden_dim * 2)
        # x, _ = self.lstm(x)

        # Reshape back to (time, node, feature)
        # x = x.permute(1, 0, 2)

        # Decode missing values
        # Shape: (batch_size, time_steps, num_nodes, feature_dim)
        # imputed_x = self.gnn_decoder(x)
        imputed_x = x

        # Compute the batch MSE
        x_loss = torch.sum(mask * (imputed_x - ori_x) ** 2) / (torch.sum(mask) + 1e-8)
        x_final = torch.where(mask.bool(), ori_x, imputed_x)
        return x_final, x_loss
