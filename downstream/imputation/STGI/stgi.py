import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.layer1 = ModelClass(in_dim, hidden_dim, **kwargs)
        self.layer2 = ModelClass(hidden_dim, out_dim, **kwargs)

        # Temporal Bi-GRU
        self.lstm = nn.LSTM(
            out_dim,
            lstm_hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            proj_size=0,
        )

        # Decoder (MLP for imputation)
        self.decoder = nn.Linear(
            lstm_hidden_dim * 2, in_dim
        )  # *2 for bidirectional GRU

    def forward(self, x, edge_index, mask):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        mask: Binary mask (1 = observed, 0 = missing)
        """
        print(f"{x.shape=}")
        time_steps, num_nodes, feature_dim = x.shape
        ori_x = x.detach().clone()
        x = x.reshape(-1, feature_dim)
        x = F.relu(self.layer1(x, edge_index))
        x = F.relu(self.layer2(x, edge_index))
        x = x.reshape(time_steps, num_nodes, -1)

        # Apply Bi-GRU for temporal modeling
        # Output shape: (batch_size, time_steps, num_nodes, lstm_hidden_dim * 2)
        x, _ = self.lstm(x)

        # Decode missing values
        # Shape: (batch_size, time_steps, num_nodes, feature_dim)
        imputed_x = self.decoder(x)

        # Compute the batch MSE
        x_loss = torch.sum(mask * (imputed_x - ori_x) ** 2) / (torch.sum(mask) + 1e-8)
        x_final = torch.where(mask.bool(), ori_x, imputed_x)
        return x_final, x_loss
