import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class STGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, gcn_out_dim, lstm_hidden_dim, num_layers):
        super(STGI, self).__init__()

        # Spatial GCN
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, gcn_out_dim)

        # Temporal Bi-GRU
        self.lstm = nn.LSTM(
            gcn_out_dim,
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

    def forward(self, x_missing, edge_index, mask):
        """
        x: (batch_size, time_steps, num_nodes, feature_dim)
        edge_index: Graph edges (from adjacency matrix)
        mask: Binary mask (1 = observed, 0 = missing)
        """
        # print(f"{x_missing.shape=}")
        time_steps, num_nodes, feature_dim = x_missing.shape
        # x = x.view(-1, num_nodes, feature_dim)  # Reshape for GCN
        x = x_missing.reshape(-1, feature_dim)
        # Apply GCN to spatial features
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = x.reshape(time_steps, num_nodes, -1)

        # Apply Bi-GRU for temporal modeling
        x, _ = self.lstm(
            x
        )  # Output shape: (batch_size, time_steps, num_nodes, lstm_hidden_dim * 2)

        # Decode missing values
        imputed_x = self.decoder(
            x
        )  # Shape: (batch_size, time_steps, num_nodes, feature_dim)
        # print(f"{imputed_x.shape=} {x_missing.shape=}")
        # Ensure only missing values are replaced
        # x_final = mask * x_missing + (1 - mask) * imputed_x
        x_loss = torch.sum(mask * (imputed_x - x_missing) ** 2) / (
            torch.sum(mask) + 1e-8
        )
        x_final = torch.where(mask.bool(), x_missing, imputed_x)
        # print(f"{x_final.shape=}")
        return x_final, x_loss
