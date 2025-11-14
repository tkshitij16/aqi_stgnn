import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TemporalEncoder(nn.Module):
    def __init__(self, dyn_dim, hidden):
        super().__init__()
        self.gru = nn.GRU(input_size=dyn_dim, hidden_size=hidden, num_layers=1, batch_first=True)

    def forward(self, x_dyn):  # [N,W,D]
        out, _ = self.gru(x_dyn)
        return out[:, -1, :]    # [N,H]

class STTGNN(nn.Module):
    def __init__(self, dyn_dim, static_dim, gnn_hidden=64, gru_hidden=64, gnn_layers=2, dropout=0.2):
        super().__init__()
        self.temporal = TemporalEncoder(dyn_dim, gru_hidden)
        in_dim = gru_hidden + static_dim
        self.g1 = GCNConv(in_dim, gnn_hidden, add_self_loops=False, normalize=False)
        self.g2 = GCNConv(gnn_hidden, gnn_hidden, add_self_loops=False, normalize=False) if gnn_layers>1 else None
        self.drop = nn.Dropout(dropout)
        self.head_aqi = nn.Linear(gnn_hidden, 1)
        self.head_pm  = nn.Linear(gnn_hidden, 1)

    def forward(self, x_dyn, x_stat, edge_index, edge_weight):
        z = self.temporal(x_dyn)
        h = torch.cat([z, x_stat], dim=1)
        h = F.relu(self.g1(h, edge_index, edge_weight=edge_weight))
        h = self.drop(h)
        if self.g2 is not None:
            h = F.relu(self.g2(h, edge_index, edge_weight=edge_weight))
            h = self.drop(h)
        return self.head_aqi(h).squeeze(-1), self.head_pm(h).squeeze(-1)
