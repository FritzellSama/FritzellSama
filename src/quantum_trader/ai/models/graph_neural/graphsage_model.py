"""GraphSAGE Model."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim * 2, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.matmul(adj, x) / (adj.sum(dim=1, keepdim=True) + 1e-10)
        return self.fc(torch.cat([x, neigh], dim=-1))

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x, adj))
        return self.layer2(x, adj)
