"""Graph Convolutional Network."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.matmul(adj, x))

class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, out_dim)
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x, adj))
        return self.layer2(x, adj)
