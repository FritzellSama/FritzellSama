"""Informer Model for Long Sequences."""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class InformerModel(nn.Module):
    def __init__(self, input_dim: int, d_model: int, output_dim: int):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 8), 2)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.output_proj(x[:, -1, :])
