"""Temporal Fusion Transformer."""
from decimal import Decimal
from typing import Dict
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 128)
        self.num_heads = config.get("num_heads", 4)
        
        self.attention = nn.MultiheadAttention(self.hidden_size, self.num_heads)
        self.fc = nn.Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        output = self.fc(attn_output.mean(dim=0))
        return output
