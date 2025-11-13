"""Neural ODE for continuous-time modeling."""
from decimal import Decimal
from typing import Dict
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class NeuralODE(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.get("hidden_dim", 64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
