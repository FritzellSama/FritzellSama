"""WaveNet architecture for time series."""
from decimal import Decimal
from typing import Dict
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class WaveNetModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        n_channels = config.get("n_channels", 64)
        kernel_size = config.get("kernel_size", 2)
        n_layers = config.get("n_layers", 10)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(
                nn.Conv1d(n_channels if i > 0 else 1, n_channels, kernel_size,
                         dilation=dilation, padding=dilation)
            )
            
        self.output = nn.Linear(n_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = x.mean(dim=-1)
        return self.output(x)
