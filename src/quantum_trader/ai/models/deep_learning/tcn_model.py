"""Temporal Convolutional Network."""
from decimal import Decimal
from typing import Dict
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class TCNModel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        input_channels = config.get("input_channels", 1)
        hidden_channels = config.get("hidden_channels", 64)
        kernel_size = config.get("kernel_size", 3)
        
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(hidden_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=-1)  # Global average pooling
        return self.fc(x)
