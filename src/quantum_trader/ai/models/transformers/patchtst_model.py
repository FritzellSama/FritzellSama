"""PatchTST transformer model."""
from decimal import Decimal
from typing import Dict
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class PatchTST(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.patch_len = config.get("patch_len", 16)
        self.stride = config.get("stride", 8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
