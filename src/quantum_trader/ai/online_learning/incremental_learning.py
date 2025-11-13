"""Incremental Learning."""
import logging
from decimal import Decimal
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class IncrementalLearner:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    async def update(self, X: torch.Tensor, y: torch.Tensor, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        logger.info(f"Incremental update: loss={loss.item():.4f}")
        return {"loss": Decimal(str(loss.item()))}
