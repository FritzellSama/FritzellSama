"""Few-Shot Learning."""
import logging
from decimal import Decimal
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class FewShotLearner:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    async def adapt(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.model(support_x)
            loss = criterion(outputs, support_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return self.model(query_x)
