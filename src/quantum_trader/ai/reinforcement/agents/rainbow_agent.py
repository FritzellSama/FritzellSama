"""Rainbow DQN agent."""
from decimal import Decimal
from typing import Dict, List
import torch
import torch.nn as nn
from structlog import get_logger

logger = get_logger(__name__)

class RainbowAgent:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = float(config.get("gamma", 0.99))
        
    def select_action(self, state) -> int:
        return 0
        
    def update(self, batch) -> Dict[str, Decimal]:
        return {"loss": Decimal("0.01")}
