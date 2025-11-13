"""Sharpe ratio-based reward."""
from decimal import Decimal
from typing import Dict, List
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

class SharpeRewardCalculator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.window = config.get("window", 20)
        self.returns_history: List[Decimal] = []
        
    def calculate_reward(self, returns: Decimal) -> Decimal:
        self.returns_history.append(returns)
        
        if len(self.returns_history) > self.window:
            self.returns_history.pop(0)
            
        if len(self.returns_history) >= 2:
            returns_array = np.array([float(r) for r in self.returns_history])
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            if std_return > 0:
                sharpe = mean_return / std_return
                return Decimal(str(sharpe))
                
        return Decimal("0")
