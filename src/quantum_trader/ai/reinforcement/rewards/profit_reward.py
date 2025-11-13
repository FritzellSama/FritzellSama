"""Profit-based reward function."""
from decimal import Decimal
from typing import Dict
from structlog import get_logger

logger = get_logger(__name__)

class ProfitRewardCalculator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.scale = Decimal(str(config.get("scale", 1.0)))
        
    def calculate_reward(self, prev_value: Decimal, current_value: Decimal) -> Decimal:
        if prev_value > 0:
            return ((current_value - prev_value) / prev_value) * self.scale
        return Decimal("0")
