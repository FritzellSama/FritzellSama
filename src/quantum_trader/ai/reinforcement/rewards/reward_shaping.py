"""Reward shaping for better learning."""
from decimal import Decimal
from typing import Dict
from structlog import get_logger

logger = get_logger(__name__)

class RewardShaper:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.shape_weight = Decimal(str(config.get("shape_weight", 0.1)))
        
    def shape_reward(self, base_reward: Decimal, state_info: Dict) -> Decimal:
        shaped = base_reward
        if "risk" in state_info:
            risk_penalty = Decimal(str(state_info["risk"])) * self.shape_weight
            shaped -= risk_penalty
        return shaped
