"""Risk-adjusted reward functions."""
from decimal import Decimal
from typing import Dict
from structlog import get_logger

logger = get_logger(__name__)

class RiskAdjustedReward:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.risk_aversion = Decimal(str(config.get("risk_aversion", 0.5)))
        
    def calculate(self, returns: Decimal, volatility: Decimal) -> Decimal:
        if volatility > 0:
            return returns / volatility - self.risk_aversion * volatility
        return returns
