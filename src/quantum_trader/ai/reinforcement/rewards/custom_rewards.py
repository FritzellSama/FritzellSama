"""Custom Reward Functions for Trading RL."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    profit_weight: Decimal = Decimal("1.0")
    risk_penalty: Decimal = Decimal("0.1")
    transaction_cost: Decimal = Decimal("0.001")
    drawdown_penalty: Decimal = Decimal("0.5")

class SharpeReward:
    """Sharpe ratio based reward."""

    def __init__(self, window: int = 100):
        self.window = window
        self.returns = []

    def calculate(self, profit: Decimal) -> Decimal:
        """Calculate Sharpe-based reward."""
        self.returns.append(float(profit))
        if len(self.returns) > self.window:
            self.returns.pop(0)

        if len(self.returns) < 2:
            return profit

        returns_arr = np.array(self.returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr) + 1e-10
        sharpe = mean_return / std_return

        return Decimal(str(sharpe))

class RiskAdjustedReward:
    """Risk-adjusted reward function."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self.max_value = Decimal("0")

    def calculate(
        self,
        profit: Decimal,
        position_size: Decimal,
        volatility: Decimal,
        portfolio_value: Decimal
    ) -> Decimal:
        """Calculate risk-adjusted reward."""
        # Base profit reward
        reward = profit * self.config.profit_weight

        # Risk penalty
        risk = position_size * volatility
        reward -= risk * self.config.risk_penalty

        # Transaction cost
        reward -= abs(position_size) * self.config.transaction_cost

        # Drawdown penalty
        self.max_value = max(self.max_value, portfolio_value)
        if self.max_value > 0:
            drawdown = (self.max_value - portfolio_value) / self.max_value
            reward -= drawdown * self.config.drawdown_penalty

        logger.debug("Calculated risk-adjusted reward", extra={"reward": str(reward), "timestamp": datetime.now(timezone.utc).isoformat()})

        return reward
