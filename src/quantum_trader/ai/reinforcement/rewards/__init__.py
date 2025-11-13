"""Reward functions for reinforcement learning agents.

This module contains various reward function implementations for training
reinforcement learning agents in trading environments. Reward functions
define the learning objective and significantly impact agent behavior.

Includes implementations for:
- Profit/Loss-based rewards
- Sharpe ratio rewards
- Risk-adjusted returns
- Drawdown penalties
- Customizable composite rewards
"""

from typing import Callable, Optional

__all__ = [
    "RewardFunction",
    "ProfitReward",
    "SharpeRatioReward",
    "RiskAdjustedReward",
    "CompositeReward",
]

# Placeholder exports - import actual classes when available
# from .base_reward import RewardFunction
# from .profit_reward import ProfitReward
# from .sharpe_reward import SharpeRatioReward
# from .risk_adjusted_reward import RiskAdjustedReward
# from .composite_reward import CompositeReward

__version__ = "1.0.0"
