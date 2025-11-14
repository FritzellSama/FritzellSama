"""Profit-based reward function for trading reinforcement learning.

This module implements reward functions based on profit, returns, and risk-adjusted metrics.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RewardMetrics:
    """Metrics for reward calculation.

    Attributes:
        profit: Absolute profit/loss (Decimal)
        return_pct: Return percentage (Decimal)
        sharpe: Sharpe ratio (Decimal)
        drawdown: Current drawdown (Decimal)
        trade_count: Number of trades
        timestamp: UTC timestamp
    """
    profit: Decimal
    return_pct: Decimal
    sharpe: Decimal
    drawdown: Decimal
    trade_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class ProfitReward:
    """Profit-based reward calculator for trading RL.

    Calculates rewards based on realized and unrealized P&L, with optional
    risk adjustments and transaction cost penalties.

    Attributes:
        config: Configuration dictionary
        reward_scale: Scaling factor for rewards
        risk_penalty: Penalty coefficient for risk metrics
        transaction_cost_penalty: Penalty for trading costs
        drawdown_penalty: Penalty coefficient for drawdowns

    Example:
        >>> config = {
        ...     'reward_scale': 1000.0,
        ...     'risk_penalty': 0.1,
        ...     'transaction_cost_penalty': 0.01,
        ...     'drawdown_penalty': 0.5,
        ...     'normalize_rewards': True,
        ...     'clip_rewards': True,
        ...     'clip_min': -10.0,
        ...     'clip_max': 10.0
        ... }
        >>> reward_fn = ProfitReward(config)
        >>> reward = reward_fn.calculate(
        ...     current_value=Decimal('10500'),
        ...     previous_value=Decimal('10000'),
        ...     transaction_cost=Decimal('5'),
        ...     position_size=Decimal('1.0')
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize profit reward calculator.

        Args:
            config: Configuration with keys:
                - reward_scale: Scaling factor
                - risk_penalty: Risk penalty coefficient
                - transaction_cost_penalty: Trading cost penalty
                - drawdown_penalty: Drawdown penalty coefficient
                - normalize_rewards: Whether to normalize
                - clip_rewards: Whether to clip
                - clip_min: Minimum reward value
                - clip_max: Maximum reward value
        """
        self.config = config
        self._validate_config()

        self.reward_scale = Decimal(str(config['reward_scale']))
        self.risk_penalty = Decimal(str(config['risk_penalty']))
        self.transaction_cost_penalty = Decimal(str(config['transaction_cost_penalty']))
        self.drawdown_penalty = Decimal(str(config['drawdown_penalty']))
        self.normalize_rewards = config['normalize_rewards']
        self.clip_rewards = config['clip_rewards']
        self.clip_min = Decimal(str(config['clip_min']))
        self.clip_max = Decimal(str(config['clip_max']))

        # Track statistics for normalization
        self.reward_history: List[Decimal] = []
        self.max_history_size = config.get('max_history_size', 10000)

        # Track portfolio metrics
        self.peak_value = Decimal('0')
        self.total_transaction_costs = Decimal('0')

        logger.info(
            "initialized_profit_reward",
            reward_scale=str(self.reward_scale),
            risk_penalty=str(self.risk_penalty),
            normalize=self.normalize_rewards
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'reward_scale', 'risk_penalty', 'transaction_cost_penalty',
            'drawdown_penalty', 'normalize_rewards', 'clip_rewards',
            'clip_min', 'clip_max'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['reward_scale'] <= 0:
            raise ValueError("reward_scale must be positive")

        if self.config['clip_min'] >= self.config['clip_max']:
            raise ValueError("clip_min must be less than clip_max")

    def calculate(
        self,
        current_value: Decimal,
        previous_value: Decimal,
        transaction_cost: Decimal = Decimal('0'),
        position_size: Decimal = Decimal('0'),
        volatility: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate reward for current step.

        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            transaction_cost: Transaction costs incurred
            position_size: Current position size (for risk penalty)
            volatility: Current volatility (optional, for risk adjustment)

        Returns:
            Calculated reward (Decimal)

        Raises:
            ValueError: If values are invalid
        """
        try:
            if current_value <= 0 or previous_value <= 0:
                raise ValueError("Portfolio values must be positive")

            # Calculate raw profit
            profit = current_value - previous_value

            # Calculate return
            return_pct = (profit / previous_value) * Decimal('100')

            # Base reward: scaled return
            reward = return_pct * self.reward_scale

            # Penalty for transaction costs
            if transaction_cost > 0:
                cost_penalty = transaction_cost * self.transaction_cost_penalty * self.reward_scale
                reward -= cost_penalty
                self.total_transaction_costs += transaction_cost

                logger.debug(
                    "applied_transaction_cost_penalty",
                    cost=str(transaction_cost),
                    penalty=str(cost_penalty)
                )

            # Penalty for drawdown
            self.peak_value = max(self.peak_value, current_value)
            if self.peak_value > 0:
                drawdown = (self.peak_value - current_value) / self.peak_value
                if drawdown > 0:
                    drawdown_penalty = drawdown * self.drawdown_penalty * self.reward_scale
                    reward -= drawdown_penalty

                    logger.debug(
                        "applied_drawdown_penalty",
                        drawdown=str(drawdown * Decimal('100')),
                        penalty=str(drawdown_penalty)
                    )

            # Risk penalty based on position size and volatility
            if volatility is not None and volatility > 0:
                risk_metric = abs(position_size) * volatility
                risk_penalty_value = risk_metric * self.risk_penalty * self.reward_scale
                reward -= risk_penalty_value

                logger.debug(
                    "applied_risk_penalty",
                    position_size=str(position_size),
                    volatility=str(volatility),
                    penalty=str(risk_penalty_value)
                )

            # Store in history for normalization
            self.reward_history.append(reward)
            if len(self.reward_history) > self.max_history_size:
                self.reward_history.pop(0)

            # Normalize if configured
            if self.normalize_rewards and len(self.reward_history) > 10:
                reward = self._normalize_reward(reward)

            # Clip if configured
            if self.clip_rewards:
                reward = max(self.clip_min, min(self.clip_max, reward))

            logger.debug(
                "calculated_reward",
                profit=str(profit),
                return_pct=str(return_pct),
                reward=str(reward),
                current_value=str(current_value)
            )

            return reward

        except Exception as e:
            logger.error("failed_to_calculate_reward", error=str(e))
            raise

    def _normalize_reward(self, reward: Decimal) -> Decimal:
        """Normalize reward using running statistics.

        Args:
            reward: Raw reward value

        Returns:
            Normalized reward
        """
        # Calculate mean and std from history
        rewards_array = np.array([float(r) for r in self.reward_history])
        mean = Decimal(str(np.mean(rewards_array)))
        std = Decimal(str(np.std(rewards_array)))

        if std > Decimal('1e-8'):
            normalized = (reward - mean) / std
        else:
            normalized = reward

        return normalized

    def calculate_sharpe_reward(
        self,
        returns: List[Decimal],
        risk_free_rate: Decimal = Decimal('0')
    ) -> Decimal:
        """Calculate Sharpe ratio as reward.

        Args:
            returns: List of returns (Decimal)
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio reward

        Raises:
            ValueError: If returns list is empty
        """
        try:
            if not returns:
                raise ValueError("Returns list cannot be empty")

            returns_array = np.array([float(r) for r in returns])

            mean_return = Decimal(str(np.mean(returns_array)))
            std_return = Decimal(str(np.std(returns_array)))

            if std_return > Decimal('1e-8'):
                sharpe = (mean_return - risk_free_rate) / std_return
            else:
                sharpe = Decimal('0')

            # Scale Sharpe ratio for RL
            reward = sharpe * self.reward_scale

            logger.debug(
                "calculated_sharpe_reward",
                sharpe=str(sharpe),
                mean_return=str(mean_return),
                std_return=str(std_return),
                reward=str(reward)
            )

            return reward

        except Exception as e:
            logger.error("failed_to_calculate_sharpe_reward", error=str(e))
            raise

    def calculate_sortino_reward(
        self,
        returns: List[Decimal],
        target_return: Decimal = Decimal('0')
    ) -> Decimal:
        """Calculate Sortino ratio as reward.

        Args:
            returns: List of returns (Decimal)
            target_return: Target return threshold

        Returns:
            Sortino ratio reward
        """
        try:
            if not returns:
                raise ValueError("Returns list cannot be empty")

            returns_array = np.array([float(r) for r in returns])
            mean_return = Decimal(str(np.mean(returns_array)))

            # Calculate downside deviation
            downside_returns = returns_array[returns_array < float(target_return)]
            if len(downside_returns) > 0:
                downside_std = Decimal(str(np.std(downside_returns)))
            else:
                downside_std = Decimal('1e-8')

            if downside_std > Decimal('1e-8'):
                sortino = (mean_return - target_return) / downside_std
            else:
                sortino = Decimal('0')

            reward = sortino * self.reward_scale

            logger.debug(
                "calculated_sortino_reward",
                sortino=str(sortino),
                mean_return=str(mean_return),
                downside_std=str(downside_std),
                reward=str(reward)
            )

            return reward

        except Exception as e:
            logger.error("failed_to_calculate_sortino_reward", error=str(e))
            raise

    def calculate_calmar_reward(
        self,
        returns: List[Decimal],
        max_drawdown: Decimal
    ) -> Decimal:
        """Calculate Calmar ratio as reward.

        Args:
            returns: List of returns (Decimal)
            max_drawdown: Maximum drawdown observed

        Returns:
            Calmar ratio reward
        """
        try:
            if not returns or max_drawdown <= 0:
                return Decimal('0')

            returns_array = np.array([float(r) for r in returns])
            annualized_return = Decimal(str(np.mean(returns_array))) * Decimal('252')

            calmar = annualized_return / max_drawdown

            reward = calmar * self.reward_scale

            logger.debug(
                "calculated_calmar_reward",
                calmar=str(calmar),
                annualized_return=str(annualized_return),
                max_drawdown=str(max_drawdown),
                reward=str(reward)
            )

            return reward

        except Exception as e:
            logger.error("failed_to_calculate_calmar_reward", error=str(e))
            raise

    def reset(self) -> None:
        """Reset reward calculator state."""
        self.reward_history.clear()
        self.peak_value = Decimal('0')
        self.total_transaction_costs = Decimal('0')

        logger.info("reset_profit_reward")

    def get_stats(self) -> Dict[str, Any]:
        """Get reward statistics.

        Returns:
            Dictionary of reward statistics
        """
        if len(self.reward_history) > 0:
            rewards_array = np.array([float(r) for r in self.reward_history])
            mean_reward = Decimal(str(np.mean(rewards_array)))
            std_reward = Decimal(str(np.std(rewards_array)))
            min_reward = Decimal(str(np.min(rewards_array)))
            max_reward = Decimal(str(np.max(rewards_array)))
        else:
            mean_reward = Decimal('0')
            std_reward = Decimal('0')
            min_reward = Decimal('0')
            max_reward = Decimal('0')

        return {
            'mean_reward': str(mean_reward),
            'std_reward': str(std_reward),
            'min_reward': str(min_reward),
            'max_reward': str(max_reward),
            'history_size': len(self.reward_history),
            'peak_value': str(self.peak_value),
            'total_transaction_costs': str(self.total_transaction_costs)
        }
