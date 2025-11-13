"""Custom reward functions for reinforcement learning trading agents.

This module provides various reward function implementations for training
RL agents in trading environments, including risk-adjusted returns, drawdown
penalties, and multi-objective rewards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BaseReward(ABC):
    """Abstract base class for reward functions.

    All reward functions must inherit from this class and implement
    the calculate method.

    Attributes:
        config: Configuration dictionary for the reward function
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the reward function.

        Args:
            config: Configuration dictionary with reward parameters

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        logger.info(
            "reward_function_initialized",
            reward_type=self.__class__.__name__,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    @abstractmethod
    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate reward for a state transition.

        Args:
            state: Current state dictionary
            action: Action taken
            next_state: Next state after action

        Returns:
            Reward value as Decimal

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass


class ProfitReward(BaseReward):
    """Simple profit-based reward function.

    Rewards based on realized profit/loss from trades.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize profit reward.

        Args:
            config: Configuration with parameters:
                - reward_scale: Scaling factor for reward (default: 1.0)
                - penalty_per_trade: Fixed penalty per trade (default: 0.0)

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        self.reward_scale = Decimal(str(config.get("reward_scale", "1.0")))
        self.penalty_per_trade = Decimal(str(config.get("penalty_per_trade", "0.0")))

        logger.info(
            "profit_reward_initialized",
            reward_scale=str(self.reward_scale),
            penalty_per_trade=str(self.penalty_per_trade),
        )

    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate reward based on profit.

        Args:
            state: Current state with 'balance' key
            action: Action taken (0=hold, 1=buy, 2=sell)
            next_state: Next state with 'balance' key

        Returns:
            Reward based on balance change

        Raises:
            ValueError: If required keys missing
        """
        try:
            current_balance = Decimal(str(state.get("balance", 0)))
            next_balance = Decimal(str(next_state.get("balance", 0)))

            # Calculate profit
            profit = next_balance - current_balance

            # Apply scaling
            reward = profit * self.reward_scale

            # Apply trade penalty if action taken
            if action != 0:  # Not holding
                reward -= self.penalty_per_trade

            logger.debug(
                "profit_reward_calculated",
                profit=str(profit),
                reward=str(reward),
                action=action,
            )

            return reward

        except Exception as e:
            logger.error("profit_reward_calculation_failed", error=str(e))
            raise


class SharpeRatioReward(BaseReward):
    """Risk-adjusted reward based on Sharpe ratio.

    Rewards consider both returns and volatility.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Sharpe ratio reward.

        Args:
            config: Configuration with parameters:
                - risk_free_rate: Annual risk-free rate (default: 0.02)
                - window_size: Window for calculating statistics (default: 30)
                - reward_scale: Scaling factor (default: 1.0)

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        self.risk_free_rate = Decimal(str(config.get("risk_free_rate", "0.02")))
        self.window_size = int(config.get("window_size", 30))
        self.reward_scale = Decimal(str(config.get("reward_scale", "1.0")))

        # Track returns history
        self.returns_history: List[Decimal] = []

        logger.info(
            "sharpe_reward_initialized",
            risk_free_rate=str(self.risk_free_rate),
            window_size=self.window_size,
        )

    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate Sharpe ratio-based reward.

        Args:
            state: Current state with 'portfolio_value' key
            action: Action taken
            next_state: Next state with 'portfolio_value' key

        Returns:
            Sharpe ratio-based reward

        Raises:
            ValueError: If calculation fails
        """
        try:
            current_value = Decimal(str(state.get("portfolio_value", 1)))
            next_value = Decimal(str(next_state.get("portfolio_value", 1)))

            # Calculate return
            if current_value > Decimal("0.0"):
                period_return = (next_value - current_value) / current_value
            else:
                period_return = Decimal("0.0")

            # Add to history
            self.returns_history.append(period_return)
            if len(self.returns_history) > self.window_size:
                self.returns_history.pop(0)

            # Calculate Sharpe ratio if enough data
            if len(self.returns_history) >= self.window_size:
                mean_return = sum(self.returns_history) / Decimal(
                    str(len(self.returns_history))
                )

                # Calculate standard deviation
                variance = sum(
                    (r - mean_return) ** Decimal("2") for r in self.returns_history
                ) / Decimal(str(len(self.returns_history)))
                std_dev = variance ** Decimal("0.5")

                # Daily risk-free rate
                daily_rf = self.risk_free_rate / Decimal("252")

                # Sharpe ratio
                if std_dev > Decimal("0.0"):
                    sharpe = (mean_return - daily_rf) / std_dev
                else:
                    sharpe = Decimal("0.0")

                reward = sharpe * self.reward_scale
            else:
                # Not enough data yet, use simple return
                reward = period_return * self.reward_scale

            logger.debug(
                "sharpe_reward_calculated",
                reward=str(reward),
                returns_count=len(self.returns_history),
            )

            return reward

        except Exception as e:
            logger.error("sharpe_reward_calculation_failed", error=str(e))
            raise


class DrawdownPenaltyReward(BaseReward):
    """Reward with drawdown penalty.

    Penalizes large drawdowns to encourage risk management.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize drawdown penalty reward.

        Args:
            config: Configuration with parameters:
                - base_reward_scale: Scale for base profit reward (default: 1.0)
                - drawdown_penalty_scale: Scale for drawdown penalty (default: 2.0)
                - max_drawdown_threshold: Max allowed drawdown before penalty (default: 0.1)

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        self.base_reward_scale = Decimal(str(config.get("base_reward_scale", "1.0")))
        self.drawdown_penalty_scale = Decimal(
            str(config.get("drawdown_penalty_scale", "2.0"))
        )
        self.max_drawdown_threshold = Decimal(
            str(config.get("max_drawdown_threshold", "0.1"))
        )

        # Track peak portfolio value
        self.peak_value = Decimal("0.0")

        logger.info(
            "drawdown_penalty_reward_initialized",
            drawdown_threshold=str(self.max_drawdown_threshold),
        )

    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate reward with drawdown penalty.

        Args:
            state: Current state with 'portfolio_value' key
            action: Action taken
            next_state: Next state with 'portfolio_value' key

        Returns:
            Reward with drawdown penalty applied

        Raises:
            ValueError: If calculation fails
        """
        try:
            current_value = Decimal(str(state.get("portfolio_value", 0)))
            next_value = Decimal(str(next_state.get("portfolio_value", 0)))

            # Update peak value
            if next_value > self.peak_value:
                self.peak_value = next_value

            # Calculate base reward (profit)
            profit = next_value - current_value
            base_reward = profit * self.base_reward_scale

            # Calculate drawdown
            if self.peak_value > Decimal("0.0"):
                drawdown = (self.peak_value - next_value) / self.peak_value
            else:
                drawdown = Decimal("0.0")

            # Apply penalty if drawdown exceeds threshold
            if drawdown > self.max_drawdown_threshold:
                excess_drawdown = drawdown - self.max_drawdown_threshold
                penalty = excess_drawdown * self.drawdown_penalty_scale
                reward = base_reward - penalty
            else:
                reward = base_reward

            logger.debug(
                "drawdown_penalty_reward_calculated",
                base_reward=str(base_reward),
                drawdown=str(drawdown),
                reward=str(reward),
            )

            return reward

        except Exception as e:
            logger.error("drawdown_penalty_reward_failed", error=str(e))
            raise


class MultiObjectiveReward(BaseReward):
    """Multi-objective reward combining multiple factors.

    Combines profit, risk, trading frequency, and other objectives.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize multi-objective reward.

        Args:
            config: Configuration with parameters:
                - profit_weight: Weight for profit component (default: 1.0)
                - risk_weight: Weight for risk penalty (default: 0.5)
                - trade_freq_weight: Weight for trading frequency penalty (default: 0.1)
                - position_size_weight: Weight for position size penalty (default: 0.2)

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        self.profit_weight = Decimal(str(config.get("profit_weight", "1.0")))
        self.risk_weight = Decimal(str(config.get("risk_weight", "0.5")))
        self.trade_freq_weight = Decimal(str(config.get("trade_freq_weight", "0.1")))
        self.position_size_weight = Decimal(
            str(config.get("position_size_weight", "0.2"))
        )

        # State tracking
        self.trade_count = 0
        self.total_steps = 0

        logger.info(
            "multi_objective_reward_initialized",
            profit_weight=str(self.profit_weight),
            risk_weight=str(self.risk_weight),
        )

    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate multi-objective reward.

        Args:
            state: Current state with portfolio metrics
            action: Action taken
            next_state: Next state with portfolio metrics

        Returns:
            Combined multi-objective reward

        Raises:
            ValueError: If calculation fails
        """
        try:
            self.total_steps += 1
            if action != 0:  # Not holding
                self.trade_count += 1

            # Profit component
            current_value = Decimal(str(state.get("portfolio_value", 0)))
            next_value = Decimal(str(next_state.get("portfolio_value", 0)))
            profit = next_value - current_value
            profit_reward = profit * self.profit_weight

            # Risk component (portfolio volatility)
            volatility = Decimal(str(next_state.get("volatility", 0)))
            risk_penalty = volatility * self.risk_weight

            # Trading frequency penalty
            if self.total_steps > 0:
                trade_frequency = Decimal(str(self.trade_count)) / Decimal(
                    str(self.total_steps)
                )
            else:
                trade_frequency = Decimal("0.0")
            freq_penalty = trade_frequency * self.trade_freq_weight

            # Position size penalty (discourage over-leveraging)
            position_size = Decimal(str(next_state.get("position_size", 0)))
            max_position = Decimal(str(next_state.get("max_position_size", 1)))
            if max_position > Decimal("0.0"):
                position_ratio = position_size / max_position
                if position_ratio > Decimal("0.8"):  # Over 80% max position
                    size_penalty = (position_ratio - Decimal("0.8")) * self.position_size_weight
                else:
                    size_penalty = Decimal("0.0")
            else:
                size_penalty = Decimal("0.0")

            # Combine components
            reward = profit_reward - risk_penalty - freq_penalty - size_penalty

            logger.debug(
                "multi_objective_reward_calculated",
                profit_reward=str(profit_reward),
                risk_penalty=str(risk_penalty),
                freq_penalty=str(freq_penalty),
                size_penalty=str(size_penalty),
                total_reward=str(reward),
            )

            return reward

        except Exception as e:
            logger.error("multi_objective_reward_failed", error=str(e))
            raise


class InformationRatioReward(BaseReward):
    """Reward based on Information Ratio.

    Measures risk-adjusted excess returns relative to a benchmark.

    Attributes:
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize information ratio reward.

        Args:
            config: Configuration with parameters:
                - benchmark_return: Expected benchmark return (default: 0.0)
                - window_size: Window for statistics (default: 30)
                - reward_scale: Scaling factor (default: 1.0)

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        self.benchmark_return = Decimal(str(config.get("benchmark_return", "0.0")))
        self.window_size = int(config.get("window_size", 30))
        self.reward_scale = Decimal(str(config.get("reward_scale", "1.0")))

        # Track excess returns
        self.excess_returns: List[Decimal] = []

        logger.info(
            "information_ratio_reward_initialized",
            benchmark_return=str(self.benchmark_return),
            window_size=self.window_size,
        )

    def calculate(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> Decimal:
        """Calculate information ratio-based reward.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Information ratio-based reward

        Raises:
            ValueError: If calculation fails
        """
        try:
            current_value = Decimal(str(state.get("portfolio_value", 1)))
            next_value = Decimal(str(next_state.get("portfolio_value", 1)))

            # Calculate return
            if current_value > Decimal("0.0"):
                period_return = (next_value - current_value) / current_value
            else:
                period_return = Decimal("0.0")

            # Calculate excess return
            excess_return = period_return - self.benchmark_return

            # Add to history
            self.excess_returns.append(excess_return)
            if len(self.excess_returns) > self.window_size:
                self.excess_returns.pop(0)

            # Calculate information ratio if enough data
            if len(self.excess_returns) >= self.window_size:
                mean_excess = sum(self.excess_returns) / Decimal(
                    str(len(self.excess_returns))
                )

                # Calculate tracking error (std of excess returns)
                variance = sum(
                    (r - mean_excess) ** Decimal("2") for r in self.excess_returns
                ) / Decimal(str(len(self.excess_returns)))
                tracking_error = variance ** Decimal("0.5")

                # Information ratio
                if tracking_error > Decimal("0.0"):
                    info_ratio = mean_excess / tracking_error
                else:
                    info_ratio = Decimal("0.0")

                reward = info_ratio * self.reward_scale
            else:
                # Not enough data, use excess return
                reward = excess_return * self.reward_scale

            logger.debug(
                "information_ratio_reward_calculated",
                reward=str(reward),
                excess_returns_count=len(self.excess_returns),
            )

            return reward

        except Exception as e:
            logger.error("information_ratio_reward_failed", error=str(e))
            raise


class RewardFactory:
    """Factory for creating reward functions.

    Provides centralized reward function instantiation.
    """

    _reward_types = {
        "profit": ProfitReward,
        "sharpe": SharpeRatioReward,
        "drawdown_penalty": DrawdownPenaltyReward,
        "multi_objective": MultiObjectiveReward,
        "information_ratio": InformationRatioReward,
    }

    @classmethod
    def create(cls, reward_type: str, config: Dict[str, Any]) -> BaseReward:
        """Create a reward function instance.

        Args:
            reward_type: Type of reward function
            config: Configuration dictionary

        Returns:
            Reward function instance

        Raises:
            ValueError: If reward type is unknown
        """
        try:
            if reward_type not in cls._reward_types:
                raise ValueError(
                    f"Unknown reward type: {reward_type}. "
                    f"Available: {list(cls._reward_types.keys())}"
                )

            reward_class = cls._reward_types[reward_type]
            reward_instance = reward_class(config)

            logger.info("reward_created", reward_type=reward_type)

            return reward_instance

        except Exception as e:
            logger.error("reward_creation_failed", reward_type=reward_type, error=str(e))
            raise

    @classmethod
    def register_reward(cls, name: str, reward_class: type) -> None:
        """Register a custom reward function.

        Args:
            name: Name for the reward function
            reward_class: Reward class to register

        Raises:
            ValueError: If registration fails
        """
        try:
            if not issubclass(reward_class, BaseReward):
                raise ValueError(
                    f"Reward class must inherit from BaseReward, got {reward_class}"
                )

            cls._reward_types[name] = reward_class

            logger.info("custom_reward_registered", name=name, class_name=reward_class.__name__)

        except Exception as e:
            logger.error("reward_registration_failed", name=name, error=str(e))
            raise


__all__ = [
    "BaseReward",
    "ProfitReward",
    "SharpeRatioReward",
    "DrawdownPenaltyReward",
    "MultiObjectiveReward",
    "InformationRatioReward",
    "RewardFactory",
]
