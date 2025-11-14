"""Sharpe ratio-based reward function for reinforcement learning.

This module implements a reward function based on the Sharpe ratio,
a key metric for risk-adjusted returns in trading.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from datetime import datetime, timezone
import os

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class SharpeReward:
    """Sharpe ratio-based reward function for RL agents.

    Calculates rewards based on the Sharpe ratio of returns, promoting
    strategies that maximize risk-adjusted performance.

    Attributes:
        config: Configuration dictionary
        window_size: Rolling window for Sharpe calculation
        risk_free_rate: Risk-free rate from config
        returns_buffer: Circular buffer for returns
        annualization_factor: Factor to annualize Sharpe ratio

    Examples:
        >>> config = {"window_size": 100, "risk_free_rate": "0.02"}
        >>> reward_fn = SharpeReward(config)
        >>> reward = await reward_fn.calculate(
        ...     returns=pl.DataFrame({"returns": [0.01, 0.02, -0.01]})
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Sharpe reward function.

        Args:
            config: Configuration with window_size, risk_free_rate, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        self.window_size: int = int(config.get("window_size", os.getenv("RL_SHARPE_WINDOW", "100")))
        self.risk_free_rate: Decimal = Decimal(str(config.get("risk_free_rate", os.getenv("RL_RISK_FREE_RATE", "0.02"))))
        self.annualization_factor: Decimal = Decimal(str(config.get("annualization_factor", os.getenv("RL_ANNUALIZATION_FACTOR", "252"))))
        self.min_periods: int = int(config.get("min_periods", os.getenv("RL_MIN_PERIODS", "30")))

        self.returns_buffer: Deque[Decimal] = deque(maxlen=self.window_size)

        logger.info(
            "Sharpe reward initialized",
            window_size=self.window_size,
            risk_free_rate=str(self.risk_free_rate),
            annualization_factor=str(self.annualization_factor)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "window_size" in self.config:
            window = int(self.config["window_size"])
            if window <= 0:
                raise ValueError(f"window_size must be positive, got {window}")

        if "min_periods" in self.config:
            min_p = int(self.config["min_periods"])
            if min_p <= 0:
                raise ValueError(f"min_periods must be positive, got {min_p}")

    async def calculate(
        self,
        returns: pl.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calculate Sharpe-based reward.

        Args:
            returns: DataFrame with 'returns' column (Decimal values)
            metadata: Optional metadata for context

        Returns:
            Sharpe ratio as reward value

        Raises:
            ValueError: If returns DataFrame invalid
        """
        try:
            if returns.is_empty():
                logger.warning("Empty returns DataFrame provided")
                return Decimal("0")

            if "returns" not in returns.columns:
                raise ValueError("returns DataFrame must have 'returns' column")

            # Add returns to buffer
            for ret in returns["returns"].to_list():
                if ret is not None:
                    self.returns_buffer.append(Decimal(str(ret)))

            # Need minimum periods to calculate Sharpe
            if len(self.returns_buffer) < self.min_periods:
                logger.debug(
                    "Insufficient data for Sharpe calculation",
                    buffer_size=len(self.returns_buffer),
                    min_periods=self.min_periods
                )
                return Decimal("0")

            # Calculate Sharpe ratio
            sharpe = await self._calculate_sharpe()

            logger.debug(
                "Sharpe reward calculated",
                sharpe=str(sharpe),
                buffer_size=len(self.returns_buffer)
            )

            return sharpe

        except Exception as e:
            logger.error(
                "Sharpe reward calculation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _calculate_sharpe(self) -> Decimal:
        """Calculate Sharpe ratio from returns buffer.

        Returns:
            Annualized Sharpe ratio
        """
        # Convert to numpy for calculation (using float64 internally, convert back to Decimal)
        returns_array = np.array([float(r) for r in self.returns_buffer], dtype=np.float64)

        # Calculate excess returns
        rf_rate_period = float(self.risk_free_rate) / float(self.annualization_factor)
        excess_returns = returns_array - rf_rate_period

        # Calculate mean and std
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        # Avoid division by zero
        if std_excess == 0:
            logger.warning("Zero standard deviation in returns")
            return Decimal("0")

        # Calculate Sharpe ratio
        sharpe_ratio = mean_excess / std_excess

        # Annualize
        annualized_sharpe = sharpe_ratio * np.sqrt(float(self.annualization_factor))

        return Decimal(str(annualized_sharpe))

    async def calculate_batch(
        self,
        returns_batch: List[pl.DataFrame],
        metadata_batch: Optional[List[Dict[str, Any]]] = None
    ) -> List[Decimal]:
        """Calculate rewards for batch of return sequences.

        Args:
            returns_batch: List of returns DataFrames
            metadata_batch: Optional list of metadata dicts

        Returns:
            List of Sharpe rewards
        """
        try:
            rewards = []
            metadata_batch = metadata_batch or [None] * len(returns_batch)

            for returns_df, metadata in zip(returns_batch, metadata_batch):
                reward = await self.calculate(returns_df, metadata)
                rewards.append(reward)

            logger.debug(
                "Batch rewards calculated",
                batch_size=len(rewards),
                mean_reward=str(Decimal(str(np.mean([float(r) for r in rewards]))))
            )

            return rewards

        except Exception as e:
            logger.error(
                "Batch reward calculation failed",
                error=str(e),
                batch_size=len(returns_batch)
            )
            raise

    def reset(self) -> None:
        """Reset the returns buffer."""
        self.returns_buffer.clear()
        logger.debug("Sharpe reward buffer reset")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of current metrics
        """
        if len(self.returns_buffer) == 0:
            return {
                "buffer_size": 0,
                "mean_return": None,
                "std_return": None,
                "current_sharpe": None
            }

        returns_array = np.array([float(r) for r in self.returns_buffer], dtype=np.float64)

        return {
            "buffer_size": len(self.returns_buffer),
            "mean_return": str(Decimal(str(np.mean(returns_array)))),
            "std_return": str(Decimal(str(np.std(returns_array, ddof=1)))),
            "current_sharpe": str(Decimal(str(np.mean(returns_array) / (np.std(returns_array, ddof=1) + 1e-10))))
        }
