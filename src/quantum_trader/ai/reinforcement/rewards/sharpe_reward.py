"""
Sharpe Ratio Reward Function for Reinforcement Learning

Production-ready implementation of Sharpe ratio-based reward calculation
for trading reinforcement learning agents.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from datetime import datetime, timezone
import os

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision for Decimal operations
getcontext().prec = 28


class SharpeReward:
    """
    Sharpe ratio-based reward function for RL trading agents.

    Calculates risk-adjusted returns to encourage profitable trading
    while penalizing excessive volatility.

    Attributes:
        config: Configuration dictionary
        window_size: Lookback window for Sharpe calculation
        risk_free_rate: Annual risk-free rate
        annualization_factor: Factor to annualize returns
        returns_buffer: Rolling buffer of returns
        min_samples: Minimum samples required for calculation

    Example:
        >>> config = {"reward": {"sharpe_window": 100, "risk_free_rate": "0.02"}}
        >>> sharpe = SharpeReward(config)
        >>> reward = sharpe.calculate_reward(
        ...     returns=Decimal("0.05"),
        ...     timestamp=datetime.now(timezone.utc)
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Sharpe reward calculator.

        Args:
            config: Configuration dictionary with reward parameters

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        reward_config = self.config.get("reward", {})

        # Load configuration parameters
        self.window_size: int = reward_config.get("sharpe_window", int(os.getenv("SHARPE_WINDOW", "100")))
        self.risk_free_rate: Decimal = Decimal(
            str(reward_config.get("risk_free_rate", os.getenv("RISK_FREE_RATE", "0.02")))
        )
        self.annualization_factor: Decimal = Decimal(
            str(reward_config.get("annualization_factor", os.getenv("ANNUALIZATION_FACTOR", "252")))
        )
        self.min_samples: int = reward_config.get("min_samples", int(os.getenv("SHARPE_MIN_SAMPLES", "30")))

        # Penalty factors
        self.volatility_penalty: Decimal = Decimal(
            str(reward_config.get("volatility_penalty", os.getenv("VOLATILITY_PENALTY", "0.1")))
        )
        self.drawdown_penalty: Decimal = Decimal(
            str(reward_config.get("drawdown_penalty", os.getenv("DRAWDOWN_PENALTY", "0.5")))
        )

        # Rolling buffer for returns
        self.returns_buffer: Deque[Decimal] = deque(maxlen=self.window_size)
        self.equity_curve: List[Decimal] = []
        self.peak_equity: Decimal = Decimal("0")

        logger.info(
            "sharpe_reward_initialized",
            window_size=self.window_size,
            risk_free_rate=str(self.risk_free_rate),
            annualization_factor=str(self.annualization_factor)
        )

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        reward_config = self.config.get("reward", {})

        if reward_config:
            window_size = reward_config.get("sharpe_window", 100)
            if window_size <= 0:
                raise ValueError("sharpe_window must be positive")

            risk_free = reward_config.get("risk_free_rate", "0.02")
            try:
                rf_decimal = Decimal(str(risk_free))
                if rf_decimal < Decimal("0") or rf_decimal > Decimal("1"):
                    raise ValueError("risk_free_rate must be between 0 and 1")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid risk_free_rate: {e}")

    def calculate_reward(
        self,
        returns: Decimal,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate Sharpe ratio-based reward.

        Args:
            returns: Portfolio returns for current period
            timestamp: UTC timestamp
            metadata: Additional context (drawdown, volatility, etc.)

        Returns:
            Calculated reward value

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> reward = sharpe.calculate_reward(
            ...     returns=Decimal("0.05"),
            ...     timestamp=datetime.now(timezone.utc)
            ... )
        """
        try:
            # Validate inputs
            if not isinstance(returns, Decimal):
                raise ValueError("Returns must be Decimal type")

            if timestamp.tzinfo is None:
                raise ValueError("Timestamp must be timezone-aware (UTC)")

            # Add to buffer
            self.returns_buffer.append(returns)

            # Calculate base Sharpe reward
            if len(self.returns_buffer) < self.min_samples:
                # Not enough data - return simple returns
                reward = returns
                logger.debug(
                    "insufficient_samples_for_sharpe",
                    samples=len(self.returns_buffer),
                    min_samples=self.min_samples,
                    reward=str(reward)
                )
            else:
                # Calculate Sharpe ratio
                sharpe = self._calculate_sharpe_ratio()
                reward = sharpe

                # Apply penalties if provided
                if metadata:
                    reward = self._apply_penalties(reward, metadata)

                logger.debug(
                    "sharpe_reward_calculated",
                    sharpe_ratio=str(sharpe),
                    final_reward=str(reward),
                    samples=len(self.returns_buffer)
                )

            return reward

        except Exception as e:
            logger.error("reward_calculation_failed", error=str(e))
            raise

    def _calculate_sharpe_ratio(self) -> Decimal:
        """
        Calculate Sharpe ratio from returns buffer.

        Returns:
            Sharpe ratio value
        """
        # Convert to numpy array for calculation
        returns_array = np.array([float(r) for r in self.returns_buffer], dtype=np.float64)

        # Calculate mean and std
        mean_return = Decimal(str(np.mean(returns_array)))
        std_return = Decimal(str(np.std(returns_array, ddof=1)))

        # Avoid division by zero
        if std_return == Decimal("0"):
            return Decimal("0")

        # Calculate daily risk-free rate
        daily_rf = self.risk_free_rate / self.annualization_factor

        # Sharpe ratio = (mean_return - risk_free) / std_return
        excess_return = mean_return - daily_rf
        sharpe = excess_return / std_return

        # Annualize
        sharpe_annualized = sharpe * (self.annualization_factor ** Decimal("0.5"))

        return sharpe_annualized

    def _apply_penalties(self, base_reward: Decimal, metadata: Dict[str, Any]) -> Decimal:
        """
        Apply penalty adjustments to base reward.

        Args:
            base_reward: Base Sharpe reward
            metadata: Metadata containing penalty signals

        Returns:
            Adjusted reward
        """
        reward = base_reward

        # Volatility penalty
        if "volatility" in metadata:
            vol = Decimal(str(metadata["volatility"]))
            vol_threshold = Decimal(str(metadata.get("volatility_threshold", "0.5")))

            if vol > vol_threshold:
                penalty = (vol - vol_threshold) * self.volatility_penalty
                reward -= penalty
                logger.debug("volatility_penalty_applied", penalty=str(penalty))

        # Drawdown penalty
        if "drawdown" in metadata:
            dd = Decimal(str(metadata["drawdown"]))
            dd_threshold = Decimal(str(metadata.get("drawdown_threshold", "0.1")))

            if abs(dd) > dd_threshold:
                penalty = abs(dd) * self.drawdown_penalty
                reward -= penalty
                logger.debug("drawdown_penalty_applied", penalty=str(penalty))

        # Trade frequency penalty (discourage overtrading)
        if "num_trades" in metadata:
            num_trades = int(metadata["num_trades"])
            max_trades = int(metadata.get("max_trades_per_period", 100))

            if num_trades > max_trades:
                penalty = Decimal(str(num_trades - max_trades)) * Decimal("0.01")
                reward -= penalty
                logger.debug("overtrading_penalty_applied", penalty=str(penalty))

        return reward

    def calculate_batch_rewards(
        self,
        returns_df: pl.DataFrame,
        metadata_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Calculate rewards for batch of returns.

        Args:
            returns_df: Polars DataFrame with 'timestamp' and 'returns' columns
            metadata_df: Optional metadata DataFrame

        Returns:
            DataFrame with added 'reward' column

        Raises:
            ValueError: If DataFrame schema is invalid

        Example:
            >>> df = pl.DataFrame({
            ...     "timestamp": [...],
            ...     "returns": [Decimal("0.01"), Decimal("0.02"), ...]
            ... })
            >>> result = sharpe.calculate_batch_rewards(df)
        """
        try:
            # Validate schema
            if "returns" not in returns_df.columns or "timestamp" not in returns_df.columns:
                raise ValueError("DataFrame must have 'timestamp' and 'returns' columns")

            rewards = []

            for row in returns_df.iter_rows(named=True):
                returns = Decimal(str(row["returns"]))
                timestamp = row["timestamp"]

                # Get metadata for this row if available
                metadata = None
                if metadata_df is not None:
                    # Find matching row by timestamp
                    meta_row = metadata_df.filter(pl.col("timestamp") == timestamp)
                    if len(meta_row) > 0:
                        metadata = meta_row.to_dicts()[0]

                reward = self.calculate_reward(returns, timestamp, metadata)
                rewards.append(float(reward))

            # Add reward column
            result = returns_df.with_columns(
                pl.Series("reward", rewards)
            )

            logger.info(
                "batch_rewards_calculated",
                num_rows=len(result),
                mean_reward=str(Decimal(str(np.mean(rewards))))
            )

            return result

        except Exception as e:
            logger.error("batch_reward_calculation_failed", error=str(e))
            raise

    def reset(self) -> None:
        """Reset internal state (for new episode)."""
        self.returns_buffer.clear()
        self.equity_curve.clear()
        self.peak_equity = Decimal("0")
        logger.info("sharpe_reward_reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with current metrics
        """
        if len(self.returns_buffer) < self.min_samples:
            return {
                "samples": len(self.returns_buffer),
                "sharpe_ratio": None,
                "mean_return": None,
                "volatility": None
            }

        returns_array = np.array([float(r) for r in self.returns_buffer], dtype=np.float64)
        mean_return = Decimal(str(np.mean(returns_array)))
        volatility = Decimal(str(np.std(returns_array, ddof=1)))
        sharpe = self._calculate_sharpe_ratio()

        return {
            "samples": len(self.returns_buffer),
            "sharpe_ratio": str(sharpe),
            "mean_return": str(mean_return),
            "volatility": str(volatility),
            "annualization_factor": str(self.annualization_factor)
        }
