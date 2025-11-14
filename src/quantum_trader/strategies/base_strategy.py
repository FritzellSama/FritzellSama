"""
Base Strategy - Abstract base class for all trading strategies.

This module provides the foundation for all trading strategies in the Quantum Trader AI system.
All strategies must inherit from BaseStrategy and implement required abstract methods.
"""

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, Order, SignalAction
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies.

    This class defines the interface that all trading strategies must implement.
    It provides common functionality for signal validation, configuration management,
    and state tracking.

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk manager instance for position sizing and validation
        name: Strategy name for identification
        active: Whether strategy is currently active
        state: Internal strategy state

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def generate_signals(self, market_data):
        ...         # Implementation
        ...         pass
        ...     def calculate_indicators(self, data):
        ...         # Implementation
        ...         pass
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize base strategy.

        Args:
            config: Strategy configuration dictionary
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self.risk_manager = risk_manager
        self.name = config.get("name", self.__class__.__name__)
        self.active = config.get("active", True)
        self.state: Dict[str, Any] = {}

        self._validate_config()

        logger.info(
            "Strategy initialized",
            strategy=self.name,
            config_keys=list(config.keys())
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_params = self.config.get("required_params", [])

        for param in required_params:
            if param not in self.config:
                error_msg = f"Required config parameter missing: {param}"
                logger.error("Config validation failed", strategy=self.name, missing_param=param)
                raise ValueError(error_msg)

        # Validate timeframe if present
        if "timeframe" in self.config:
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            if self.config["timeframe"] not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {self.config['timeframe']}")

        # Validate symbols
        if "symbols" in self.config:
            if not isinstance(self.config["symbols"], list) or not self.config["symbols"]:
                raise ValueError("symbols must be a non-empty list")

    @abstractmethod
    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals from market data.

        This method must be implemented by all concrete strategy classes.
        It analyzes market data and returns a list of trading signals.

        Args:
            market_data: Polars DataFrame with OHLCV data
                Required columns: [timestamp, open, high, low, close, volume]

        Returns:
            List of Signal objects

        Raises:
            ValueError: If market_data invalid or missing required columns

        Example:
            >>> data = pl.DataFrame({...})
            >>> signals = strategy.generate_signals(data)
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate technical indicators for strategy.

        This method must be implemented by all concrete strategy classes.
        It computes technical indicators needed for signal generation.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary mapping indicator names to their Decimal values

        Raises:
            ValueError: If data invalid or insufficient

        Example:
            >>> indicators = strategy.calculate_indicators(data)
            >>> {'rsi': Decimal('65.5'), 'macd': Decimal('0.023')}
        """
        pass

    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal before execution.

        Checks signal for validity including required fields, reasonable values,
        and consistency.

        Args:
            signal: Signal to validate

        Returns:
            True if signal valid, False otherwise
        """
        try:
            # Check required fields
            if not signal.symbol or not signal.strategy:
                logger.warning("Signal missing required fields", signal=signal)
                return False

            # Validate strength and confidence ranges
            if not (Decimal("0") <= signal.strength <= Decimal("1")):
                logger.warning("Signal strength out of range", strength=signal.strength)
                return False

            if not (Decimal("0") <= signal.confidence <= Decimal("1")):
                logger.warning("Signal confidence out of range", confidence=signal.confidence)
                return False

            # Validate action
            if signal.action not in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD, SignalAction.CLOSE]:
                logger.warning("Invalid signal action", action=signal.action)
                return False

            # Check timestamp is not in future
            now = datetime.now(timezone.utc)
            if signal.timestamp > now:
                logger.warning("Signal timestamp in future", timestamp=signal.timestamp)
                return False

            # Check minimum confidence threshold from config
            min_confidence = Decimal(str(self.config.get("min_confidence", "0.5")))
            if signal.confidence < min_confidence:
                logger.debug(
                    "Signal below confidence threshold",
                    confidence=signal.confidence,
                    threshold=min_confidence
                )
                return False

            return True

        except Exception as e:
            logger.error("Signal validation error", error=str(e), signal=signal)
            return False

    def validate_market_data(self, data: pl.DataFrame) -> Tuple[bool, Optional[str]]:
        """Validate market data DataFrame.

        Args:
            data: Market data DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Check DataFrame is not empty
            if data.is_empty():
                return False, "Market data DataFrame is empty"

            # Check required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"

            # Check for null values
            null_counts = data.null_count()
            if null_counts.sum_horizontal()[0] > 0:
                return False, "Market data contains null values"

            # Validate price relationships (high >= low, etc.)
            invalid_prices = data.filter(
                (pl.col("high") < pl.col("low")) |
                (pl.col("high") < pl.col("open")) |
                (pl.col("high") < pl.col("close")) |
                (pl.col("low") > pl.col("open")) |
                (pl.col("low") > pl.col("close"))
            )

            if len(invalid_prices) > 0:
                return False, f"Invalid OHLC relationships in {len(invalid_prices)} rows"

            # Check minimum data points
            min_data_points = self.config.get("min_data_points", 20)
            if len(data) < min_data_points:
                return False, f"Insufficient data points: {len(data)} < {min_data_points}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_state(self) -> Dict[str, Any]:
        """Get current strategy state.

        Returns:
            Dictionary containing strategy state
        """
        return {
            "name": self.name,
            "active": self.active,
            "state": self.state.copy(),
            "config": self.config.copy()
        }

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update strategy state.

        Args:
            updates: Dictionary of state updates
        """
        self.state.update(updates)
        logger.debug("Strategy state updated", strategy=self.name, updates=list(updates.keys()))

    def reset_state(self) -> None:
        """Reset strategy state to initial conditions."""
        self.state = {}
        logger.info("Strategy state reset", strategy=self.name)

    def enable(self) -> None:
        """Enable strategy."""
        self.active = True
        logger.info("Strategy enabled", strategy=self.name)

    def disable(self) -> None:
        """Disable strategy."""
        self.active = False
        logger.info("Strategy disabled", strategy=self.name)

    def get_performance_metrics(self) -> Dict[str, Decimal]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return {
            "signals_generated": Decimal(str(self.state.get("signals_generated", 0))),
            "signals_validated": Decimal(str(self.state.get("signals_validated", 0))),
            "last_signal_time": self.state.get("last_signal_time"),
        }

    async def warmup(self, historical_data: pl.DataFrame) -> None:
        """Warm up strategy with historical data.

        Optional method for strategies that need initialization with historical data.

        Args:
            historical_data: Historical market data for warmup
        """
        logger.info("Strategy warmup", strategy=self.name, data_points=len(historical_data))
        # Default implementation does nothing, subclasses can override
        pass

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}', active={self.active})"
