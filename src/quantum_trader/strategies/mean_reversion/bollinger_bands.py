"""
Bollinger Bands Mean Reversion Strategy.

Implements a mean reversion strategy using Bollinger Bands to identify
overbought and oversold conditions for trading opportunities.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import math

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, OrderSide
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion trading strategy.

    This strategy uses Bollinger Bands to identify overbought/oversold conditions:
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    - Close positions when price returns to middle band

    The strategy uses multiple timeframes and confluence for signal generation.

    Attributes:
        period: Bollinger Bands period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        threshold: Touch threshold as percentage of bandwidth

    Example:
        >>> config = {
        ...     "period": "20",
        ...     "std_dev": "2.0",
        ...     "threshold": "0.05"
        ... }
        >>> strategy = BollingerBandsStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize Bollinger Bands strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Bollinger Bands parameters
        self.period = int(config.get("period", 20))
        self.std_dev = Decimal(str(config.get("std_dev", "2.0")))
        self.threshold = Decimal(str(config.get("threshold", "0.05")))  # 5% of bandwidth

        # Mean reversion parameters
        self.mean_reversion_strength = Decimal(str(config.get("mean_reversion_strength", "0.7")))
        self.exit_at_middle = config.get("exit_at_middle", True)
        self.use_volume_filter = config.get("use_volume_filter", True)

        # State tracking
        self.last_signals: Dict[str, Signal] = {}
        self.band_values: Dict[str, Dict[str, Decimal]] = {}

        logger.info(
            "Bollinger Bands strategy initialized",
            strategy=self.name,
            period=self.period,
            std_dev=self.std_dev
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        period = int(self.config.get("period", 20))
        if period < 5 or period > 200:
            raise ValueError(f"period must be between 5 and 200: {period}")

        std_dev = Decimal(str(self.config.get("std_dev", "2.0")))
        if std_dev <= Decimal("0") or std_dev > Decimal("5"):
            raise ValueError(f"std_dev must be between 0 and 5: {std_dev}")

    def calculate_bollinger_bands(
        self,
        data: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """Calculate Bollinger Bands indicators.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with upper, middle, lower bands and bandwidth

        Raises:
            ValueError: If insufficient data
        """
        try:
            if len(data) < self.period:
                raise ValueError(f"Insufficient data: need {self.period}, got {len(data)}")

            # Calculate middle band (SMA)
            closes = data.select(pl.col("close")).to_series().to_list()
            recent_closes = closes[-self.period:]

            middle_band = Decimal(str(sum(recent_closes) / len(recent_closes)))

            # Calculate standard deviation
            variance = sum((Decimal(str(c)) - middle_band) ** 2 for c in recent_closes) / Decimal(str(len(recent_closes)))
            std = variance.sqrt()

            # Calculate upper and lower bands
            upper_band = middle_band + (self.std_dev * std)
            lower_band = middle_band - (self.std_dev * std)

            # Calculate bandwidth (for squeeze detection)
            bandwidth = (upper_band - lower_band) / middle_band

            # Get current price
            current_price = Decimal(str(closes[-1]))

            # Calculate %B indicator (position within bands)
            if upper_band != lower_band:
                percent_b = (current_price - lower_band) / (upper_band - lower_band)
            else:
                percent_b = Decimal("0.5")

            return {
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band,
                "bandwidth": bandwidth,
                "percent_b": percent_b,
                "current_price": current_price,
                "std_dev": std
            }

        except Exception as e:
            logger.error("Error calculating Bollinger Bands", error=str(e))
            raise

    def calculate_volume_filter(self, data: pl.DataFrame) -> bool:
        """Check if volume conditions are met.

        Args:
            data: DataFrame with volume data

        Returns:
            True if volume filter passed
        """
        try:
            if not self.use_volume_filter or len(data) < self.period:
                return True

            volumes = data.select(pl.col("volume")).to_series().to_list()
            recent_volumes = volumes[-self.period:]

            current_volume = Decimal(str(volumes[-1]))
            avg_volume = Decimal(str(sum(recent_volumes) / len(recent_volumes)))

            # Require volume above average for mean reversion signals
            volume_threshold = Decimal(str(self.config.get("volume_threshold", "0.8")))
            return current_volume >= (avg_volume * volume_threshold)

        except Exception as e:
            logger.error("Error in volume filter", error=str(e))
            return True

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals from Bollinger Bands.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            # Validate data
            is_valid, error = self.validate_market_data(market_data)
            if not is_valid:
                logger.warning("Invalid market data", error=error)
                return signals

            if not self.active:
                return signals

            # Get symbol from config
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            symbol = symbols[0]  # Process first symbol

            # Calculate Bollinger Bands
            bands = self.calculate_bollinger_bands(market_data)
            self.band_values[symbol] = bands

            current_price = bands["current_price"]
            upper_band = bands["upper_band"]
            middle_band = bands["middle_band"]
            lower_band = bands["lower_band"]
            percent_b = bands["percent_b"]

            # Check volume filter
            volume_ok = self.calculate_volume_filter(market_data)

            # Calculate distance from bands
            distance_to_lower = abs(current_price - lower_band) / (upper_band - lower_band)
            distance_to_upper = abs(current_price - upper_band) / (upper_band - lower_band)

            # Generate signals based on band touches
            action = SignalAction.HOLD
            strength = Decimal("0")
            confidence = Decimal("0.5")

            # Oversold condition - buy signal
            if percent_b <= self.threshold and volume_ok:
                action = SignalAction.BUY
                strength = Decimal("1") - percent_b  # Stronger when deeper oversold
                confidence = min(Decimal("1"), Decimal("0.6") + (distance_to_lower * Decimal("0.4")))

                logger.info(
                    "Bollinger Bands BUY signal",
                    symbol=symbol,
                    price=current_price,
                    lower_band=lower_band,
                    percent_b=percent_b
                )

            # Overbought condition - sell signal
            elif percent_b >= (Decimal("1") - self.threshold) and volume_ok:
                action = SignalAction.SELL
                strength = percent_b  # Stronger when deeper overbought
                confidence = min(Decimal("1"), Decimal("0.6") + (distance_to_upper * Decimal("0.4")))

                logger.info(
                    "Bollinger Bands SELL signal",
                    symbol=symbol,
                    price=current_price,
                    upper_band=upper_band,
                    percent_b=percent_b
                )

            # Return to middle band - close signal
            elif self.exit_at_middle and abs(percent_b - Decimal("0.5")) < Decimal("0.1"):
                action = SignalAction.CLOSE
                strength = Decimal("0.7")
                confidence = Decimal("0.6")

                logger.info(
                    "Bollinger Bands CLOSE signal",
                    symbol=symbol,
                    price=current_price,
                    middle_band=middle_band,
                    percent_b=percent_b
                )

            # Create signal if action determined
            if action != SignalAction.HOLD:
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=self.config.get("timeframe", "1h"),
                    indicators={
                        "upper_band": upper_band,
                        "middle_band": middle_band,
                        "lower_band": lower_band,
                        "bandwidth": bands["bandwidth"],
                        "percent_b": percent_b,
                        "current_price": current_price
                    },
                    metadata={
                        "strategy_type": "mean_reversion",
                        "indicator": "bollinger_bands",
                        "period": self.period,
                        "std_dev": float(self.std_dev)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signals[symbol] = signal

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Bollinger Bands indicators.

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary of calculated indicators
        """
        try:
            if len(data) < self.period:
                logger.warning("Insufficient data for indicators")
                return {}

            bands = self.calculate_bollinger_bands(data)

            # Add additional technical indicators
            closes = data.select(pl.col("close")).to_series().to_list()

            # Calculate momentum
            if len(closes) >= 10:
                momentum = (Decimal(str(closes[-1])) - Decimal(str(closes[-10]))) / Decimal(str(closes[-10]))
            else:
                momentum = Decimal("0")

            indicators = {
                **bands,
                "momentum": momentum,
                "period": Decimal(str(self.period)),
                "std_dev_multiplier": self.std_dev
            }

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        bb_metrics = {
            "band_values": {
                symbol: {k: float(v) for k, v in bands.items()}
                for symbol, bands in self.band_values.items()
            },
            "last_signals": {
                symbol: {
                    "action": signal.action.value,
                    "strength": float(signal.strength),
                    "confidence": float(signal.confidence),
                    "timestamp": signal.timestamp.isoformat()
                }
                for symbol, signal in self.last_signals.items()
            }
        }

        return {**base_metrics, **bb_metrics}
