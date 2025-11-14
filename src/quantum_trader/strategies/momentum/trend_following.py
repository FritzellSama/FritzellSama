"""Trend Following Momentum Strategy.

High-performance trend following strategy using multiple timeframe analysis,
adaptive indicators, and institutional-grade signal generation.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, Order, OrderSide, OrderType
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """Multi-timeframe trend following strategy with adaptive indicators.

    Combines EMA crossovers, ADX trend strength, and momentum oscillators
    to identify high-probability trend continuation opportunities.

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management instance
        name: Strategy identifier
        ema_fast_period: Fast EMA period from config
        ema_slow_period: Slow EMA period from config
        adx_period: ADX period from config
        adx_threshold: Minimum ADX for trend confirmation from config
        atr_period: ATR period for volatility measurement from config
        min_confidence: Minimum signal confidence from config
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize trend following strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: Risk manager instance for validation

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)
        self.name = "TrendFollowing"

        self._validate_config()

        # Load parameters from config
        self.ema_fast_period = config["indicators"]["ema_fast_period"]
        self.ema_slow_period = config["indicators"]["ema_slow_period"]
        self.adx_period = config["indicators"]["adx_period"]
        self.adx_threshold = Decimal(str(config["indicators"]["adx_threshold"]))
        self.atr_period = config["indicators"]["atr_period"]
        self.min_confidence = Decimal(str(config["signal"]["min_confidence"]))
        self.position_size_pct = Decimal(str(config["risk"]["position_size_pct"]))

        logger.info(
            "trend_following_initialized",
            ema_fast=self.ema_fast_period,
            ema_slow=self.ema_slow_period,
            adx_period=self.adx_period,
            adx_threshold=str(self.adx_threshold)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_keys = ["indicators", "signal", "risk"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        required_indicators = ["ema_fast_period", "ema_slow_period", "adx_period",
                              "adx_threshold", "atr_period"]
        for key in required_indicators:
            if key not in self.config["indicators"]:
                raise ValueError(f"Missing required indicator config: {key}")

        if "min_confidence" not in self.config["signal"]:
            raise ValueError("Missing required signal.min_confidence")

        if "position_size_pct" not in self.config["risk"]:
            raise ValueError("Missing required risk.position_size_pct")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals from market data.

        Analyzes market data using trend indicators and generates signals
        when strong trends are identified with sufficient confidence.

        Args:
            market_data: Polars DataFrame with OHLCV data

        Returns:
            List of Signal objects for detected opportunities

        Raises:
            ValueError: If market data invalid or insufficient
        """
        try:
            if market_data.height < max(self.ema_slow_period, self.adx_period, self.atr_period) + 20:
                logger.warning(
                    "insufficient_data",
                    rows=market_data.height,
                    required=max(self.ema_slow_period, self.adx_period, self.atr_period) + 20
                )
                return []

            # Calculate indicators
            indicators = self.calculate_indicators(market_data)

            signals = []

            # Get latest values
            latest_ema_fast = indicators["ema_fast"]
            latest_ema_slow = indicators["ema_slow"]
            latest_adx = indicators["adx"]
            latest_di_plus = indicators["di_plus"]
            latest_di_minus = indicators["di_minus"]
            latest_price = indicators["close"]
            latest_atr = indicators["atr"]

            symbol = market_data["symbol"][0] if "symbol" in market_data.columns else "UNKNOWN"
            timeframe = market_data["timeframe"][0] if "timeframe" in market_data.columns else "1m"

            # Bullish trend signal
            if (latest_ema_fast > latest_ema_slow and
                latest_adx > self.adx_threshold and
                latest_di_plus > latest_di_minus):

                strength = self._calculate_signal_strength(
                    latest_ema_fast, latest_ema_slow, latest_adx, "BUY"
                )
                confidence = self._calculate_confidence(
                    latest_adx, latest_di_plus, latest_di_minus, latest_atr
                )

                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.BUY,
                        strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        strategy=self.name,
                        timeframe=timeframe,
                        indicators={
                            "ema_fast": latest_ema_fast,
                            "ema_slow": latest_ema_slow,
                            "adx": latest_adx,
                            "di_plus": latest_di_plus,
                            "di_minus": latest_di_minus,
                            "atr": latest_atr,
                            "price": latest_price
                        },
                        metadata={
                            "trend_direction": "bullish",
                            "ema_spread_pct": ((latest_ema_fast - latest_ema_slow) / latest_ema_slow * Decimal("100"))
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bullish_signal_generated",
                            symbol=symbol,
                            confidence=str(confidence),
                            strength=str(strength)
                        )

            # Bearish trend signal
            elif (latest_ema_fast < latest_ema_slow and
                  latest_adx > self.adx_threshold and
                  latest_di_minus > latest_di_plus):

                strength = self._calculate_signal_strength(
                    latest_ema_fast, latest_ema_slow, latest_adx, "SELL"
                )
                confidence = self._calculate_confidence(
                    latest_adx, latest_di_plus, latest_di_minus, latest_atr
                )

                if confidence >= self.min_confidence:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.SELL,
                        strength=strength,
                        confidence=confidence,
                        timestamp=datetime.now(timezone.utc),
                        strategy=self.name,
                        timeframe=timeframe,
                        indicators={
                            "ema_fast": latest_ema_fast,
                            "ema_slow": latest_ema_slow,
                            "adx": latest_adx,
                            "di_plus": latest_di_plus,
                            "di_minus": latest_di_minus,
                            "atr": latest_atr,
                            "price": latest_price
                        },
                        metadata={
                            "trend_direction": "bearish",
                            "ema_spread_pct": ((latest_ema_slow - latest_ema_fast) / latest_ema_slow * Decimal("100"))
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bearish_signal_generated",
                            symbol=symbol,
                            confidence=str(confidence),
                            strength=str(strength)
                        )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate technical indicators for trend analysis.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of calculated indicator values

        Raises:
            ValueError: If data invalid or calculation fails
        """
        try:
            # Calculate EMAs
            close_prices = data["close"].to_numpy()

            ema_fast = self._calculate_ema(close_prices, self.ema_fast_period)
            ema_slow = self._calculate_ema(close_prices, self.ema_slow_period)

            # Calculate ADX and directional indicators
            high_prices = data["high"].to_numpy()
            low_prices = data["low"].to_numpy()

            adx, di_plus, di_minus = self._calculate_adx(
                high_prices, low_prices, close_prices, self.adx_period
            )

            # Calculate ATR
            atr = self._calculate_atr(high_prices, low_prices, close_prices, self.atr_period)

            return {
                "ema_fast": Decimal(str(ema_fast[-1])),
                "ema_slow": Decimal(str(ema_slow[-1])),
                "adx": Decimal(str(adx[-1])),
                "di_plus": Decimal(str(di_plus[-1])),
                "di_minus": Decimal(str(di_minus[-1])),
                "atr": Decimal(str(atr[-1])),
                "close": Decimal(str(close_prices[-1]))
            }

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            raise ValueError(f"Failed to calculate indicators: {e}")

    def _calculate_ema(self, prices: Any, period: int) -> Any:
        """Calculate Exponential Moving Average.

        Args:
            prices: Array of prices
            period: EMA period

        Returns:
            Array of EMA values
        """
        import numpy as np

        alpha = Decimal("2") / Decimal(str(period + 1))
        ema = np.zeros_like(prices, dtype=np.float64)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = float(alpha) * prices[i] + (1 - float(alpha)) * ema[i-1]

        return ema

    def _calculate_adx(
        self,
        high: Any,
        low: Any,
        close: Any,
        period: int
    ) -> Tuple[Any, Any, Any]:
        """Calculate Average Directional Index and DI+/DI-.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            Tuple of (ADX, DI+, DI-) arrays
        """
        import numpy as np

        # Calculate True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Calculate Directional Movement
        dm_plus = np.maximum(high[1:] - high[:-1], 0)
        dm_minus = np.maximum(low[:-1] - low[1:], 0)

        dm_plus = np.where(dm_plus > dm_minus, dm_plus, 0)
        dm_minus = np.where(dm_minus > dm_plus, dm_minus, 0)

        # Smooth with EMA
        atr = self._calculate_ema(np.concatenate([[tr[0]], tr]), period)
        di_plus_smooth = self._calculate_ema(np.concatenate([[dm_plus[0]], dm_plus]), period)
        di_minus_smooth = self._calculate_ema(np.concatenate([[dm_minus[0]], dm_minus]), period)

        # Calculate DI
        di_plus = 100 * di_plus_smooth / (atr + 1e-10)
        di_minus = 100 * di_minus_smooth / (atr + 1e-10)

        # Calculate DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = self._calculate_ema(dx, period)

        return adx, di_plus, di_minus

    def _calculate_atr(self, high: Any, low: Any, close: Any, period: int) -> Any:
        """Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            Array of ATR values
        """
        import numpy as np

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        atr = self._calculate_ema(np.concatenate([[tr[0]], tr]), period)
        return atr

    def _calculate_signal_strength(
        self,
        ema_fast: Decimal,
        ema_slow: Decimal,
        adx: Decimal,
        direction: str
    ) -> Decimal:
        """Calculate signal strength based on indicator values.

        Args:
            ema_fast: Fast EMA value
            ema_slow: Slow EMA value
            adx: ADX value
            direction: Signal direction ("BUY" or "SELL")

        Returns:
            Signal strength between 0 and 1
        """
        # EMA spread strength (normalized)
        ema_spread = abs(ema_fast - ema_slow) / ema_slow
        ema_strength = min(ema_spread * Decimal("10"), Decimal("1.0"))

        # ADX strength (normalized to 0-1, assumes max ADX of 100)
        adx_strength = min(adx / Decimal("100"), Decimal("1.0"))

        # Combined strength
        strength = (ema_strength + adx_strength) / Decimal("2")

        return min(max(strength, Decimal("0")), Decimal("1"))

    def _calculate_confidence(
        self,
        adx: Decimal,
        di_plus: Decimal,
        di_minus: Decimal,
        atr: Decimal
    ) -> Decimal:
        """Calculate signal confidence based on trend strength and volatility.

        Args:
            adx: ADX value
            di_plus: DI+ value
            di_minus: DI- value
            atr: ATR value

        Returns:
            Confidence level between 0 and 1
        """
        # Higher ADX = higher confidence
        adx_conf = min(adx / Decimal("100"), Decimal("1.0"))

        # Larger DI spread = higher confidence
        di_spread = abs(di_plus - di_minus)
        di_conf = min(di_spread / Decimal("50"), Decimal("1.0"))

        # Combine factors
        confidence = (adx_conf * Decimal("0.6") + di_conf * Decimal("0.4"))

        return min(max(confidence, Decimal("0")), Decimal("1"))
