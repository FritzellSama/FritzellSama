"""Turtle Trading Strategy.

Implementation of the legendary Turtle Trading system with modern enhancements.
Uses Donchian Channels for breakouts and ATR-based position sizing.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction, Order, OrderSide, OrderType
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class TurtleTradingStrategy(BaseStrategy):
    """Turtle Trading strategy with Donchian Channel breakouts.

    Classic trend-following system using:
    - Donchian Channel breakouts for entries
    - ATR-based stop losses and position sizing
    - Multiple timeframe confirmation
    - Pyramiding on strong trends

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management instance
        name: Strategy identifier
        entry_period: Donchian entry period from config
        exit_period: Donchian exit period from config
        atr_period: ATR period from config
        atr_multiplier: Stop loss ATR multiplier from config
        risk_per_trade: Risk percentage per trade from config
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize turtle trading strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: Risk manager instance for validation

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)
        self.name = "TurtleTrading"

        self._validate_config()

        # Load parameters from config
        self.entry_period = config["indicators"]["entry_period"]
        self.exit_period = config["indicators"]["exit_period"]
        self.atr_period = config["indicators"]["atr_period"]
        self.atr_multiplier = Decimal(str(config["risk"]["atr_multiplier"]))
        self.risk_per_trade = Decimal(str(config["risk"]["risk_per_trade"]))
        self.min_confidence = Decimal(str(config["signal"]["min_confidence"]))
        self.enable_pyramiding = config.get("pyramiding", {}).get("enabled", False)
        self.max_pyramid_units = config.get("pyramiding", {}).get("max_units", 4)

        logger.info(
            "turtle_trading_initialized",
            entry_period=self.entry_period,
            exit_period=self.exit_period,
            atr_period=self.atr_period,
            atr_multiplier=str(self.atr_multiplier)
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

        required_indicators = ["entry_period", "exit_period", "atr_period"]
        for key in required_indicators:
            if key not in self.config["indicators"]:
                raise ValueError(f"Missing required indicator config: {key}")

        required_risk = ["atr_multiplier", "risk_per_trade"]
        for key in required_risk:
            if key not in self.config["risk"]:
                raise ValueError(f"Missing required risk config: {key}")

        if "min_confidence" not in self.config["signal"]:
            raise ValueError("Missing required signal.min_confidence")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate trading signals from market data.

        Detects Donchian Channel breakouts and generates entry/exit signals
        with proper risk management based on ATR.

        Args:
            market_data: Polars DataFrame with OHLCV data

        Returns:
            List of Signal objects for detected opportunities

        Raises:
            ValueError: If market data invalid or insufficient
        """
        try:
            min_periods = max(self.entry_period, self.exit_period, self.atr_period) + 10
            if market_data.height < min_periods:
                logger.warning(
                    "insufficient_data",
                    rows=market_data.height,
                    required=min_periods
                )
                return []

            # Calculate indicators
            indicators = self.calculate_indicators(market_data)

            signals = []

            # Get latest values
            latest_close = indicators["close"]
            entry_high = indicators["entry_high"]
            entry_low = indicators["entry_low"]
            exit_high = indicators["exit_high"]
            exit_low = indicators["exit_low"]
            atr = indicators["atr"]

            symbol = market_data["symbol"][0] if "symbol" in market_data.columns else "UNKNOWN"
            timeframe = market_data["timeframe"][0] if "timeframe" in market_data.columns else "1m"

            # Bullish breakout - price breaks above entry channel
            if latest_close > entry_high:
                strength = self._calculate_breakout_strength(
                    latest_close, entry_high, atr, "BUY"
                )
                confidence = self._calculate_confidence(
                    latest_close, entry_high, entry_low, atr
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
                            "close": latest_close,
                            "entry_high": entry_high,
                            "entry_low": entry_low,
                            "exit_high": exit_high,
                            "exit_low": exit_low,
                            "atr": atr
                        },
                        metadata={
                            "breakout_type": "bullish",
                            "stop_loss": entry_low - (atr * self.atr_multiplier),
                            "risk_amount": atr * self.atr_multiplier,
                            "position_size_atr": atr
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bullish_breakout_detected",
                            symbol=symbol,
                            price=str(latest_close),
                            entry_high=str(entry_high),
                            confidence=str(confidence)
                        )

            # Bearish breakout - price breaks below entry channel
            elif latest_close < entry_low:
                strength = self._calculate_breakout_strength(
                    latest_close, entry_low, atr, "SELL"
                )
                confidence = self._calculate_confidence(
                    latest_close, entry_high, entry_low, atr
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
                            "close": latest_close,
                            "entry_high": entry_high,
                            "entry_low": entry_low,
                            "exit_high": exit_high,
                            "exit_low": exit_low,
                            "atr": atr
                        },
                        metadata={
                            "breakout_type": "bearish",
                            "stop_loss": entry_high + (atr * self.atr_multiplier),
                            "risk_amount": atr * self.atr_multiplier,
                            "position_size_atr": atr
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "bearish_breakout_detected",
                            symbol=symbol,
                            price=str(latest_close),
                            entry_low=str(entry_low),
                            confidence=str(confidence)
                        )

            # Exit signals
            if latest_close < exit_low:
                # Exit long position
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    strength=Decimal("0.8"),
                    confidence=Decimal("0.9"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=timeframe,
                    indicators={"close": latest_close, "exit_low": exit_low},
                    metadata={"exit_type": "channel_exit", "position_type": "long"}
                )
                signals.append(signal)
                logger.info("long_exit_signal", symbol=symbol, price=str(latest_close))

            elif latest_close > exit_high:
                # Exit short position
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    strength=Decimal("0.8"),
                    confidence=Decimal("0.9"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=timeframe,
                    indicators={"close": latest_close, "exit_high": exit_high},
                    metadata={"exit_type": "channel_exit", "position_type": "short"}
                )
                signals.append(signal)
                logger.info("short_exit_signal", symbol=symbol, price=str(latest_close))

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate Turtle Trading indicators.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of calculated indicator values

        Raises:
            ValueError: If data invalid or calculation fails
        """
        try:
            high_prices = data["high"].to_numpy()
            low_prices = data["low"].to_numpy()
            close_prices = data["close"].to_numpy()

            # Calculate Donchian Channels for entry
            entry_high = self._calculate_donchian_high(high_prices, self.entry_period)
            entry_low = self._calculate_donchian_low(low_prices, self.entry_period)

            # Calculate Donchian Channels for exit
            exit_high = self._calculate_donchian_high(high_prices, self.exit_period)
            exit_low = self._calculate_donchian_low(low_prices, self.exit_period)

            # Calculate ATR
            atr = self._calculate_atr(high_prices, low_prices, close_prices, self.atr_period)

            return {
                "close": Decimal(str(close_prices[-1])),
                "entry_high": Decimal(str(entry_high)),
                "entry_low": Decimal(str(entry_low)),
                "exit_high": Decimal(str(exit_high)),
                "exit_low": Decimal(str(exit_low)),
                "atr": Decimal(str(atr))
            }

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            raise ValueError(f"Failed to calculate indicators: {e}")

    def _calculate_donchian_high(self, high: Any, period: int) -> float:
        """Calculate Donchian Channel high.

        Args:
            high: Array of high prices
            period: Lookback period

        Returns:
            Highest high over period
        """
        import numpy as np
        return float(np.max(high[-period:]))

    def _calculate_donchian_low(self, low: Any, period: int) -> float:
        """Calculate Donchian Channel low.

        Args:
            low: Array of low prices
            period: Lookback period

        Returns:
            Lowest low over period
        """
        import numpy as np
        return float(np.min(low[-period:]))

    def _calculate_atr(self, high: Any, low: Any, close: Any, period: int) -> float:
        """Calculate Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            Current ATR value
        """
        import numpy as np

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Simple moving average of TR
        atr = np.mean(tr[-period:])
        return float(atr)

    def _calculate_breakout_strength(
        self,
        price: Decimal,
        channel_level: Decimal,
        atr: Decimal,
        direction: str
    ) -> Decimal:
        """Calculate breakout signal strength.

        Args:
            price: Current price
            channel_level: Donchian channel level
            atr: Current ATR
            direction: Breakout direction ("BUY" or "SELL")

        Returns:
            Signal strength between 0 and 1
        """
        # Measure breakout distance relative to ATR
        breakout_distance = abs(price - channel_level)
        breakout_strength = min(breakout_distance / (atr * Decimal("2")), Decimal("1.0"))

        return min(max(breakout_strength, Decimal("0.3")), Decimal("1.0"))

    def _calculate_confidence(
        self,
        price: Decimal,
        channel_high: Decimal,
        channel_low: Decimal,
        atr: Decimal
    ) -> Decimal:
        """Calculate signal confidence.

        Args:
            price: Current price
            channel_high: Upper channel
            channel_low: Lower channel
            atr: Current ATR

        Returns:
            Confidence level between 0 and 1
        """
        # Channel width relative to ATR
        channel_width = channel_high - channel_low
        width_ratio = channel_width / (atr * Decimal("4"))

        # Narrower channels = clearer breakouts = higher confidence
        confidence = Decimal("1.0") - min(width_ratio / Decimal("2"), Decimal("0.3"))

        return min(max(confidence, Decimal("0.5")), Decimal("1.0"))

    def calculate_position_size(
        self,
        account_balance: Decimal,
        atr: Decimal,
        price: Decimal
    ) -> Decimal:
        """Calculate position size using Turtle ATR method.

        Args:
            account_balance: Current account balance
            atr: Current ATR value
            price: Current price

        Returns:
            Position size in base currency

        Raises:
            ValueError: If inputs invalid
        """
        try:
            if account_balance <= Decimal("0"):
                raise ValueError("Account balance must be positive")
            if atr <= Decimal("0"):
                raise ValueError("ATR must be positive")
            if price <= Decimal("0"):
                raise ValueError("Price must be positive")

            # Calculate dollar risk per unit (1 ATR move)
            dollar_volatility = atr

            # Calculate risk amount
            risk_amount = account_balance * self.risk_per_trade

            # Calculate position size
            position_size = risk_amount / (dollar_volatility * self.atr_multiplier)

            logger.debug(
                "position_size_calculated",
                balance=str(account_balance),
                risk_amount=str(risk_amount),
                atr=str(atr),
                position_size=str(position_size)
            )

            return position_size

        except Exception as e:
            logger.error("position_size_calculation_failed", error=str(e))
            raise
