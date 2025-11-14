"""Volatility Arbitrage Strategy for Options Trading.

Exploits mispricings between implied and realized volatility using delta-neutral
positions and statistical arbitrage techniques.
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


class VolatilityArbitrageStrategy(BaseStrategy):
    """Volatility arbitrage using implied vs realized volatility spread.

    Identifies opportunities when options are mispriced relative to expected
    realized volatility. Maintains delta-neutral positions through dynamic hedging.

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management instance
        name: Strategy identifier
        iv_rv_threshold: Min IV-RV spread for entry from config
        lookback_period: Period for realized volatility from config
        min_confidence: Minimum signal confidence from config
        delta_neutral_threshold: Max acceptable delta from config
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize volatility arbitrage strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: Risk manager instance for validation

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)
        self.name = "VolatilityArbitrage"

        self._validate_config()

        # Load parameters from config
        self.iv_rv_threshold = Decimal(str(config["thresholds"]["iv_rv_spread"]))
        self.lookback_period = config["indicators"]["lookback_period"]
        self.min_confidence = Decimal(str(config["signal"]["min_confidence"]))
        self.delta_neutral_threshold = Decimal(str(config["risk"]["delta_neutral_threshold"]))
        self.max_position_vega = Decimal(str(config["risk"]["max_position_vega"]))
        self.parkinson_hl_enabled = config.get("indicators", {}).get("parkinson_hl", True)

        logger.info(
            "volatility_arb_initialized",
            iv_rv_threshold=str(self.iv_rv_threshold),
            lookback_period=self.lookback_period,
            delta_neutral=str(self.delta_neutral_threshold)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_keys = ["thresholds", "indicators", "signal", "risk"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if "iv_rv_spread" not in self.config["thresholds"]:
            raise ValueError("Missing required thresholds.iv_rv_spread")

        if "lookback_period" not in self.config["indicators"]:
            raise ValueError("Missing required indicators.lookback_period")

        if "min_confidence" not in self.config["signal"]:
            raise ValueError("Missing required signal.min_confidence")

        required_risk = ["delta_neutral_threshold", "max_position_vega"]
        for key in required_risk:
            if key not in self.config["risk"]:
                raise ValueError(f"Missing required risk config: {key}")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate volatility arbitrage signals.

        Analyzes IV-RV spread and generates signals for option positions
        when significant mispricings are detected.

        Args:
            market_data: Polars DataFrame with OHLCV and options data

        Returns:
            List of Signal objects for detected opportunities

        Raises:
            ValueError: If market data invalid or insufficient
        """
        try:
            if market_data.height < self.lookback_period + 20:
                logger.warning(
                    "insufficient_data",
                    rows=market_data.height,
                    required=self.lookback_period + 20
                )
                return []

            # Calculate indicators
            indicators = self.calculate_indicators(market_data)

            signals = []

            # Get latest values
            implied_vol = indicators["implied_volatility"]
            realized_vol = indicators["realized_volatility"]
            iv_percentile = indicators["iv_percentile"]
            rv_percentile = indicators["rv_percentile"]

            symbol = market_data["symbol"][0] if "symbol" in market_data.columns else "UNKNOWN"
            timeframe = market_data["timeframe"][0] if "timeframe" in market_data.columns else "1h"

            # Calculate spread
            iv_rv_spread = implied_vol - realized_vol
            spread_pct = (iv_rv_spread / realized_vol) * Decimal("100")

            # Long volatility signal (IV < RV - volatility is cheap)
            if iv_rv_spread < -self.iv_rv_threshold:
                strength = self._calculate_signal_strength(
                    abs(iv_rv_spread), realized_vol, "LONG_VOL"
                )
                confidence = self._calculate_confidence(
                    implied_vol, realized_vol, iv_percentile, rv_percentile
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
                            "implied_volatility": implied_vol,
                            "realized_volatility": realized_vol,
                            "iv_rv_spread": iv_rv_spread,
                            "iv_percentile": iv_percentile,
                            "rv_percentile": rv_percentile
                        },
                        metadata={
                            "position_type": "long_volatility",
                            "spread_pct": spread_pct,
                            "trade_type": "straddle_long",
                            "hedge_ratio": Decimal("1.0"),
                            "target_delta": Decimal("0.0")
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "long_volatility_signal",
                            symbol=symbol,
                            iv=str(implied_vol),
                            rv=str(realized_vol),
                            spread_pct=str(spread_pct),
                            confidence=str(confidence)
                        )

            # Short volatility signal (IV > RV - volatility is expensive)
            elif iv_rv_spread > self.iv_rv_threshold:
                strength = self._calculate_signal_strength(
                    iv_rv_spread, realized_vol, "SHORT_VOL"
                )
                confidence = self._calculate_confidence(
                    implied_vol, realized_vol, iv_percentile, rv_percentile
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
                            "implied_volatility": implied_vol,
                            "realized_volatility": realized_vol,
                            "iv_rv_spread": iv_rv_spread,
                            "iv_percentile": iv_percentile,
                            "rv_percentile": rv_percentile
                        },
                        metadata={
                            "position_type": "short_volatility",
                            "spread_pct": spread_pct,
                            "trade_type": "straddle_short",
                            "hedge_ratio": Decimal("1.0"),
                            "target_delta": Decimal("0.0")
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "short_volatility_signal",
                            symbol=symbol,
                            iv=str(implied_vol),
                            rv=str(realized_vol),
                            spread_pct=str(spread_pct),
                            confidence=str(confidence)
                        )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate volatility indicators.

        Args:
            data: Polars DataFrame with OHLCV and options data

        Returns:
            Dictionary of calculated indicator values

        Raises:
            ValueError: If data invalid or calculation fails
        """
        try:
            close_prices = data["close"].to_numpy()

            # Calculate realized volatility (annualized)
            if self.parkinson_hl_enabled and "high" in data.columns and "low" in data.columns:
                high_prices = data["high"].to_numpy()
                low_prices = data["low"].to_numpy()
                realized_vol = self._calculate_parkinson_volatility(
                    high_prices, low_prices, self.lookback_period
                )
            else:
                realized_vol = self._calculate_close_to_close_volatility(
                    close_prices, self.lookback_period
                )

            # Get implied volatility from data or estimate
            if "implied_volatility" in data.columns:
                implied_vol = Decimal(str(data["implied_volatility"][-1]))
            else:
                # If IV not provided, estimate from ATM options or use proxy
                logger.warning("implied_volatility_not_in_data_using_proxy")
                implied_vol = realized_vol * Decimal("1.2")  # Common IV/RV ratio

            # Calculate percentiles for context
            rv_history = [
                self._calculate_close_to_close_volatility(close_prices[:i+1], min(self.lookback_period, i+1))
                for i in range(max(self.lookback_period, len(close_prices) - 100), len(close_prices))
            ]

            rv_percentile = self._calculate_percentile(realized_vol, rv_history)

            # For IV percentile, use similar approach if historical IV available
            iv_percentile = Decimal("50.0")  # Default to median if no history

            return {
                "implied_volatility": implied_vol,
                "realized_volatility": Decimal(str(realized_vol)),
                "iv_percentile": iv_percentile,
                "rv_percentile": Decimal(str(rv_percentile))
            }

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            raise ValueError(f"Failed to calculate indicators: {e}")

    def _calculate_close_to_close_volatility(
        self,
        prices: Any,
        period: int
    ) -> float:
        """Calculate close-to-close historical volatility.

        Args:
            prices: Array of closing prices
            period: Lookback period

        Returns:
            Annualized volatility
        """
        import numpy as np

        # Calculate log returns
        returns = np.diff(np.log(prices[-period-1:]))

        # Annualize (assuming 365 days)
        volatility = np.std(returns) * np.sqrt(365)

        return float(volatility * 100)  # Return as percentage

    def _calculate_parkinson_volatility(
        self,
        high: Any,
        low: Any,
        period: int
    ) -> float:
        """Calculate Parkinson high-low volatility estimator.

        More efficient than close-to-close, uses high-low range.

        Args:
            high: High prices
            low: Low prices
            period: Lookback period

        Returns:
            Annualized volatility
        """
        import numpy as np

        # Parkinson formula
        hl_ratio = np.log(high[-period:] / low[-period:])
        parkinson_var = np.mean(hl_ratio ** 2) / (4 * np.log(2))

        # Annualize
        volatility = np.sqrt(parkinson_var * 365)

        return float(volatility * 100)  # Return as percentage

    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile rank of value in history.

        Args:
            value: Current value
            history: Historical values

        Returns:
            Percentile rank (0-100)
        """
        import numpy as np

        if not history:
            return 50.0

        sorted_history = sorted(history)
        rank = sum(1 for x in sorted_history if x <= value)
        percentile = (rank / len(sorted_history)) * 100

        return float(percentile)

    def _calculate_signal_strength(
        self,
        spread: Decimal,
        realized_vol: Decimal,
        position_type: str
    ) -> Decimal:
        """Calculate signal strength based on spread magnitude.

        Args:
            spread: Absolute IV-RV spread
            realized_vol: Realized volatility
            position_type: "LONG_VOL" or "SHORT_VOL"

        Returns:
            Signal strength between 0 and 1
        """
        # Normalize spread by realized vol
        normalized_spread = spread / realized_vol

        # Larger spreads = stronger signals
        strength = min(normalized_spread * Decimal("2"), Decimal("1.0"))

        return min(max(strength, Decimal("0.2")), Decimal("1.0"))

    def _calculate_confidence(
        self,
        implied_vol: Decimal,
        realized_vol: Decimal,
        iv_percentile: Decimal,
        rv_percentile: Decimal
    ) -> Decimal:
        """Calculate signal confidence.

        Args:
            implied_vol: Current implied volatility
            realized_vol: Current realized volatility
            iv_percentile: IV percentile rank
            rv_percentile: RV percentile rank

        Returns:
            Confidence level between 0 and 1
        """
        # Spread magnitude component
        spread = abs(implied_vol - realized_vol)
        spread_conf = min(spread / realized_vol, Decimal("1.0"))

        # Percentile extremes increase confidence
        iv_extreme = abs(iv_percentile - Decimal("50")) / Decimal("50")
        rv_extreme = abs(rv_percentile - Decimal("50")) / Decimal("50")
        percentile_conf = (iv_extreme + rv_extreme) / Decimal("2")

        # Combined confidence
        confidence = (spread_conf * Decimal("0.7") + percentile_conf * Decimal("0.3"))

        return min(max(confidence, Decimal("0.4")), Decimal("1.0"))

    def calculate_hedge_ratio(
        self,
        option_delta: Decimal,
        position_size: Decimal
    ) -> Decimal:
        """Calculate hedge ratio for delta-neutral position.

        Args:
            option_delta: Option delta
            position_size: Number of option contracts

        Returns:
            Number of underlying shares to hedge

        Raises:
            ValueError: If inputs invalid
        """
        try:
            if position_size <= Decimal("0"):
                raise ValueError("Position size must be positive")

            # Delta hedge calculation
            hedge_ratio = abs(option_delta * position_size)

            logger.debug(
                "hedge_ratio_calculated",
                option_delta=str(option_delta),
                position_size=str(position_size),
                hedge_ratio=str(hedge_ratio)
            )

            return hedge_ratio

        except Exception as e:
            logger.error("hedge_ratio_calculation_failed", error=str(e))
            raise
