"""Z-Score Mean Reversion Strategy.

Statistical arbitrage strategy using z-score to identify overbought/oversold
conditions and mean reversion opportunities across multiple timeframes.
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


class ZScoreStrategy(BaseStrategy):
    """Z-Score based mean reversion strategy.

    Uses statistical z-score to identify price deviations from mean:
    - Positive z-score: Price above mean (potential short)
    - Negative z-score: Price below mean (potential long)
    - Combines with momentum filters for confirmation

    Attributes:
        config: Strategy configuration parameters
        risk_manager: Risk management instance
        name: Strategy identifier
        lookback_period: Period for mean calculation from config
        entry_zscore: Z-score threshold for entry from config
        exit_zscore: Z-score threshold for exit from config
        min_confidence: Minimum signal confidence from config
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize z-score strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: Risk manager instance for validation

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config, risk_manager)
        self.name = "ZScore"

        self._validate_config()

        # Load parameters from config
        self.lookback_period = config["indicators"]["lookback_period"]
        self.entry_zscore = Decimal(str(config["thresholds"]["entry_zscore"]))
        self.exit_zscore = Decimal(str(config["thresholds"]["exit_zscore"]))
        self.min_confidence = Decimal(str(config["signal"]["min_confidence"]))
        self.use_bollinger_filter = config.get("filters", {}).get("bollinger_enabled", False)
        self.bollinger_period = config.get("filters", {}).get("bollinger_period", 20)
        self.bollinger_std = Decimal(str(config.get("filters", {}).get("bollinger_std", 2.0)))

        logger.info(
            "zscore_strategy_initialized",
            lookback_period=self.lookback_period,
            entry_zscore=str(self.entry_zscore),
            exit_zscore=str(self.exit_zscore)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        required_keys = ["indicators", "thresholds", "signal"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if "lookback_period" not in self.config["indicators"]:
            raise ValueError("Missing required indicators.lookback_period")

        required_thresholds = ["entry_zscore", "exit_zscore"]
        for key in required_thresholds:
            if key not in self.config["thresholds"]:
                raise ValueError(f"Missing required threshold config: {key}")

        if "min_confidence" not in self.config["signal"]:
            raise ValueError("Missing required signal.min_confidence")

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate mean reversion signals based on z-score.

        Identifies extreme price deviations and generates signals
        for mean reversion trades.

        Args:
            market_data: Polars DataFrame with OHLCV data

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
            current_price = indicators["current_price"]
            zscore = indicators["zscore"]
            mean_price = indicators["mean"]
            std_dev = indicators["std_dev"]

            symbol = market_data["symbol"][0] if "symbol" in market_data.columns else "UNKNOWN"
            timeframe = market_data["timeframe"][0] if "timeframe" in market_data.columns else "15m"

            # Additional filters if enabled
            bollinger_confirmed = True
            if self.use_bollinger_filter:
                bb_upper = indicators.get("bb_upper")
                bb_lower = indicators.get("bb_lower")
                if bb_upper is not None and bb_lower is not None:
                    bollinger_confirmed = (
                        (zscore < -self.entry_zscore and current_price < bb_lower) or
                        (zscore > self.entry_zscore and current_price > bb_upper)
                    )

            # Oversold condition - potential long
            if zscore <= -self.entry_zscore and bollinger_confirmed:
                strength = self._calculate_signal_strength(abs(zscore), self.entry_zscore)
                confidence = self._calculate_confidence(
                    zscore, std_dev, current_price, mean_price
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
                            "current_price": current_price,
                            "zscore": zscore,
                            "mean": mean_price,
                            "std_dev": std_dev
                        },
                        metadata={
                            "signal_type": "oversold",
                            "target_price": mean_price,
                            "exit_zscore": self.exit_zscore,
                            "deviation_pct": ((current_price - mean_price) / mean_price * Decimal("100")),
                            "std_devs_from_mean": abs(zscore)
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "oversold_signal",
                            symbol=symbol,
                            price=str(current_price),
                            zscore=str(zscore),
                            confidence=str(confidence)
                        )

            # Overbought condition - potential short
            elif zscore >= self.entry_zscore and bollinger_confirmed:
                strength = self._calculate_signal_strength(zscore, self.entry_zscore)
                confidence = self._calculate_confidence(
                    zscore, std_dev, current_price, mean_price
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
                            "current_price": current_price,
                            "zscore": zscore,
                            "mean": mean_price,
                            "std_dev": std_dev
                        },
                        metadata={
                            "signal_type": "overbought",
                            "target_price": mean_price,
                            "exit_zscore": self.exit_zscore,
                            "deviation_pct": ((current_price - mean_price) / mean_price * Decimal("100")),
                            "std_devs_from_mean": zscore
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        logger.info(
                            "overbought_signal",
                            symbol=symbol,
                            price=str(current_price),
                            zscore=str(zscore),
                            confidence=str(confidence)
                        )

            # Exit signals
            elif abs(zscore) <= self.exit_zscore:
                # Price has reverted to mean
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.CLOSE,
                    strength=Decimal("0.7"),
                    confidence=Decimal("0.8"),
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=timeframe,
                    indicators={
                        "current_price": current_price,
                        "zscore": zscore,
                        "mean": mean_price
                    },
                    metadata={
                        "signal_type": "mean_reversion_complete",
                        "exit_reason": "zscore_normalized"
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    logger.info(
                        "reversion_complete",
                        symbol=symbol,
                        price=str(current_price),
                        zscore=str(zscore)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate z-score and related indicators.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of calculated indicator values

        Raises:
            ValueError: If data invalid or calculation fails
        """
        try:
            close_prices = data["close"].to_numpy()

            # Calculate rolling mean and std dev
            prices_lookback = close_prices[-self.lookback_period:]
            mean_price = self._calculate_mean(prices_lookback)
            std_dev = self._calculate_std_dev(prices_lookback, mean_price)

            # Calculate z-score
            current_price = close_prices[-1]
            if std_dev > 0:
                zscore = (current_price - mean_price) / std_dev
            else:
                zscore = 0.0

            result = {
                "current_price": Decimal(str(current_price)),
                "mean": Decimal(str(mean_price)),
                "std_dev": Decimal(str(std_dev)),
                "zscore": Decimal(str(zscore))
            }

            # Add Bollinger Bands if enabled
            if self.use_bollinger_filter:
                bb_mean = self._calculate_mean(close_prices[-self.bollinger_period:])
                bb_std = self._calculate_std_dev(
                    close_prices[-self.bollinger_period:],
                    bb_mean
                )
                result["bb_upper"] = Decimal(str(bb_mean + float(self.bollinger_std) * bb_std))
                result["bb_lower"] = Decimal(str(bb_mean - float(self.bollinger_std) * bb_std))
                result["bb_middle"] = Decimal(str(bb_mean))

            return result

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            raise ValueError(f"Failed to calculate indicators: {e}")

    def _calculate_mean(self, prices: Any) -> float:
        """Calculate arithmetic mean of prices.

        Args:
            prices: Array of prices

        Returns:
            Mean value
        """
        import numpy as np
        return float(np.mean(prices))

    def _calculate_std_dev(self, prices: Any, mean: float) -> float:
        """Calculate standard deviation.

        Args:
            prices: Array of prices
            mean: Mean value

        Returns:
            Standard deviation
        """
        import numpy as np
        return float(np.std(prices))

    def _calculate_signal_strength(
        self,
        zscore_abs: Decimal,
        entry_threshold: Decimal
    ) -> Decimal:
        """Calculate signal strength based on z-score magnitude.

        Args:
            zscore_abs: Absolute z-score value
            entry_threshold: Entry threshold

        Returns:
            Signal strength between 0 and 1
        """
        # Higher z-score = stronger signal
        # Normalize: 2x threshold = max strength
        strength = min(zscore_abs / (entry_threshold * Decimal("2")), Decimal("1.0"))

        return min(max(strength, Decimal("0.3")), Decimal("1.0"))

    def _calculate_confidence(
        self,
        zscore: Decimal,
        std_dev: Decimal,
        current_price: Decimal,
        mean_price: Decimal
    ) -> Decimal:
        """Calculate signal confidence.

        Args:
            zscore: Current z-score
            std_dev: Standard deviation
            current_price: Current price
            mean_price: Mean price

        Returns:
            Confidence level between 0 and 1
        """
        # Higher absolute z-score = higher confidence
        zscore_conf = min(abs(zscore) / Decimal("5.0"), Decimal("1.0"))

        # Lower std dev relative to price = more stable mean = higher confidence
        cv = std_dev / mean_price  # Coefficient of variation
        cv_conf = Decimal("1.0") - min(Decimal(str(cv)) * Decimal("10"), Decimal("0.4"))

        # Combined confidence
        confidence = (zscore_conf * Decimal("0.7") + cv_conf * Decimal("0.3"))

        return min(max(confidence, Decimal("0.4")), Decimal("1.0"))

    def calculate_position_size(
        self,
        account_balance: Decimal,
        zscore: Decimal,
        std_dev: Decimal
    ) -> Decimal:
        """Calculate position size based on z-score and volatility.

        Higher z-score = larger position (stronger signal)
        Higher volatility = smaller position (more risk)

        Args:
            account_balance: Current account balance
            zscore: Current z-score
            std_dev: Standard deviation

        Returns:
            Position size as percentage of balance

        Raises:
            ValueError: If inputs invalid
        """
        try:
            if account_balance <= Decimal("0"):
                raise ValueError("Account balance must be positive")
            if std_dev <= Decimal("0"):
                raise ValueError("Standard deviation must be positive")

            # Base position size from config
            base_size_pct = Decimal(str(self.config.get("risk", {}).get("base_position_pct", 0.05)))

            # Scale by z-score strength (max 2x at zscore=4)
            zscore_multiplier = Decimal("1.0") + min(abs(zscore) / Decimal("4.0"), Decimal("1.0"))

            # Scale by inverse of volatility (lower vol = larger position)
            # Assuming reasonable CV range of 0-0.2
            cv = std_dev / account_balance
            vol_multiplier = Decimal("1.0") / (Decimal("1.0") + Decimal(str(cv)) * Decimal("5"))

            # Calculate final position size
            position_size_pct = base_size_pct * zscore_multiplier * vol_multiplier

            # Cap at max from config
            max_position_pct = Decimal(str(self.config.get("risk", {}).get("max_position_pct", 0.15)))
            position_size_pct = min(position_size_pct, max_position_pct)

            logger.debug(
                "position_size_calculated",
                zscore=str(zscore),
                cv=str(cv),
                position_pct=str(position_size_pct)
            )

            return position_size_pct

        except Exception as e:
            logger.error("position_size_calculation_failed", error=str(e))
            raise
