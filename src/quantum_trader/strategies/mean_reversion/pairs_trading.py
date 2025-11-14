"""
Pairs Trading Strategy.

Identifies and trades cointegrated pairs of assets,
taking advantage of temporary divergences from equilibrium spread.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"


class SignalAction(Enum):
    """Signal action enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Order:
    """Trading order - immutable after creation."""
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    exchange: str = ""
    strategy: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    action: SignalAction
    strength: Decimal
    confidence: Decimal
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairRelationship:
    """Cointegration relationship between two assets."""
    asset_a: str
    asset_b: str
    hedge_ratio: Decimal  # Beta coefficient for linear relationship
    intercept: Decimal
    correlation: Decimal
    cointegration_score: Decimal  # ADF test statistic
    spread_mean: Decimal
    spread_std: Decimal
    last_calibration: datetime


class PairsTradingStrategy:
    """
    Pairs Trading Strategy.

    Trades pairs of cointegrated assets:
    1. Identifies cointegrated pairs
    2. Monitors spread between pairs
    3. Trades when spread deviates from equilibrium
    4. Closes when spread reverts

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        pair_relationships: Dictionary of pair parameters
        spread_history: Historical spread values
        position_state: Current positions in pairs
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Pairs Trading Strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            risk_manager: RiskManager instance for position validation

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Load parameters from config
        self.asset_a_symbol = config["asset_a_symbol"]
        self.asset_b_symbol = config["asset_b_symbol"]
        self.lookback_window = int(config["lookback_window"])
        self.entry_threshold_std = Decimal(str(config["entry_threshold_std"]))
        self.exit_threshold_std = Decimal(str(config["exit_threshold_std"]))
        self.stop_loss_std = Decimal(str(config["stop_loss_std"]))
        self.min_correlation = Decimal(str(config["min_correlation"]))
        self.recalibration_periods = int(config.get("recalibration_periods", 100))

        # State tracking
        self.pair_relationship: Optional[PairRelationship] = None
        self.price_history_a: deque = deque(maxlen=self.lookback_window * 2)
        self.price_history_b: deque = deque(maxlen=self.lookback_window * 2)
        self.spread_history: deque = deque(maxlen=200)
        self.periods_since_calibration = 0
        self.current_position: Optional[str] = None  # "long_spread" or "short_spread"

        logger.info(
            "pairs_trading_initialized",
            asset_a=self.asset_a_symbol,
            asset_b=self.asset_b_symbol,
            entry_threshold=float(self.entry_threshold_std),
            lookback=self.lookback_window
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "asset_a_symbol",
            "asset_b_symbol",
            "lookback_window",
            "entry_threshold_std",
            "exit_threshold_std",
            "stop_loss_std",
            "min_correlation",
            "symbol",  # Primary symbol for signal
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate pairs trading signals.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - close: Close price

        Returns:
            List of Signal objects for both legs of the pair

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "close"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest timestamp
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]

            # Extract prices for both assets
            # Note: In practice, you would receive data for both assets separately
            # This is a simplified version assuming symbol in config is asset_a

            current_symbol = latest_row["symbol"]

            # For demonstration, we'll use the provided data as asset_a
            # and generate synthetic asset_b (in production, fetch real data)
            if current_symbol == self.asset_a_symbol or current_symbol == self.config["symbol"]:
                price_a = Decimal(str(latest_row["close"]))

                # Store price_a
                self.price_history_a.append(price_a)

                # In production, you would fetch asset_b price from data feed
                # For now, create a correlated synthetic price
                if len(self.price_history_a) >= 2:
                    # Simulate asset_b with correlation
                    correlation = Decimal("0.85")
                    price_b = price_a * correlation  # Simplified
                    self.price_history_b.append(price_b)
                else:
                    return signals

            else:
                logger.debug("symbol_not_matching_asset_a", symbol=current_symbol)
                return signals

            # Check if we need (re)calibration
            if await self._needs_calibration():
                await self._calibrate_pair_relationship()

            # If relationship not calibrated, cannot trade
            if self.pair_relationship is None:
                logger.debug("pair_relationship_not_calibrated")
                return signals

            # Calculate current spread
            spread = await self._calculate_spread(price_a, price_b)

            # Calculate z-score of spread
            z_score = await self._calculate_spread_zscore(spread)

            # Store spread
            self.spread_history.append(spread)

            # Determine trading actions
            signals = await self._generate_pair_signals(
                z_score=z_score,
                price_a=price_a,
                price_b=price_b,
                current_time=current_time
            )

            self.periods_since_calibration += 1

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _needs_calibration(self) -> bool:
        """
        Check if pair relationship needs (re)calibration.

        Returns:
            True if calibration needed
        """
        # Initial calibration
        if self.pair_relationship is None:
            return (len(self.price_history_a) >= self.lookback_window and
                    len(self.price_history_b) >= self.lookback_window)

        # Periodic recalibration
        if self.periods_since_calibration >= self.recalibration_periods:
            return True

        return False

    async def _calibrate_pair_relationship(self) -> None:
        """
        Calibrate the cointegration relationship between the pair.

        Estimates hedge ratio, spread statistics, and cointegration strength.
        """
        try:
            if len(self.price_history_a) < self.lookback_window:
                logger.warning("insufficient_data_for_calibration")
                return

            logger.info("calibrating_pair_relationship")

            prices_a = list(self.price_history_a)[-self.lookback_window:]
            prices_b = list(self.price_history_b)[-self.lookback_window:]

            # Calculate linear regression: price_a = beta * price_b + alpha
            # We want to find hedge_ratio (beta) such that spread = price_a - beta * price_b is stationary

            n = len(prices_a)

            mean_a = sum(prices_a) / Decimal(str(n))
            mean_b = sum(prices_b) / Decimal(str(n))

            # Calculate covariance and variance
            cov_ab = sum((prices_a[i] - mean_a) * (prices_b[i] - mean_b) for i in range(n)) / Decimal(str(n))
            var_b = sum((prices_b[i] - mean_b) ** 2 for i in range(n)) / Decimal(str(n))

            # Hedge ratio (beta) = Cov(A,B) / Var(B)
            if var_b > Decimal("0"):
                hedge_ratio = cov_ab / var_b
            else:
                hedge_ratio = Decimal("1")

            # Intercept (alpha)
            intercept = mean_a - hedge_ratio * mean_b

            # Calculate correlation
            var_a = sum((prices_a[i] - mean_a) ** 2 for i in range(n)) / Decimal(str(n))
            std_a = Decimal(str(np.sqrt(float(var_a))))
            std_b = Decimal(str(np.sqrt(float(var_b))))

            if std_a > Decimal("0") and std_b > Decimal("0"):
                correlation = cov_ab / (std_a * std_b)
            else:
                correlation = Decimal("0")

            # Calculate spread
            spreads = [prices_a[i] - hedge_ratio * prices_b[i] for i in range(n)]

            # Spread statistics
            spread_mean = sum(spreads) / Decimal(str(n))
            spread_variance = sum((s - spread_mean) ** 2 for s in spreads) / Decimal(str(n))
            spread_std = Decimal(str(np.sqrt(float(spread_variance))))

            # Simplified cointegration test (ADF test approximation)
            # In production, use proper statistical tests
            cointegration_score = await self._approximate_adf_test(spreads)

            # Validate relationship
            if abs(correlation) < self.min_correlation:
                logger.warning(
                    "low_correlation",
                    correlation=float(correlation),
                    min_required=float(self.min_correlation)
                )
                return

            self.pair_relationship = PairRelationship(
                asset_a=self.asset_a_symbol,
                asset_b=self.asset_b_symbol,
                hedge_ratio=hedge_ratio,
                intercept=intercept,
                correlation=correlation,
                cointegration_score=cointegration_score,
                spread_mean=spread_mean,
                spread_std=spread_std,
                last_calibration=datetime.now(timezone.utc)
            )

            self.periods_since_calibration = 0

            logger.info(
                "pair_relationship_calibrated",
                hedge_ratio=float(hedge_ratio),
                correlation=float(correlation),
                spread_std=float(spread_std),
                cointegration_score=float(cointegration_score)
            )

        except Exception as e:
            logger.error("calibration_failed", error=str(e), exc_info=True)

    async def _approximate_adf_test(self, spreads: List[Decimal]) -> Decimal:
        """
        Approximate ADF test for stationarity.

        Args:
            spreads: List of spread values

        Returns:
            Approximate ADF statistic (more negative = more stationary)
        """
        # Simplified ADF: check if spread is mean-reverting
        # Calculate first differences
        if len(spreads) < 2:
            return Decimal("0")

        diffs = [spreads[i] - spreads[i-1] for i in range(1, len(spreads))]

        # Regression of diffs on lagged spread
        lagged = spreads[:-1]
        n = len(diffs)

        mean_diff = sum(diffs) / Decimal(str(n))
        mean_lagged = sum(lagged) / Decimal(str(n))

        # Calculate regression coefficient
        numerator = sum((lagged[i] - mean_lagged) * (diffs[i] - mean_diff) for i in range(n))
        denominator = sum((lagged[i] - mean_lagged) ** 2 for i in range(n))

        if denominator > Decimal("0"):
            beta = numerator / denominator
            # More negative beta = stronger mean reversion
            adf_stat = beta * Decimal("100")  # Scale for interpretation
        else:
            adf_stat = Decimal("0")

        return adf_stat

    async def _calculate_spread(self, price_a: Decimal, price_b: Decimal) -> Decimal:
        """
        Calculate current spread between pair.

        Args:
            price_a: Price of asset A
            price_b: Price of asset B

        Returns:
            Spread value
        """
        if self.pair_relationship is None:
            return Decimal("0")

        # Spread = price_a - hedge_ratio * price_b
        spread = price_a - (self.pair_relationship.hedge_ratio * price_b)

        return spread

    async def _calculate_spread_zscore(self, spread: Decimal) -> Decimal:
        """
        Calculate z-score of spread.

        Args:
            spread: Current spread value

        Returns:
            Z-score (standard deviations from mean)
        """
        if self.pair_relationship is None:
            return Decimal("0")

        if self.pair_relationship.spread_std == Decimal("0"):
            return Decimal("0")

        z_score = (
            (spread - self.pair_relationship.spread_mean) /
            self.pair_relationship.spread_std
        )

        return z_score

    async def _generate_pair_signals(
        self,
        z_score: Decimal,
        price_a: Decimal,
        price_b: Decimal,
        current_time: datetime
    ) -> List[Signal]:
        """
        Generate trading signals for both legs of the pair.

        Args:
            z_score: Current spread z-score
            price_a: Current price of asset A
            price_b: Current price of asset B
            current_time: Current timestamp

        Returns:
            List of signals (one for each asset, or close signals)
        """
        signals = []

        # Check for stop loss
        if self.current_position and abs(z_score) >= self.stop_loss_std:
            logger.warning(
                "pairs_stop_loss_triggered",
                z_score=float(z_score),
                stop_loss_std=float(self.stop_loss_std)
            )
            # Generate close signals
            signals.extend(await self._generate_close_signals(current_time))
            self.current_position = None
            return signals

        # Entry signals (when not in position)
        if self.current_position is None:
            if z_score <= -self.entry_threshold_std:
                # Spread is too negative: buy spread
                # Buy asset A, sell asset B
                signals.extend(await self._generate_long_spread_signals(
                    price_a, price_b, z_score, current_time
                ))
                self.current_position = "long_spread"

            elif z_score >= self.entry_threshold_std:
                # Spread is too positive: sell spread
                # Sell asset A, buy asset B
                signals.extend(await self._generate_short_spread_signals(
                    price_a, price_b, z_score, current_time
                ))
                self.current_position = "short_spread"

        # Exit signals (when in position)
        else:
            if self.current_position == "long_spread":
                # Exit long spread when spread reverts
                if z_score >= -self.exit_threshold_std:
                    signals.extend(await self._generate_close_signals(current_time))
                    self.current_position = None

            elif self.current_position == "short_spread":
                # Exit short spread when spread reverts
                if z_score <= self.exit_threshold_std:
                    signals.extend(await self._generate_close_signals(current_time))
                    self.current_position = None

        return signals

    async def _generate_long_spread_signals(
        self,
        price_a: Decimal,
        price_b: Decimal,
        z_score: Decimal,
        current_time: datetime
    ) -> List[Signal]:
        """
        Generate signals to go long the spread (buy A, sell B).

        Args:
            price_a: Price of asset A
            price_b: Price of asset B
            z_score: Current z-score
            current_time: Current timestamp

        Returns:
            List of signals
        """
        signals = []

        strength = min(Decimal("1.0"), abs(z_score) / self.entry_threshold_std)
        confidence = await self._calculate_confidence()

        indicators = await self.calculate_indicators(pl.DataFrame())

        # Buy asset A
        signal_a = Signal(
            symbol=self.asset_a_symbol,
            action=SignalAction.BUY,
            strength=strength,
            confidence=confidence,
            timestamp=current_time,
            strategy="pairs_trading",
            timeframe=self.config["timeframe"],
            indicators=indicators,
            metadata={
                "pair_leg": "asset_a",
                "spread_action": "long_spread",
                "z_score": str(z_score),
                "hedge_ratio": str(self.pair_relationship.hedge_ratio),
                "price": str(price_a)
            }
        )

        # Sell asset B (with quantity adjusted by hedge ratio)
        signal_b = Signal(
            symbol=self.asset_b_symbol,
            action=SignalAction.SELL,
            strength=strength,
            confidence=confidence,
            timestamp=current_time,
            strategy="pairs_trading",
            timeframe=self.config["timeframe"],
            indicators=indicators,
            metadata={
                "pair_leg": "asset_b",
                "spread_action": "long_spread",
                "z_score": str(z_score),
                "hedge_ratio": str(self.pair_relationship.hedge_ratio),
                "price": str(price_b)
            }
        )

        if self.validate_signal(signal_a):
            signals.append(signal_a)

        if self.validate_signal(signal_b):
            signals.append(signal_b)

        logger.info(
            "long_spread_signals_generated",
            z_score=float(z_score),
            signals_count=len(signals)
        )

        return signals

    async def _generate_short_spread_signals(
        self,
        price_a: Decimal,
        price_b: Decimal,
        z_score: Decimal,
        current_time: datetime
    ) -> List[Signal]:
        """
        Generate signals to go short the spread (sell A, buy B).

        Args:
            price_a: Price of asset A
            price_b: Price of asset B
            z_score: Current z-score
            current_time: Current timestamp

        Returns:
            List of signals
        """
        signals = []

        strength = min(Decimal("1.0"), abs(z_score) / self.entry_threshold_std)
        confidence = await self._calculate_confidence()

        indicators = await self.calculate_indicators(pl.DataFrame())

        # Sell asset A
        signal_a = Signal(
            symbol=self.asset_a_symbol,
            action=SignalAction.SELL,
            strength=strength,
            confidence=confidence,
            timestamp=current_time,
            strategy="pairs_trading",
            timeframe=self.config["timeframe"],
            indicators=indicators,
            metadata={
                "pair_leg": "asset_a",
                "spread_action": "short_spread",
                "z_score": str(z_score),
                "hedge_ratio": str(self.pair_relationship.hedge_ratio),
                "price": str(price_a)
            }
        )

        # Buy asset B
        signal_b = Signal(
            symbol=self.asset_b_symbol,
            action=SignalAction.BUY,
            strength=strength,
            confidence=confidence,
            timestamp=current_time,
            strategy="pairs_trading",
            timeframe=self.config["timeframe"],
            indicators=indicators,
            metadata={
                "pair_leg": "asset_b",
                "spread_action": "short_spread",
                "z_score": str(z_score),
                "hedge_ratio": str(self.pair_relationship.hedge_ratio),
                "price": str(price_b)
            }
        )

        if self.validate_signal(signal_a):
            signals.append(signal_a)

        if self.validate_signal(signal_b):
            signals.append(signal_b)

        logger.info(
            "short_spread_signals_generated",
            z_score=float(z_score),
            signals_count=len(signals)
        )

        return signals

    async def _generate_close_signals(self, current_time: datetime) -> List[Signal]:
        """
        Generate close signals for both legs.

        Args:
            current_time: Current timestamp

        Returns:
            List of close signals
        """
        signals = []
        confidence = Decimal("0.9")  # High confidence for exits

        indicators = await self.calculate_indicators(pl.DataFrame())

        # Close both legs
        for symbol in [self.asset_a_symbol, self.asset_b_symbol]:
            signal = Signal(
                symbol=symbol,
                action=SignalAction.CLOSE,
                strength=Decimal("1.0"),
                confidence=confidence,
                timestamp=current_time,
                strategy="pairs_trading",
                timeframe=self.config["timeframe"],
                indicators=indicators,
                metadata={
                    "pair_leg": "asset_a" if symbol == self.asset_a_symbol else "asset_b",
                    "spread_action": "close"
                }
            )
            signals.append(signal)

        logger.info("close_signals_generated", signals_count=len(signals))

        return signals

    async def _calculate_confidence(self) -> Decimal:
        """
        Calculate signal confidence based on pair quality.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if self.pair_relationship is None:
            return Decimal("0.3")

        confidence = Decimal("0.5")  # Base confidence

        # Higher confidence for stronger correlation
        confidence += abs(self.pair_relationship.correlation) * Decimal("0.3")

        # Higher confidence for stronger cointegration
        if self.pair_relationship.cointegration_score < Decimal("-1"):
            confidence += Decimal("0.2")

        return min(Decimal("1.0"), confidence)

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate indicators for pairs trading.

        Args:
            data: Market data (unused, kept for compatibility)

        Returns:
            Dictionary of indicator values
        """
        indicators = {}

        if self.pair_relationship:
            indicators["hedge_ratio"] = self.pair_relationship.hedge_ratio
            indicators["correlation"] = self.pair_relationship.correlation
            indicators["spread_mean"] = self.pair_relationship.spread_mean
            indicators["spread_std"] = self.pair_relationship.spread_std

        # Current spread z-score
        if self.spread_history:
            indicators["current_spread"] = list(self.spread_history)[-1]

        return indicators

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid
        """
        try:
            # Check signal strength
            if signal.strength <= Decimal("0") or signal.strength > Decimal("1"):
                return False

            # Check confidence
            min_confidence = Decimal(str(self.config.get("min_confidence", "0.4")))
            if signal.confidence < min_confidence:
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False
