"""
Momentum Scanner Strategy.

Scans multiple assets for momentum breakouts and generates signals
based on price acceleration and volume confirmation.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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
class MomentumMetrics:
    """Momentum metrics for an asset."""
    symbol: str
    momentum_score: Decimal
    price_acceleration: Decimal
    volume_surge: Decimal
    relative_strength: Decimal
    trend_quality: Decimal
    timestamp: datetime


class MomentumScanner:
    """
    Momentum Scanner Strategy.

    Identifies assets with strong momentum characteristics:
    - Price acceleration
    - Volume confirmation
    - Relative strength vs market
    - Trend persistence

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        momentum_rankings: Current momentum rankings by asset
        historical_momentum: Historical momentum data
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Momentum Scanner.

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
        self.momentum_threshold = Decimal(str(config["momentum_threshold"]))
        self.min_volume_surge = Decimal(str(config["min_volume_surge"]))
        self.lookback_short = int(config["lookback_short_periods"])
        self.lookback_long = int(config["lookback_long_periods"])
        self.acceleration_threshold = Decimal(str(config["acceleration_threshold"]))
        self.min_trend_quality = Decimal(str(config["min_trend_quality"]))
        self.top_n_assets = int(config.get("top_n_assets", 5))

        # State tracking
        self.momentum_rankings: Dict[str, MomentumMetrics] = {}
        self.historical_momentum: Dict[str, deque] = {}
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.last_scan_time: Optional[datetime] = None

        logger.info(
            "momentum_scanner_initialized",
            momentum_threshold=float(self.momentum_threshold),
            lookback_short=self.lookback_short,
            lookback_long=self.lookback_long
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "momentum_threshold",
            "min_volume_surge",
            "lookback_short_periods",
            "lookback_long_periods",
            "acceleration_threshold",
            "min_trend_quality",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate momentum-based trading signals.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Trading volume

        Returns:
            List of Signal objects for high-momentum assets

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest timestamp
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]

            # Calculate momentum metrics
            momentum_metrics = await self._calculate_momentum_metrics(
                market_data=market_data,
                symbol=symbol,
                current_time=current_time
            )

            if not momentum_metrics:
                return signals

            # Store metrics
            self.momentum_rankings[symbol] = momentum_metrics

            # Initialize historical tracking if needed
            if symbol not in self.historical_momentum:
                self.historical_momentum[symbol] = deque(maxlen=100)
            self.historical_momentum[symbol].append(momentum_metrics)

            # Check if momentum threshold met
            if momentum_metrics.momentum_score >= self.momentum_threshold:
                # Determine signal action based on momentum direction
                action = await self._determine_action(momentum_metrics)

                # Calculate signal strength from momentum score
                strength = min(
                    Decimal("1.0"),
                    momentum_metrics.momentum_score / self.momentum_threshold
                )

                # Calculate confidence from multiple factors
                confidence = await self._calculate_confidence(momentum_metrics)

                indicators = await self.calculate_indicators(market_data)

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=current_time,
                    strategy="momentum_scanner",
                    timeframe=self.config["timeframe"],
                    indicators=indicators,
                    metadata={
                        "momentum_score": str(momentum_metrics.momentum_score),
                        "price_acceleration": str(momentum_metrics.price_acceleration),
                        "volume_surge": str(momentum_metrics.volume_surge),
                        "relative_strength": str(momentum_metrics.relative_strength),
                        "trend_quality": str(momentum_metrics.trend_quality)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    logger.info(
                        "momentum_signal_generated",
                        symbol=symbol,
                        action=action.value,
                        momentum_score=float(momentum_metrics.momentum_score),
                        strength=float(strength)
                    )

            self.last_scan_time = current_time

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _calculate_momentum_metrics(
        self,
        market_data: pl.DataFrame,
        symbol: str,
        current_time: datetime
    ) -> Optional[MomentumMetrics]:
        """
        Calculate comprehensive momentum metrics.

        Args:
            market_data: Market data DataFrame
            symbol: Asset symbol
            current_time: Current timestamp

        Returns:
            MomentumMetrics object or None if insufficient data
        """
        try:
            if len(market_data) < self.lookback_long:
                logger.debug(
                    "insufficient_data_for_momentum",
                    symbol=symbol,
                    data_length=len(market_data),
                    required=self.lookback_long
                )
                return None

            # Extract price and volume data
            closes = [Decimal(str(row["close"])) for row in market_data.iter_rows(named=True)]
            volumes = [Decimal(str(row["volume"])) for row in market_data.iter_rows(named=True)]

            # Store price/volume history
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=self.lookback_long * 2)
                self.volume_history[symbol] = deque(maxlen=self.lookback_long * 2)

            self.price_history[symbol].extend(closes)
            self.volume_history[symbol].extend(volumes)

            # Calculate price momentum (rate of change)
            price_momentum = await self._calculate_price_momentum(closes)

            # Calculate price acceleration (second derivative)
            price_acceleration = await self._calculate_price_acceleration(closes)

            # Calculate volume surge
            volume_surge = await self._calculate_volume_surge(volumes)

            # Calculate relative strength
            relative_strength = await self._calculate_relative_strength(closes)

            # Calculate trend quality (how smooth/persistent is the trend)
            trend_quality = await self._calculate_trend_quality(closes)

            # Combine into overall momentum score
            momentum_score = await self._calculate_composite_momentum(
                price_momentum=price_momentum,
                price_acceleration=price_acceleration,
                volume_surge=volume_surge,
                relative_strength=relative_strength,
                trend_quality=trend_quality
            )

            metrics = MomentumMetrics(
                symbol=symbol,
                momentum_score=momentum_score,
                price_acceleration=price_acceleration,
                volume_surge=volume_surge,
                relative_strength=relative_strength,
                trend_quality=trend_quality,
                timestamp=current_time
            )

            return metrics

        except Exception as e:
            logger.error("momentum_calculation_error", symbol=symbol, error=str(e))
            return None

    async def _calculate_price_momentum(self, closes: List[Decimal]) -> Decimal:
        """
        Calculate price momentum (rate of change).

        Args:
            closes: List of closing prices

        Returns:
            Momentum value
        """
        if len(closes) < self.lookback_short:
            return Decimal("0")

        # Short-term momentum
        short_return = (closes[-1] - closes[-self.lookback_short]) / closes[-self.lookback_short]

        # Long-term momentum
        if len(closes) >= self.lookback_long:
            long_return = (closes[-1] - closes[-self.lookback_long]) / closes[-self.lookback_long]
        else:
            long_return = short_return

        # Weight short-term more heavily
        momentum = (short_return * Decimal("0.6")) + (long_return * Decimal("0.4"))

        return momentum

    async def _calculate_price_acceleration(self, closes: List[Decimal]) -> Decimal:
        """
        Calculate price acceleration (change in momentum).

        Args:
            closes: List of closing prices

        Returns:
            Acceleration value
        """
        if len(closes) < self.lookback_short * 2:
            return Decimal("0")

        # Calculate momentum at two different points
        recent_momentum = (
            (closes[-1] - closes[-self.lookback_short]) / closes[-self.lookback_short]
        )

        prior_momentum = (
            (closes[-self.lookback_short] - closes[-self.lookback_short * 2]) /
            closes[-self.lookback_short * 2]
        )

        # Acceleration is change in momentum
        acceleration = recent_momentum - prior_momentum

        return acceleration

    async def _calculate_volume_surge(self, volumes: List[Decimal]) -> Decimal:
        """
        Calculate volume surge ratio.

        Args:
            volumes: List of volume values

        Returns:
            Volume surge ratio
        """
        if len(volumes) < self.lookback_short * 2:
            return Decimal("1")

        # Recent average volume
        recent_avg = sum(volumes[-self.lookback_short:]) / Decimal(str(self.lookback_short))

        # Historical average volume
        historical_avg = sum(
            volumes[-self.lookback_short * 2:-self.lookback_short]
        ) / Decimal(str(self.lookback_short))

        if historical_avg > Decimal("0"):
            surge_ratio = recent_avg / historical_avg
        else:
            surge_ratio = Decimal("1")

        return surge_ratio

    async def _calculate_relative_strength(self, closes: List[Decimal]) -> Decimal:
        """
        Calculate relative strength (simplified RSI-like indicator).

        Args:
            closes: List of closing prices

        Returns:
            Relative strength value (0-1)
        """
        if len(closes) < 14:
            return Decimal("0.5")

        # Calculate gains and losses
        gains = []
        losses = []

        for i in range(1, min(15, len(closes))):
            change = closes[-i] - closes[-i-1]
            if change > Decimal("0"):
                gains.append(change)
            else:
                losses.append(abs(change))

        avg_gain = sum(gains) / Decimal("14") if gains else Decimal("0.0001")
        avg_loss = sum(losses) / Decimal("14") if losses else Decimal("0.0001")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        # Normalize to 0-1
        normalized_rs = rsi / Decimal("100")

        return normalized_rs

    async def _calculate_trend_quality(self, closes: List[Decimal]) -> Decimal:
        """
        Calculate trend quality (smoothness and persistence).

        Args:
            closes: List of closing prices

        Returns:
            Trend quality score (0-1)
        """
        if len(closes) < self.lookback_short:
            return Decimal("0")

        recent_closes = closes[-self.lookback_short:]

        # Calculate linear regression R-squared
        x_values = list(range(len(recent_closes)))
        y_values = [float(c) for c in recent_closes]

        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)

        # Calculate slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return Decimal("0")

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = [slope * x + intercept for x in x_values]
        ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)

        if ss_tot == 0:
            return Decimal("0")

        r_squared = Decimal(str(1 - (ss_res / ss_tot)))

        # Ensure non-negative
        trend_quality = max(Decimal("0"), r_squared)

        return trend_quality

    async def _calculate_composite_momentum(
        self,
        price_momentum: Decimal,
        price_acceleration: Decimal,
        volume_surge: Decimal,
        relative_strength: Decimal,
        trend_quality: Decimal
    ) -> Decimal:
        """
        Calculate composite momentum score from components.

        Args:
            price_momentum: Price momentum value
            price_acceleration: Price acceleration value
            volume_surge: Volume surge ratio
            relative_strength: Relative strength value
            trend_quality: Trend quality score

        Returns:
            Composite momentum score
        """
        # Normalize components
        norm_momentum = price_momentum * Decimal("10")  # Scale returns
        norm_acceleration = price_acceleration * Decimal("20")  # Scale acceleration
        norm_volume = (volume_surge - Decimal("1")) * Decimal("0.5")  # Center around 1
        norm_rs = (relative_strength - Decimal("0.5")) * Decimal("2")  # Center around 0.5

        # Weight components
        weights = {
            "momentum": Decimal("0.30"),
            "acceleration": Decimal("0.25"),
            "volume": Decimal("0.20"),
            "relative_strength": Decimal("0.15"),
            "trend_quality": Decimal("0.10")
        }

        composite = (
            norm_momentum * weights["momentum"] +
            norm_acceleration * weights["acceleration"] +
            norm_volume * weights["volume"] +
            norm_rs * weights["relative_strength"] +
            trend_quality * weights["trend_quality"]
        )

        # Take absolute value for momentum magnitude
        return abs(composite)

    async def _determine_action(self, metrics: MomentumMetrics) -> SignalAction:
        """
        Determine trading action from momentum metrics.

        Args:
            metrics: MomentumMetrics object

        Returns:
            SignalAction
        """
        # Check price acceleration direction
        if metrics.price_acceleration > self.acceleration_threshold:
            return SignalAction.BUY
        elif metrics.price_acceleration < -self.acceleration_threshold:
            return SignalAction.SELL
        else:
            # Use relative strength to determine direction
            if metrics.relative_strength > Decimal("0.6"):
                return SignalAction.BUY
            elif metrics.relative_strength < Decimal("0.4"):
                return SignalAction.SELL
            else:
                return SignalAction.HOLD

    async def _calculate_confidence(self, metrics: MomentumMetrics) -> Decimal:
        """
        Calculate signal confidence from metrics.

        Args:
            metrics: MomentumMetrics object

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal("0.5")  # Base confidence

        # Increase confidence with strong trend quality
        confidence += metrics.trend_quality * Decimal("0.2")

        # Increase confidence with volume confirmation
        if metrics.volume_surge >= self.min_volume_surge:
            confidence += Decimal("0.2")

        # Increase confidence with strong relative strength extreme
        rs_deviation = abs(metrics.relative_strength - Decimal("0.5"))
        confidence += rs_deviation * Decimal("0.2")

        return min(Decimal("1.0"), confidence)

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            if len(data) < 2:
                return indicators

            closes = [Decimal(str(row["close"])) for row in data.iter_rows(named=True)]
            volumes = [Decimal(str(row["volume"])) for row in data.iter_rows(named=True)]

            # Rate of change
            if len(closes) >= self.lookback_short:
                roc = (closes[-1] - closes[-self.lookback_short]) / closes[-self.lookback_short]
                indicators["roc"] = roc * Decimal("100")  # As percentage

            # Average volume
            if volumes:
                avg_volume = sum(volumes[-20:]) / Decimal(str(min(20, len(volumes))))
                indicators["avg_volume"] = avg_volume

            # Volatility
            if len(closes) >= 20:
                returns = [
                    (closes[i] - closes[i-1]) / closes[i-1]
                    for i in range(1, min(21, len(closes)))
                ]
                mean_ret = sum(returns) / len(returns)
                variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
                indicators["volatility"] = Decimal(str(np.sqrt(float(variance))))

            return indicators

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            return {}

    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate signal before execution.

        Args:
            signal: Signal to validate

        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Check signal strength
            if signal.strength <= Decimal("0") or signal.strength > Decimal("1"):
                logger.warning("invalid_signal_strength", strength=float(signal.strength))
                return False

            # Check confidence
            if signal.confidence < Decimal("0.5"):
                logger.warning("low_confidence_signal", confidence=float(signal.confidence))
                return False

            # Verify momentum metadata
            if "momentum_score" not in signal.metadata:
                logger.warning("missing_momentum_score")
                return False

            # Check volume surge requirement
            volume_surge = Decimal(str(signal.metadata.get("volume_surge", "0")))
            if volume_surge < self.min_volume_surge:
                logger.debug(
                    "insufficient_volume_surge",
                    surge=float(volume_surge),
                    min_required=float(self.min_volume_surge)
                )
                return False

            # Check trend quality
            trend_quality = Decimal(str(signal.metadata.get("trend_quality", "0")))
            if trend_quality < self.min_trend_quality:
                logger.debug(
                    "insufficient_trend_quality",
                    quality=float(trend_quality),
                    min_required=float(self.min_trend_quality)
                )
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def get_momentum_rankings(self) -> List[Tuple[str, Decimal]]:
        """
        Get ranked list of assets by momentum score.

        Returns:
            List of (symbol, momentum_score) tuples, sorted descending
        """
        rankings = [
            (symbol, metrics.momentum_score)
            for symbol, metrics in self.momentum_rankings.items()
        ]

        # Sort by momentum score descending
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings[:self.top_n_assets]
