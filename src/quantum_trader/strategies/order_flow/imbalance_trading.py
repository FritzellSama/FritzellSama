"""
Order Flow Imbalance Trading Strategy.

Analyzes orderbook imbalances and aggressive buying/selling pressure
to predict short-term price movements.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import polars as pl
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
class OrderBookSnapshot:
    """Orderbook snapshot at a point in time."""
    timestamp: datetime
    bid_volume: Decimal
    ask_volume: Decimal
    bid_price: Decimal
    ask_price: Decimal
    imbalance_ratio: Decimal


class ImbalanceTradingStrategy:
    """
    Order Flow Imbalance Trading Strategy.

    Monitors orderbook dynamics and trade flow to identify
    directional pressure and liquidity imbalances.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        imbalance_history: Recent imbalance measurements
        trade_flow_history: Recent trade flow data
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Order Flow Imbalance Strategy.

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
        self.imbalance_threshold = Decimal(str(config["imbalance_threshold"]))
        self.min_volume_threshold = Decimal(str(config["min_volume_threshold"]))
        self.lookback_periods = int(config["lookback_periods"])
        self.aggressive_flow_weight = Decimal(str(config["aggressive_flow_weight"]))
        self.position_size_base = Decimal(str(config["position_size_base"]))
        self.max_holding_seconds = int(config["max_holding_seconds"])
        self.depth_levels = int(config.get("depth_levels", 5))

        # State tracking
        self.imbalance_history: deque = deque(maxlen=self.lookback_periods)
        self.trade_flow_history: deque = deque(maxlen=self.lookback_periods)
        self.orderbook_snapshots: deque = deque(maxlen=self.lookback_periods)
        self.current_position = Decimal("0")
        self.last_signal_time: Optional[datetime] = None

        logger.info(
            "imbalance_trading_initialized",
            imbalance_threshold=float(self.imbalance_threshold),
            lookback_periods=self.lookback_periods
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "imbalance_threshold",
            "min_volume_threshold",
            "lookback_periods",
            "aggressive_flow_weight",
            "position_size_base",
            "max_holding_seconds",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on order flow imbalance.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - bid_volume: Best bid volume
                - ask_volume: Best ask volume
                - bid_price: Best bid price
                - ask_price: Best ask price
                - trade_volume: Recent trade volume
                - trade_side: Dominant trade side (1=buy, -1=sell)

        Returns:
            List of Signal objects

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest market data
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]

            # Extract orderbook data
            bid_volume = Decimal(str(latest_row.get("bid_volume", 0)))
            ask_volume = Decimal(str(latest_row.get("ask_volume", 0)))
            bid_price = Decimal(str(latest_row.get("bid_price", latest_row.get("close", 0))))
            ask_price = Decimal(str(latest_row.get("ask_price", latest_row.get("close", 0))))

            # Check minimum volume requirement
            total_volume = bid_volume + ask_volume
            if total_volume < self.min_volume_threshold:
                logger.debug(
                    "insufficient_volume",
                    total_volume=float(total_volume),
                    threshold=float(self.min_volume_threshold)
                )
                return signals

            # Calculate orderbook imbalance
            imbalance = await self._calculate_orderbook_imbalance(
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                bid_price=bid_price,
                ask_price=ask_price,
                timestamp=current_time
            )

            # Calculate trade flow imbalance
            trade_flow = await self._calculate_trade_flow(market_data)

            # Combine orderbook and trade flow signals
            combined_signal = await self._combine_signals(imbalance, trade_flow)

            # Generate trading signal if threshold exceeded
            if abs(combined_signal) >= self.imbalance_threshold:
                action = (
                    SignalAction.BUY if combined_signal > 0
                    else SignalAction.SELL
                )

                # Calculate signal strength based on magnitude
                strength = min(
                    Decimal("1.0"),
                    abs(combined_signal) / self.imbalance_threshold
                )

                # Calculate confidence based on persistence
                confidence = await self._calculate_confidence()

                indicators = await self.calculate_indicators(market_data)

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=current_time,
                    strategy="imbalance_trading",
                    timeframe=self.config["timeframe"],
                    indicators=indicators,
                    metadata={
                        "orderbook_imbalance": str(imbalance),
                        "trade_flow_imbalance": str(trade_flow),
                        "combined_signal": str(combined_signal),
                        "bid_volume": str(bid_volume),
                        "ask_volume": str(ask_volume)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    self.last_signal_time = current_time

                    logger.info(
                        "imbalance_signal_generated",
                        symbol=symbol,
                        action=action.value,
                        strength=float(strength),
                        imbalance=float(combined_signal)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _calculate_orderbook_imbalance(
        self,
        bid_volume: Decimal,
        ask_volume: Decimal,
        bid_price: Decimal,
        ask_price: Decimal,
        timestamp: datetime
    ) -> Decimal:
        """
        Calculate orderbook imbalance ratio.

        Args:
            bid_volume: Total bid volume
            ask_volume: Total ask volume
            bid_price: Best bid price
            ask_price: Best ask price
            timestamp: Current timestamp

        Returns:
            Imbalance ratio (-1 to +1, positive = buying pressure)
        """
        total_volume = bid_volume + ask_volume

        if total_volume == Decimal("0"):
            return Decimal("0")

        # Basic imbalance: (bid_volume - ask_volume) / total_volume
        basic_imbalance = (bid_volume - ask_volume) / total_volume

        # Weight by spread (tighter spread = more reliable signal)
        mid_price = (bid_price + ask_price) / Decimal("2")
        if mid_price > Decimal("0"):
            spread_pct = (ask_price - bid_price) / mid_price
            # Reduce signal strength if spread is wide
            spread_weight = Decimal("1") / (Decimal("1") + spread_pct * Decimal("100"))
            weighted_imbalance = basic_imbalance * spread_weight
        else:
            weighted_imbalance = basic_imbalance

        # Store snapshot
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            bid_price=bid_price,
            ask_price=ask_price,
            imbalance_ratio=weighted_imbalance
        )
        self.orderbook_snapshots.append(snapshot)
        self.imbalance_history.append(weighted_imbalance)

        return weighted_imbalance

    async def _calculate_trade_flow(self, market_data: pl.DataFrame) -> Decimal:
        """
        Calculate trade flow imbalance from recent trades.

        Args:
            market_data: Market data with trade information

        Returns:
            Trade flow imbalance (-1 to +1)
        """
        try:
            # Look for trade-related columns
            if "trade_side" in market_data.columns:
                # Use provided trade side indicator
                recent_trades = market_data.tail(self.lookback_periods)
                trade_sides = recent_trades["trade_side"].to_list()

                # Calculate net buying pressure
                net_pressure = sum(Decimal(str(side)) for side in trade_sides if side is not None)
                count = len([s for s in trade_sides if s is not None])

                if count > 0:
                    flow_imbalance = net_pressure / Decimal(str(count))
                    self.trade_flow_history.append(flow_imbalance)
                    return flow_imbalance

            # Fallback: infer from price and volume
            if len(market_data) >= 2 and "close" in market_data.columns:
                recent_data = market_data.tail(self.lookback_periods)

                # Calculate price changes weighted by volume
                price_changes = []
                for i in range(1, len(recent_data)):
                    prev_row = recent_data.row(i - 1, named=True)
                    curr_row = recent_data.row(i, named=True)

                    price_change = Decimal(str(curr_row["close"])) - Decimal(str(prev_row["close"]))
                    volume = Decimal(str(curr_row.get("volume", 1)))

                    # Positive price change = buying, negative = selling
                    weighted_change = price_change * volume
                    price_changes.append(weighted_change)

                if price_changes:
                    net_change = sum(price_changes)
                    total_volume = sum(abs(change) for change in price_changes)

                    if total_volume > Decimal("0"):
                        flow_imbalance = net_change / total_volume
                        # Normalize to -1 to +1 range
                        flow_imbalance = max(
                            Decimal("-1"),
                            min(Decimal("1"), flow_imbalance * Decimal("100"))
                        )
                        self.trade_flow_history.append(flow_imbalance)
                        return flow_imbalance

            return Decimal("0")

        except Exception as e:
            logger.error("trade_flow_calculation_error", error=str(e))
            return Decimal("0")

    async def _combine_signals(
        self,
        orderbook_imbalance: Decimal,
        trade_flow: Decimal
    ) -> Decimal:
        """
        Combine orderbook and trade flow signals.

        Args:
            orderbook_imbalance: Orderbook imbalance ratio
            trade_flow: Trade flow imbalance

        Returns:
            Combined signal (-1 to +1)
        """
        # Weight trade flow more heavily (aggressive orders more predictive)
        orderbook_weight = Decimal("1") - self.aggressive_flow_weight
        trade_flow_weight = self.aggressive_flow_weight

        combined = (
            orderbook_imbalance * orderbook_weight +
            trade_flow * trade_flow_weight
        )

        # Normalize to -1 to +1
        combined = max(Decimal("-1"), min(Decimal("1"), combined))

        return combined

    async def _calculate_confidence(self) -> Decimal:
        """
        Calculate signal confidence based on persistence.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if len(self.imbalance_history) < 3:
            return Decimal("0.5")

        # Check if recent imbalances are persistent (same direction)
        recent_imbalances = list(self.imbalance_history)[-3:]

        # Count how many are in the same direction
        positive_count = sum(1 for x in recent_imbalances if x > Decimal("0"))
        negative_count = sum(1 for x in recent_imbalances if x < Decimal("0"))

        # Higher confidence if all in same direction
        max_direction = max(positive_count, negative_count)
        consistency = Decimal(str(max_direction)) / Decimal("3")

        # Also consider magnitude
        avg_magnitude = sum(abs(x) for x in recent_imbalances) / Decimal("3")

        # Combine consistency and magnitude
        confidence = (consistency * Decimal("0.6")) + (avg_magnitude * Decimal("0.4"))

        return min(Decimal("1.0"), confidence)

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators for order flow analysis.

        Args:
            data: Polars DataFrame with market data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            # Average imbalance over lookback period
            if self.imbalance_history:
                avg_imbalance = sum(self.imbalance_history) / Decimal(str(len(self.imbalance_history)))
                indicators["avg_imbalance"] = avg_imbalance

                # Imbalance momentum (change in imbalance)
                if len(self.imbalance_history) >= 2:
                    imbalance_change = list(self.imbalance_history)[-1] - list(self.imbalance_history)[-2]
                    indicators["imbalance_momentum"] = imbalance_change

            # Trade flow indicators
            if self.trade_flow_history:
                avg_flow = sum(self.trade_flow_history) / Decimal(str(len(self.trade_flow_history)))
                indicators["avg_trade_flow"] = avg_flow

            # Volume metrics
            if "volume" in data.columns and len(data) >= 10:
                recent_volume = data.tail(10)["volume"].to_list()
                avg_volume = Decimal(str(sum(recent_volume))) / Decimal("10")
                current_volume = Decimal(str(recent_volume[-1]))
                indicators["volume_ratio"] = (
                    current_volume / avg_volume if avg_volume > 0 else Decimal("1")
                )

            # Spread analysis
            if self.orderbook_snapshots:
                recent_snapshot = list(self.orderbook_snapshots)[-1]
                mid_price = (recent_snapshot.bid_price + recent_snapshot.ask_price) / Decimal("2")
                if mid_price > Decimal("0"):
                    spread_bps = (
                        (recent_snapshot.ask_price - recent_snapshot.bid_price) / mid_price
                    ) * Decimal("10000")
                    indicators["spread_bps"] = spread_bps

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

            # Check confidence threshold
            min_confidence = Decimal(str(self.config.get("min_confidence", "0.4")))
            if signal.confidence < min_confidence:
                logger.warning(
                    "low_confidence_signal",
                    confidence=float(signal.confidence),
                    min_required=float(min_confidence)
                )
                return False

            # Avoid over-trading (minimum time between signals)
            if self.last_signal_time:
                time_delta = (signal.timestamp - self.last_signal_time).total_seconds()
                min_signal_interval = int(self.config.get("min_signal_interval_seconds", 5))
                if time_delta < min_signal_interval:
                    logger.debug("signal_too_soon", time_delta=time_delta)
                    return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def update_orderbook(
        self,
        bid_levels: List[Tuple[Decimal, Decimal]],
        ask_levels: List[Tuple[Decimal, Decimal]],
        timestamp: datetime
    ) -> None:
        """
        Update orderbook snapshot with multi-level depth.

        Args:
            bid_levels: List of (price, volume) tuples for bids
            ask_levels: List of (price, volume) tuples for asks
            timestamp: Snapshot timestamp
        """
        # Aggregate volume across depth levels
        total_bid_volume = sum(volume for _, volume in bid_levels[:self.depth_levels])
        total_ask_volume = sum(volume for _, volume in ask_levels[:self.depth_levels])

        best_bid = bid_levels[0][0] if bid_levels else Decimal("0")
        best_ask = ask_levels[0][0] if ask_levels else Decimal("0")

        await self._calculate_orderbook_imbalance(
            bid_volume=total_bid_volume,
            ask_volume=total_ask_volume,
            bid_price=best_bid,
            ask_price=best_ask,
            timestamp=timestamp
        )
