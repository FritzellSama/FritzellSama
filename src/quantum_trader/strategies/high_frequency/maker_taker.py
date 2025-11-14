"""
Maker-Taker High-Frequency Trading Strategy.

Dynamically switches between providing liquidity (maker) and taking
liquidity (taker) based on market microstructure and fee rebates.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
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


class ExecutionMode(Enum):
    """Execution mode selection."""
    MAKER = "MAKER"  # Post limit orders
    TAKER = "TAKER"  # Take liquidity with market orders
    HYBRID = "HYBRID"  # Mix of both


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
class FeeStructure:
    """Exchange fee structure."""
    maker_fee_bps: Decimal
    taker_fee_bps: Decimal
    has_rebate: bool
    maker_rebate_bps: Decimal = Decimal("0")


class MakerTakerStrategy:
    """
    Maker-Taker High-Frequency Strategy.

    Optimizes order execution between maker and taker modes based on:
    - Fee/rebate structure
    - Market microstructure
    - Predicted short-term price movement
    - Queue position and fill probability

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        current_mode: Active execution mode
        fill_rate_tracker: Tracks maker order fill rates
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Maker-Taker Strategy.

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
        self.maker_threshold_bps = Decimal(str(config["maker_threshold_bps"]))
        self.taker_threshold_bps = Decimal(str(config["taker_threshold_bps"]))
        self.min_spread_bps = Decimal(str(config["min_spread_bps"]))
        self.max_quote_age_ms = Decimal(str(config["max_quote_age_ms"]))
        self.queue_position_weight = Decimal(str(config["queue_position_weight"]))
        self.prediction_horizon_ms = Decimal(str(config["prediction_horizon_ms"]))
        self.min_edge_bps = Decimal(str(config["min_edge_bps"]))

        # Fee structure (loaded from config)
        self.fee_structure = FeeStructure(
            maker_fee_bps=Decimal(str(config["maker_fee_bps"])),
            taker_fee_bps=Decimal(str(config["taker_fee_bps"])),
            has_rebate=config.get("maker_rebate_enabled", False),
            maker_rebate_bps=Decimal(str(config.get("maker_rebate_bps", "0")))
        )

        # State tracking
        self.current_mode = ExecutionMode.HYBRID
        self.fill_rate_tracker: deque = deque(maxlen=100)
        self.maker_fill_count = 0
        self.maker_attempt_count = 0
        self.recent_signals: deque = deque(maxlen=50)
        self.last_mode_switch: Optional[datetime] = None

        logger.info(
            "maker_taker_initialized",
            maker_threshold=float(self.maker_threshold_bps),
            taker_threshold=float(self.taker_threshold_bps),
            maker_fee=float(self.fee_structure.maker_fee_bps),
            taker_fee=float(self.fee_structure.taker_fee_bps)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "maker_threshold_bps",
            "taker_threshold_bps",
            "min_spread_bps",
            "max_quote_age_ms",
            "queue_position_weight",
            "prediction_horizon_ms",
            "min_edge_bps",
            "maker_fee_bps",
            "taker_fee_bps",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate trading signals with optimal execution mode.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp (microsecond precision)
                - symbol: Trading pair
                - bid: Best bid price
                - ask: Best ask price
                - bid_volume: Best bid volume
                - ask_volume: Best ask volume
                - last_trade_price: Most recent trade price
                - last_trade_side: Side of last trade (1=buy, -1=sell)

        Returns:
            List of Signal objects with execution mode metadata

        Raises:
            ValueError: If market_data format invalid
        """
        try:
            signals = []

            if market_data.is_empty():
                logger.warning("empty_market_data")
                return signals

            # Validate required columns
            required_cols = ["timestamp", "symbol", "bid", "ask"]
            missing_cols = [c for c in required_cols if c not in market_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Get latest market data
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]
            bid_price = Decimal(str(latest_row["bid"]))
            ask_price = Decimal(str(latest_row["ask"]))

            # Calculate mid price and spread
            mid_price = (bid_price + ask_price) / Decimal("2")
            spread_bps = ((ask_price - bid_price) / mid_price) * Decimal("10000")

            # Check if spread is wide enough to trade
            if spread_bps < self.min_spread_bps:
                logger.debug(
                    "spread_too_tight",
                    spread_bps=float(spread_bps),
                    min_required=float(self.min_spread_bps)
                )
                return signals

            # Predict short-term price direction
            price_prediction = await self._predict_short_term_direction(market_data)

            # Calculate expected edge for maker and taker
            maker_edge = await self._calculate_maker_edge(
                bid_price=bid_price,
                ask_price=ask_price,
                prediction=price_prediction,
                spread_bps=spread_bps
            )

            taker_edge = await self._calculate_taker_edge(
                bid_price=bid_price,
                ask_price=ask_price,
                prediction=price_prediction
            )

            # Determine optimal execution mode
            execution_mode, action, target_price = await self._select_execution_mode(
                maker_edge=maker_edge,
                taker_edge=taker_edge,
                prediction=price_prediction,
                bid_price=bid_price,
                ask_price=ask_price
            )

            # Generate signal if edge is sufficient
            if action != SignalAction.HOLD:
                # Calculate signal strength based on edge magnitude
                if execution_mode == ExecutionMode.MAKER:
                    strength = min(Decimal("1.0"), abs(maker_edge) / self.maker_threshold_bps)
                else:
                    strength = min(Decimal("1.0"), abs(taker_edge) / self.taker_threshold_bps)

                # Calculate confidence based on prediction strength and fill probability
                confidence = await self._calculate_confidence(
                    prediction=price_prediction,
                    execution_mode=execution_mode,
                    spread_bps=spread_bps
                )

                indicators = await self.calculate_indicators(market_data)

                # Determine order type
                order_type = (
                    OrderType.LIMIT if execution_mode == ExecutionMode.MAKER
                    else OrderType.MARKET
                )

                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=current_time,
                    strategy="maker_taker",
                    timeframe=self.config["timeframe"],
                    indicators=indicators,
                    metadata={
                        "execution_mode": execution_mode.value,
                        "target_price": str(target_price),
                        "order_type": order_type.value,
                        "maker_edge_bps": str(maker_edge),
                        "taker_edge_bps": str(taker_edge),
                        "prediction": str(price_prediction),
                        "spread_bps": str(spread_bps),
                        "mid_price": str(mid_price)
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)
                    self.recent_signals.append(signal)

                    logger.info(
                        "maker_taker_signal_generated",
                        symbol=symbol,
                        action=action.value,
                        mode=execution_mode.value,
                        edge=float(maker_edge if execution_mode == ExecutionMode.MAKER else taker_edge)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _predict_short_term_direction(
        self,
        market_data: pl.DataFrame
    ) -> Decimal:
        """
        Predict short-term price direction using microstructure signals.

        Args:
            market_data: Recent market data

        Returns:
            Prediction value (-1 to +1, positive = bullish)
        """
        try:
            prediction = Decimal("0")
            signal_count = 0

            # Signal 1: Order book imbalance
            if "bid_volume" in market_data.columns and "ask_volume" in market_data.columns:
                latest = market_data.tail(1).row(0, named=True)
                bid_vol = Decimal(str(latest["bid_volume"]))
                ask_vol = Decimal(str(latest["ask_volume"]))

                if (bid_vol + ask_vol) > Decimal("0"):
                    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
                    prediction += imbalance * Decimal("0.4")
                    signal_count += 1

            # Signal 2: Recent trade direction
            if "last_trade_side" in market_data.columns:
                recent_trades = market_data.tail(10)["last_trade_side"].to_list()
                trade_pressure = sum(
                    Decimal(str(side)) for side in recent_trades if side is not None
                )
                if recent_trades:
                    trade_signal = trade_pressure / Decimal(str(len(recent_trades)))
                    prediction += trade_signal * Decimal("0.3")
                    signal_count += 1

            # Signal 3: Price momentum
            if len(market_data) >= 5:
                recent_prices = [
                    (Decimal(str(row["bid"])) + Decimal(str(row["ask"]))) / Decimal("2")
                    for row in market_data.tail(5).iter_rows(named=True)
                ]

                if len(recent_prices) >= 2:
                    price_change = recent_prices[-1] - recent_prices[0]
                    # Normalize to -1 to +1 range
                    momentum = price_change / recent_prices[0] * Decimal("1000")
                    momentum = max(Decimal("-1"), min(Decimal("1"), momentum))
                    prediction += momentum * Decimal("0.3")
                    signal_count += 1

            # Normalize prediction
            if signal_count > 0:
                prediction = max(Decimal("-1"), min(Decimal("1"), prediction))

            return prediction

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            return Decimal("0")

    async def _calculate_maker_edge(
        self,
        bid_price: Decimal,
        ask_price: Decimal,
        prediction: Decimal,
        spread_bps: Decimal
    ) -> Decimal:
        """
        Calculate expected edge from maker orders.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            prediction: Price direction prediction
            spread_bps: Current spread in bps

        Returns:
            Expected edge in bps (can be negative)
        """
        mid_price = (bid_price + ask_price) / Decimal("2")

        # If prediction is bullish, we want to buy (post on bid)
        # If prediction is bearish, we want to sell (post on ask)

        if prediction > Decimal("0"):
            # Bullish: post buy order at bid
            # Edge = (rebate - fee) + (spread/2) - predicted adverse selection
            fee_edge = self.fee_structure.maker_rebate_bps - self.fee_structure.maker_fee_bps
            spread_edge = spread_bps / Decimal("2")

            # Adverse selection: if price moves against us before fill
            fill_probability = await self._estimate_maker_fill_probability(spread_bps)
            adverse_selection = prediction * spread_bps * (Decimal("1") - fill_probability)

            edge = fee_edge + spread_edge - adverse_selection

        else:
            # Bearish: post sell order at ask
            fee_edge = self.fee_structure.maker_rebate_bps - self.fee_structure.maker_fee_bps
            spread_edge = spread_bps / Decimal("2")

            # Adverse selection
            fill_probability = await self._estimate_maker_fill_probability(spread_bps)
            adverse_selection = abs(prediction) * spread_bps * (Decimal("1") - fill_probability)

            edge = fee_edge + spread_edge - adverse_selection

        return edge

    async def _calculate_taker_edge(
        self,
        bid_price: Decimal,
        ask_price: Decimal,
        prediction: Decimal
    ) -> Decimal:
        """
        Calculate expected edge from taker orders.

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            prediction: Price direction prediction

        Returns:
            Expected edge in bps
        """
        mid_price = (bid_price + ask_price) / Decimal("2")
        spread_bps = ((ask_price - bid_price) / mid_price) * Decimal("10000")

        # Taker edge = predicted price move - (taker fee + spread_cost)

        if prediction > Decimal("0"):
            # Bullish: buy at ask (lift the offer)
            # Cost: half spread + taker fee
            cost = (spread_bps / Decimal("2")) + self.fee_structure.taker_fee_bps

            # Expected gain from predicted move (simplified)
            expected_move_bps = prediction * spread_bps * Decimal("2")

            edge = expected_move_bps - cost

        else:
            # Bearish: sell at bid (hit the bid)
            cost = (spread_bps / Decimal("2")) + self.fee_structure.taker_fee_bps
            expected_move_bps = abs(prediction) * spread_bps * Decimal("2")

            edge = expected_move_bps - cost

        return edge

    async def _select_execution_mode(
        self,
        maker_edge: Decimal,
        taker_edge: Decimal,
        prediction: Decimal,
        bid_price: Decimal,
        ask_price: Decimal
    ) -> Tuple[ExecutionMode, SignalAction, Decimal]:
        """
        Select optimal execution mode and action.

        Args:
            maker_edge: Expected maker edge
            taker_edge: Expected taker edge
            prediction: Price prediction
            bid_price: Best bid
            ask_price: Best ask

        Returns:
            Tuple of (ExecutionMode, SignalAction, target_price)
        """
        # Compare edges
        if maker_edge >= self.maker_threshold_bps and maker_edge >= taker_edge:
            # Use maker mode
            mode = ExecutionMode.MAKER
            action = SignalAction.BUY if prediction > 0 else SignalAction.SELL
            target_price = bid_price if prediction > 0 else ask_price

        elif taker_edge >= self.taker_threshold_bps and taker_edge > maker_edge:
            # Use taker mode
            mode = ExecutionMode.TAKER
            action = SignalAction.BUY if prediction > 0 else SignalAction.SELL
            target_price = ask_price if prediction > 0 else bid_price

        else:
            # No sufficient edge
            mode = ExecutionMode.HYBRID
            action = SignalAction.HOLD
            target_price = (bid_price + ask_price) / Decimal("2")

        # Update current mode
        if mode != self.current_mode:
            self.current_mode = mode
            self.last_mode_switch = datetime.now(timezone.utc)

        return mode, action, target_price

    async def _estimate_maker_fill_probability(self, spread_bps: Decimal) -> Decimal:
        """
        Estimate probability of maker order getting filled.

        Args:
            spread_bps: Current spread in bps

        Returns:
            Fill probability (0.0 to 1.0)
        """
        # Use historical fill rate if available
        if self.maker_attempt_count > 10:
            historical_rate = Decimal(str(self.maker_fill_count)) / Decimal(str(self.maker_attempt_count))
        else:
            historical_rate = Decimal("0.5")  # Default assumption

        # Adjust based on spread (wider spread = lower fill probability)
        spread_factor = Decimal("1") / (Decimal("1") + spread_bps / Decimal("100"))

        # Combine historical and spread-based estimates
        fill_prob = (historical_rate * Decimal("0.7")) + (spread_factor * Decimal("0.3"))

        return min(Decimal("1.0"), max(Decimal("0.0"), fill_prob))

    async def _calculate_confidence(
        self,
        prediction: Decimal,
        execution_mode: ExecutionMode,
        spread_bps: Decimal
    ) -> Decimal:
        """
        Calculate signal confidence.

        Args:
            prediction: Price prediction
            execution_mode: Selected execution mode
            spread_bps: Current spread

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from prediction strength
        base_confidence = abs(prediction)

        # Adjust for execution mode
        if execution_mode == ExecutionMode.MAKER:
            fill_prob = await self._estimate_maker_fill_probability(spread_bps)
            mode_factor = fill_prob
        else:
            mode_factor = Decimal("0.95")  # Taker orders have high certainty

        # Combine factors
        confidence = base_confidence * Decimal("0.6") + mode_factor * Decimal("0.4")

        return min(Decimal("1.0"), confidence)

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators for HFT.

        Args:
            data: Polars DataFrame with market data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            # Maker fill rate
            if self.maker_attempt_count > 0:
                fill_rate = Decimal(str(self.maker_fill_count)) / Decimal(str(self.maker_attempt_count))
                indicators["maker_fill_rate"] = fill_rate

            # Average spread
            if "bid" in data.columns and "ask" in data.columns:
                recent_data = data.tail(10)
                spreads = []
                for row in recent_data.iter_rows(named=True):
                    bid = Decimal(str(row["bid"]))
                    ask = Decimal(str(row["ask"]))
                    mid = (bid + ask) / Decimal("2")
                    if mid > Decimal("0"):
                        spread_bps = ((ask - bid) / mid) * Decimal("10000")
                        spreads.append(spread_bps)

                if spreads:
                    indicators["avg_spread_bps"] = sum(spreads) / Decimal(str(len(spreads)))

            # Quote stability (lower = more stable)
            if "bid" in data.columns and len(data) >= 5:
                recent_bids = [Decimal(str(row["bid"])) for row in data.tail(5).iter_rows(named=True)]
                bid_changes = [
                    abs(recent_bids[i] - recent_bids[i-1]) / recent_bids[i-1]
                    for i in range(1, len(recent_bids))
                    if recent_bids[i-1] > Decimal("0")
                ]
                if bid_changes:
                    indicators["quote_volatility"] = sum(bid_changes) / Decimal(str(len(bid_changes)))

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
            if signal.confidence < Decimal("0.3"):
                logger.warning("low_confidence_signal", confidence=float(signal.confidence))
                return False

            # Check quote freshness
            now = datetime.now(timezone.utc)
            age_ms = (now - signal.timestamp).total_seconds() * 1000

            if Decimal(str(age_ms)) > self.max_quote_age_ms:
                logger.warning("stale_quote", age_ms=age_ms)
                return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def on_maker_order_placed(self) -> None:
        """Record maker order placement for fill rate tracking."""
        self.maker_attempt_count += 1

    async def on_maker_order_filled(self) -> None:
        """Record maker order fill for fill rate tracking."""
        self.maker_fill_count += 1
        self.fill_rate_tracker.append(True)

    async def on_maker_order_cancelled(self) -> None:
        """Record maker order cancellation."""
        self.fill_rate_tracker.append(False)
