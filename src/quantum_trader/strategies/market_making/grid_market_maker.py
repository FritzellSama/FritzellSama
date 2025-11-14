"""
Grid Market Making Strategy.

Places multiple limit orders above and below the current price in a grid pattern,
profiting from bid-ask spread and price oscillations.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
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
class GridLevel:
    """Represents a single grid level."""
    price: Decimal
    quantity: Decimal
    side: OrderSide
    order_id: Optional[str] = None
    is_filled: bool = False
    fill_time: Optional[datetime] = None


class GridMarketMaker:
    """
    Grid Market Making Strategy.

    Places limit orders in a grid pattern above and below mid-price,
    capturing spread and profiting from price oscillation.

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        grid_levels: Current grid levels (buy and sell)
        active_orders: Dictionary of active order IDs
        total_pnl: Cumulative profit and loss
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """
        Initialize Grid Market Maker.

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
        self.grid_levels_count = int(config["grid_levels_count"])
        self.grid_spacing_pct = Decimal(str(config["grid_spacing_pct"]))
        self.order_quantity_base = Decimal(str(config["order_quantity_base"]))
        self.max_position_size = Decimal(str(config["max_position_size"]))
        self.min_spread_bps = Decimal(str(config["min_spread_bps"]))
        self.rebalance_threshold_pct = Decimal(str(config["rebalance_threshold_pct"]))
        self.inventory_target = Decimal(str(config.get("inventory_target", "0")))

        # State tracking
        self.grid_levels: List[GridLevel] = []
        self.active_orders: Dict[str, GridLevel] = {}
        self.current_position = Decimal("0")
        self.total_pnl = Decimal("0")
        self.grid_center_price: Optional[Decimal] = None
        self.last_rebalance_time: Optional[datetime] = None

        logger.info(
            "grid_market_maker_initialized",
            levels=self.grid_levels_count,
            spacing=float(self.grid_spacing_pct)
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required_params = [
            "grid_levels_count",
            "grid_spacing_pct",
            "order_quantity_base",
            "max_position_size",
            "min_spread_bps",
            "rebalance_threshold_pct",
            "symbol",
            "exchange",
            "timeframe"
        ]

        missing = [p for p in required_params if p not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        # Validate ranges
        if int(self.config["grid_levels_count"]) < 2:
            raise ValueError("grid_levels_count must be at least 2")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """
        Generate grid order placement signals.

        Args:
            market_data: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - symbol: Trading pair
                - close: Close price
                - bid: Best bid price
                - ask: Best ask price
                - volume: Trading volume

        Returns:
            List of Signal objects for grid orders

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

            # Get latest market data
            latest_row = market_data.sort("timestamp", descending=True).row(0, named=True)
            current_price = Decimal(str(latest_row["close"]))
            current_time = latest_row["timestamp"]
            symbol = latest_row["symbol"]

            # Get bid/ask if available, otherwise use close price
            bid_price = Decimal(str(latest_row.get("bid", latest_row["close"])))
            ask_price = Decimal(str(latest_row.get("ask", latest_row["close"])))

            # Calculate mid price
            mid_price = (bid_price + ask_price) / Decimal("2")

            # Check if spread is wide enough
            spread_bps = ((ask_price - bid_price) / mid_price) * Decimal("10000")
            if spread_bps < self.min_spread_bps:
                logger.warning(
                    "spread_too_tight",
                    spread_bps=float(spread_bps),
                    min_required=float(self.min_spread_bps)
                )
                return signals

            # Check if grid needs initialization or rebalancing
            needs_rebalance = await self._needs_rebalance(mid_price, current_time)

            if needs_rebalance:
                # Generate grid levels
                grid_levels = await self._generate_grid_levels(mid_price)

                # Create signals for each grid level
                for level in grid_levels:
                    action = (
                        SignalAction.BUY if level.side == OrderSide.BUY
                        else SignalAction.SELL
                    )

                    # Calculate strength based on distance from mid price
                    distance_pct = abs(level.price - mid_price) / mid_price
                    strength = Decimal("1.0") - min(distance_pct * Decimal("10"), Decimal("0.5"))

                    indicators = await self.calculate_indicators(market_data)

                    signal = Signal(
                        symbol=symbol,
                        action=action,
                        strength=strength,
                        confidence=Decimal("0.85"),  # High confidence for grid orders
                        timestamp=current_time,
                        strategy="grid_market_maker",
                        timeframe=self.config["timeframe"],
                        indicators=indicators,
                        metadata={
                            "grid_price": str(level.price),
                            "grid_quantity": str(level.quantity),
                            "mid_price": str(mid_price),
                            "spread_bps": str(spread_bps),
                            "is_limit_order": True
                        }
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)

                if signals:
                    logger.info(
                        "grid_signals_generated",
                        symbol=symbol,
                        signal_count=len(signals),
                        mid_price=float(mid_price)
                    )

            return signals

        except Exception as e:
            logger.error("signal_generation_failed", error=str(e), exc_info=True)
            raise

    async def _needs_rebalance(
        self,
        current_price: Decimal,
        current_time: datetime
    ) -> bool:
        """
        Check if grid needs rebalancing.

        Args:
            current_price: Current mid market price
            current_time: Current timestamp

        Returns:
            True if grid should be rebalanced
        """
        # First time initialization
        if self.grid_center_price is None:
            return True

        # Check if price has moved beyond threshold
        price_move_pct = abs(
            (current_price - self.grid_center_price) / self.grid_center_price
        )

        if price_move_pct > self.rebalance_threshold_pct:
            logger.info(
                "grid_rebalance_triggered",
                price_move_pct=float(price_move_pct),
                threshold=float(self.rebalance_threshold_pct)
            )
            return True

        # Check if enough time has passed (avoid over-rebalancing)
        if self.last_rebalance_time:
            time_delta = (current_time - self.last_rebalance_time).total_seconds()
            min_rebalance_interval = int(self.config.get("min_rebalance_interval_seconds", 300))
            if time_delta < min_rebalance_interval:
                return False

        return False

    async def _generate_grid_levels(self, center_price: Decimal) -> List[GridLevel]:
        """
        Generate grid levels around center price.

        Args:
            center_price: Center price for grid

        Returns:
            List of GridLevel objects
        """
        grid_levels = []

        # Generate buy levels (below center price)
        for i in range(1, self.grid_levels_count + 1):
            price_offset = Decimal(str(i)) * self.grid_spacing_pct / Decimal("100")
            buy_price = center_price * (Decimal("1") - price_offset)

            # Adjust quantity based on distance from center (more aggressive near center)
            quantity_multiplier = Decimal("1") + (Decimal(str(self.grid_levels_count - i)) / Decimal("10"))
            quantity = self.order_quantity_base * quantity_multiplier

            grid_levels.append(GridLevel(
                price=buy_price,
                quantity=quantity,
                side=OrderSide.BUY
            ))

        # Generate sell levels (above center price)
        for i in range(1, self.grid_levels_count + 1):
            price_offset = Decimal(str(i)) * self.grid_spacing_pct / Decimal("100")
            sell_price = center_price * (Decimal("1") + price_offset)

            # Adjust quantity based on inventory position
            inventory_adjustment = self._calculate_inventory_adjustment()
            quantity_multiplier = Decimal("1") + (Decimal(str(self.grid_levels_count - i)) / Decimal("10"))
            quantity = self.order_quantity_base * quantity_multiplier * inventory_adjustment

            grid_levels.append(GridLevel(
                price=sell_price,
                quantity=quantity,
                side=OrderSide.SELL
            ))

        self.grid_levels = grid_levels
        self.grid_center_price = center_price
        self.last_rebalance_time = datetime.now(timezone.utc)

        logger.info(
            "grid_levels_generated",
            center_price=float(center_price),
            total_levels=len(grid_levels)
        )

        return grid_levels

    def _calculate_inventory_adjustment(self) -> Decimal:
        """
        Calculate inventory-based quantity adjustment.

        Returns:
            Multiplier for order quantity (higher when inventory is off-target)
        """
        if self.max_position_size == Decimal("0"):
            return Decimal("1")

        # Calculate inventory deviation from target
        inventory_deviation = self.current_position - self.inventory_target

        # If we have too much inventory, increase sell quantities
        # If we have too little, increase buy quantities
        inventory_pct = inventory_deviation / self.max_position_size

        # Adjustment ranges from 0.5x to 1.5x
        adjustment = Decimal("1") + (inventory_pct * Decimal("0.5"))
        adjustment = max(Decimal("0.5"), min(Decimal("1.5"), adjustment))

        return adjustment

    async def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators for grid trading.

        Args:
            data: Polars DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}

            # Calculate volatility (important for grid spacing)
            if len(data) >= 20:
                returns = data.select([
                    (pl.col("close").pct_change().alias("returns"))
                ])["returns"].to_list()[1:]

                decimal_returns = [
                    Decimal(str(r)) if r is not None else Decimal("0")
                    for r in returns
                ]

                mean_return = sum(decimal_returns) / len(decimal_returns)
                variance = sum(
                    (r - mean_return) ** 2 for r in decimal_returns
                ) / len(decimal_returns)

                std_dev = Decimal(str(variance ** 0.5))
                indicators["volatility"] = std_dev

            # Calculate average spread
            if "bid" in data.columns and "ask" in data.columns:
                spreads = [
                    Decimal(str(row["ask"])) - Decimal(str(row["bid"]))
                    for row in data.tail(10).iter_rows(named=True)
                ]
                avg_spread = sum(spreads) / len(spreads) if spreads else Decimal("0")
                indicators["avg_spread"] = avg_spread

            # Volume profile
            if len(data) >= 10:
                recent_volume = data.tail(10)["volume"].to_list()
                avg_volume = Decimal(str(sum(recent_volume))) / Decimal("10")
                current_volume = Decimal(str(recent_volume[-1]))
                indicators["volume_ratio"] = (
                    current_volume / avg_volume if avg_volume > 0 else Decimal("1")
                )

            # Price range (for grid sizing)
            if len(data) >= 20:
                recent_prices = data.tail(20)["close"].to_list()
                price_high = Decimal(str(max(recent_prices)))
                price_low = Decimal(str(min(recent_prices)))
                price_range = price_high - price_low
                indicators["price_range_pct"] = (
                    price_range / price_low if price_low > 0 else Decimal("0")
                )

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

            # Verify grid price is set
            if "grid_price" not in signal.metadata:
                logger.warning("missing_grid_price")
                return False

            # Check position limits
            if signal.action == SignalAction.BUY:
                if self.current_position >= self.max_position_size:
                    logger.warning("max_long_position_reached")
                    return False
            elif signal.action == SignalAction.SELL:
                if self.current_position <= -self.max_position_size:
                    logger.warning("max_short_position_reached")
                    return False

            return True

        except Exception as e:
            logger.error("signal_validation_error", error=str(e))
            return False

    async def on_order_fill(
        self,
        order_id: str,
        fill_price: Decimal,
        fill_quantity: Decimal,
        side: OrderSide,
        timestamp: datetime
    ) -> None:
        """
        Handle order fill event.

        Args:
            order_id: Order identifier
            fill_price: Fill price
            fill_quantity: Filled quantity
            side: Order side (BUY/SELL)
            timestamp: Fill timestamp
        """
        # Update position
        position_change = fill_quantity if side == OrderSide.BUY else -fill_quantity
        self.current_position += position_change

        # Update grid level if it exists
        if order_id in self.active_orders:
            level = self.active_orders[order_id]
            level.is_filled = True
            level.fill_time = timestamp
            level.order_id = order_id

            # Calculate PnL (simplified - actual PnL calculation more complex)
            if side == OrderSide.SELL:
                self.total_pnl += fill_price * fill_quantity
            else:
                self.total_pnl -= fill_price * fill_quantity

            del self.active_orders[order_id]

            logger.info(
                "grid_order_filled",
                order_id=order_id,
                price=float(fill_price),
                quantity=float(fill_quantity),
                side=side.value,
                current_position=float(self.current_position),
                total_pnl=float(self.total_pnl)
            )

    async def get_active_grid_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of active grid orders.

        Returns:
            List of dictionaries with order details
        """
        return [
            {
                "order_id": order_id,
                "price": str(level.price),
                "quantity": str(level.quantity),
                "side": level.side.value,
                "is_filled": level.is_filled
            }
            for order_id, level in self.active_orders.items()
        ]
