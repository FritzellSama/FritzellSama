"""Liquidity Model for Market Simulation.

Models market depth and liquidity for realistic order fill simulation.
"""

import asyncio
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class LiquidityLevel:
    """Represents liquidity at a specific price level.

    Attributes:
        price: Price level
        quantity: Available quantity at this price
        num_orders: Number of orders at this level
    """
    price: Decimal
    quantity: Decimal
    num_orders: int = 1


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bid and ask liquidity.

    Attributes:
        timestamp: Snapshot timestamp
        symbol: Trading symbol
        bids: List of bid liquidity levels (descending price)
        asks: List of ask liquidity levels (ascending price)
        spread: Bid-ask spread
    """
    timestamp: datetime
    symbol: str
    bids: List[LiquidityLevel]
    asks: List[LiquidityLevel]

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return Decimal("0")
        return self.asks[0].price - self.bids[0].price

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return Decimal("0")
        return (self.bids[0].price + self.asks[0].price) / Decimal("2")


class LiquidityModel:
    """Model market liquidity and depth for backtesting.

    Simulates realistic order book depth and liquidity dynamics based
    on historical patterns and market conditions.

    Attributes:
        config: Model configuration from config files
        symbol: Trading symbol
        liquidity_profiles: Liquidity profiles by market condition
    """

    def __init__(self, config: Dict[str, Any], symbol: str) -> None:
        """Initialize liquidity model.

        Args:
            config: Configuration dictionary with liquidity parameters
            symbol: Trading symbol to model

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "depth_levels": 10,
            ...     "base_liquidity": {"BTC/USDT": 100.0},
            ...     "spread_bps": 5,
            ...     "volatility_multiplier": 1.5
            ... }
            >>> model = LiquidityModel(config, "BTC/USDT")
        """
        self.config = config
        self.symbol = symbol
        self._validate_config()

        # Model parameters
        self.depth_levels = config["depth_levels"]
        self.base_liquidity = Decimal(str(config["base_liquidity"].get(symbol, config.get("default_base_liquidity", 100.0))))
        self.spread_bps = Decimal(str(config["spread_bps"]))
        self.volatility_multiplier = Decimal(str(config.get("volatility_multiplier", 1.0)))

        # Liquidity decay parameters
        self.level_decay_factor = Decimal(str(config.get("level_decay_factor", 0.8)))

        # Current state
        self.current_orderbook: Optional[OrderBookSnapshot] = None

        logger.info(
            "LiquidityModel initialized",
            symbol=symbol,
            depth_levels=self.depth_levels,
            base_liquidity=float(self.base_liquidity)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters missing or invalid
        """
        required_keys = ["depth_levels", "base_liquidity", "spread_bps"]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("Config validation failed", missing_keys=missing_keys)
            raise ValueError(error_msg)

        if self.config["depth_levels"] <= 0:
            raise ValueError("depth_levels must be positive")

        if not isinstance(self.config["base_liquidity"], dict):
            raise ValueError("base_liquidity must be a dictionary")

    async def generate_orderbook(
        self,
        mid_price: Decimal,
        timestamp: datetime,
        volatility: Optional[Decimal] = None
    ) -> OrderBookSnapshot:
        """Generate synthetic order book based on mid price.

        Args:
            mid_price: Current mid market price
            timestamp: Current timestamp
            volatility: Optional volatility factor (0-1)

        Returns:
            Generated order book snapshot

        Raises:
            ValueError: If mid_price is invalid

        Example:
            >>> model = LiquidityModel(config, "BTC/USDT")
            >>> mid_price = Decimal("50000.00")
            >>> orderbook = await model.generate_orderbook(mid_price, datetime.utcnow())
        """
        if mid_price <= Decimal("0"):
            error_msg = f"Invalid mid_price: {mid_price}"
            logger.error("Invalid mid_price", mid_price=str(mid_price))
            raise ValueError(error_msg)

        try:
            # Calculate spread based on volatility
            effective_volatility = volatility or Decimal("1.0")
            spread = self._calculate_spread(mid_price, effective_volatility)

            # Generate bid and ask sides
            bid_price = mid_price - (spread / Decimal("2"))
            ask_price = mid_price + (spread / Decimal("2"))

            bids = await self._generate_liquidity_levels(
                bid_price,
                self.depth_levels,
                is_bid=True,
                volatility=effective_volatility
            )

            asks = await self._generate_liquidity_levels(
                ask_price,
                self.depth_levels,
                is_bid=False,
                volatility=effective_volatility
            )

            orderbook = OrderBookSnapshot(
                timestamp=timestamp,
                symbol=self.symbol,
                bids=bids,
                asks=asks
            )

            self.current_orderbook = orderbook

            logger.debug(
                "Order book generated",
                symbol=self.symbol,
                mid_price=str(mid_price),
                spread=str(spread),
                bid_levels=len(bids),
                ask_levels=len(asks)
            )

            return orderbook

        except Exception as e:
            logger.error("Failed to generate order book", error=str(e), symbol=self.symbol)
            raise

    def _calculate_spread(self, mid_price: Decimal, volatility: Decimal) -> Decimal:
        """Calculate bid-ask spread based on price and volatility.

        Args:
            mid_price: Mid market price
            volatility: Volatility factor

        Returns:
            Calculated spread
        """
        # Base spread in basis points
        base_spread = (self.spread_bps / Decimal("10000")) * mid_price

        # Adjust for volatility
        volatility_adjusted_spread = base_spread * (Decimal("1") + (volatility - Decimal("1")) * self.volatility_multiplier)

        return max(base_spread, volatility_adjusted_spread)

    async def _generate_liquidity_levels(
        self,
        start_price: Decimal,
        num_levels: int,
        is_bid: bool,
        volatility: Decimal
    ) -> List[LiquidityLevel]:
        """Generate liquidity levels for one side of the order book.

        Args:
            start_price: Starting price (best bid/ask)
            num_levels: Number of levels to generate
            is_bid: True for bid side, False for ask side
            volatility: Volatility factor

        Returns:
            List of liquidity levels
        """
        levels = []
        current_price = start_price
        current_liquidity = self.base_liquidity

        # Price tick size (0.01% of price)
        tick_size = start_price * Decimal("0.0001")

        for i in range(num_levels):
            # Add liquidity level
            levels.append(LiquidityLevel(
                price=current_price,
                quantity=current_liquidity,
                num_orders=max(1, int(float(current_liquidity) / 10))
            ))

            # Decay liquidity as we move away from best price
            current_liquidity = current_liquidity * self.level_decay_factor

            # Move to next price level
            if is_bid:
                current_price = current_price - tick_size * Decimal(str(i + 1))
            else:
                current_price = current_price + tick_size * Decimal(str(i + 1))

        return levels

    async def calculate_available_liquidity(
        self,
        price: Decimal,
        is_buy: bool,
        max_slippage_pct: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate available liquidity at a given price.

        Args:
            price: Target price
            is_buy: True for buy order, False for sell order
            max_slippage_pct: Maximum acceptable slippage percentage

        Returns:
            Available liquidity quantity

        Raises:
            ValueError: If no order book available

        Example:
            >>> model = LiquidityModel(config, "BTC/USDT")
            >>> orderbook = await model.generate_orderbook(Decimal("50000"), datetime.utcnow())
            >>> liquidity = await model.calculate_available_liquidity(Decimal("50010"), is_buy=True)
        """
        if self.current_orderbook is None:
            error_msg = "No order book available. Call generate_orderbook first."
            logger.error("No order book available")
            raise ValueError(error_msg)

        try:
            # Select appropriate side
            levels = self.current_orderbook.asks if is_buy else self.current_orderbook.bids

            if not levels:
                logger.warning("No liquidity levels available", is_buy=is_buy)
                return Decimal("0")

            # Calculate maximum acceptable price based on slippage
            if max_slippage_pct is not None:
                reference_price = levels[0].price
                if is_buy:
                    max_price = reference_price * (Decimal("1") + max_slippage_pct / Decimal("100"))
                else:
                    max_price = reference_price * (Decimal("1") - max_slippage_pct / Decimal("100"))
            else:
                max_price = price

            # Sum available liquidity up to max price
            total_liquidity = Decimal("0")

            for level in levels:
                if is_buy:
                    if level.price > max_price:
                        break
                else:
                    if level.price < max_price:
                        break

                total_liquidity += level.quantity

            logger.debug(
                "Available liquidity calculated",
                price=str(price),
                is_buy=is_buy,
                total_liquidity=str(total_liquidity)
            )

            return total_liquidity

        except Exception as e:
            logger.error("Failed to calculate available liquidity", error=str(e))
            raise

    async def simulate_order_impact(
        self,
        order_quantity: Decimal,
        is_buy: bool
    ) -> Tuple[Decimal, Decimal]:
        """Simulate market impact of an order.

        Args:
            order_quantity: Size of the order
            is_buy: True for buy order, False for sell order

        Returns:
            Tuple of (average_fill_price, total_quantity_filled)

        Raises:
            ValueError: If no order book available or order_quantity invalid

        Example:
            >>> model = LiquidityModel(config, "BTC/USDT")
            >>> orderbook = await model.generate_orderbook(Decimal("50000"), datetime.utcnow())
            >>> avg_price, filled_qty = await model.simulate_order_impact(Decimal("10"), is_buy=True)
        """
        if self.current_orderbook is None:
            error_msg = "No order book available. Call generate_orderbook first."
            logger.error("No order book available")
            raise ValueError(error_msg)

        if order_quantity <= Decimal("0"):
            error_msg = f"Invalid order_quantity: {order_quantity}"
            logger.error("Invalid order quantity", order_quantity=str(order_quantity))
            raise ValueError(error_msg)

        try:
            # Select appropriate side
            levels = self.current_orderbook.asks if is_buy else self.current_orderbook.bids

            if not levels:
                logger.warning("No liquidity available for order")
                return Decimal("0"), Decimal("0")

            remaining_quantity = order_quantity
            total_cost = Decimal("0")
            total_filled = Decimal("0")

            # Walk through order book levels
            for level in levels:
                if remaining_quantity <= Decimal("0"):
                    break

                # Quantity available at this level
                available_qty = min(remaining_quantity, level.quantity)

                # Fill at this price level
                total_cost += available_qty * level.price
                total_filled += available_qty
                remaining_quantity -= available_qty

            # Calculate average fill price
            if total_filled > Decimal("0"):
                average_fill_price = total_cost / total_filled
            else:
                average_fill_price = levels[0].price

            logger.debug(
                "Order impact simulated",
                order_quantity=str(order_quantity),
                is_buy=is_buy,
                average_fill_price=str(average_fill_price),
                total_filled=str(total_filled)
            )

            return average_fill_price, total_filled

        except Exception as e:
            logger.error("Failed to simulate order impact", error=str(e))
            raise

    def update_liquidity_after_fill(
        self,
        fill_price: Decimal,
        fill_quantity: Decimal,
        is_buy: bool
    ) -> None:
        """Update order book after order fill.

        Args:
            fill_price: Price at which order was filled
            fill_quantity: Quantity that was filled
            is_buy: True if buy order, False if sell order

        Example:
            >>> model = LiquidityModel(config, "BTC/USDT")
            >>> # ... generate orderbook and fill order ...
            >>> model.update_liquidity_after_fill(Decimal("50010"), Decimal("5"), is_buy=True)
        """
        if self.current_orderbook is None:
            logger.warning("No order book to update")
            return

        try:
            # Select appropriate side
            levels = self.current_orderbook.asks if is_buy else self.current_orderbook.bids

            remaining_quantity = fill_quantity

            # Remove filled quantity from order book levels
            for level in levels:
                if remaining_quantity <= Decimal("0"):
                    break

                if level.price == fill_price or (
                    is_buy and level.price <= fill_price
                ) or (
                    not is_buy and level.price >= fill_price
                ):
                    quantity_to_remove = min(remaining_quantity, level.quantity)
                    level.quantity -= quantity_to_remove
                    remaining_quantity -= quantity_to_remove

            logger.debug(
                "Liquidity updated after fill",
                fill_price=str(fill_price),
                fill_quantity=str(fill_quantity),
                is_buy=is_buy
            )

        except Exception as e:
            logger.error("Failed to update liquidity", error=str(e))
