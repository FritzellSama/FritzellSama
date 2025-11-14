"""Order Fill Simulator for Backtesting.

Simulates realistic order fill behavior including partial fills,
fill probability, and time to fill based on market conditions.
"""

import asyncio
import random
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FillStrategy(Enum):
    """Order fill simulation strategies."""
    IMMEDIATE = "IMMEDIATE"  # Fill immediately at market price
    REALISTIC = "REALISTIC"  # Realistic fill based on market conditions
    CONSERVATIVE = "CONSERVATIVE"  # Conservative fill estimates
    AGGRESSIVE = "AGGRESSIVE"  # Aggressive fill assumptions


@dataclass
class FillSimulation:
    """Result of order fill simulation.

    Attributes:
        filled_quantity: Quantity that was filled
        fill_price: Average fill price
        fill_timestamp: When order was filled
        partial_fill: Whether order was partially filled
        fill_probability: Probability that fill occurred (0-1)
        time_to_fill_ms: Time from order to fill in milliseconds
        slippage_bps: Slippage in basis points
    """
    filled_quantity: Decimal
    fill_price: Decimal
    fill_timestamp: datetime
    partial_fill: bool
    fill_probability: Decimal
    time_to_fill_ms: int
    slippage_bps: Decimal


class OrderFillSimulator:
    """Simulate realistic order fill behavior for backtesting.

    Models order fill dynamics including:
    - Probability of fill based on market conditions
    - Time to fill estimation
    - Partial fill scenarios
    - Price slippage during fill

    Attributes:
        config: Simulator configuration from config files
        fill_strategy: Fill simulation strategy to use
        enable_partial_fills: Whether to allow partial fills
        seed: Random seed for reproducibility
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize order fill simulator.

        Args:
            config: Configuration dictionary with fill parameters

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "fill_strategy": "REALISTIC",
            ...     "enable_partial_fills": True,
            ...     "min_fill_probability": 0.95,
            ...     "max_time_to_fill_ms": 1000
            ... }
            >>> simulator = OrderFillSimulator(config)
        """
        self.config = config
        self._validate_config()

        # Load strategy
        strategy_str = config.get("fill_strategy", "REALISTIC").upper()
        self.fill_strategy = FillStrategy[strategy_str]

        self.enable_partial_fills = config.get("enable_partial_fills", True)
        self.min_fill_probability = Decimal(str(config.get("min_fill_probability", 0.95)))
        self.max_time_to_fill_ms = config.get("max_time_to_fill_ms", 1000)

        # Random seed for reproducibility
        self.seed = config.get("random_seed")
        if self.seed is not None:
            random.seed(self.seed)

        # Fill statistics
        self.fill_stats = {
            "total_orders": 0,
            "full_fills": 0,
            "partial_fills": 0,
            "no_fills": 0,
            "avg_fill_time_ms": Decimal("0"),
            "avg_fill_ratio": Decimal("0")
        }

        logger.info(
            "OrderFillSimulator initialized",
            strategy=self.fill_strategy.value,
            partial_fills_enabled=self.enable_partial_fills
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

        # Validate fill strategy if provided
        if "fill_strategy" in self.config:
            strategy_str = self.config["fill_strategy"].upper()
            if strategy_str not in [s.value for s in FillStrategy]:
                error_msg = f"Invalid fill_strategy: {strategy_str}"
                logger.error("Invalid fill strategy", strategy=strategy_str)
                raise ValueError(error_msg)

    async def simulate_market_order_fill(
        self,
        quantity: Decimal,
        order_price: Decimal,
        is_buy: bool,
        order_timestamp: datetime,
        market_liquidity: Optional[Decimal] = None,
        market_volatility: Optional[Decimal] = None
    ) -> FillSimulation:
        """Simulate fill for a market order.

        Args:
            quantity: Order quantity
            order_price: Price when order was placed
            is_buy: True for buy order, False for sell order
            order_timestamp: When order was placed
            market_liquidity: Available market liquidity
            market_volatility: Current market volatility

        Returns:
            Fill simulation result

        Raises:
            ValueError: If inputs are invalid

        Example:
            >>> simulator = OrderFillSimulator(config)
            >>> fill = await simulator.simulate_market_order_fill(
            ...     Decimal("10.0"),
            ...     Decimal("50000.0"),
            ...     is_buy=True,
            ...     order_timestamp=datetime.utcnow(),
            ...     market_liquidity=Decimal("1000")
            ... )
            >>> print(f"Filled {fill.filled_quantity} at ${fill.fill_price}")
        """
        if quantity <= Decimal("0"):
            error_msg = f"Invalid quantity: {quantity}"
            logger.error("Invalid order quantity", quantity=str(quantity))
            raise ValueError(error_msg)

        try:
            self.fill_stats["total_orders"] += 1

            # Market orders typically fill quickly
            base_fill_probability = Decimal("0.99")

            # Calculate fill probability based on strategy
            fill_probability = await self._calculate_fill_probability(
                quantity,
                market_liquidity,
                market_volatility,
                is_market_order=True
            )

            # Determine if order fills
            will_fill = random.random() < float(fill_probability)

            if not will_fill:
                self.fill_stats["no_fills"] += 1
                return FillSimulation(
                    filled_quantity=Decimal("0"),
                    fill_price=order_price,
                    fill_timestamp=order_timestamp,
                    partial_fill=False,
                    fill_probability=fill_probability,
                    time_to_fill_ms=0,
                    slippage_bps=Decimal("0")
                )

            # Calculate filled quantity
            filled_quantity = await self._calculate_filled_quantity(
                quantity,
                market_liquidity,
                is_market_order=True
            )

            # Calculate time to fill (market orders fill quickly)
            time_to_fill_ms = await self._calculate_time_to_fill(
                filled_quantity,
                market_liquidity,
                is_market_order=True
            )

            fill_timestamp = order_timestamp + timedelta(milliseconds=time_to_fill_ms)

            # Calculate fill price with slippage
            fill_price, slippage_bps = await self._calculate_fill_price(
                order_price,
                filled_quantity,
                is_buy,
                market_volatility,
                is_market_order=True
            )

            # Update statistics
            if filled_quantity == quantity:
                self.fill_stats["full_fills"] += 1
            else:
                self.fill_stats["partial_fills"] += 1

            result = FillSimulation(
                filled_quantity=filled_quantity,
                fill_price=fill_price,
                fill_timestamp=fill_timestamp,
                partial_fill=(filled_quantity < quantity),
                fill_probability=fill_probability,
                time_to_fill_ms=time_to_fill_ms,
                slippage_bps=slippage_bps
            )

            logger.debug(
                "Market order fill simulated",
                quantity=str(quantity),
                filled=str(filled_quantity),
                time_to_fill_ms=time_to_fill_ms
            )

            return result

        except Exception as e:
            logger.error("Failed to simulate market order fill", error=str(e))
            raise

    async def simulate_limit_order_fill(
        self,
        quantity: Decimal,
        limit_price: Decimal,
        is_buy: bool,
        order_timestamp: datetime,
        market_price_high: Decimal,
        market_price_low: Decimal,
        market_liquidity: Optional[Decimal] = None
    ) -> FillSimulation:
        """Simulate fill for a limit order.

        Args:
            quantity: Order quantity
            limit_price: Limit price
            is_buy: True for buy order, False for sell order
            order_timestamp: When order was placed
            market_price_high: Market high price during period
            market_price_low: Market low price during period
            market_liquidity: Available market liquidity

        Returns:
            Fill simulation result

        Example:
            >>> simulator = OrderFillSimulator(config)
            >>> fill = await simulator.simulate_limit_order_fill(
            ...     Decimal("10.0"),
            ...     Decimal("49900.0"),
            ...     is_buy=True,
            ...     order_timestamp=datetime.utcnow(),
            ...     market_price_high=Decimal("50100"),
            ...     market_price_low=Decimal("49800")
            ... )
        """
        if quantity <= Decimal("0"):
            error_msg = f"Invalid quantity: {quantity}"
            logger.error("Invalid order quantity", quantity=str(quantity))
            raise ValueError(error_msg)

        try:
            self.fill_stats["total_orders"] += 1

            # Check if limit price was reached
            price_reached = False

            if is_buy:
                # Buy limit: fills if market goes at or below limit price
                price_reached = market_price_low <= limit_price
            else:
                # Sell limit: fills if market goes at or above limit price
                price_reached = market_price_high >= limit_price

            if not price_reached:
                # Price never reached, order doesn't fill
                self.fill_stats["no_fills"] += 1
                return FillSimulation(
                    filled_quantity=Decimal("0"),
                    fill_price=limit_price,
                    fill_timestamp=order_timestamp,
                    partial_fill=False,
                    fill_probability=Decimal("0"),
                    time_to_fill_ms=0,
                    slippage_bps=Decimal("0")
                )

            # Calculate fill probability (even if price reached, might not fill)
            fill_probability = await self._calculate_fill_probability(
                quantity,
                market_liquidity,
                None,  # volatility
                is_market_order=False
            )

            will_fill = random.random() < float(fill_probability)

            if not will_fill:
                self.fill_stats["no_fills"] += 1
                return FillSimulation(
                    filled_quantity=Decimal("0"),
                    fill_price=limit_price,
                    fill_timestamp=order_timestamp,
                    partial_fill=False,
                    fill_probability=fill_probability,
                    time_to_fill_ms=0,
                    slippage_bps=Decimal("0")
                )

            # Calculate filled quantity
            filled_quantity = await self._calculate_filled_quantity(
                quantity,
                market_liquidity,
                is_market_order=False
            )

            # Calculate time to fill (limit orders take longer)
            time_to_fill_ms = await self._calculate_time_to_fill(
                filled_quantity,
                market_liquidity,
                is_market_order=False
            )

            fill_timestamp = order_timestamp + timedelta(milliseconds=time_to_fill_ms)

            # Limit orders fill at limit price or better
            fill_price = limit_price
            slippage_bps = Decimal("0")  # No slippage for limit orders (price guaranteed)

            # Update statistics
            if filled_quantity == quantity:
                self.fill_stats["full_fills"] += 1
            else:
                self.fill_stats["partial_fills"] += 1

            result = FillSimulation(
                filled_quantity=filled_quantity,
                fill_price=fill_price,
                fill_timestamp=fill_timestamp,
                partial_fill=(filled_quantity < quantity),
                fill_probability=fill_probability,
                time_to_fill_ms=time_to_fill_ms,
                slippage_bps=slippage_bps
            )

            logger.debug(
                "Limit order fill simulated",
                quantity=str(quantity),
                filled=str(filled_quantity),
                limit_price=str(limit_price),
                time_to_fill_ms=time_to_fill_ms
            )

            return result

        except Exception as e:
            logger.error("Failed to simulate limit order fill", error=str(e))
            raise

    async def _calculate_fill_probability(
        self,
        quantity: Decimal,
        market_liquidity: Optional[Decimal],
        market_volatility: Optional[Decimal],
        is_market_order: bool
    ) -> Decimal:
        """Calculate probability that order will fill.

        Args:
            quantity: Order quantity
            market_liquidity: Available liquidity
            market_volatility: Market volatility
            is_market_order: Whether this is a market order

        Returns:
            Fill probability (0-1)
        """
        # Base probability depends on strategy
        if self.fill_strategy == FillStrategy.IMMEDIATE:
            base_probability = Decimal("1.0")
        elif self.fill_strategy == FillStrategy.AGGRESSIVE:
            base_probability = Decimal("0.99")
        elif self.fill_strategy == FillStrategy.REALISTIC:
            base_probability = Decimal("0.97") if is_market_order else Decimal("0.90")
        elif self.fill_strategy == FillStrategy.CONSERVATIVE:
            base_probability = Decimal("0.95") if is_market_order else Decimal("0.80")
        else:
            base_probability = Decimal("0.95")

        # Adjust based on liquidity if provided
        if market_liquidity is not None and market_liquidity > Decimal("0"):
            liquidity_ratio = quantity / market_liquidity

            # Lower probability if order is large relative to liquidity
            if liquidity_ratio > Decimal("0.5"):
                base_probability *= Decimal("0.8")
            elif liquidity_ratio > Decimal("0.3"):
                base_probability *= Decimal("0.9")

        # Adjust based on volatility if provided
        if market_volatility is not None:
            if market_volatility > Decimal("0.05"):  # High volatility
                base_probability *= Decimal("0.95")

        # Ensure within bounds
        fill_probability = max(self.min_fill_probability, min(Decimal("1.0"), base_probability))

        return fill_probability

    async def _calculate_filled_quantity(
        self,
        quantity: Decimal,
        market_liquidity: Optional[Decimal],
        is_market_order: bool
    ) -> Decimal:
        """Calculate how much of the order gets filled.

        Args:
            quantity: Requested quantity
            market_liquidity: Available liquidity
            is_market_order: Whether this is a market order

        Returns:
            Filled quantity
        """
        if not self.enable_partial_fills or self.fill_strategy == FillStrategy.IMMEDIATE:
            return quantity

        # Check liquidity constraints
        if market_liquidity is not None:
            if quantity > market_liquidity:
                # Partial fill based on available liquidity
                fill_ratio = market_liquidity / quantity

                # Add some randomness
                if self.fill_strategy == FillStrategy.REALISTIC:
                    fill_ratio *= Decimal(str(random.uniform(0.9, 1.0)))

                filled_quantity = quantity * fill_ratio

                logger.debug(
                    "Partial fill due to liquidity",
                    requested=str(quantity),
                    filled=str(filled_quantity),
                    liquidity=str(market_liquidity)
                )

                return filled_quantity

        # Randomly determine partial fill for realism
        if self.fill_strategy == FillStrategy.CONSERVATIVE and not is_market_order:
            if random.random() < 0.2:  # 20% chance of partial fill
                fill_ratio = Decimal(str(random.uniform(0.7, 0.95)))
                return quantity * fill_ratio

        return quantity

    async def _calculate_time_to_fill(
        self,
        quantity: Decimal,
        market_liquidity: Optional[Decimal],
        is_market_order: bool
    ) -> int:
        """Calculate time required to fill order.

        Args:
            quantity: Order quantity
            market_liquidity: Available liquidity
            is_market_order: Whether this is a market order

        Returns:
            Time to fill in milliseconds
        """
        # Market orders fill quickly
        if is_market_order:
            if self.fill_strategy == FillStrategy.IMMEDIATE:
                return 0
            else:
                # Small random delay for market orders
                base_time_ms = self.config.get("market_order_base_time_ms", 50)
                return int(base_time_ms + random.randint(0, 50))

        # Limit orders take longer
        base_time_ms = self.config.get("limit_order_base_time_ms", 200)

        # Adjust based on quantity and liquidity
        if market_liquidity is not None and market_liquidity > Decimal("0"):
            liquidity_ratio = quantity / market_liquidity

            # Larger orders relative to liquidity take longer
            if liquidity_ratio > Decimal("0.3"):
                multiplier = float(Decimal("1") + liquidity_ratio)
                base_time_ms = int(base_time_ms * multiplier)

        # Add randomness
        time_variance = int(base_time_ms * 0.3)
        time_to_fill_ms = base_time_ms + random.randint(-time_variance, time_variance)

        # Clamp to maximum
        time_to_fill_ms = min(time_to_fill_ms, self.max_time_to_fill_ms)

        return max(0, time_to_fill_ms)

    async def _calculate_fill_price(
        self,
        order_price: Decimal,
        quantity: Decimal,
        is_buy: bool,
        market_volatility: Optional[Decimal],
        is_market_order: bool
    ) -> Tuple[Decimal, Decimal]:
        """Calculate fill price with slippage.

        Args:
            order_price: Original order price
            quantity: Order quantity
            is_buy: Order direction
            market_volatility: Market volatility
            is_market_order: Whether this is a market order

        Returns:
            Tuple of (fill_price, slippage_bps)
        """
        if not is_market_order:
            # Limit orders don't have slippage
            return order_price, Decimal("0")

        # Calculate slippage based on strategy
        if self.fill_strategy == FillStrategy.IMMEDIATE:
            base_slippage_bps = Decimal("1")
        elif self.fill_strategy == FillStrategy.AGGRESSIVE:
            base_slippage_bps = Decimal("2")
        elif self.fill_strategy == FillStrategy.REALISTIC:
            base_slippage_bps = Decimal("5")
        elif self.fill_strategy == FillStrategy.CONSERVATIVE:
            base_slippage_bps = Decimal("10")
        else:
            base_slippage_bps = Decimal("5")

        # Adjust for volatility
        if market_volatility is not None:
            volatility_multiplier = Decimal("1") + (market_volatility * Decimal("2"))
            base_slippage_bps *= volatility_multiplier

        # Add random component
        slippage_variance = float(base_slippage_bps) * 0.3
        slippage_bps = Decimal(str(float(base_slippage_bps) + random.uniform(-slippage_variance, slippage_variance)))

        # Ensure non-negative
        slippage_bps = max(Decimal("0"), slippage_bps)

        # Calculate fill price
        slippage_amount = order_price * slippage_bps / Decimal("10000")

        if is_buy:
            fill_price = order_price + slippage_amount
        else:
            fill_price = order_price - slippage_amount

        return fill_price, slippage_bps

    def get_fill_statistics(self) -> Dict[str, Any]:
        """Get order fill statistics.

        Returns:
            Dictionary with fill statistics

        Example:
            >>> simulator = OrderFillSimulator(config)
            >>> # ... simulate fills ...
            >>> stats = simulator.get_fill_statistics()
            >>> print(f"Fill rate: {stats['fill_rate']:.2%}")
        """
        total = self.fill_stats["total_orders"]

        if total > 0:
            fill_rate = (self.fill_stats["full_fills"] + self.fill_stats["partial_fills"]) / total
            full_fill_rate = self.fill_stats["full_fills"] / total
        else:
            fill_rate = Decimal("0")
            full_fill_rate = Decimal("0")

        return {
            "total_orders": total,
            "full_fills": self.fill_stats["full_fills"],
            "partial_fills": self.fill_stats["partial_fills"],
            "no_fills": self.fill_stats["no_fills"],
            "fill_rate": float(fill_rate),
            "full_fill_rate": float(full_fill_rate),
            "strategy": self.fill_strategy.value
        }

    def reset_statistics(self) -> None:
        """Reset fill statistics.

        Example:
            >>> simulator = OrderFillSimulator(config)
            >>> simulator.reset_statistics()
        """
        self.fill_stats = {
            "total_orders": 0,
            "full_fills": 0,
            "partial_fills": 0,
            "no_fills": 0,
            "avg_fill_time_ms": Decimal("0"),
            "avg_fill_ratio": Decimal("0")
        }

        logger.info("Fill statistics reset")
