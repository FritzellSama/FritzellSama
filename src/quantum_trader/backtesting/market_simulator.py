"""Market Simulator for Backtesting.

Simulates realistic market conditions including order book dynamics,
liquidity, latency, and market impact for comprehensive backtesting.
"""

import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class MarketState:
    """Current market state snapshot.

    Attributes:
        timestamp: Current market time
        symbol: Trading symbol
        bid_price: Best bid price
        ask_price: Best ask price
        last_price: Last traded price
        volume_24h: 24-hour volume
        volatility: Current volatility
        liquidity_score: Market liquidity score (0-1)
    """
    timestamp: datetime
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    last_price: Decimal
    volume_24h: Decimal
    volatility: Decimal
    liquidity_score: Decimal

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / Decimal("2")


@dataclass
class SimulationConfig:
    """Simulation configuration parameters.

    Attributes:
        enable_latency: Enable latency simulation
        enable_slippage: Enable slippage simulation
        enable_market_impact: Enable market impact simulation
        enable_liquidity_constraints: Enable liquidity constraints
        realistic_fills: Use realistic fill simulation
    """
    enable_latency: bool = True
    enable_slippage: bool = True
    enable_market_impact: bool = True
    enable_liquidity_constraints: bool = True
    realistic_fills: bool = True


class MarketSimulator:
    """Comprehensive market simulator for backtesting.

    Simulates realistic market conditions by combining:
    - Latency simulation
    - Liquidity modeling
    - Market impact
    - Order fill simulation
    - Slippage calculation

    Attributes:
        config: Simulator configuration from config files
        symbol: Trading symbol
        simulation_config: Simulation feature flags
        current_state: Current market state
    """

    def __init__(
        self,
        config: Dict[str, Any],
        symbol: str,
        simulation_config: Optional[SimulationConfig] = None
    ) -> None:
        """Initialize market simulator.

        Args:
            config: Configuration dictionary with all simulation parameters
            symbol: Trading symbol to simulate
            simulation_config: Optional simulation feature configuration

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {
            ...     "latency": {...},
            ...     "liquidity": {...},
            ...     "market_impact": {...},
            ...     "slippage": {...}
            ... }
            >>> simulator = MarketSimulator(config, "BTC/USDT")
        """
        self.config = config
        self.symbol = symbol
        self.simulation_config = simulation_config or SimulationConfig()

        self._validate_config()

        # Current market state
        self.current_state: Optional[MarketState] = None

        # Historical data cache
        self.market_data_cache: Optional[pl.DataFrame] = None

        # Component simulators (would be initialized from respective modules)
        self.latency_config = config.get("latency", {})
        self.liquidity_config = config.get("liquidity", {})
        self.impact_config = config.get("market_impact", {})
        self.slippage_config = config.get("slippage", {})

        # Statistics tracking
        self.simulation_stats = {
            "total_simulated_orders": 0,
            "total_fills": 0,
            "total_rejects": 0,
            "avg_latency_ms": Decimal("0"),
            "avg_slippage_bps": Decimal("0")
        }

        logger.info(
            "MarketSimulator initialized",
            symbol=symbol,
            latency_enabled=self.simulation_config.enable_latency,
            slippage_enabled=self.simulation_config.enable_slippage,
            impact_enabled=self.simulation_config.enable_market_impact
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

        # Check for required sub-configurations
        required_sections = ["latency", "liquidity", "market_impact", "slippage"]
        missing_sections = [s for s in required_sections if s not in self.config]

        if missing_sections:
            logger.warning(
                "Missing optional config sections",
                missing=missing_sections,
                symbol=self.symbol
            )

    async def initialize(self, market_data: pl.DataFrame) -> None:
        """Initialize simulator with historical market data.

        Args:
            market_data: DataFrame with historical OHLCV data

        Raises:
            ValueError: If market data is invalid

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> await simulator.initialize(historical_data_df)
        """
        try:
            # Validate market data
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in market_data.columns]

            if missing_cols:
                error_msg = f"Missing required columns in market_data: {missing_cols}"
                logger.error("Invalid market data", missing_cols=missing_cols)
                raise ValueError(error_msg)

            if market_data.height == 0:
                error_msg = "market_data cannot be empty"
                logger.error("Empty market data")
                raise ValueError(error_msg)

            # Cache market data
            self.market_data_cache = market_data.sort("timestamp")

            # Initialize market state from first data point
            first_row = market_data.row(0, named=True)
            await self._update_market_state(first_row)

            logger.info(
                "MarketSimulator initialized with data",
                symbol=self.symbol,
                data_points=market_data.height,
                start_time=first_row["timestamp"],
                end_time=market_data.row(-1, named=True)["timestamp"]
            )

        except Exception as e:
            logger.error("Failed to initialize simulator", error=str(e))
            raise

    async def _update_market_state(self, market_row: Dict[str, Any]) -> None:
        """Update current market state from market data.

        Args:
            market_row: Dictionary with OHLCV data
        """
        close_price = Decimal(str(market_row["close"]))
        volume = Decimal(str(market_row["volume"]))

        # Estimate bid-ask spread (simplified)
        spread_bps = Decimal(str(self.config.get("default_spread_bps", 5)))
        spread = close_price * spread_bps / Decimal("10000")

        # Calculate volatility from price range
        high = Decimal(str(market_row["high"]))
        low = Decimal(str(market_row["low"]))

        if close_price > Decimal("0"):
            volatility = (high - low) / close_price
        else:
            volatility = Decimal("0")

        # Estimate liquidity score based on volume
        avg_volume = Decimal(str(self.config.get("average_volume", {}).get(self.symbol, 1000000)))
        if avg_volume > Decimal("0"):
            liquidity_score = min(Decimal("1"), volume / avg_volume)
        else:
            liquidity_score = Decimal("0.5")

        self.current_state = MarketState(
            timestamp=market_row["timestamp"],
            symbol=self.symbol,
            bid_price=close_price - (spread / Decimal("2")),
            ask_price=close_price + (spread / Decimal("2")),
            last_price=close_price,
            volume_24h=volume,
            volatility=volatility,
            liquidity_score=liquidity_score
        )

    async def simulate_market_order(
        self,
        quantity: Decimal,
        is_buy: bool,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Simulate execution of a market order.

        Args:
            quantity: Order quantity
            is_buy: True for buy order, False for sell order
            timestamp: Order placement timestamp

        Returns:
            Dictionary with execution details

        Raises:
            ValueError: If inputs are invalid or simulator not initialized

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> await simulator.initialize(market_data_df)
            >>> result = await simulator.simulate_market_order(
            ...     Decimal("1.5"),
            ...     is_buy=True,
            ...     timestamp=datetime.utcnow()
            ... )
            >>> print(f"Filled at: ${result['fill_price']}")
        """
        if self.current_state is None:
            error_msg = "Simulator not initialized. Call initialize() first."
            logger.error("Simulator not initialized")
            raise ValueError(error_msg)

        if quantity <= Decimal("0"):
            error_msg = f"Invalid quantity: {quantity}"
            logger.error("Invalid order quantity", quantity=str(quantity))
            raise ValueError(error_msg)

        try:
            self.simulation_stats["total_simulated_orders"] += 1

            # Update market state to current timestamp
            await self._advance_to_timestamp(timestamp)

            # Simulate latency
            execution_timestamp = timestamp
            if self.simulation_config.enable_latency:
                latency_ms = await self._simulate_latency("order_placement")
                execution_timestamp = timestamp + timedelta(milliseconds=float(latency_ms))

            # Check liquidity constraints
            if self.simulation_config.enable_liquidity_constraints:
                available_liquidity = await self._calculate_available_liquidity(is_buy)
                if quantity > available_liquidity:
                    logger.warning(
                        "Insufficient liquidity",
                        requested=str(quantity),
                        available=str(available_liquidity)
                    )
                    # Partial fill or reject based on config
                    reject_on_insufficient_liquidity = self.config.get("reject_on_insufficient_liquidity", False)
                    if reject_on_insufficient_liquidity:
                        self.simulation_stats["total_rejects"] += 1
                        return {
                            "status": "rejected",
                            "reason": "insufficient_liquidity",
                            "timestamp": execution_timestamp
                        }
                    else:
                        quantity = available_liquidity  # Partial fill

            # Calculate base fill price
            if is_buy:
                base_price = self.current_state.ask_price
            else:
                base_price = self.current_state.bid_price

            # Apply market impact
            fill_price = base_price
            if self.simulation_config.enable_market_impact:
                impact = await self._calculate_market_impact(quantity, base_price, is_buy)
                if is_buy:
                    fill_price = base_price + impact
                else:
                    fill_price = base_price - impact

            # Apply slippage
            if self.simulation_config.enable_slippage:
                slippage = await self._calculate_slippage(quantity, fill_price, is_buy)
                if is_buy:
                    fill_price = fill_price + slippage
                else:
                    fill_price = fill_price - slippage

            # Calculate fees
            fees = await self._calculate_fees(quantity, fill_price)

            self.simulation_stats["total_fills"] += 1

            result = {
                "status": "filled",
                "fill_price": fill_price,
                "filled_quantity": quantity,
                "fees": fees,
                "execution_timestamp": execution_timestamp,
                "latency_ms": float((execution_timestamp - timestamp).total_seconds() * 1000),
                "slippage_bps": float(abs((fill_price - base_price) / base_price) * Decimal("10000"))
            }

            logger.debug(
                "Market order simulated",
                symbol=self.symbol,
                quantity=str(quantity),
                is_buy=is_buy,
                fill_price=str(fill_price)
            )

            return result

        except Exception as e:
            logger.error("Failed to simulate market order", error=str(e))
            raise

    async def simulate_limit_order(
        self,
        quantity: Decimal,
        limit_price: Decimal,
        is_buy: bool,
        timestamp: datetime,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """Simulate execution of a limit order.

        Args:
            quantity: Order quantity
            limit_price: Limit price
            is_buy: True for buy order, False for sell order
            timestamp: Order placement timestamp
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Dictionary with execution details

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> result = await simulator.simulate_limit_order(
            ...     Decimal("1.0"),
            ...     Decimal("50000.00"),
            ...     is_buy=True,
            ...     timestamp=datetime.utcnow()
            ... )
        """
        if self.current_state is None:
            error_msg = "Simulator not initialized. Call initialize() first."
            logger.error("Simulator not initialized")
            raise ValueError(error_msg)

        try:
            self.simulation_stats["total_simulated_orders"] += 1

            # Update market state
            await self._advance_to_timestamp(timestamp)

            # Simulate latency
            execution_timestamp = timestamp
            if self.simulation_config.enable_latency:
                latency_ms = await self._simulate_latency("order_placement")
                execution_timestamp = timestamp + timedelta(milliseconds=float(latency_ms))

            # Check if limit order would fill immediately
            if is_buy:
                can_fill = limit_price >= self.current_state.ask_price
                fill_price = min(limit_price, self.current_state.ask_price)
            else:
                can_fill = limit_price <= self.current_state.bid_price
                fill_price = max(limit_price, self.current_state.bid_price)

            if can_fill:
                # Check liquidity
                if self.simulation_config.enable_liquidity_constraints:
                    available_liquidity = await self._calculate_available_liquidity(is_buy)
                    filled_quantity = min(quantity, available_liquidity)
                else:
                    filled_quantity = quantity

                # Calculate fees
                fees = await self._calculate_fees(filled_quantity, fill_price)

                self.simulation_stats["total_fills"] += 1

                status = "filled" if filled_quantity == quantity else "partially_filled"

                return {
                    "status": status,
                    "fill_price": fill_price,
                    "filled_quantity": filled_quantity,
                    "fees": fees,
                    "execution_timestamp": execution_timestamp,
                    "latency_ms": float((execution_timestamp - timestamp).total_seconds() * 1000)
                }
            else:
                # Order not filled (would sit on book in real trading)
                return {
                    "status": "open",
                    "fill_price": Decimal("0"),
                    "filled_quantity": Decimal("0"),
                    "fees": Decimal("0"),
                    "execution_timestamp": execution_timestamp
                }

        except Exception as e:
            logger.error("Failed to simulate limit order", error=str(e))
            raise

    async def _advance_to_timestamp(self, timestamp: datetime) -> None:
        """Advance market state to specific timestamp.

        Args:
            timestamp: Target timestamp
        """
        if self.market_data_cache is None:
            return

        # Find market data at or before target timestamp
        past_data = self.market_data_cache.filter(
            pl.col("timestamp") <= timestamp
        )

        if past_data.height > 0:
            latest_row = past_data.row(-1, named=True)
            await self._update_market_state(latest_row)

    async def _simulate_latency(self, operation_type: str) -> Decimal:
        """Simulate latency for operation.

        Args:
            operation_type: Type of operation

        Returns:
            Latency in milliseconds
        """
        if not self.latency_config:
            return Decimal("0")

        # Get latency parameters
        op_config = self.latency_config.get(operation_type, {})
        avg_latency = Decimal(str(op_config.get("avg_ms", 10)))

        # For simplicity, return average latency
        # Real implementation would use LatencySimulator
        return avg_latency

    async def _calculate_available_liquidity(self, is_buy: bool) -> Decimal:
        """Calculate available liquidity.

        Args:
            is_buy: True for buy side, False for sell side

        Returns:
            Available liquidity quantity
        """
        if self.current_state is None:
            return Decimal("0")

        # Estimate liquidity based on market state
        base_liquidity = Decimal(str(self.liquidity_config.get("base_liquidity", {}).get(self.symbol, 100)))

        # Adjust by liquidity score
        available = base_liquidity * self.current_state.liquidity_score

        return available

    async def _calculate_market_impact(
        self,
        quantity: Decimal,
        price: Decimal,
        is_buy: bool
    ) -> Decimal:
        """Calculate market impact.

        Args:
            quantity: Order quantity
            price: Current price
            is_buy: Order direction

        Returns:
            Price impact
        """
        if not self.impact_config:
            return Decimal("0")

        # Simplified impact calculation
        impact_factor = Decimal(str(self.impact_config.get("temporary_impact_factor", 0.05)))
        liquidity = await self._calculate_available_liquidity(is_buy)

        if liquidity > Decimal("0"):
            participation = quantity / liquidity
            impact = price * impact_factor * participation.sqrt()
        else:
            impact = Decimal("0")

        return impact

    async def _calculate_slippage(
        self,
        quantity: Decimal,
        price: Decimal,
        is_buy: bool
    ) -> Decimal:
        """Calculate slippage.

        Args:
            quantity: Order quantity
            price: Fill price
            is_buy: Order direction

        Returns:
            Slippage amount
        """
        if not self.slippage_config or self.current_state is None:
            return Decimal("0")

        # Slippage increases with volatility and quantity
        base_slippage_bps = Decimal(str(self.slippage_config.get("base_slippage_bps", 2)))
        volatility_multiplier = Decimal(str(self.slippage_config.get("volatility_multiplier", 1.5)))

        slippage_bps = base_slippage_bps * (Decimal("1") + self.current_state.volatility * volatility_multiplier)

        slippage = price * slippage_bps / Decimal("10000")

        return slippage

    async def _calculate_fees(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate trading fees.

        Args:
            quantity: Order quantity
            price: Fill price

        Returns:
            Fee amount
        """
        fee_rate = Decimal(str(self.config.get("fee_rate_bps", 10))) / Decimal("10000")
        total_value = quantity * price
        fees = total_value * fee_rate

        return fees

    def get_current_state(self) -> Optional[MarketState]:
        """Get current market state.

        Returns:
            Current market state or None if not initialized

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> state = simulator.get_current_state()
            >>> if state:
            ...     print(f"Mid price: ${state.mid_price}")
        """
        return self.current_state

    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics.

        Returns:
            Dictionary with simulation metrics

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> # ... simulate orders ...
            >>> stats = simulator.get_simulation_statistics()
            >>> print(f"Total orders: {stats['total_simulated_orders']}")
        """
        return self.simulation_stats.copy()

    def reset_statistics(self) -> None:
        """Reset simulation statistics.

        Example:
            >>> simulator = MarketSimulator(config, "BTC/USDT")
            >>> simulator.reset_statistics()
        """
        self.simulation_stats = {
            "total_simulated_orders": 0,
            "total_fills": 0,
            "total_rejects": 0,
            "avg_latency_ms": Decimal("0"),
            "avg_slippage_bps": Decimal("0")
        }

        logger.debug("Simulation statistics reset", symbol=self.symbol)
