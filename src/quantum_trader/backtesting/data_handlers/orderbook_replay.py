"""Order Book Replay Handler for Backtesting.

Replays historical order book data for high-fidelity backtesting
of market making and liquidity-sensitive strategies.
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
class OrderBookLevel:
    """Single order book price level.

    Attributes:
        price: Price level
        quantity: Total quantity at this level
        order_count: Number of orders at this level
    """
    price: Decimal
    quantity: Decimal
    order_count: int = 1


@dataclass
class OrderBookState:
    """Complete order book state at a timestamp.

    Attributes:
        timestamp: Snapshot timestamp
        symbol: Trading symbol
        bids: Bid levels (sorted descending by price)
        asks: Ask levels (sorted ascending by price)
        sequence: Sequence number for ordering
    """
    timestamp: datetime
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence: int

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return Decimal("0")

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / Decimal("2")
        return Decimal("0")


class OrderBookReplayer:
    """Replay historical order book data for backtesting.

    Provides time-series access to historical order book states
    for realistic simulation of order execution and market making.

    Attributes:
        config: Replayer configuration from config files
        symbol: Trading symbol
        orderbook_data: Cached order book snapshots
        current_index: Current replay position
    """

    def __init__(self, config: Dict[str, Any], symbol: str) -> None:
        """Initialize order book replayer.

        Args:
            config: Configuration dictionary
            symbol: Trading symbol

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = {"depth_levels": 10, "update_frequency_ms": 100}
            >>> replayer = OrderBookReplayer(config, "BTC/USDT")
        """
        self.config = config
        self.symbol = symbol
        self._validate_config()

        self.depth_levels = config.get("depth_levels", 10)
        self.update_frequency_ms = config.get("update_frequency_ms", 100)

        # Data storage
        self.orderbook_data: List[OrderBookState] = []
        self.current_index = 0

        # Caching
        self.current_state: Optional[OrderBookState] = None

        logger.info(
            "OrderBookReplayer initialized",
            symbol=symbol,
            depth_levels=self.depth_levels
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("config must be a dictionary")

    async def load_orderbook_data(self, data: pl.DataFrame) -> None:
        """Load historical order book data.

        Args:
            data: DataFrame with order book snapshots

        Raises:
            ValueError: If data is invalid

        Example:
            >>> replayer = OrderBookReplayer(config, "BTC/USDT")
            >>> await replayer.load_orderbook_data(orderbook_df)
        """
        try:
            required_cols = ["timestamp", "side", "price", "quantity"]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error("Invalid orderbook data", missing_cols=missing_cols)
                raise ValueError(error_msg)

            # Parse and structure order book data
            await self._parse_orderbook_data(data)

            logger.info(
                "Order book data loaded",
                symbol=self.symbol,
                snapshots=len(self.orderbook_data),
                start_time=self.orderbook_data[0].timestamp if self.orderbook_data else None
            )

        except Exception as e:
            logger.error("Failed to load orderbook data", error=str(e))
            raise

    async def _parse_orderbook_data(self, data: pl.DataFrame) -> None:
        """Parse raw orderbook data into structured snapshots.

        Args:
            data: Raw order book DataFrame
        """
        # Group by timestamp to create snapshots
        data_sorted = data.sort("timestamp")

        # Get unique timestamps
        timestamps = data_sorted["timestamp"].unique().sort()

        for idx, ts in enumerate(timestamps.to_list()):
            # Filter data for this timestamp
            snapshot_data = data_sorted.filter(pl.col("timestamp") == ts)

            # Separate bids and asks
            bids_data = snapshot_data.filter(pl.col("side") == "bid")
            asks_data = snapshot_data.filter(pl.col("side") == "ask")

            # Create bid levels
            bids = []
            for row in bids_data.sort("price", descending=True).iter_rows(named=True):
                bids.append(OrderBookLevel(
                    price=Decimal(str(row["price"])),
                    quantity=Decimal(str(row["quantity"])),
                    order_count=row.get("order_count", 1)
                ))

            # Create ask levels
            asks = []
            for row in asks_data.sort("price").iter_rows(named=True):
                asks.append(OrderBookLevel(
                    price=Decimal(str(row["price"])),
                    quantity=Decimal(str(row["quantity"])),
                    order_count=row.get("order_count", 1)
                ))

            # Limit to configured depth
            bids = bids[:self.depth_levels]
            asks = asks[:self.depth_levels]

            # Create order book state
            ob_state = OrderBookState(
                timestamp=ts,
                symbol=self.symbol,
                bids=bids,
                asks=asks,
                sequence=idx
            )

            self.orderbook_data.append(ob_state)

    async def get_orderbook_at_time(self, timestamp: datetime) -> Optional[OrderBookState]:
        """Get order book state at specific timestamp.

        Args:
            timestamp: Target timestamp

        Returns:
            Order book state or None if no data available

        Example:
            >>> replayer = OrderBookReplayer(config, "BTC/USDT")
            >>> await replayer.load_orderbook_data(data_df)
            >>> ob = await replayer.get_orderbook_at_time(datetime.utcnow())
            >>> if ob:
            ...     print(f"Spread: {ob.spread}")
        """
        if not self.orderbook_data:
            logger.warning("No orderbook data loaded")
            return None

        # Binary search for closest timestamp
        left, right = 0, len(self.orderbook_data) - 1
        result_idx = None

        while left <= right:
            mid = (left + right) // 2
            mid_time = self.orderbook_data[mid].timestamp

            if mid_time <= timestamp:
                result_idx = mid
                left = mid + 1
            else:
                right = mid - 1

        if result_idx is not None:
            self.current_state = self.orderbook_data[result_idx]
            self.current_index = result_idx
            return self.current_state

        return None

    async def get_next_orderbook(self) -> Optional[OrderBookState]:
        """Get next order book state in sequence.

        Returns:
            Next order book state or None if end reached

        Example:
            >>> replayer = OrderBookReplayer(config, "BTC/USDT")
            >>> await replayer.load_orderbook_data(data_df)
            >>> while ob := await replayer.get_next_orderbook():
            ...     process_orderbook(ob)
        """
        if self.current_index >= len(self.orderbook_data):
            return None

        state = self.orderbook_data[self.current_index]
        self.current_index += 1
        self.current_state = state

        return state

    def reset(self) -> None:
        """Reset replay to beginning.

        Example:
            >>> replayer = OrderBookReplayer(config, "BTC/USDT")
            >>> replayer.reset()
        """
        self.current_index = 0
        self.current_state = None
        logger.debug("OrderBook replayer reset", symbol=self.symbol)

    def get_current_state(self) -> Optional[OrderBookState]:
        """Get current order book state.

        Returns:
            Current order book state or None
        """
        return self.current_state

    async def calculate_liquidity_at_price(
        self,
        price: Decimal,
        is_buy: bool,
        max_levels: int = 5
    ) -> Decimal:
        """Calculate available liquidity at price level.

        Args:
            price: Target price
            is_buy: True for buy side, False for sell side
            max_levels: Maximum levels to check

        Returns:
            Total available liquidity
        """
        if self.current_state is None:
            return Decimal("0")

        total_liquidity = Decimal("0")
        levels = self.current_state.asks if is_buy else self.current_state.bids

        for level in levels[:max_levels]:
            if is_buy:
                if level.price <= price:
                    total_liquidity += level.quantity
            else:
                if level.price >= price:
                    total_liquidity += level.quantity

        return total_liquidity
