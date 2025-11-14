"""Order book snapshot collector.

This module collects order book snapshots from exchanges via WebSocket streams
or REST API polling. Order books are fundamental for market making, arbitrage,
and order flow analysis strategies.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from structlog import get_logger

from quantum_trader.models import OrderBook

logger = get_logger(__name__)


class OrderbookCollector:
    """Collects order book snapshots from exchanges.

    Maintains real-time order book data via WebSocket streams with automatic
    reconnection and snapshot validation. Supports both full snapshots and
    incremental updates.

    Attributes:
        config: Configuration dictionary
        symbols: List of symbols to collect
        exchanges: Dictionary of exchange connectors
        data_writer: Data writer instance
        collection_mode: Mode of collection (websocket or polling)
        depth: Order book depth to collect
        snapshot_interval: Interval for snapshot collection
        _running: Flag indicating if collector is active
        _websocket_tasks: WebSocket connection tasks
        _orderbooks: Current order book state cache

    Example:
        >>> config = {
        ...     "orderbook_collector": {
        ...         "symbols": ["BTC/USDT", "ETH/USDT"],
        ...         "depth": 20,
        ...         "collection_mode": "websocket"
        ...     }
        ... }
        >>> async with OrderbookCollector(config, exchanges, writer) as collector:
        ...     await collector.start()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize orderbook collector.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        collector_config = config.get("orderbook_collector", {})
        self.symbols = collector_config.get("symbols", [])
        self.enabled_exchanges = collector_config.get("enabled_exchanges", [])
        self.collection_mode = collector_config.get("collection_mode", "websocket")
        self.depth = collector_config.get("depth", 20)
        self.snapshot_interval = collector_config.get("snapshot_interval", 1.0)
        self.batch_size = collector_config.get("batch_size", 100)
        self.max_retries = collector_config.get("max_retries", 3)
        self.retry_delay = collector_config.get("retry_delay_ms", 1000) / 1000.0

        self.exchanges = exchanges
        self.data_writer = data_writer

        self._running = False
        self._websocket_tasks: Dict[str, asyncio.Task] = {}
        self._polling_task: Optional[asyncio.Task] = None

        # Cache current order books
        self._orderbooks: Dict[str, OrderBook] = {}
        self._orderbook_lock = asyncio.Lock()

        self._metrics: Dict[str, int] = {
            "total_collected": 0,
            "total_errors": 0,
            "snapshots": 0,
            "updates": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "orderbook_collector" not in self.config:
            raise ValueError("Missing orderbook_collector configuration")

        collector_config = self.config["orderbook_collector"]

        if not collector_config.get("symbols"):
            raise ValueError("No symbols configured for orderbook collection")

        if not collector_config.get("enabled_exchanges"):
            raise ValueError("No exchanges enabled for orderbook collection")

    async def start(self) -> None:
        """Start collecting order book data.

        Launches WebSocket streams or polling loop based on configuration.
        """
        if self._running:
            logger.warning("orderbook_collector_already_running")
            return

        self._running = True

        if self.collection_mode == "websocket":
            # Start WebSocket connections
            for exchange_name in self.enabled_exchanges:
                if exchange_name in self.exchanges:
                    task = asyncio.create_task(
                        self._websocket_collection_loop(
                            exchange_name,
                            self.exchanges[exchange_name]
                        )
                    )
                    self._websocket_tasks[exchange_name] = task
        else:
            # Start polling loop
            self._polling_task = asyncio.create_task(self._polling_collection_loop())

        logger.info(
            "orderbook_collector_started",
            symbols=len(self.symbols),
            exchanges=len(self.enabled_exchanges),
            mode=self.collection_mode,
            depth=self.depth
        )

    async def stop(self) -> None:
        """Stop collecting order book data.

        Gracefully stops all collection tasks.
        """
        if not self._running:
            return

        self._running = False

        # Cancel polling task
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        # Cancel WebSocket tasks
        for task in self._websocket_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(
            "orderbook_collector_stopped",
            metrics=self._metrics
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def _websocket_collection_loop(
        self,
        exchange_name: str,
        exchange: Any
    ) -> None:
        """WebSocket collection loop for a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance
        """
        while self._running:
            try:
                # Subscribe to order book streams
                async for orderbook_data in exchange.watch_order_book(
                    self.symbols,
                    limit=self.depth
                ):
                    if not self._running:
                        break

                    # Parse order book
                    orderbook = self._parse_orderbook(
                        orderbook_data,
                        exchange_name
                    )

                    if orderbook:
                        # Update cache
                        cache_key = f"{exchange_name}:{orderbook.symbol}"
                        async with self._orderbook_lock:
                            self._orderbooks[cache_key] = orderbook

                        # Write to storage
                        await self.data_writer.write_orderbook(orderbook)

                        self._metrics["total_collected"] += 1
                        self._metrics["snapshots"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "websocket_orderbook_error",
                    exchange=exchange_name,
                    error=str(e)
                )
                self._metrics["total_errors"] += 1
                # Retry connection after delay
                await asyncio.sleep(self.retry_delay * 2)

    async def _polling_collection_loop(self) -> None:
        """Polling collection loop for all exchanges."""
        while self._running:
            try:
                start_time = datetime.now(timezone.utc)

                # Collect from all exchanges concurrently
                await self._collect_all_exchanges()

                # Calculate sleep time
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, self.snapshot_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("orderbook_polling_error", error=str(e))
                await asyncio.sleep(self.snapshot_interval)

    async def _collect_all_exchanges(self) -> None:
        """Collect order books from all exchanges."""
        tasks = []
        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                task = self._collect_exchange(
                    exchange_name,
                    self.exchanges[exchange_name]
                )
                tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_exchange(
        self,
        exchange_name: str,
        exchange: Any
    ) -> None:
        """Collect order books from a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance
        """
        for symbol in self.symbols:
            try:
                # Fetch order book with retry
                orderbook_data = await self._fetch_with_retry(
                    exchange,
                    symbol,
                    exchange_name
                )

                if orderbook_data:
                    # Parse order book
                    orderbook = self._parse_orderbook(
                        orderbook_data,
                        exchange_name
                    )

                    if orderbook:
                        # Update cache
                        cache_key = f"{exchange_name}:{symbol}"
                        async with self._orderbook_lock:
                            self._orderbooks[cache_key] = orderbook

                        # Write to storage
                        await self.data_writer.write_orderbook(orderbook)

                        self._metrics["total_collected"] += 1
                        self._metrics["snapshots"] += 1

            except Exception as e:
                logger.error(
                    "orderbook_collection_failed",
                    exchange=exchange_name,
                    symbol=symbol,
                    error=str(e)
                )
                self._metrics["total_errors"] += 1

    async def _fetch_with_retry(
        self,
        exchange: Any,
        symbol: str,
        exchange_name: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch order book with exponential backoff retry.

        Args:
            exchange: Exchange connector
            symbol: Trading symbol
            exchange_name: Exchange name for logging

        Returns:
            Order book data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                orderbook = await exchange.fetch_order_book(
                    symbol,
                    limit=self.depth
                )
                return orderbook

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "orderbook_fetch_retry",
                        exchange=exchange_name,
                        symbol=symbol,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "orderbook_fetch_failed_all_retries",
                        exchange=exchange_name,
                        symbol=symbol,
                        error=str(e)
                    )
                    raise

        return None

    def _parse_orderbook(
        self,
        data: Dict[str, Any],
        exchange_name: str
    ) -> Optional[OrderBook]:
        """Parse order book data from exchange format.

        Args:
            data: Raw order book data from exchange
            exchange_name: Exchange name

        Returns:
            Normalized OrderBook instance or None if invalid
        """
        try:
            # Extract symbol
            symbol = data.get("symbol")
            if not symbol:
                raise ValueError("Missing symbol in order book data")

            # Extract bids and asks
            bids_raw = data.get("bids", [])
            asks_raw = data.get("asks", [])

            if not bids_raw or not asks_raw:
                logger.warning(
                    "empty_orderbook",
                    exchange=exchange_name,
                    symbol=symbol
                )
                return None

            # Convert to Decimal tuples
            bids: List[Tuple[Decimal, Decimal]] = [
                (Decimal(str(price)), Decimal(str(size)))
                for price, size in bids_raw[:self.depth]
            ]

            asks: List[Tuple[Decimal, Decimal]] = [
                (Decimal(str(price)), Decimal(str(size)))
                for price, size in asks_raw[:self.depth]
            ]

            # Extract timestamp
            timestamp = data.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Validate spread
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]

                if best_bid >= best_ask:
                    logger.warning(
                        "invalid_spread",
                        exchange=exchange_name,
                        symbol=symbol,
                        best_bid=str(best_bid),
                        best_ask=str(best_ask)
                    )
                    return None

            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp
            )

            return orderbook

        except Exception as e:
            logger.error(
                "orderbook_parse_failed",
                exchange=exchange_name,
                error=str(e)
            )
            return None

    def get_orderbook(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[OrderBook]:
        """Get cached order book.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Cached OrderBook or None if not available
        """
        cache_key = f"{exchange}:{symbol}"
        return self._orderbooks.get(cache_key)

    def get_all_orderbooks(self) -> Dict[str, OrderBook]:
        """Get all cached order books.

        Returns:
            Dictionary of cache keys to OrderBook instances
        """
        return self._orderbooks.copy()

    def get_metrics(self) -> Dict[str, int]:
        """Get collector metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = self._metrics.copy()
        metrics["cached_orderbooks"] = len(self._orderbooks)
        return metrics
