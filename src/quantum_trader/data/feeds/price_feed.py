"""Real-time price feed for strategy consumption.

This module provides a high-performance event-driven feed of price ticker updates
for trading strategies. Supports multiple subscribers with filtering and
rate limiting capabilities.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from structlog import get_logger

from quantum_trader.models import Ticker

logger = get_logger(__name__)


class PriceFeed:
    """Real-time price ticker feed with pub/sub pattern.

    Distributes ticker updates to multiple subscribers with filtering
    by symbol and exchange. Provides buffering and backpressure handling.

    Attributes:
        config: Configuration dictionary
        exchanges: Dictionary of exchange connectors
        data_writer: Data writer instance
        symbols: Set of symbols to subscribe to
        _running: Flag indicating if feed is active
        _subscribers: Dictionary of subscriber callbacks
        _feed_task: Background feed task
        _buffer: Event buffer for backpressure handling

    Example:
        >>> async def on_ticker(ticker: Ticker):
        ...     print(f"{ticker.symbol}: ${ticker.last}")
        ...
        >>> feed = PriceFeed(config, exchanges, writer)
        >>> await feed.start()
        >>> sub_id = feed.subscribe(on_ticker, symbols=["BTC/USDT"])
        >>> # ... later ...
        >>> feed.unsubscribe(sub_id)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize price feed.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        feed_config = config.get("price_feed", {})
        self.symbols = set(feed_config.get("symbols", []))
        self.enabled_exchanges = feed_config.get("enabled_exchanges", [])
        self.buffer_size = feed_config.get("buffer_size", 1000)
        self.max_retries = feed_config.get("max_retries", 3)
        self.retry_delay = feed_config.get("retry_delay_ms", 1000) / 1000.0

        self.exchanges = exchanges
        self.data_writer = data_writer

        self._running = False
        self._feed_task: Optional[asyncio.Task] = None
        self._exchange_tasks: Dict[str, asyncio.Task] = {}

        # Subscriber management
        self._subscribers: Dict[str, Dict[str, Any]] = {}
        self._subscriber_lock = asyncio.Lock()
        self._next_subscriber_id = 0

        # Event buffer
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=self.buffer_size)

        self._metrics: Dict[str, int] = {
            "total_updates": 0,
            "total_errors": 0,
            "total_delivered": 0,
            "delivery_errors": 0,
            "buffer_overflows": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "price_feed" not in self.config:
            raise ValueError("Missing price_feed configuration")

        feed_config = self.config["price_feed"]

        if not feed_config.get("symbols"):
            raise ValueError("No symbols configured for price feed")

        if not feed_config.get("enabled_exchanges"):
            raise ValueError("No exchanges enabled for price feed")

    async def start(self) -> None:
        """Start the price feed.

        Launches WebSocket connections and subscriber dispatch task.
        """
        if self._running:
            logger.warning("price_feed_already_running")
            return

        self._running = True

        # Start exchange WebSocket feeds
        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                task = asyncio.create_task(
                    self._exchange_feed_loop(
                        exchange_name,
                        self.exchanges[exchange_name]
                    )
                )
                self._exchange_tasks[exchange_name] = task

        # Start subscriber dispatch task
        self._feed_task = asyncio.create_task(self._dispatch_loop())

        logger.info(
            "price_feed_started",
            symbols=len(self.symbols),
            exchanges=len(self.enabled_exchanges)
        )

    async def stop(self) -> None:
        """Stop the price feed.

        Gracefully stops all tasks and notifies subscribers.
        """
        if not self._running:
            return

        self._running = False

        # Cancel exchange tasks
        for task in self._exchange_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel dispatch task
        if self._feed_task and not self._feed_task.done():
            self._feed_task.cancel()
            try:
                await self._feed_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "price_feed_stopped",
            metrics=self._metrics
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def subscribe(
        self,
        callback: Callable[[Ticker], None],
        symbols: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None
    ) -> str:
        """Subscribe to ticker updates.

        Args:
            callback: Async callback function to receive updates
            symbols: Optional list of symbols to filter (None = all)
            exchanges: Optional list of exchanges to filter (None = all)

        Returns:
            Subscription ID for later unsubscribe

        Example:
            >>> async def handler(ticker: Ticker):
            ...     print(f"{ticker.symbol}: ${ticker.last}")
            >>> sub_id = feed.subscribe(handler, symbols=["BTC/USDT"])
        """
        subscriber_id = f"sub_{self._next_subscriber_id}"
        self._next_subscriber_id += 1

        subscriber = {
            "callback": callback,
            "symbols": set(symbols) if symbols else None,
            "exchanges": set(exchanges) if exchanges else None,
            "created_at": datetime.now(timezone.utc),
            "delivered": 0,
            "errors": 0
        }

        self._subscribers[subscriber_id] = subscriber

        logger.info(
            "subscriber_added",
            subscriber_id=subscriber_id,
            symbols=len(symbols) if symbols else "all",
            exchanges=len(exchanges) if exchanges else "all"
        )

        return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from ticker updates.

        Args:
            subscriber_id: Subscription ID returned from subscribe()

        Returns:
            True if unsubscribed, False if ID not found
        """
        if subscriber_id in self._subscribers:
            subscriber = self._subscribers.pop(subscriber_id)
            logger.info(
                "subscriber_removed",
                subscriber_id=subscriber_id,
                delivered=subscriber["delivered"],
                errors=subscriber["errors"]
            )
            return True

        logger.warning("subscriber_not_found", subscriber_id=subscriber_id)
        return False

    async def _exchange_feed_loop(
        self,
        exchange_name: str,
        exchange: Any
    ) -> None:
        """WebSocket feed loop for a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance
        """
        while self._running:
            try:
                # Subscribe to ticker stream
                async for ticker_data in exchange.watch_tickers(list(self.symbols)):
                    if not self._running:
                        break

                    # Parse ticker
                    ticker = self._parse_ticker(
                        ticker_data,
                        exchange_name
                    )

                    if ticker:
                        # Add to buffer
                        try:
                            self._buffer.put_nowait({
                                "ticker": ticker,
                                "exchange": exchange_name
                            })
                            self._metrics["total_updates"] += 1

                        except asyncio.QueueFull:
                            self._metrics["buffer_overflows"] += 1
                            logger.warning(
                                "ticker_buffer_full",
                                exchange=exchange_name,
                                symbol=ticker.symbol
                            )
                            # Drop oldest item and retry
                            try:
                                self._buffer.get_nowait()
                                self._buffer.put_nowait({
                                    "ticker": ticker,
                                    "exchange": exchange_name
                                })
                            except Exception as e:
                                logger.error("buffer_recovery_failed", error=str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "exchange_feed_error",
                    exchange=exchange_name,
                    error=str(e)
                )
                self._metrics["total_errors"] += 1
                await asyncio.sleep(self.retry_delay * 2)

    async def _dispatch_loop(self) -> None:
        """Dispatch ticker updates to subscribers."""
        while self._running:
            try:
                # Get next update from buffer
                update = await asyncio.wait_for(
                    self._buffer.get(),
                    timeout=1.0
                )

                ticker = update["ticker"]
                exchange = update["exchange"]

                # Dispatch to subscribers
                await self._dispatch_to_subscribers(ticker, exchange)

            except asyncio.TimeoutError:
                # No updates available, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("dispatch_loop_error", error=str(e))

    async def _dispatch_to_subscribers(
        self,
        ticker: Ticker,
        exchange: str
    ) -> None:
        """Dispatch ticker to matching subscribers.

        Args:
            ticker: Ticker instance
            exchange: Exchange name
        """
        async with self._subscriber_lock:
            for sub_id, subscriber in self._subscribers.items():
                # Check filters
                if subscriber["symbols"] and ticker.symbol not in subscriber["symbols"]:
                    continue

                if subscriber["exchanges"] and exchange not in subscriber["exchanges"]:
                    continue

                # Call subscriber callback
                try:
                    callback = subscriber["callback"]

                    # Handle both sync and async callbacks
                    if asyncio.iscoroutinefunction(callback):
                        await callback(ticker)
                    else:
                        callback(ticker)

                    subscriber["delivered"] += 1
                    self._metrics["total_delivered"] += 1

                except Exception as e:
                    logger.error(
                        "subscriber_callback_error",
                        subscriber_id=sub_id,
                        symbol=ticker.symbol,
                        error=str(e)
                    )
                    subscriber["errors"] += 1
                    self._metrics["delivery_errors"] += 1

    def _parse_ticker(
        self,
        data: Dict[str, Any],
        exchange_name: str
    ) -> Optional[Ticker]:
        """Parse ticker from exchange format.

        Args:
            data: Raw ticker data
            exchange_name: Exchange name

        Returns:
            Ticker instance or None if invalid
        """
        try:
            from decimal import Decimal

            symbol = data.get("symbol")
            if not symbol:
                return None

            bid = data.get("bid")
            ask = data.get("ask")
            last = data.get("last")

            if bid is None or ask is None or last is None:
                return None

            volume = data.get("baseVolume") or data.get("volume") or 0

            timestamp = data.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

            return Ticker(
                symbol=symbol,
                bid=Decimal(str(bid)),
                ask=Decimal(str(ask)),
                last=Decimal(str(last)),
                volume=Decimal(str(volume)),
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(
                "ticker_parse_failed",
                exchange=exchange_name,
                error=str(e)
            )
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get feed metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        metrics["active_subscribers"] = len(self._subscribers)
        metrics["buffer_size"] = self._buffer.qsize()
        return metrics

    def get_subscriber_count(self) -> int:
        """Get number of active subscribers.

        Returns:
            Count of subscribers
        """
        return len(self._subscribers)
