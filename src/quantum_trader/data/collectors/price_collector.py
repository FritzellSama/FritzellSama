"""Price ticker collector for real-time market prices.

This module collects ticker data (bid, ask, last, volume) from exchanges
via WebSocket streams or REST API. Provides foundation for pricing,
spread analysis, and arbitrage detection.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import polars as pl
from structlog import get_logger

from quantum_trader.models import Ticker

logger = get_logger(__name__)


class PriceCollector:
    """Collects real-time price ticker data from exchanges.

    Monitors bid/ask/last prices and volumes across multiple symbols and
    exchanges. Supports both WebSocket streaming and REST polling modes.

    Attributes:
        config: Configuration dictionary
        symbols: List of symbols to collect
        exchanges: Dictionary of exchange connectors
        data_writer: Data writer instance
        collection_mode: Mode of collection (websocket or polling)
        collection_interval: Interval for polling mode
        _running: Flag indicating if collector is active
        _websocket_tasks: WebSocket connection tasks
        _tickers: Current ticker cache

    Example:
        >>> config = {
        ...     "price_collector": {
        ...         "symbols": ["BTC/USDT", "ETH/USDT"],
        ...         "collection_mode": "websocket",
        ...         "enabled_exchanges": ["binance", "bybit"]
        ...     }
        ... }
        >>> async with PriceCollector(config, exchanges, writer) as collector:
        ...     await collector.start()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize price collector.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        collector_config = config.get("price_collector", {})
        self.symbols = collector_config.get("symbols", [])
        self.enabled_exchanges = collector_config.get("enabled_exchanges", [])
        self.collection_mode = collector_config.get("collection_mode", "websocket")
        self.collection_interval = collector_config.get("collection_interval", 1.0)
        self.batch_size = collector_config.get("batch_size", 100)
        self.max_retries = collector_config.get("max_retries", 3)
        self.retry_delay = collector_config.get("retry_delay_ms", 1000) / 1000.0

        self.exchanges = exchanges
        self.data_writer = data_writer

        self._running = False
        self._websocket_tasks: Dict[str, asyncio.Task] = {}
        self._polling_task: Optional[asyncio.Task] = None

        # Cache current tickers
        self._tickers: Dict[str, Ticker] = {}
        self._ticker_lock = asyncio.Lock()

        self._metrics: Dict[str, int] = {
            "total_collected": 0,
            "total_errors": 0,
            "updates": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "price_collector" not in self.config:
            raise ValueError("Missing price_collector configuration")

        collector_config = self.config["price_collector"]

        if not collector_config.get("symbols"):
            raise ValueError("No symbols configured for price collection")

        if not collector_config.get("enabled_exchanges"):
            raise ValueError("No exchanges enabled for price collection")

    async def start(self) -> None:
        """Start collecting price data.

        Launches WebSocket streams or polling loop based on configuration.
        """
        if self._running:
            logger.warning("price_collector_already_running")
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
            "price_collector_started",
            symbols=len(self.symbols),
            exchanges=len(self.enabled_exchanges),
            mode=self.collection_mode
        )

    async def stop(self) -> None:
        """Stop collecting price data.

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
            "price_collector_stopped",
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
                # Subscribe to ticker stream
                async for ticker_data in exchange.watch_tickers(self.symbols):
                    if not self._running:
                        break

                    # Parse ticker
                    ticker = self._parse_ticker(
                        ticker_data,
                        exchange_name
                    )

                    if ticker:
                        # Update cache
                        cache_key = f"{exchange_name}:{ticker.symbol}"
                        async with self._ticker_lock:
                            self._tickers[cache_key] = ticker

                        # Write to storage
                        await self.data_writer.write_ticker(ticker)

                        self._metrics["total_collected"] += 1
                        self._metrics["updates"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "websocket_ticker_error",
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
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ticker_polling_error", error=str(e))
                await asyncio.sleep(self.collection_interval)

    async def _collect_all_exchanges(self) -> None:
        """Collect tickers from all exchanges."""
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
        """Collect tickers from a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance
        """
        for symbol in self.symbols:
            try:
                # Fetch ticker with retry
                ticker_data = await self._fetch_with_retry(
                    exchange,
                    symbol,
                    exchange_name
                )

                if ticker_data:
                    # Parse ticker
                    ticker = self._parse_ticker(
                        ticker_data,
                        exchange_name
                    )

                    if ticker:
                        # Update cache
                        cache_key = f"{exchange_name}:{symbol}"
                        async with self._ticker_lock:
                            self._tickers[cache_key] = ticker

                        # Write to storage
                        await self.data_writer.write_ticker(ticker)

                        self._metrics["total_collected"] += 1

            except Exception as e:
                logger.error(
                    "ticker_collection_failed",
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
        """Fetch ticker with exponential backoff retry.

        Args:
            exchange: Exchange connector
            symbol: Trading symbol
            exchange_name: Exchange name for logging

        Returns:
            Ticker data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                ticker = await exchange.fetch_ticker(symbol)
                return ticker

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "ticker_fetch_retry",
                        exchange=exchange_name,
                        symbol=symbol,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "ticker_fetch_failed_all_retries",
                        exchange=exchange_name,
                        symbol=symbol,
                        error=str(e)
                    )
                    raise

        return None

    def _parse_ticker(
        self,
        data: Dict[str, Any],
        exchange_name: str
    ) -> Optional[Ticker]:
        """Parse ticker data from exchange format.

        Args:
            data: Raw ticker data from exchange
            exchange_name: Exchange name

        Returns:
            Normalized Ticker instance or None if invalid
        """
        try:
            # Extract symbol
            symbol = data.get("symbol")
            if not symbol:
                raise ValueError("Missing symbol in ticker data")

            # Extract prices
            bid = data.get("bid")
            ask = data.get("ask")
            last = data.get("last")

            if bid is None or ask is None or last is None:
                raise ValueError("Missing required price fields")

            # Extract volume
            volume = data.get("baseVolume") or data.get("volume") or 0

            # Extract timestamp
            timestamp = data.get("timestamp")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Validate prices
            bid_decimal = Decimal(str(bid))
            ask_decimal = Decimal(str(ask))
            last_decimal = Decimal(str(last))
            volume_decimal = Decimal(str(volume))

            if bid_decimal <= 0 or ask_decimal <= 0 or last_decimal <= 0:
                raise ValueError("Invalid price values (must be positive)")

            if bid_decimal >= ask_decimal:
                logger.warning(
                    "invalid_ticker_spread",
                    exchange=exchange_name,
                    symbol=symbol,
                    bid=str(bid_decimal),
                    ask=str(ask_decimal)
                )
                # Don't return None, just log warning
                # Some exchanges may have crossed quotes momentarily

            ticker = Ticker(
                symbol=symbol,
                bid=bid_decimal,
                ask=ask_decimal,
                last=last_decimal,
                volume=volume_decimal,
                timestamp=timestamp
            )

            return ticker

        except Exception as e:
            logger.error(
                "ticker_parse_failed",
                exchange=exchange_name,
                error=str(e),
                data=data
            )
            return None

    def get_ticker(
        self,
        exchange: str,
        symbol: str
    ) -> Optional[Ticker]:
        """Get cached ticker.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Cached Ticker or None if not available
        """
        cache_key = f"{exchange}:{symbol}"
        return self._tickers.get(cache_key)

    def get_all_tickers(self) -> Dict[str, Ticker]:
        """Get all cached tickers.

        Returns:
            Dictionary of cache keys to Ticker instances
        """
        return self._tickers.copy()

    def get_metrics(self) -> Dict[str, int]:
        """Get collector metrics.

        Returns:
            Dictionary of metric names and values
        """
        metrics = self._metrics.copy()
        metrics["cached_tickers"] = len(self._tickers)
        return metrics

    async def collect_once(self) -> pl.DataFrame:
        """Collect tickers once and return as DataFrame.

        Returns:
            DataFrame containing collected tickers

        Example:
            >>> df = await collector.collect_once()
            >>> print(df.select(["symbol", "bid", "ask"]))
        """
        all_tickers = []

        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                try:
                    exchange = self.exchanges[exchange_name]

                    for symbol in self.symbols:
                        ticker_data = await self._fetch_with_retry(
                            exchange,
                            symbol,
                            exchange_name
                        )

                        if ticker_data:
                            ticker = self._parse_ticker(
                                ticker_data,
                                exchange_name
                            )

                            if ticker:
                                all_tickers.append({
                                    "exchange": exchange_name,
                                    "symbol": ticker.symbol,
                                    "bid": str(ticker.bid),
                                    "ask": str(ticker.ask),
                                    "last": str(ticker.last),
                                    "volume": str(ticker.volume),
                                    "timestamp": ticker.timestamp
                                })

                except Exception as e:
                    logger.error(
                        "collect_once_failed",
                        exchange=exchange_name,
                        error=str(e)
                    )

        if all_tickers:
            return pl.DataFrame(all_tickers)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "exchange": pl.Utf8,
                    "symbol": pl.Utf8,
                    "bid": pl.Utf8,
                    "ask": pl.Utf8,
                    "last": pl.Utf8,
                    "volume": pl.Utf8,
                    "timestamp": pl.Datetime
                }
            )
