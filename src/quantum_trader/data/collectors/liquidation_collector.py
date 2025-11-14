"""Liquidation data collector for futures markets.

This module collects liquidation data from exchanges. Liquidations are forced
closures of positions and provide valuable signals about market leverage,
volatility, and potential reversal points.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class LiquidationCollector:
    """Collects liquidation data from exchanges.

    Monitors forced liquidations in futures markets across multiple exchanges.
    Large liquidations can signal market turning points and volatility spikes.

    Attributes:
        config: Configuration dictionary
        symbols: List of symbols to monitor
        exchanges: Dictionary of exchange connectors
        data_writer: Data writer instance
        collection_interval: Interval between collection cycles
        _running: Flag indicating if collector is active
        _collection_task: Background collection task
        _websocket_tasks: WebSocket connection tasks per exchange

    Example:
        >>> config = {
        ...     "liquidation_collector": {
        ...         "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        ...         "collection_mode": "websocket",
        ...         "enabled_exchanges": ["binance", "bybit"]
        ...     }
        ... }
        >>> async with LiquidationCollector(config, exchanges, writer) as collector:
        ...     await collector.start()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize liquidation collector.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        collector_config = config.get("liquidation_collector", {})
        self.symbols = collector_config.get("symbols", [])
        self.enabled_exchanges = collector_config.get("enabled_exchanges", [])
        self.collection_mode = collector_config.get("collection_mode", "websocket")
        self.collection_interval = collector_config.get("collection_interval", 5)
        self.batch_size = collector_config.get("batch_size", 100)
        self.max_retries = collector_config.get("max_retries", 3)
        self.retry_delay = collector_config.get("retry_delay_ms", 1000) / 1000.0
        self.lookback_seconds = collector_config.get("lookback_seconds", 300)

        self.exchanges = exchanges
        self.data_writer = data_writer

        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._websocket_tasks: Dict[str, asyncio.Task] = {}
        self._liquidation_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = asyncio.Lock()

        self._metrics: Dict[str, int] = {
            "total_collected": 0,
            "total_errors": 0,
            "collections": 0,
            "buy_liquidations": 0,
            "sell_liquidations": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "liquidation_collector" not in self.config:
            raise ValueError("Missing liquidation_collector configuration")

        collector_config = self.config["liquidation_collector"]

        if not collector_config.get("symbols"):
            raise ValueError("No symbols configured for liquidation collection")

        if not collector_config.get("enabled_exchanges"):
            raise ValueError("No exchanges enabled for liquidation collection")

    async def start(self) -> None:
        """Start collecting liquidation data.

        Launches background tasks based on collection mode (websocket or polling).
        """
        if self._running:
            logger.warning("liquidation_collector_already_running")
            return

        self._running = True

        if self.collection_mode == "websocket":
            # Start WebSocket connections for each exchange
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
            self._collection_task = asyncio.create_task(self._polling_collection_loop())

        logger.info(
            "liquidation_collector_started",
            symbols=len(self.symbols),
            exchanges=len(self.enabled_exchanges),
            mode=self.collection_mode
        )

    async def stop(self) -> None:
        """Stop collecting liquidation data.

        Gracefully stops all collection tasks and flushes remaining data.
        """
        if not self._running:
            return

        self._running = False

        # Cancel polling task
        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
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

        # Flush remaining data
        await self._flush_buffer()

        logger.info(
            "liquidation_collector_stopped",
            metrics=self._metrics
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def _polling_collection_loop(self) -> None:
        """Main collection loop for polling mode."""
        while self._running:
            try:
                start_time = datetime.now(timezone.utc)

                # Collect from all exchanges concurrently
                await self._collect_all_exchanges()

                # Update metrics
                self._metrics["collections"] += 1

                # Calculate sleep time
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("liquidation_polling_error", error=str(e))
                await asyncio.sleep(self.collection_interval)

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
                # Subscribe to liquidation stream
                async for liquidation_data in exchange.watch_liquidations(self.symbols):
                    if not self._running:
                        break

                    # Parse liquidation data
                    liquidation = self._parse_liquidation(
                        liquidation_data,
                        exchange_name
                    )

                    if liquidation:
                        async with self._buffer_lock:
                            self._liquidation_buffer.append(liquidation)

                            # Update metrics
                            self._metrics["total_collected"] += 1
                            if liquidation["side"].lower() == "buy":
                                self._metrics["buy_liquidations"] += 1
                            else:
                                self._metrics["sell_liquidations"] += 1

                            # Flush if buffer full
                            if len(self._liquidation_buffer) >= self.batch_size:
                                await self._flush_buffer()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "websocket_liquidation_error",
                    exchange=exchange_name,
                    error=str(e)
                )
                # Retry connection after delay
                await asyncio.sleep(self.retry_delay * 2)

    async def _collect_all_exchanges(self) -> None:
        """Collect liquidations from all exchanges (polling mode)."""
        tasks = []
        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                task = self._collect_exchange(
                    exchange_name,
                    self.exchanges[exchange_name]
                )
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            all_liquidations = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("exchange_liquidation_collection_failed", error=str(result))
                    self._metrics["total_errors"] += 1
                elif result:
                    all_liquidations.extend(result)

            # Write batch
            if all_liquidations:
                async with self._buffer_lock:
                    self._liquidation_buffer.extend(all_liquidations)
                    self._metrics["total_collected"] += len(all_liquidations)

                    if len(self._liquidation_buffer) >= self.batch_size:
                        await self._flush_buffer()

    async def _collect_exchange(
        self,
        exchange_name: str,
        exchange: Any
    ) -> List[Dict[str, Any]]:
        """Collect liquidations from a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance

        Returns:
            List of liquidation records
        """
        liquidations = []

        for symbol in self.symbols:
            try:
                # Fetch recent liquidations
                liq_data = await self._fetch_with_retry(
                    exchange,
                    symbol,
                    exchange_name
                )

                if liq_data:
                    for item in liq_data:
                        liquidation = self._parse_liquidation(item, exchange_name)
                        if liquidation:
                            liquidations.append(liquidation)

            except Exception as e:
                logger.error(
                    "liquidation_collection_failed",
                    exchange=exchange_name,
                    symbol=symbol,
                    error=str(e)
                )
                self._metrics["total_errors"] += 1

        logger.debug(
            "liquidations_collected_from_exchange",
            exchange=exchange_name,
            count=len(liquidations)
        )

        return liquidations

    async def _fetch_with_retry(
        self,
        exchange: Any,
        symbol: str,
        exchange_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch liquidations with exponential backoff retry.

        Args:
            exchange: Exchange connector
            symbol: Trading symbol
            exchange_name: Exchange name for logging

        Returns:
            List of liquidation data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch recent liquidations (last N seconds)
                end_time = datetime.now(timezone.utc)
                start_time = end_time.timestamp() - self.lookback_seconds

                liquidations = await exchange.fetch_liquidations(
                    symbol=symbol,
                    since=int(start_time * 1000)
                )

                return liquidations

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "liquidation_fetch_retry",
                        exchange=exchange_name,
                        symbol=symbol,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "liquidation_fetch_failed_all_retries",
                        exchange=exchange_name,
                        symbol=symbol,
                        error=str(e)
                    )
                    raise

        return None

    def _parse_liquidation(
        self,
        data: Dict[str, Any],
        exchange_name: str
    ) -> Optional[Dict[str, Any]]:
        """Parse liquidation data from exchange format.

        Args:
            data: Raw liquidation data from exchange
            exchange_name: Exchange name

        Returns:
            Normalized liquidation record or None if invalid
        """
        try:
            # Extract symbol
            symbol = data.get("symbol") or data.get("s")
            if not symbol:
                raise ValueError("Missing symbol in liquidation data")

            # Extract side (buy/sell)
            side = data.get("side") or data.get("S")
            if not side:
                raise ValueError("Missing side in liquidation data")

            # Extract price
            price = data.get("price") or data.get("p")
            if price is None:
                raise ValueError("Missing price in liquidation data")

            # Extract quantity
            quantity = data.get("quantity") or data.get("q") or data.get("amount")
            if quantity is None:
                raise ValueError("Missing quantity in liquidation data")

            # Extract timestamp
            timestamp = data.get("timestamp") or data.get("time") or data.get("T")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            record = {
                "symbol": symbol,
                "exchange": exchange_name,
                "side": str(side).upper(),
                "price": Decimal(str(price)),
                "quantity": Decimal(str(quantity)),
                "value": Decimal(str(price)) * Decimal(str(quantity)),
                "timestamp": timestamp
            }

            return record

        except Exception as e:
            logger.error(
                "liquidation_parse_failed",
                exchange=exchange_name,
                error=str(e),
                data=data
            )
            return None

    async def _flush_buffer(self) -> None:
        """Flush liquidation buffer to storage."""
        if not self._liquidation_buffer:
            return

        try:
            # Copy and clear buffer
            liquidations = self._liquidation_buffer.copy()
            self._liquidation_buffer.clear()

            # Convert to Polars DataFrame
            df = pl.DataFrame(liquidations)

            # Write to database
            await self.data_writer.write_batch("liquidations", df)

            logger.debug(
                "liquidation_batch_written",
                count=len(liquidations)
            )

        except Exception as e:
            logger.error(
                "liquidation_batch_write_failed",
                count=len(liquidations),
                error=str(e)
            )
            # Re-add to buffer on failure
            async with self._buffer_lock:
                self._liquidation_buffer.extend(liquidations)
            raise

    def get_metrics(self) -> Dict[str, int]:
        """Get collector metrics.

        Returns:
            Dictionary of metric names and values
        """
        return self._metrics.copy()

    async def collect_once(self) -> pl.DataFrame:
        """Collect liquidations once and return as DataFrame.

        Returns:
            DataFrame containing collected liquidations

        Example:
            >>> df = await collector.collect_once()
            >>> print(df.filter(pl.col("side") == "SELL"))
        """
        all_liquidations = []

        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                try:
                    liquidations = await self._collect_exchange(
                        exchange_name,
                        self.exchanges[exchange_name]
                    )
                    all_liquidations.extend(liquidations)
                except Exception as e:
                    logger.error(
                        "collect_once_failed",
                        exchange=exchange_name,
                        error=str(e)
                    )

        if all_liquidations:
            return pl.DataFrame(all_liquidations)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "exchange": pl.Utf8,
                    "side": pl.Utf8,
                    "price": pl.Decimal,
                    "quantity": pl.Decimal,
                    "value": pl.Decimal,
                    "timestamp": pl.Datetime
                }
            )
