"""Funding rate collector for perpetual futures contracts.

This module collects funding rate data from multiple exchanges for perpetual futures
contracts. Funding rates are critical for futures arbitrage and market sentiment analysis.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class FundingCollector:
    """Collects funding rate data from exchanges.

    Monitors funding rates for perpetual futures contracts across multiple exchanges.
    Funding rates indicate the premium/discount of perpetual contracts vs spot prices.

    Attributes:
        config: Configuration dictionary
        symbols: List of symbols to collect funding rates for
        exchanges: List of exchange connectors
        data_writer: Data writer instance for persisting data
        collection_interval: Interval between collection cycles in seconds
        _running: Flag indicating if collector is active
        _collection_task: Background collection task

    Example:
        >>> config = {
        ...     "funding_collector": {
        ...         "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        ...         "collection_interval": 60,
        ...         "enabled_exchanges": ["binance", "bybit"]
        ...     }
        ... }
        >>> async with FundingCollector(config, exchanges, writer) as collector:
        ...     await collector.start()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize funding collector.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        collector_config = config.get("funding_collector", {})
        self.symbols = collector_config.get("symbols", [])
        self.enabled_exchanges = collector_config.get("enabled_exchanges", [])
        self.collection_interval = collector_config.get("collection_interval", 60)
        self.batch_size = collector_config.get("batch_size", 100)
        self.max_retries = collector_config.get("max_retries", 3)
        self.retry_delay = collector_config.get("retry_delay_ms", 1000) / 1000.0

        self.exchanges = exchanges
        self.data_writer = data_writer

        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._metrics: Dict[str, int] = {
            "total_collected": 0,
            "total_errors": 0,
            "collections": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "funding_collector" not in self.config:
            raise ValueError("Missing funding_collector configuration")

        collector_config = self.config["funding_collector"]

        if not collector_config.get("symbols"):
            raise ValueError("No symbols configured for funding collection")

        if not collector_config.get("enabled_exchanges"):
            raise ValueError("No exchanges enabled for funding collection")

    async def start(self) -> None:
        """Start collecting funding rates.

        Launches background task that periodically collects funding rates
        from all configured exchanges.
        """
        if self._running:
            logger.warning("funding_collector_already_running")
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info(
            "funding_collector_started",
            symbols=len(self.symbols),
            exchanges=len(self.enabled_exchanges),
            interval=self.collection_interval
        )

    async def stop(self) -> None:
        """Stop collecting funding rates.

        Gracefully stops the collection task.
        """
        if not self._running:
            return

        self._running = False

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "funding_collector_stopped",
            metrics=self._metrics
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def _collection_loop(self) -> None:
        """Main collection loop running in background."""
        while self._running:
            try:
                start_time = datetime.now(timezone.utc)

                # Collect from all exchanges concurrently
                await self._collect_all_exchanges()

                # Update metrics
                self._metrics["collections"] += 1

                # Calculate sleep time to maintain interval
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                sleep_time = max(0, self.collection_interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(
                        "funding_collection_slow",
                        elapsed=elapsed,
                        interval=self.collection_interval
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("funding_collection_loop_error", error=str(e))
                await asyncio.sleep(self.collection_interval)

    async def _collect_all_exchanges(self) -> None:
        """Collect funding rates from all exchanges concurrently."""
        tasks = []
        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                task = self._collect_exchange(exchange_name, self.exchanges[exchange_name])
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            all_funding_rates = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("exchange_funding_collection_failed", error=str(result))
                    self._metrics["total_errors"] += 1
                elif result:
                    all_funding_rates.extend(result)

            # Write batch to storage
            if all_funding_rates:
                await self._write_funding_batch(all_funding_rates)
                self._metrics["total_collected"] += len(all_funding_rates)

    async def _collect_exchange(
        self,
        exchange_name: str,
        exchange: Any
    ) -> List[Dict[str, Any]]:
        """Collect funding rates from a single exchange.

        Args:
            exchange_name: Name of the exchange
            exchange: Exchange connector instance

        Returns:
            List of funding rate records
        """
        funding_rates = []

        for symbol in self.symbols:
            try:
                # Collect with retry logic
                funding = await self._fetch_with_retry(
                    exchange,
                    symbol,
                    exchange_name
                )

                if funding:
                    funding_rates.append(funding)

            except Exception as e:
                logger.error(
                    "funding_collection_failed",
                    exchange=exchange_name,
                    symbol=symbol,
                    error=str(e)
                )
                self._metrics["total_errors"] += 1

        logger.debug(
            "funding_collected_from_exchange",
            exchange=exchange_name,
            count=len(funding_rates)
        )

        return funding_rates

    async def _fetch_with_retry(
        self,
        exchange: Any,
        symbol: str,
        exchange_name: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch funding rate with exponential backoff retry.

        Args:
            exchange: Exchange connector
            symbol: Trading symbol
            exchange_name: Exchange name for logging

        Returns:
            Funding rate data or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch funding rate from exchange
                funding_data = await exchange.fetch_funding_rate(symbol)

                if not funding_data:
                    return None

                # Parse and validate
                funding_rate = self._parse_funding_rate(
                    funding_data,
                    symbol,
                    exchange_name
                )

                return funding_rate

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "funding_fetch_retry",
                        exchange=exchange_name,
                        symbol=symbol,
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e)
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "funding_fetch_failed_all_retries",
                        exchange=exchange_name,
                        symbol=symbol,
                        error=str(e)
                    )
                    raise

        return None

    def _parse_funding_rate(
        self,
        data: Dict[str, Any],
        symbol: str,
        exchange_name: str
    ) -> Dict[str, Any]:
        """Parse funding rate data from exchange format.

        Args:
            data: Raw funding rate data from exchange
            symbol: Trading symbol
            exchange_name: Exchange name

        Returns:
            Normalized funding rate record

        Raises:
            ValueError: If data format is invalid
        """
        try:
            # Extract funding rate (typically in data['fundingRate'])
            funding_rate = data.get("fundingRate") or data.get("funding_rate")
            if funding_rate is None:
                raise ValueError("Missing funding rate in response")

            # Extract timestamp
            timestamp = data.get("timestamp") or data.get("fundingTimestamp")
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif isinstance(timestamp, (int, float)):
                # Convert millisecond timestamp to datetime
                timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Get next funding time if available
            next_funding = data.get("fundingTime") or data.get("nextFundingTime")
            if next_funding and isinstance(next_funding, (int, float)):
                next_funding = datetime.fromtimestamp(next_funding / 1000, tz=timezone.utc)

            record = {
                "symbol": symbol,
                "exchange": exchange_name,
                "rate": Decimal(str(funding_rate)),
                "timestamp": timestamp,
                "next_funding_time": next_funding,
                "mark_price": Decimal(str(data.get("markPrice", 0))) if data.get("markPrice") else None,
                "index_price": Decimal(str(data.get("indexPrice", 0))) if data.get("indexPrice") else None
            }

            return record

        except Exception as e:
            logger.error(
                "funding_rate_parse_failed",
                exchange=exchange_name,
                symbol=symbol,
                error=str(e)
            )
            raise ValueError(f"Failed to parse funding rate: {e}")

    async def _write_funding_batch(self, funding_rates: List[Dict[str, Any]]) -> None:
        """Write batch of funding rates to storage.

        Args:
            funding_rates: List of funding rate records
        """
        try:
            # Convert to Polars DataFrame
            df = pl.DataFrame(funding_rates)

            # Write to database via data writer
            await self.data_writer.write_batch("funding", df)

            logger.debug(
                "funding_batch_written",
                count=len(funding_rates)
            )

        except Exception as e:
            logger.error(
                "funding_batch_write_failed",
                count=len(funding_rates),
                error=str(e)
            )
            raise

    def get_metrics(self) -> Dict[str, int]:
        """Get collector metrics.

        Returns:
            Dictionary of metric names and values
        """
        return self._metrics.copy()

    async def collect_once(self) -> pl.DataFrame:
        """Collect funding rates once and return as DataFrame.

        Returns:
            DataFrame containing collected funding rates

        Example:
            >>> df = await collector.collect_once()
            >>> print(df)
        """
        all_funding_rates = []

        for exchange_name in self.enabled_exchanges:
            if exchange_name in self.exchanges:
                try:
                    rates = await self._collect_exchange(
                        exchange_name,
                        self.exchanges[exchange_name]
                    )
                    all_funding_rates.extend(rates)
                except Exception as e:
                    logger.error(
                        "collect_once_failed",
                        exchange=exchange_name,
                        error=str(e)
                    )

        if all_funding_rates:
            return pl.DataFrame(all_funding_rates)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "exchange": pl.Utf8,
                    "rate": pl.Decimal,
                    "timestamp": pl.Datetime,
                    "next_funding_time": pl.Datetime,
                    "mark_price": pl.Decimal,
                    "index_price": pl.Decimal
                }
            )
