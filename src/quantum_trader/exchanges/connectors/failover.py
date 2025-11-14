"""Failover connector for exchange redundancy.

This module provides a failover mechanism that automatically switches between
multiple exchange connectors when one fails, ensuring high availability.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


@dataclass
class ExchangeHealth:
    """Health status for an exchange connector."""

    exchange_name: str
    is_healthy: bool
    last_success: Optional[datetime]
    last_failure: Optional[datetime]
    consecutive_failures: int
    total_requests: int
    total_failures: int
    average_latency_ms: Decimal


class FailoverConnector:
    """Failover connector that manages multiple exchange connectors.

    This connector automatically switches between configured exchanges when
    failures occur, providing high availability and redundancy.
    """

    def __init__(
        self,
        primary_exchanges: List[Any],
        fallback_exchanges: Optional[List[Any]] = None,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
        health_check_interval: Optional[int] = None,
    ) -> None:
        """Initialize failover connector.

        Args:
            primary_exchanges: List of primary exchange connectors
            fallback_exchanges: List of fallback exchange connectors
            failure_threshold: Number of failures before switching
            recovery_timeout: Seconds before retrying failed exchange
            health_check_interval: Seconds between health checks
        """
        self.primary_exchanges = primary_exchanges
        self.fallback_exchanges = fallback_exchanges or []
        self.all_exchanges = primary_exchanges + self.fallback_exchanges

        self.failure_threshold = int(
            os.getenv("FAILOVER_FAILURE_THRESHOLD", str(failure_threshold or 3))
        )
        self.recovery_timeout = int(
            os.getenv("FAILOVER_RECOVERY_TIMEOUT", str(recovery_timeout or 300))
        )
        self.health_check_interval = int(
            os.getenv("FAILOVER_HEALTH_CHECK_INTERVAL", str(health_check_interval or 60))
        )

        # Track exchange health
        self.health_status: Dict[str, ExchangeHealth] = {}
        self._initialize_health_tracking()

        # Current active exchange
        self.current_exchange: Optional[Any] = None
        self._select_healthy_exchange()

        # Health monitoring task
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info(
            "failover_connector_initialized",
            primary_count=len(primary_exchanges),
            fallback_count=len(self.fallback_exchanges),
            failure_threshold=self.failure_threshold,
        )

    def _initialize_health_tracking(self) -> None:
        """Initialize health tracking for all exchanges."""
        for exchange in self.all_exchanges:
            exchange_name = getattr(exchange, "name", exchange.__class__.__name__)
            self.health_status[exchange_name] = ExchangeHealth(
                exchange_name=exchange_name,
                is_healthy=True,
                last_success=None,
                last_failure=None,
                consecutive_failures=0,
                total_requests=0,
                total_failures=0,
                average_latency_ms=Decimal("0"),
            )

    def _select_healthy_exchange(self) -> None:
        """Select a healthy exchange as current active."""
        # Try primary exchanges first
        for exchange in self.primary_exchanges:
            exchange_name = getattr(exchange, "name", exchange.__class__.__name__)
            if self.health_status[exchange_name].is_healthy:
                self.current_exchange = exchange
                logger.info("active_exchange_selected", exchange=exchange_name, tier="primary")
                return

        # Fall back to fallback exchanges
        for exchange in self.fallback_exchanges:
            exchange_name = getattr(exchange, "name", exchange.__class__.__name__)
            if self.health_status[exchange_name].is_healthy:
                self.current_exchange = exchange
                logger.warning("active_exchange_selected", exchange=exchange_name, tier="fallback")
                return

        # No healthy exchanges available
        logger.error("no_healthy_exchanges_available")
        self.current_exchange = self.primary_exchanges[0] if self.primary_exchanges else None

    def _mark_success(self, exchange_name: str, latency_ms: Decimal) -> None:
        """Mark successful request for exchange.

        Args:
            exchange_name: Exchange name
            latency_ms: Request latency in milliseconds
        """
        health = self.health_status[exchange_name]
        health.last_success = datetime.utcnow()
        health.consecutive_failures = 0
        health.total_requests += 1

        # Update average latency
        if health.average_latency_ms == 0:
            health.average_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = Decimal("0.3")
            health.average_latency_ms = (
                alpha * latency_ms + (Decimal("1") - alpha) * health.average_latency_ms
            )

        if not health.is_healthy:
            health.is_healthy = True
            logger.info("exchange_recovered", exchange=exchange_name)

    def _mark_failure(self, exchange_name: str) -> None:
        """Mark failed request for exchange.

        Args:
            exchange_name: Exchange name
        """
        health = self.health_status[exchange_name]
        health.last_failure = datetime.utcnow()
        health.consecutive_failures += 1
        health.total_requests += 1
        health.total_failures += 1

        if health.consecutive_failures >= self.failure_threshold:
            if health.is_healthy:
                health.is_healthy = False
                logger.error(
                    "exchange_marked_unhealthy",
                    exchange=exchange_name,
                    consecutive_failures=health.consecutive_failures,
                )
                self._select_healthy_exchange()

    def _should_retry_exchange(self, exchange_name: str) -> bool:
        """Check if failed exchange should be retried.

        Args:
            exchange_name: Exchange name

        Returns:
            True if exchange should be retried
        """
        health = self.health_status[exchange_name]

        if health.is_healthy:
            return True

        if health.last_failure is None:
            return True

        time_since_failure = (datetime.utcnow() - health.last_failure).total_seconds()
        return time_since_failure >= self.recovery_timeout

    async def _execute_with_failover(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute method with automatic failover.

        Args:
            method_name: Method name to call on exchange
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method result

        Raises:
            Exception: If all exchanges fail
        """
        attempted_exchanges: List[str] = []
        last_exception: Optional[Exception] = None

        # Try current exchange first
        if self.current_exchange:
            exchange_name = getattr(
                self.current_exchange, "name", self.current_exchange.__class__.__name__
            )
            if self._should_retry_exchange(exchange_name):
                try:
                    start_time = datetime.utcnow()
                    method = getattr(self.current_exchange, method_name)
                    result = await method(*args, **kwargs)
                    latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._mark_success(exchange_name, Decimal(str(latency)))
                    return result

                except Exception as e:
                    last_exception = e
                    self._mark_failure(exchange_name)
                    attempted_exchanges.append(exchange_name)
                    logger.warning(
                        "exchange_request_failed",
                        exchange=exchange_name,
                        method=method_name,
                        error=str(e),
                    )

        # Try all other exchanges
        for exchange in self.all_exchanges:
            exchange_name = getattr(exchange, "name", exchange.__class__.__name__)

            if exchange_name in attempted_exchanges:
                continue

            if not self._should_retry_exchange(exchange_name):
                continue

            try:
                start_time = datetime.utcnow()
                method = getattr(exchange, method_name)
                result = await method(*args, **kwargs)
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._mark_success(exchange_name, Decimal(str(latency)))

                # Switch to this exchange if successful
                if exchange != self.current_exchange:
                    logger.info("failover_switched_exchange", new_exchange=exchange_name)
                    self.current_exchange = exchange

                return result

            except Exception as e:
                last_exception = e
                self._mark_failure(exchange_name)
                attempted_exchanges.append(exchange_name)
                logger.warning(
                    "exchange_request_failed",
                    exchange=exchange_name,
                    method=method_name,
                    error=str(e),
                )

        # All exchanges failed
        logger.error(
            "all_exchanges_failed",
            method=method_name,
            attempted=attempted_exchanges,
        )

        if last_exception:
            raise last_exception
        raise RuntimeError(f"All exchanges failed for method: {method_name}")

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 100, since: Optional[int] = None
    ) -> pl.DataFrame:
        """Fetch OHLCV data with failover.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles
            since: Start timestamp

        Returns:
            DataFrame with OHLCV data
        """
        return await self._execute_with_failover(
            "fetch_ohlcv", symbol, timeframe, limit, since
        )

    async def fetch_ticker(self, symbol: str) -> Any:
        """Fetch ticker with failover.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data
        """
        return await self._execute_with_failover("fetch_ticker", symbol)

    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Any:
        """Fetch order book with failover.

        Args:
            symbol: Trading pair symbol
            depth: Order book depth

        Returns:
            Order book data
        """
        return await self._execute_with_failover("fetch_orderbook", symbol, depth)

    async def fetch_balance(self) -> Dict[str, Decimal]:
        """Fetch balance with failover.

        Returns:
            Balance dictionary
        """
        return await self._execute_with_failover("fetch_balance")

    async def create_order(self, order: Any) -> str:
        """Create order with failover.

        Args:
            order: Order object

        Returns:
            Order ID
        """
        return await self._execute_with_failover("create_order", order)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order with failover.

        Args:
            order_id: Order ID
            symbol: Trading pair symbol

        Returns:
            True if cancelled
        """
        return await self._execute_with_failover("cancel_order", order_id, symbol)

    def get_health_status(self) -> Dict[str, ExchangeHealth]:
        """Get health status for all exchanges.

        Returns:
            Dictionary of exchange health status
        """
        return self.health_status.copy()

    def get_current_exchange(self) -> Optional[str]:
        """Get name of current active exchange.

        Returns:
            Exchange name or None
        """
        if self.current_exchange:
            return getattr(
                self.current_exchange, "name", self.current_exchange.__class__.__name__
            )
        return None

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("health_monitoring_started")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("health_monitoring_stopped")

    async def _health_check_loop(self) -> None:
        """Background loop for health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all exchanges."""
        for exchange in self.all_exchanges:
            exchange_name = getattr(exchange, "name", exchange.__class__.__name__)

            # Check if exchange should be tested for recovery
            if not self._should_retry_exchange(exchange_name):
                continue

            try:
                # Simple ping test
                if hasattr(exchange, "ping"):
                    await exchange.ping()
                    if not self.health_status[exchange_name].is_healthy:
                        logger.info("exchange_health_check_passed", exchange=exchange_name)
                        self.health_status[exchange_name].is_healthy = True

            except Exception as e:
                logger.debug(
                    "exchange_health_check_failed",
                    exchange=exchange_name,
                    error=str(e),
                )
