"""Metrics collector for aggregating and collecting system metrics.

This module provides functionality to collect metrics from various system
components and expose them for Prometheus scraping.
"""

import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
import psutil
from prometheus_client import CollectorRegistry
from structlog import get_logger

from quantum_trader.monitoring.prometheus.metric_definitions import MetricDefinitions

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and aggregates metrics from the trading system.

    This collector gathers metrics from various sources including:
    - Trading operations (orders, trades, positions)
    - System resources (CPU, memory, disk)
    - Exchange connectivity
    - Strategy performance

    Attributes:
        config: Configuration dictionary
        metrics: MetricDefinitions instance
        collection_interval: Interval for collecting metrics in seconds
        is_running: Flag indicating if collector is running

    Example:
        >>> config = {
        ...     "collection_interval": 5,
        ...     "namespace": "quantum_trader"
        ... }
        >>> collector = MetricsCollector(config)
        >>> await collector.start()
        >>> # ... collect metrics ...
        >>> await collector.stop()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize metrics collector.

        Args:
            config: Configuration dictionary containing:
                - collection_interval: Metrics collection interval in seconds
                - namespace: Metrics namespace
                - enabled_collectors: List of collector types to enable

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.collection_interval = Decimal(
            str(self.config.get(
                "collection_interval",
                os.getenv("METRICS_COLLECTION_INTERVAL", "5")
            ))
        )

        # Initialize metric definitions
        self.metrics = MetricDefinitions(config)

        self.is_running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._collectors: Set[str] = set(
            self.config.get(
                "enabled_collectors",
                os.getenv("METRICS_ENABLED_COLLECTORS", "system,exchange,strategy").split(",")
            )
        )

        # Tracking state
        self._last_collection_time: Optional[datetime] = None
        self._collection_count = 0
        self._collection_errors = 0

        logger.info(
            "MetricsCollector initialized",
            interval=str(self.collection_interval),
            enabled_collectors=list(self._collectors)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Validation passed
        logger.debug("Config validation passed")

    async def start(self) -> None:
        """Start the metrics collection process.

        Raises:
            RuntimeError: If collector is already running
        """
        if self.is_running:
            raise RuntimeError("MetricsCollector is already running")

        self.is_running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info("MetricsCollector started")

    async def stop(self) -> None:
        """Stop the metrics collection process."""
        if not self.is_running:
            logger.warning("MetricsCollector is not running")
            return

        self.is_running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "MetricsCollector stopped",
            total_collections=self._collection_count,
            total_errors=self._collection_errors
        )

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        logger.info("Starting metrics collection loop")

        while self.is_running:
            try:
                start_time = time.time()

                await self._collect_all_metrics()

                self._collection_count += 1
                self._last_collection_time = datetime.now(timezone.utc)

                elapsed = time.time() - start_time
                logger.debug(
                    "Metrics collection completed",
                    elapsed_seconds=elapsed,
                    count=self._collection_count
                )

                # Sleep until next collection
                sleep_duration = max(0, float(self.collection_interval) - elapsed)
                await asyncio.sleep(sleep_duration)

            except asyncio.CancelledError:
                logger.info("Collection loop cancelled")
                break
            except Exception as e:
                self._collection_errors += 1
                logger.error(
                    "Error in collection loop",
                    error=str(e),
                    error_count=self._collection_errors,
                    exc_info=True
                )
                await asyncio.sleep(float(self.collection_interval))

    async def _collect_all_metrics(self) -> None:
        """Collect all enabled metrics."""
        collection_tasks = []

        if "system" in self._collectors:
            collection_tasks.append(self._collect_system_metrics())

        if "exchange" in self._collectors:
            collection_tasks.append(self._collect_exchange_metrics())

        if "strategy" in self._collectors:
            collection_tasks.append(self._collect_strategy_metrics())

        if collection_tasks:
            await asyncio.gather(*collection_tasks, return_exceptions=True)

    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics.cpu_usage.labels(component="main").set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.memory_usage.labels(component="main").set(memory.used)

            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss
            self.metrics.memory_usage.labels(component="process").set(process_memory)

            process_cpu = process.cpu_percent(interval=0.1)
            self.metrics.cpu_usage.labels(component="process").set(process_cpu)

            logger.debug(
                "System metrics collected",
                cpu_percent=cpu_percent,
                memory_used_mb=memory.used / (1024 * 1024),
                process_memory_mb=process_memory / (1024 * 1024)
            )

        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e), exc_info=True)

    async def _collect_exchange_metrics(self) -> None:
        """Collect exchange connectivity metrics."""
        try:
            # This would normally query actual exchange connectors
            # For now, we just update metrics based on configuration

            exchanges = self.config.get("exchanges", os.getenv("ENABLED_EXCHANGES", "").split(","))

            for exchange in exchanges:
                if exchange:
                    # In production, check actual connection status
                    # For now, assume connected
                    pass

            logger.debug("Exchange metrics collected", exchanges=exchanges)

        except Exception as e:
            logger.error("Error collecting exchange metrics", error=str(e), exc_info=True)

    async def _collect_strategy_metrics(self) -> None:
        """Collect strategy performance metrics."""
        try:
            # This would normally query actual strategy instances
            # Metrics are updated by strategies themselves via record_* methods

            logger.debug("Strategy metrics collected")

        except Exception as e:
            logger.error("Error collecting strategy metrics", error=str(e), exc_info=True)

    def record_order(
        self,
        exchange: str,
        order_type: str,
        side: str,
        status: str,
        size_usd: Decimal,
        latency_ms: Optional[Decimal] = None
    ) -> None:
        """Record an order event.

        Args:
            exchange: Exchange name
            order_type: Type of order (MARKET, LIMIT, etc.)
            side: Order side (BUY/SELL)
            status: Order status
            size_usd: Order size in USD
            latency_ms: Order placement latency in milliseconds
        """
        try:
            self.metrics.order_total.labels(
                exchange=exchange,
                order_type=order_type,
                side=side,
                status=status
            ).inc()

            self.metrics.order_size.labels(
                exchange=exchange,
                order_type=order_type,
                side=side
            ).observe(float(size_usd))

            if latency_ms is not None:
                self.metrics.order_latency.labels(
                    exchange=exchange,
                    order_type=order_type
                ).observe(float(latency_ms))

            logger.debug(
                "Order recorded",
                exchange=exchange,
                order_type=order_type,
                side=side,
                status=status,
                size_usd=str(size_usd)
            )

        except Exception as e:
            logger.error("Error recording order", error=str(e), exc_info=True)

    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        volume_usd: Decimal,
        fee_usd: Decimal,
        slippage_bps: Optional[Decimal] = None,
        latency_ms: Optional[Decimal] = None
    ) -> None:
        """Record a trade execution.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            volume_usd: Trade volume in USD
            fee_usd: Trading fee in USD
            slippage_bps: Trade slippage in basis points
            latency_ms: Execution latency in milliseconds
        """
        try:
            self.metrics.trades_total.labels(
                exchange=exchange,
                side=side,
                symbol=symbol
            ).inc()

            self.metrics.trade_volume.labels(
                exchange=exchange,
                symbol=symbol
            ).inc(float(volume_usd))

            self.metrics.trade_fees.labels(
                exchange=exchange,
                fee_type="trading"
            ).inc(float(fee_usd))

            if slippage_bps is not None:
                self.metrics.trade_slippage.labels(
                    exchange=exchange,
                    order_type="MARKET"
                ).observe(float(slippage_bps))

            if latency_ms is not None:
                self.metrics.execution_latency.labels(
                    exchange=exchange
                ).observe(float(latency_ms))

            logger.debug(
                "Trade recorded",
                exchange=exchange,
                symbol=symbol,
                side=side,
                volume_usd=str(volume_usd)
            )

        except Exception as e:
            logger.error("Error recording trade", error=str(e), exc_info=True)

    def record_position(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size_usd: Decimal,
        leverage: Decimal
    ) -> None:
        """Record position update.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: Position side (LONG/SHORT)
            size_usd: Position size in USD
            leverage: Leverage ratio
        """
        try:
            self.metrics.position_size.labels(
                exchange=exchange,
                symbol=symbol,
                side=side
            ).set(float(size_usd))

            self.metrics.leverage_ratio.labels(
                exchange=exchange,
                symbol=symbol
            ).set(float(leverage))

            logger.debug(
                "Position recorded",
                exchange=exchange,
                symbol=symbol,
                side=side,
                size_usd=str(size_usd)
            )

        except Exception as e:
            logger.error("Error recording position", error=str(e), exc_info=True)

    def record_pnl(
        self,
        strategy: str,
        realized_pnl: Decimal,
        unrealized_pnl: Decimal,
        total_pnl: Decimal
    ) -> None:
        """Record profit and loss metrics.

        Args:
            strategy: Strategy name
            realized_pnl: Realized PnL in USD
            unrealized_pnl: Unrealized PnL in USD
            total_pnl: Total PnL in USD
        """
        try:
            self.metrics.total_pnl.labels(strategy=strategy).set(float(total_pnl))

            logger.debug(
                "PnL recorded",
                strategy=strategy,
                realized=str(realized_pnl),
                unrealized=str(unrealized_pnl),
                total=str(total_pnl)
            )

        except Exception as e:
            logger.error("Error recording PnL", error=str(e), exc_info=True)

    def record_risk_violation(self, violation_type: str, severity: str) -> None:
        """Record a risk limit violation.

        Args:
            violation_type: Type of violation
            severity: Severity level
        """
        try:
            self.metrics.risk_violations.labels(
                violation_type=violation_type,
                severity=severity
            ).inc()

            logger.warning(
                "Risk violation recorded",
                violation_type=violation_type,
                severity=severity
            )

        except Exception as e:
            logger.error("Error recording risk violation", error=str(e), exc_info=True)

    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus collector registry.

        Returns:
            CollectorRegistry instance
        """
        return self.metrics.get_registry()

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics.

        Returns:
            Dictionary with collector stats
        """
        return {
            "is_running": self.is_running,
            "collection_count": self._collection_count,
            "collection_errors": self._collection_errors,
            "last_collection_time": self._last_collection_time.isoformat() if self._last_collection_time else None,
            "enabled_collectors": list(self._collectors),
            "collection_interval": str(self.collection_interval),
        }
