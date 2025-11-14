"""Performance monitoring for trading system components.

This module provides comprehensive performance monitoring including latency
tracking, throughput measurement, and resource utilization monitoring.
"""

import asyncio
import os
import time
from decimal import Decimal
from typing import Dict, Any, Optional, List, Deque
from datetime import datetime, timezone, timedelta
from collections import deque
from dataclasses import dataclass, field
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics.

    Attributes:
        latency_p50: 50th percentile latency
        latency_p95: 95th percentile latency
        latency_p99: 99th percentile latency
        throughput: Operations per second
        error_rate: Error rate percentage
        timestamp: Metrics timestamp
    """
    latency_p50: Decimal
    latency_p95: Decimal
    latency_p99: Decimal
    throughput: Decimal
    error_rate: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PerformanceMonitor:
    """Monitor and track performance metrics for system components.

    Tracks latency, throughput, error rates, and other performance indicators
    for trading system components.

    Attributes:
        config: Configuration dictionary
        component_name: Name of component being monitored
        window_size: Size of rolling window for metrics
        metrics_history: Deque storing recent metrics

    Example:
        >>> config = {
        ...     "window_size": 1000,
        ...     "percentiles": [50, 95, 99]
        ... }
        >>> monitor = PerformanceMonitor(config, "order_executor")
        >>> await monitor.start()
        >>> monitor.record_operation(latency_ms=5.2, success=True)
        >>> metrics = monitor.get_current_metrics()
        >>> await monitor.stop()
    """

    def __init__(self, config: Dict[str, Any], component_name: str) -> None:
        """Initialize performance monitor.

        Args:
            config: Configuration dictionary containing:
                - window_size: Number of samples to keep in rolling window
                - percentiles: List of percentiles to calculate
                - alert_thresholds: Dictionary of alert thresholds
            component_name: Name of component being monitored

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self.component_name = component_name
        self._validate_config()

        self.window_size = int(
            self.config.get("window_size", os.getenv("PERF_WINDOW_SIZE", "1000"))
        )
        self.percentiles = self.config.get(
            "percentiles",
            [int(p) for p in os.getenv("PERF_PERCENTILES", "50,95,99").split(",")]
        )

        # Rolling windows for metrics
        self._latencies: Deque[Decimal] = deque(maxlen=self.window_size)
        self._timestamps: Deque[datetime] = deque(maxlen=self.window_size)
        self._errors: Deque[bool] = deque(maxlen=self.window_size)

        # Counters
        self._total_operations = 0
        self._total_errors = 0
        self._start_time: Optional[datetime] = None

        # Alert thresholds
        self._alert_thresholds = self.config.get("alert_thresholds", {})
        self._latency_threshold = Decimal(
            str(self._alert_thresholds.get(
                "latency_ms",
                os.getenv("PERF_ALERT_LATENCY_MS", "100")
            ))
        )
        self._error_rate_threshold = Decimal(
            str(self._alert_thresholds.get(
                "error_rate_percent",
                os.getenv("PERF_ALERT_ERROR_RATE", "5.0")
            ))
        )

        # Monitoring state
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_alert_time: Optional[datetime] = None
        self._alert_cooldown = timedelta(
            seconds=int(
                self.config.get("alert_cooldown_seconds", os.getenv("PERF_ALERT_COOLDOWN", "60"))
            )
        )

        logger.info(
            "PerformanceMonitor initialized",
            component=component_name,
            window_size=self.window_size,
            percentiles=self.percentiles
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if not self.component_name:
            raise ValueError("Component name is required")

        logger.debug("Config validation passed", component=self.component_name)

    async def start(self) -> None:
        """Start performance monitoring.

        Raises:
            RuntimeError: If monitor is already running
        """
        if self.is_running:
            raise RuntimeError("PerformanceMonitor is already running")

        self.is_running = True
        self._start_time = datetime.now(timezone.utc)
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        logger.info("PerformanceMonitor started", component=self.component_name)

    async def stop(self) -> None:
        """Stop performance monitoring."""
        if not self.is_running:
            logger.warning("PerformanceMonitor is not running", component=self.component_name)
            return

        self.is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "PerformanceMonitor stopped",
            component=self.component_name,
            total_operations=self._total_operations,
            total_errors=self._total_errors
        )

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for periodic checks."""
        check_interval = int(self.config.get("check_interval", os.getenv("PERF_CHECK_INTERVAL", "10")))

        while self.is_running:
            try:
                await asyncio.sleep(check_interval)

                # Check for performance issues
                await self._check_performance_thresholds()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in monitoring loop",
                    component=self.component_name,
                    error=str(e),
                    exc_info=True
                )

    async def _check_performance_thresholds(self) -> None:
        """Check if performance metrics exceed thresholds."""
        try:
            if len(self._latencies) < int(
                self.config.get("min_samples", os.getenv("PERF_MIN_SAMPLES", "10"))
            ):
                return

            metrics = self.get_current_metrics()

            # Check latency threshold
            if metrics.latency_p95 > self._latency_threshold:
                await self._trigger_alert(
                    "high_latency",
                    f"P95 latency {metrics.latency_p95}ms exceeds threshold {self._latency_threshold}ms"
                )

            # Check error rate threshold
            if metrics.error_rate > self._error_rate_threshold:
                await self._trigger_alert(
                    "high_error_rate",
                    f"Error rate {metrics.error_rate}% exceeds threshold {self._error_rate_threshold}%"
                )

        except Exception as e:
            logger.error(
                "Error checking performance thresholds",
                component=self.component_name,
                error=str(e),
                exc_info=True
            )

    async def _trigger_alert(self, alert_type: str, message: str) -> None:
        """Trigger a performance alert.

        Args:
            alert_type: Type of alert
            message: Alert message
        """
        now = datetime.now(timezone.utc)

        # Check cooldown
        if self._last_alert_time:
            if now - self._last_alert_time < self._alert_cooldown:
                logger.debug(
                    "Alert suppressed (cooldown)",
                    component=self.component_name,
                    alert_type=alert_type
                )
                return

        self._last_alert_time = now

        logger.warning(
            "Performance alert",
            component=self.component_name,
            alert_type=alert_type,
            message=message
        )

    def record_operation(
        self,
        latency_ms: Decimal,
        success: bool = True,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record an operation for performance tracking.

        Args:
            latency_ms: Operation latency in milliseconds
            success: Whether operation succeeded
            timestamp: Operation timestamp (defaults to now)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            self._latencies.append(latency_ms)
            self._timestamps.append(timestamp)
            self._errors.append(not success)

            self._total_operations += 1
            if not success:
                self._total_errors += 1

            logger.debug(
                "Operation recorded",
                component=self.component_name,
                latency_ms=str(latency_ms),
                success=success
            )

        except Exception as e:
            logger.error(
                "Error recording operation",
                component=self.component_name,
                error=str(e),
                exc_info=True
            )

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            PerformanceMetrics object with current metrics

        Raises:
            ValueError: If insufficient data for metrics calculation
        """
        if not self._latencies:
            raise ValueError("No performance data available")

        # Calculate percentiles
        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        latency_p50 = sorted_latencies[p50_idx] if p50_idx < n else sorted_latencies[-1]
        latency_p95 = sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1]
        latency_p99 = sorted_latencies[p99_idx] if p99_idx < n else sorted_latencies[-1]

        # Calculate throughput
        if self._start_time and len(self._timestamps) > 0:
            time_window = (datetime.now(timezone.utc) - self._timestamps[0]).total_seconds()
            throughput = Decimal(str(len(self._latencies))) / Decimal(str(max(time_window, 1)))
        else:
            throughput = Decimal("0")

        # Calculate error rate
        error_count = sum(1 for e in self._errors if e)
        error_rate = (Decimal(str(error_count)) / Decimal(str(len(self._errors)))) * Decimal("100")

        return PerformanceMetrics(
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "component": self.component_name,
            "is_running": self.is_running,
            "total_operations": self._total_operations,
            "total_errors": self._total_errors,
            "window_size": self.window_size,
            "current_samples": len(self._latencies),
        }

        try:
            metrics = self.get_current_metrics()
            stats.update({
                "latency_p50_ms": str(metrics.latency_p50),
                "latency_p95_ms": str(metrics.latency_p95),
                "latency_p99_ms": str(metrics.latency_p99),
                "throughput_ops": str(metrics.throughput),
                "error_rate_percent": str(metrics.error_rate),
            })
        except ValueError:
            pass

        if self._start_time:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            stats["uptime_seconds"] = uptime

        return stats

    def reset(self) -> None:
        """Reset all performance metrics."""
        logger.info("Resetting performance metrics", component=self.component_name)

        self._latencies.clear()
        self._timestamps.clear()
        self._errors.clear()
        self._total_operations = 0
        self._total_errors = 0
        self._start_time = datetime.now(timezone.utc) if self.is_running else None
        self._last_alert_time = None


class PerformanceMonitorRegistry:
    """Registry for managing multiple performance monitors.

    Example:
        >>> config = {"window_size": 1000}
        >>> registry = PerformanceMonitorRegistry(config)
        >>> monitor = registry.get_or_create_monitor("order_executor")
        >>> monitor.record_operation(latency_ms=Decimal("5.2"))
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize monitor registry.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._monitors: Dict[str, PerformanceMonitor] = {}

        logger.info("PerformanceMonitorRegistry initialized")

    def get_or_create_monitor(self, component_name: str) -> PerformanceMonitor:
        """Get or create a performance monitor for a component.

        Args:
            component_name: Name of component

        Returns:
            PerformanceMonitor instance
        """
        if component_name not in self._monitors:
            self._monitors[component_name] = PerformanceMonitor(
                self.config,
                component_name
            )
            logger.info("Created new monitor", component=component_name)

        return self._monitors[component_name]

    def get_monitor(self, component_name: str) -> Optional[PerformanceMonitor]:
        """Get a performance monitor by component name.

        Args:
            component_name: Name of component

        Returns:
            PerformanceMonitor instance or None if not found
        """
        return self._monitors.get(component_name)

    def get_all_monitors(self) -> Dict[str, PerformanceMonitor]:
        """Get all registered monitors.

        Returns:
            Dictionary of component_name -> monitor
        """
        return self._monitors.copy()

    async def start_all(self) -> None:
        """Start all registered monitors."""
        for monitor in self._monitors.values():
            if not monitor.is_running:
                await monitor.start()

        logger.info("All monitors started", count=len(self._monitors))

    async def stop_all(self) -> None:
        """Stop all registered monitors."""
        for monitor in self._monitors.values():
            if monitor.is_running:
                await monitor.stop()

        logger.info("All monitors stopped", count=len(self._monitors))
