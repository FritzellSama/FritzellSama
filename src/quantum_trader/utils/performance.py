"""Performance Monitoring and Profiling Utilities.

Production-ready performance monitoring, profiling, and optimization tools
for tracking execution time, memory usage, and system resource utilization.
"""

import time
import functools
import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, TypeVar, List
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics container.

    Attributes:
        operation: Operation name
        start_time: Start timestamp
        end_time: End timestamp
        duration_ms: Duration in milliseconds
        success: Whether operation succeeded
        error: Error message if failed
        metadata: Additional metadata
    """
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[Decimal] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, error: Optional[str] = None) -> None:
        """Mark metrics as complete.

        Args:
            success: Whether operation succeeded
            error: Optional error message
        """
        self.end_time = datetime.now(timezone.utc)
        self.success = success
        self.error = error

        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds() * 1000
            self.duration_ms = Decimal(str(duration))


class PerformanceTracker:
    """Track performance metrics across operations.

    Attributes:
        metrics: List of recorded metrics
        enabled: Whether tracking is enabled
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize performance tracker.

        Args:
            enabled: Enable tracking
        """
        self.metrics: List[PerformanceMetrics] = []
        self.enabled = enabled

    def start(self, operation: str, **metadata: Any) -> PerformanceMetrics:
        """Start tracking an operation.

        Args:
            operation: Operation name
            **metadata: Additional metadata

        Returns:
            PerformanceMetrics object
        """
        if not self.enabled:
            return PerformanceMetrics(operation=operation, start_time=datetime.now(timezone.utc))

        metrics = PerformanceMetrics(
            operation=operation,
            start_time=datetime.now(timezone.utc),
            metadata=metadata
        )

        self.metrics.append(metrics)

        logger.debug("performance_tracking_started", operation=operation)

        return metrics

    def get_metrics(
        self,
        operation: Optional[str] = None
    ) -> List[PerformanceMetrics]:
        """Get recorded metrics.

        Args:
            operation: Optional filter by operation name

        Returns:
            List of metrics
        """
        if operation:
            return [m for m in self.metrics if m.operation == operation]
        return self.metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.

        Returns:
            Summary dictionary
        """
        if not self.metrics:
            return {"total_operations": 0}

        total = len(self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        failed = total - successful

        durations = [
            float(m.duration_ms)
            for m in self.metrics
            if m.duration_ms is not None
        ]

        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = 0
            min_duration = 0
            max_duration = 0

        return {
            "total_operations": total,
            "successful": successful,
            "failed": failed,
            "avg_duration_ms": round(avg_duration, 2),
            "min_duration_ms": round(min_duration, 2),
            "max_duration_ms": round(max_duration, 2)
        }

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        logger.debug("performance_metrics_cleared")


# Global tracker instance
_global_tracker = PerformanceTracker()


def get_global_tracker() -> PerformanceTracker:
    """Get global performance tracker.

    Returns:
        Global tracker instance
    """
    return _global_tracker


@contextmanager
def measure_time(operation: str, **metadata: Any):
    """Context manager to measure execution time.

    Args:
        operation: Operation name
        **metadata: Additional metadata

    Yields:
        PerformanceMetrics object

    Example:
        with measure_time("database_query", query_type="SELECT"):
            # Your code here
            pass
    """
    metrics = _global_tracker.start(operation, **metadata)

    try:
        yield metrics
        metrics.complete(success=True)

        logger.info(
            "operation_completed",
            operation=operation,
            duration_ms=str(metrics.duration_ms)
        )

    except Exception as e:
        metrics.complete(success=False, error=str(e))

        logger.error(
            "operation_failed",
            operation=operation,
            duration_ms=str(metrics.duration_ms) if metrics.duration_ms else None,
            error=str(e)
        )

        raise


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function

    Example:
        @timeit
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        operation = f"{func.__module__}.{func.__name__}"

        with measure_time(operation):
            result = func(*args, **kwargs)

        return result

    return wrapper


def async_timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure async function execution time.

    Args:
        func: Async function to measure

    Returns:
        Wrapped async function

    Example:
        @async_timeit
        async def my_async_function():
            pass
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        operation = f"{func.__module__}.{func.__name__}"

        with measure_time(operation):
            result = await func(*args, **kwargs)

        return result

    return wrapper


class RateLimiter:
    """Rate limiter for controlling operation frequency.

    Attributes:
        max_calls: Maximum calls per period
        period_seconds: Period in seconds
    """

    def __init__(self, max_calls: int, period_seconds: float) -> None:
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls per period
            period_seconds: Period in seconds
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls: List[float] = []

    def is_allowed(self) -> bool:
        """Check if operation is allowed.

        Returns:
            True if allowed
        """
        current_time = time.time()

        # Remove old calls
        self.calls = [
            call_time for call_time in self.calls
            if current_time - call_time < self.period_seconds
        ]

        # Check limit
        if len(self.calls) < self.max_calls:
            self.calls.append(current_time)
            return True

        return False

    async def wait_if_needed(self) -> None:
        """Wait if rate limit exceeded."""
        while not self.is_allowed():
            # Calculate wait time
            if self.calls:
                oldest_call = self.calls[0]
                wait_time = self.period_seconds - (time.time() - oldest_call)
                if wait_time > 0:
                    logger.debug("rate_limit_wait", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)
            else:
                break


class Throttle:
    """Throttle for limiting operation execution rate.

    Attributes:
        min_interval_seconds: Minimum interval between operations
    """

    def __init__(self, min_interval_seconds: float) -> None:
        """Initialize throttle.

        Args:
            min_interval_seconds: Minimum interval in seconds
        """
        self.min_interval = min_interval_seconds
        self.last_call: Optional[float] = None

    async def wait(self) -> None:
        """Wait if necessary to maintain minimum interval."""
        if self.last_call is not None:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                logger.debug("throttle_wait", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

        self.last_call = time.time()


class MemoryMonitor:
    """Monitor memory usage.

    Attributes:
        enabled: Whether monitoring is enabled
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize memory monitor.

        Args:
            enabled: Enable monitoring
        """
        self.enabled = enabled

    def get_process_memory(self) -> Optional[Dict[str, int]]:
        """Get current process memory usage.

        Returns:
            Dictionary with memory info or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                "rss_bytes": memory_info.rss,
                "vms_bytes": memory_info.vms,
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024)
            }

        except ImportError:
            logger.warning("psutil_not_available")
            return None
        except Exception as e:
            logger.error("memory_monitor_failed", error=str(e))
            return None

    def log_memory_usage(self, label: str = "current") -> None:
        """Log current memory usage.

        Args:
            label: Label for log entry
        """
        memory = self.get_process_memory()

        if memory:
            logger.info(
                "memory_usage",
                label=label,
                rss_mb=round(memory["rss_mb"], 2),
                vms_mb=round(memory["vms_mb"], 2)
            )


class CPUMonitor:
    """Monitor CPU usage.

    Attributes:
        enabled: Whether monitoring is enabled
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize CPU monitor.

        Args:
            enabled: Enable monitoring
        """
        self.enabled = enabled

    def get_cpu_usage(self, interval: float = 1.0) -> Optional[float]:
        """Get CPU usage percentage.

        Args:
            interval: Measurement interval in seconds

        Returns:
            CPU usage percentage or None if unavailable
        """
        if not self.enabled:
            return None

        try:
            import psutil

            return psutil.cpu_percent(interval=interval)

        except ImportError:
            logger.warning("psutil_not_available")
            return None
        except Exception as e:
            logger.error("cpu_monitor_failed", error=str(e))
            return None

    def log_cpu_usage(self, label: str = "current") -> None:
        """Log current CPU usage.

        Args:
            label: Label for log entry
        """
        cpu_usage = self.get_cpu_usage(interval=0.1)

        if cpu_usage is not None:
            logger.info(
                "cpu_usage",
                label=label,
                percent=round(cpu_usage, 2)
            )


class Profiler:
    """Simple profiler for code sections.

    Attributes:
        results: Profiling results
    """

    def __init__(self) -> None:
        """Initialize profiler."""
        self.results: Dict[str, List[float]] = {}

    @contextmanager
    def profile(self, section: str):
        """Profile a code section.

        Args:
            section: Section name

        Yields:
            None

        Example:
            profiler = Profiler()
            with profiler.profile("computation"):
                # Your code here
                pass
        """
        start = time.perf_counter()

        try:
            yield

        finally:
            elapsed = time.perf_counter() - start

            if section not in self.results:
                self.results[section] = []

            self.results[section].append(elapsed)

    def get_results(self) -> Dict[str, Dict[str, float]]:
        """Get profiling results.

        Returns:
            Dictionary with timing statistics
        """
        summary = {}

        for section, times in self.results.items():
            if times:
                summary[section] = {
                    "calls": len(times),
                    "total_seconds": sum(times),
                    "avg_seconds": sum(times) / len(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times)
                }

        return summary

    def print_results(self) -> None:
        """Print profiling results."""
        results = self.get_results()

        if not results:
            logger.info("No profiling results")
            return

        for section, stats in results.items():
            logger.info(
                "profiling_result",
                section=section,
                calls=stats["calls"],
                total_ms=round(stats["total_seconds"] * 1000, 2),
                avg_ms=round(stats["avg_seconds"] * 1000, 2),
                min_ms=round(stats["min_seconds"] * 1000, 2),
                max_ms=round(stats["max_seconds"] * 1000, 2)
            )

    def clear(self) -> None:
        """Clear profiling results."""
        self.results.clear()


def benchmark(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """Benchmark function execution.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Benchmark statistics
    """
    try:
        # Warmup
        for _ in range(warmup):
            func()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "iterations": iterations,
            "total_seconds": sum(times),
            "avg_seconds": sum(times) / len(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "avg_ms": (sum(times) / len(times)) * 1000
        }

    except Exception as e:
        logger.error("benchmark_failed", error=str(e))
        raise
