"""Async Utility Functions.

Production-ready async helpers for concurrent operations, retry logic,
circuit breakers, and graceful degradation in distributed systems.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable, TypeVar, Coroutine, Union
from datetime import datetime, timezone, timedelta
from functools import wraps
from enum import Enum
import time
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Failing, reject requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for fault tolerance.

    Prevents cascading failures by stopping requests to failing services.
    Automatically attempts recovery after timeout period.

    Attributes:
        failure_threshold: Max failures before opening circuit
        timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to track
        state: Current circuit state
    """

    def __init__(
        self,
        failure_threshold: int,
        timeout: float,
        expected_exception: type = Exception
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            timeout: Timeout in seconds before recovery attempt
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("circuit_breaker_half_open", timeout=self.timeout)
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)

            if self.state == CircuitState.HALF_OPEN:
                self.reset()
                logger.info("circuit_breaker_closed_recovered")

            return result

        except self.expected_exception as e:
            self.record_failure()
            logger.error(
                "circuit_breaker_failure",
                state=self.state.value,
                failures=self.failure_count,
                error=str(e)
            )
            raise

    def record_failure(self) -> None:
        """Record a failure and update circuit state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                failures=self.failure_count,
                threshold=self.failure_threshold
            )

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED


async def retry_with_exponential_backoff(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs: Any
) -> T:
    """Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        Exception: If all retries exhausted
    """
    import random

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0:
                logger.info("retry_succeeded", attempt=attempt)
            return result

        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    "retry_exhausted",
                    attempts=attempt + 1,
                    error=str(e),
                    exc_info=True
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter if enabled
            if jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e)
            )

            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise Exception("Retry failed with no exception")


async def gather_with_concurrency(
    limit: int,
    *tasks: Coroutine[Any, Any, T],
    return_exceptions: bool = False
) -> List[Union[T, Exception]]:
    """Execute coroutines with concurrency limit.

    Args:
        limit: Maximum concurrent tasks
        *tasks: Coroutines to execute
        return_exceptions: Return exceptions instead of raising

    Returns:
        List of results

    Raises:
        Exception: If any task fails and return_exceptions is False
    """
    semaphore = asyncio.Semaphore(limit)

    async def bounded_task(task: Coroutine[Any, Any, T]) -> Union[T, Exception]:
        async with semaphore:
            try:
                return await task
            except Exception as e:
                if return_exceptions:
                    return e
                raise

    results = await asyncio.gather(
        *[bounded_task(task) for task in tasks],
        return_exceptions=return_exceptions
    )

    return results  # type: ignore


async def timeout_wrapper(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    default_value: Optional[T] = None
) -> Optional[T]:
    """Wrap coroutine with timeout.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        default_value: Value to return on timeout

    Returns:
        Result or default value

    Raises:
        asyncio.TimeoutError: If timeout and no default value
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        logger.warning("operation_timeout", timeout=timeout_seconds)
        if default_value is not None:
            return default_value
        raise


def async_cached(ttl_seconds: float) -> Callable:
    """Decorator for caching async function results with TTL.

    Args:
        ttl_seconds: Time-to-live in seconds

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        cache: Dict[str, tuple[T, float]] = {}

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from args and kwargs
            cache_key = f"{args}:{sorted(kwargs.items())}"

            # Check cache
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    logger.debug("cache_hit", func=func.__name__, key=cache_key)
                    return result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache[cache_key] = (result, time.time())
            logger.debug("cache_miss", func=func.__name__, key=cache_key)

            return result

        return wrapper

    return decorator


async def run_with_rate_limit(
    func: Callable[..., Coroutine[Any, Any, T]],
    rate_limit: int,
    period_seconds: float,
    *args: Any,
    **kwargs: Any
) -> T:
    """Execute function with rate limiting.

    Args:
        func: Async function to execute
        rate_limit: Max calls per period
        period_seconds: Period in seconds
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    # Simple token bucket implementation
    if not hasattr(run_with_rate_limit, '_buckets'):
        run_with_rate_limit._buckets: Dict[str, List[float]] = {}  # type: ignore

    func_key = f"{func.__module__}.{func.__name__}"

    if func_key not in run_with_rate_limit._buckets:  # type: ignore
        run_with_rate_limit._buckets[func_key] = []  # type: ignore

    bucket = run_with_rate_limit._buckets[func_key]  # type: ignore

    # Remove old timestamps
    current_time = time.time()
    bucket[:] = [ts for ts in bucket if current_time - ts < period_seconds]

    # Wait if rate limit exceeded
    while len(bucket) >= rate_limit:
        sleep_time = period_seconds - (current_time - bucket[0])
        if sleep_time > 0:
            logger.debug(
                "rate_limit_wait",
                func=func.__name__,
                sleep=sleep_time
            )
            await asyncio.sleep(sleep_time)

        current_time = time.time()
        bucket[:] = [ts for ts in bucket if current_time - ts < period_seconds]

    # Add timestamp and execute
    bucket.append(current_time)
    return await func(*args, **kwargs)


async def wait_for_condition(
    condition: Callable[[], bool],
    timeout_seconds: float,
    poll_interval: float = 0.1
) -> bool:
    """Wait for condition to become true.

    Args:
        condition: Callable that returns bool
        timeout_seconds: Maximum wait time
        poll_interval: Check interval in seconds

    Returns:
        True if condition met, False if timeout

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        if condition():
            return True
        await asyncio.sleep(poll_interval)

    logger.warning("condition_timeout", timeout=timeout_seconds)
    raise asyncio.TimeoutError("Condition not met within timeout")


async def graceful_shutdown(
    tasks: List[asyncio.Task],
    timeout_seconds: float = 30.0
) -> None:
    """Gracefully shutdown async tasks.

    Args:
        tasks: List of tasks to cancel
        timeout_seconds: Max time to wait for cleanup
    """
    logger.info("graceful_shutdown_initiated", task_count=len(tasks))

    # Cancel all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for tasks to complete with timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout_seconds
        )
        logger.info("graceful_shutdown_completed")
    except asyncio.TimeoutError:
        logger.warning(
            "graceful_shutdown_timeout",
            timeout=timeout_seconds,
            remaining=sum(1 for t in tasks if not t.done())
        )


async def batch_process(
    items: List[Any],
    processor: Callable[[Any], Coroutine[Any, Any, T]],
    batch_size: int,
    delay_between_batches: float = 0.0
) -> List[T]:
    """Process items in batches.

    Args:
        items: Items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        delay_between_batches: Delay in seconds between batches

    Returns:
        List of results
    """
    results: List[T] = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]

        logger.debug(
            "batch_processing",
            batch_num=i // batch_size + 1,
            batch_size=len(batch),
            total=len(items)
        )

        batch_results = await asyncio.gather(
            *[processor(item) for item in batch],
            return_exceptions=False
        )

        results.extend(batch_results)

        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results


class AsyncLock:
    """Async lock with timeout and ownership tracking.

    Attributes:
        name: Lock identifier
        timeout: Max time to wait for lock
    """

    def __init__(self, name: str, timeout: float = 30.0) -> None:
        """Initialize async lock.

        Args:
            name: Lock identifier
            timeout: Timeout in seconds
        """
        self.name = name
        self.timeout = timeout
        self._lock = asyncio.Lock()
        self._owner: Optional[str] = None

    async def acquire(self, owner: str) -> bool:
        """Acquire lock with timeout.

        Args:
            owner: Owner identifier

        Returns:
            True if acquired

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self.timeout)
            self._owner = owner
            logger.debug("lock_acquired", lock=self.name, owner=owner)
            return True
        except asyncio.TimeoutError:
            logger.error(
                "lock_timeout",
                lock=self.name,
                owner=owner,
                timeout=self.timeout
            )
            raise

    def release(self) -> None:
        """Release lock."""
        if self._lock.locked():
            owner = self._owner
            self._owner = None
            self._lock.release()
            logger.debug("lock_released", lock=self.name, owner=owner)

    async def __aenter__(self) -> 'AsyncLock':
        """Context manager entry."""
        await self.acquire("context_manager")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.release()


async def parallel_map(
    func: Callable[[T], Coroutine[Any, Any, Any]],
    items: List[T],
    max_concurrency: int = 10
) -> List[Any]:
    """Map function over items in parallel with concurrency limit.

    Args:
        func: Async function to apply
        items: Items to process
        max_concurrency: Max concurrent executions

    Returns:
        List of results in same order as items
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_func(item: T) -> Any:
        async with semaphore:
            return await func(item)

    results = await asyncio.gather(*[bounded_func(item) for item in items])
    return results


def run_in_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> Coroutine[Any, Any, T]:
    """Run blocking function in executor.

    Useful for CPU-bound or blocking I/O operations that
    can't be made async.

    Args:
        func: Blocking function
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Coroutine that runs function in executor
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))
