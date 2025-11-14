"""
Decorators - Common decorators for the trading system.

This module provides reusable decorators for retry logic, caching,
rate limiting, timing, and error handling.
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional, Dict
from decimal import Decimal
from datetime import datetime, timedelta
from structlog import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int,
    backoff_base: float,
    max_backoff: float,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        backoff_base: Base delay for backoff (seconds)
        max_backoff: Maximum backoff delay (seconds)
        exceptions: Tuple of exceptions to catch

    Example:
        >>> @retry(max_attempts=3, backoff_base=1.0, max_backoff=10.0)
        >>> async def fetch_data():
        ...     # May fail and retry
        ...     return data
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            error=str(e)
                        )
                        raise

                    # Calculate backoff delay
                    delay = min(backoff_base * (2 ** (attempt - 1)), max_backoff)

                    logger.warning(
                        f"{func.__name__} failed, retrying",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e)
                    )

                    await asyncio.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            error=str(e)
                        )
                        raise

                    delay = min(backoff_base * (2 ** (attempt - 1)), max_backoff)

                    logger.warning(
                        f"{func.__name__} failed, retrying",
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=str(e)
                    )

                    time.sleep(delay)

            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def timed(log_level: str = 'debug'):
    """
    Timing decorator to log function execution time.

    Args:
        log_level: Log level for timing messages ('debug', 'info', etc.)

    Example:
        >>> @timed()
        >>> async def expensive_operation():
        ...     # Some time-consuming code
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time

                log_func = getattr(logger, log_level)
                log_func(
                    f"{func.__name__} completed",
                    duration_seconds=round(elapsed, 4)
                )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time

                log_func = getattr(logger, log_level)
                log_func(
                    f"{func.__name__} completed",
                    duration_seconds=round(elapsed, 4)
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cached(ttl_seconds: int = 300):
    """
    Simple caching decorator with TTL.

    Args:
        ttl_seconds: Time-to-live for cached values (seconds)

    Example:
        >>> @cached(ttl_seconds=60)
        >>> async def get_market_data(symbol: str):
        ...     # Expensive API call
        ...     return data
    """
    cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Create cache key from arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check cache
            if cache_key in cache:
                value, expiry = cache[cache_key]
                if datetime.utcnow() < expiry:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return value

            # Call function and cache result
            result = await func(*args, **kwargs)
            expiry = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            cache[cache_key] = (result, expiry)

            logger.debug(f"Cached result for {func.__name__}")
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            if cache_key in cache:
                value, expiry = cache[cache_key]
                if datetime.utcnow() < expiry:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return value

            result = func(*args, **kwargs)
            expiry = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            cache[cache_key] = (result, expiry)

            logger.debug(f"Cached result for {func.__name__}")
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def rate_limit(calls: int, period: float):
    """
    Rate limiting decorator.

    Args:
        calls: Maximum number of calls
        period: Time period in seconds

    Example:
        >>> @rate_limit(calls=10, period=1.0)
        >>> async def api_call():
        ...     # Rate-limited API call
        ...     pass
    """
    call_times = []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            nonlocal call_times

            now = time.time()

            # Remove old calls outside the window
            call_times = [t for t in call_times if now - t < period]

            # Check rate limit
            if len(call_times) >= calls:
                oldest = call_times[0]
                wait_time = period - (now - oldest)

                if wait_time > 0:
                    logger.debug(
                        f"Rate limit reached for {func.__name__}, waiting",
                        wait_seconds=round(wait_time, 2)
                    )
                    await asyncio.sleep(wait_time)

                # Clean up again after waiting
                now = time.time()
                call_times = [t for t in call_times if now - t < period]

            # Record this call
            call_times.append(now)

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            nonlocal call_times

            now = time.time()
            call_times = [t for t in call_times if now - t < period]

            if len(call_times) >= calls:
                oldest = call_times[0]
                wait_time = period - (now - oldest)

                if wait_time > 0:
                    logger.debug(
                        f"Rate limit reached for {func.__name__}, waiting",
                        wait_seconds=round(wait_time, 2)
                    )
                    time.sleep(wait_time)

                now = time.time()
                call_times = [t for t in call_times if now - t < period]

            call_times.append(now)
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def validate_args(**validators):
    """
    Argument validation decorator.

    Args:
        **validators: Keyword arguments mapping param names to validator functions

    Example:
        >>> def positive(x):
        ...     if x <= 0:
        ...         raise ValueError("Must be positive")
        >>> @validate_args(amount=positive)
        >>> def place_order(amount):
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    validator(value)

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    validator(value)

            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_exceptions(reraise: bool = True):
    """
    Exception logging decorator.

    Args:
        reraise: Whether to reraise the exception after logging

    Example:
        >>> @log_exceptions()
        >>> async def risky_operation():
        ...     # May raise exceptions
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}",
                    error=str(e),
                    error_type=type(e).__name__
                )
                if reraise:
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Exception in {func.__name__}",
                    error=str(e),
                    error_type=type(e).__name__
                )
                if reraise:
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def deprecated(alternative: Optional[str] = None):
    """
    Mark function as deprecated.

    Args:
        alternative: Suggested alternative function name

    Example:
        >>> @deprecated(alternative="new_function")
        >>> def old_function():
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            message = f"{func.__name__} is deprecated"
            if alternative:
                message += f", use {alternative} instead"

            logger.warning(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator
