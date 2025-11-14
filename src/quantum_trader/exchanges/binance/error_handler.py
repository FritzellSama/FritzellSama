"""Binance exchange error handling and recovery.

This module provides comprehensive error handling, retry logic, and circuit breaking
for Binance exchange API interactions.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, TypeVar
from datetime import datetime, timedelta
import os
from structlog import get_logger

from .constants import (
    BinanceErrorCode,
    RETRYABLE_ERROR_CODES,
    FATAL_ERROR_CODES,
    MAX_RETRIES,
    RETRY_DELAY_MS,
    BACKOFF_MULTIPLIER,
)

logger = get_logger(__name__)

T = TypeVar("T")


class BinanceError(Exception):
    """Base exception for Binance errors."""

    def __init__(
        self, message: str, code: Optional[str] = None, response: Optional[Dict] = None
    ) -> None:
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class BinanceAuthError(BinanceError):
    """Authentication related errors."""

    pass


class BinanceRateLimitError(BinanceError):
    """Rate limit exceeded errors."""

    pass


class BinanceOrderError(BinanceError):
    """Order related errors."""

    pass


class BinanceMarketError(BinanceError):
    """Market data related errors."""

    pass


class BinanceSystemError(BinanceError):
    """System and service errors."""

    pass


class CircuitBreaker:
    """Circuit breaker for Binance API calls.

    Implements the circuit breaker pattern to prevent cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        expected_exception: type = BinanceError,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = int(
            os.getenv("BINANCE_CB_FAILURE_THRESHOLD", str(failure_threshold))
        )
        self.recovery_timeout = int(
            os.getenv("BINANCE_CB_RECOVERY_TIMEOUT", str(recovery_timeout))
        )
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

        logger.info(
            "circuit_breaker_initialized",
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
        )

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("circuit_breaker_half_open")
            else:
                logger.warning("circuit_breaker_open", state=self.state)
                raise BinanceSystemError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True

        time_since_failure = (
            datetime.utcnow() - self.last_failure_time
        ).total_seconds()
        return time_since_failure >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            logger.info("circuit_breaker_closed")
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )


class BinanceErrorHandler:
    """Handles Binance API errors with retry logic and circuit breaking."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("BINANCE_CB_FAILURE_THRESHOLD", "5")),
            recovery_timeout=int(os.getenv("BINANCE_CB_RECOVERY_TIMEOUT", "60")),
        )

    def parse_error(self, response: Dict[str, Any]) -> BinanceError:
        """Parse Binance API error response.

        Args:
            response: API error response

        Returns:
            Appropriate BinanceError subclass
        """
        code = response.get("code")
        msg = response.get("msg", "Unknown error")

        logger.error("binance_api_error", code=code, message=msg, response=response)

        # Map error codes to exception types
        if code in [
            BinanceErrorCode.INVALID_API_KEY.value,
            BinanceErrorCode.INVALID_SIGNATURE.value,
            BinanceErrorCode.INVALID_TIMESTAMP.value,
        ]:
            return BinanceAuthError(msg, str(code), response)

        if code in [
            BinanceErrorCode.RATE_LIMIT_EXCEEDED.value,
            BinanceErrorCode.WAF_LIMIT_VIOLATED.value,
        ]:
            return BinanceRateLimitError(msg, str(code), response)

        if code in [
            BinanceErrorCode.INSUFFICIENT_BALANCE.value,
            BinanceErrorCode.INVALID_ORDER_TYPE.value,
            BinanceErrorCode.INVALID_QUANTITY.value,
        ]:
            return BinanceOrderError(msg, str(code), response)

        if code in [
            BinanceErrorCode.SYMBOL_NOT_FOUND.value,
            BinanceErrorCode.MARKET_NOT_TRADING.value,
        ]:
            return BinanceMarketError(msg, str(code), response)

        if code in [
            BinanceErrorCode.SYSTEM_ERROR.value,
            BinanceErrorCode.SERVICE_UNAVAILABLE.value,
        ]:
            return BinanceSystemError(msg, str(code), response)

        return BinanceError(msg, str(code), response)

    def is_retryable(self, error: BinanceError) -> bool:
        """Check if error should be retried.

        Args:
            error: Binance error

        Returns:
            True if error is retryable
        """
        if error.code is None:
            return False

        try:
            code_int = int(error.code)
            return code_int in RETRYABLE_ERROR_CODES
        except (ValueError, TypeError):
            return error.code in RETRYABLE_ERROR_CODES

    def is_fatal(self, error: BinanceError) -> bool:
        """Check if error is fatal (should not retry).

        Args:
            error: Binance error

        Returns:
            True if error is fatal
        """
        if error.code is None:
            return False

        try:
            code_int = int(error.code)
            return code_int in FATAL_ERROR_CODES
        except (ValueError, TypeError):
            return error.code in FATAL_ERROR_CODES

    async def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args: Any,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with exponential backoff retry.

        Args:
            func: Async function to execute
            *args: Positional arguments
            max_retries: Maximum retry attempts (uses config if None)
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            BinanceError: If all retries exhausted or fatal error
        """
        if max_retries is None:
            max_retries = MAX_RETRIES

        last_exception: Optional[Exception] = None
        retry_delay = Decimal(RETRY_DELAY_MS) / Decimal("1000")

        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info("retry_succeeded", attempt=attempt)
                return result

            except BinanceError as e:
                last_exception = e

                if self.is_fatal(e):
                    logger.error("fatal_error_no_retry", error=str(e), code=e.code)
                    raise

                if attempt >= max_retries:
                    logger.error(
                        "max_retries_exceeded",
                        attempt=attempt,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                if self.is_retryable(e):
                    wait_time = float(retry_delay * (BACKOFF_MULTIPLIER**attempt))
                    logger.warning(
                        "retrying_after_error",
                        attempt=attempt,
                        wait_seconds=wait_time,
                        error=str(e),
                        code=e.code,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("non_retryable_error", error=str(e), code=e.code)
                    raise

            except Exception as e:
                last_exception = e
                logger.error(
                    "unexpected_error",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(float(retry_delay))

        if last_exception:
            raise last_exception
        raise BinanceSystemError("Retry loop completed without result")

    def get_retry_after(self, error: BinanceError) -> Optional[int]:
        """Extract retry-after value from rate limit error.

        Args:
            error: Binance error

        Returns:
            Seconds to wait before retry, or None
        """
        if not isinstance(error, BinanceRateLimitError):
            return None

        if error.response and "retryAfter" in error.response:
            try:
                return int(error.response["retryAfter"])
            except (ValueError, TypeError):
                pass

        # Default rate limit backoff
        return int(os.getenv("BINANCE_RATE_LIMIT_BACKOFF", "60"))

    async def handle_rate_limit(self, error: BinanceRateLimitError) -> None:
        """Handle rate limit error with appropriate backoff.

        Args:
            error: Rate limit error
        """
        retry_after = self.get_retry_after(error)
        if retry_after:
            logger.warning(
                "rate_limit_backoff", retry_after_seconds=retry_after, error=str(error)
            )
            await asyncio.sleep(retry_after)
        else:
            default_wait = int(os.getenv("BINANCE_RATE_LIMIT_DEFAULT_WAIT", "10"))
            logger.warning("rate_limit_default_wait", wait_seconds=default_wait)
            await asyncio.sleep(default_wait)


# Global error handler instance
error_handler = BinanceErrorHandler()
