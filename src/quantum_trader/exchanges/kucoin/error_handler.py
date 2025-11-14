"""KuCoin exchange error handling and recovery.

This module provides comprehensive error handling, retry logic, and circuit breaking
for KuCoin exchange API interactions.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, TypeVar
from datetime import datetime
import os
from structlog import get_logger

from .constants import (
    KuCoinErrorCode,
    RETRYABLE_ERROR_CODES,
    FATAL_ERROR_CODES,
    MAX_RETRIES,
    RETRY_DELAY_MS,
    BACKOFF_MULTIPLIER,
)

logger = get_logger(__name__)

T = TypeVar("T")


class KuCoinError(Exception):
    """Base exception for KuCoin errors."""

    def __init__(
        self, message: str, code: Optional[str] = None, response: Optional[Dict] = None
    ) -> None:
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class KuCoinAuthError(KuCoinError):
    """Authentication related errors."""

    pass


class KuCoinRateLimitError(KuCoinError):
    """Rate limit exceeded errors."""

    pass


class KuCoinOrderError(KuCoinError):
    """Order related errors."""

    pass


class KuCoinMarketError(KuCoinError):
    """Market data related errors."""

    pass


class KuCoinSystemError(KuCoinError):
    """System and service errors."""

    pass


class CircuitBreaker:
    """Circuit breaker for KuCoin API calls.

    Implements the circuit breaker pattern to prevent cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        expected_exception: type = KuCoinError,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = int(
            os.getenv("KUCOIN_CB_FAILURE_THRESHOLD", str(failure_threshold))
        )
        self.recovery_timeout = int(
            os.getenv("KUCOIN_CB_RECOVERY_TIMEOUT", str(recovery_timeout))
        )
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "CLOSED"

        logger.info(
            "kucoin_circuit_breaker_initialized",
            failure_threshold=self.failure_threshold,
            recovery_timeout=self.recovery_timeout,
        )

    async def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
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
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("kucoin_circuit_breaker_half_open")
            else:
                logger.warning("kucoin_circuit_breaker_open", state=self.state)
                raise KuCoinSystemError("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
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
            logger.info("kucoin_circuit_breaker_closed")
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                "kucoin_circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )


class KuCoinErrorHandler:
    """Handles KuCoin API errors with retry logic and circuit breaking."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("KUCOIN_CB_FAILURE_THRESHOLD", "5")),
            recovery_timeout=int(os.getenv("KUCOIN_CB_RECOVERY_TIMEOUT", "60")),
        )

    def parse_error(self, response: Dict[str, Any]) -> KuCoinError:
        """Parse KuCoin API error response.

        Args:
            response: API error response

        Returns:
            Appropriate KuCoinError subclass
        """
        code = response.get("code")
        msg = response.get("msg", "Unknown error")

        logger.error("kucoin_api_error", code=code, message=msg, response=response)

        # Map error codes to exception types
        if code in [
            KuCoinErrorCode.INVALID_API_KEY.value,
            KuCoinErrorCode.INVALID_SIGNATURE.value,
            KuCoinErrorCode.INVALID_PASSPHRASE.value,
            KuCoinErrorCode.PERMISSION_DENIED.value,
        ]:
            return KuCoinAuthError(msg, str(code), response)

        if code in [
            KuCoinErrorCode.RATE_LIMIT_EXCEEDED.value,
            KuCoinErrorCode.TOO_MANY_REQUESTS.value,
        ]:
            return KuCoinRateLimitError(msg, str(code), response)

        if code in [
            KuCoinErrorCode.INSUFFICIENT_BALANCE.value,
            KuCoinErrorCode.INVALID_ORDER_SIZE.value,
            KuCoinErrorCode.BALANCE_INSUFFICIENT.value,
        ]:
            return KuCoinOrderError(msg, str(code), response)

        if code in [
            KuCoinErrorCode.SYMBOL_NOT_FOUND.value,
            KuCoinErrorCode.MARKET_NOT_AVAILABLE.value,
            KuCoinErrorCode.MARKET_SUSPENDED.value,
        ]:
            return KuCoinMarketError(msg, str(code), response)

        if code in [
            KuCoinErrorCode.SYSTEM_ERROR.value,
            KuCoinErrorCode.SERVICE_UNAVAILABLE.value,
            KuCoinErrorCode.SYSTEM_BUSY.value,
        ]:
            return KuCoinSystemError(msg, str(code), response)

        return KuCoinError(msg, str(code), response)

    def is_retryable(self, error: KuCoinError) -> bool:
        """Check if error should be retried.

        Args:
            error: KuCoin error

        Returns:
            True if error is retryable
        """
        if error.code is None:
            return False

        return error.code in RETRYABLE_ERROR_CODES

    def is_fatal(self, error: KuCoinError) -> bool:
        """Check if error is fatal (should not retry).

        Args:
            error: KuCoin error

        Returns:
            True if error is fatal
        """
        if error.code is None:
            return False

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
            KuCoinError: If all retries exhausted or fatal error
        """
        if max_retries is None:
            max_retries = MAX_RETRIES

        last_exception: Optional[Exception] = None
        retry_delay = Decimal(RETRY_DELAY_MS) / Decimal("1000")

        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info("kucoin_retry_succeeded", attempt=attempt)
                return result

            except KuCoinError as e:
                last_exception = e

                if self.is_fatal(e):
                    logger.error(
                        "kucoin_fatal_error_no_retry", error=str(e), code=e.code
                    )
                    raise

                if attempt >= max_retries:
                    logger.error(
                        "kucoin_max_retries_exceeded",
                        attempt=attempt,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                if self.is_retryable(e):
                    wait_time = float(retry_delay * (BACKOFF_MULTIPLIER**attempt))
                    logger.warning(
                        "kucoin_retrying_after_error",
                        attempt=attempt,
                        wait_seconds=wait_time,
                        error=str(e),
                        code=e.code,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "kucoin_non_retryable_error", error=str(e), code=e.code
                    )
                    raise

            except Exception as e:
                last_exception = e
                logger.error(
                    "kucoin_unexpected_error",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(float(retry_delay))

        if last_exception:
            raise last_exception
        raise KuCoinSystemError("Retry loop completed without result")

    def get_retry_after(self, error: KuCoinError) -> Optional[int]:
        """Extract retry-after value from rate limit error.

        Args:
            error: KuCoin error

        Returns:
            Seconds to wait before retry, or None
        """
        if not isinstance(error, KuCoinRateLimitError):
            return None

        # Default rate limit backoff
        return int(os.getenv("KUCOIN_RATE_LIMIT_BACKOFF", "30"))

    async def handle_rate_limit(self, error: KuCoinRateLimitError) -> None:
        """Handle rate limit error with appropriate backoff.

        Args:
            error: Rate limit error
        """
        retry_after = self.get_retry_after(error)
        if retry_after:
            logger.warning(
                "kucoin_rate_limit_backoff",
                retry_after_seconds=retry_after,
                error=str(error),
            )
            await asyncio.sleep(retry_after)
        else:
            default_wait = int(os.getenv("KUCOIN_RATE_LIMIT_DEFAULT_WAIT", "10"))
            logger.warning("kucoin_rate_limit_default_wait", wait_seconds=default_wait)
            await asyncio.sleep(default_wait)

    async def execute_with_protection(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function with full error protection (circuit breaker + retry).

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            KuCoinError: If execution fails after all retries
        """
        async def wrapped() -> T:
            return await self.retry_with_backoff(func, *args, **kwargs)

        return await self.circuit_breaker.call(wrapped)


# Global error handler instance
error_handler = KuCoinErrorHandler()
