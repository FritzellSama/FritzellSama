"""OKX exchange error handling and recovery.

This module provides comprehensive error handling, retry logic, and circuit breaking
for OKX exchange API interactions.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, Any, Callable, TypeVar
from datetime import datetime
import os
from structlog import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class OKXError(Exception):
    """Base exception for OKX errors."""

    def __init__(
        self, message: str, code: Optional[str] = None, response: Optional[Dict] = None
    ) -> None:
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class OKXAuthError(OKXError):
    """Authentication related errors."""

    pass


class OKXRateLimitError(OKXError):
    """Rate limit exceeded errors."""

    pass


class OKXOrderError(OKXError):
    """Order related errors."""

    pass


class OKXMarketError(OKXError):
    """Market data related errors."""

    pass


class OKXSystemError(OKXError):
    """System and service errors."""

    pass


# OKX Error Code Constants
OKX_RETRYABLE_CODES: set = {
    "50001",  # Service temporarily unavailable
    "50004",  # Endpoint request timeout
    "50011",  # System error
    "50013",  # System busy
    "50024",  # Parameter error
    "50026",  # System error
    "50044",  # Must be between timestamp
    "51000",  # Parameter error (sometimes retryable)
}

OKX_FATAL_CODES: set = {
    "50100",  # API key doesn't exist
    "50101",  # API key invalid
    "50102",  # Timestamp expired
    "50103",  # Request header required
    "50104",  # Content-Type required
    "50105",  # Invalid Content-Type
    "50106",  # Body required
    "50107",  # Invalid signature
    "50108",  # Invalid timestamp
    "50109",  # Invalid request method
    "50110",  # Invalid IP
    "50111",  # Invalid passphrase
}

OKX_RATE_LIMIT_CODES: set = {
    "50011",  # Rate limit exceeded
    "50014",  # Too many requests
}


class CircuitBreaker:
    """Circuit breaker for OKX API calls."""

    def __init__(
        self,
        failure_threshold: int,
        recovery_timeout: int,
        expected_exception: type = OKXError,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = int(
            os.getenv("OKX_CB_FAILURE_THRESHOLD", str(failure_threshold))
        )
        self.recovery_timeout = int(
            os.getenv("OKX_CB_RECOVERY_TIMEOUT", str(recovery_timeout))
        )
        self.expected_exception = expected_exception

        self.failure_count: int = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "CLOSED"

        logger.info(
            "okx_circuit_breaker_initialized",
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
                logger.info("okx_circuit_breaker_half_open")
            else:
                logger.warning("okx_circuit_breaker_open", state=self.state)
                raise OKXSystemError("Circuit breaker is OPEN")

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
            logger.info("okx_circuit_breaker_closed")
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                "okx_circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )


class OKXErrorHandler:
    """Handles OKX API errors with retry logic and circuit breaking."""

    def __init__(self) -> None:
        """Initialize error handler."""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("OKX_CB_FAILURE_THRESHOLD", "5")),
            recovery_timeout=int(os.getenv("OKX_CB_RECOVERY_TIMEOUT", "60")),
        )
        self.max_retries = int(os.getenv("OKX_MAX_RETRIES", "3"))
        self.retry_delay_ms = int(os.getenv("OKX_RETRY_DELAY_MS", "100"))
        self.backoff_multiplier = Decimal(os.getenv("OKX_BACKOFF_MULTIPLIER", "2"))

    def parse_error(self, response: Dict[str, Any]) -> OKXError:
        """Parse OKX API error response.

        Args:
            response: API error response

        Returns:
            Appropriate OKXError subclass
        """
        # OKX returns errors in data array
        if "code" in response and response["code"] != "0":
            code = response["code"]
            msg = response.get("msg", "Unknown error")
        elif "data" in response and isinstance(response["data"], list):
            if response["data"] and "sCode" in response["data"][0]:
                code = response["data"][0]["sCode"]
                msg = response["data"][0].get("sMsg", "Unknown error")
            else:
                code = None
                msg = "Unknown error"
        else:
            code = None
            msg = str(response)

        logger.error("okx_api_error", code=code, message=msg, response=response)

        # Map error codes to exception types
        if code in OKX_FATAL_CODES:
            return OKXAuthError(msg, code, response)

        if code in OKX_RATE_LIMIT_CODES:
            return OKXRateLimitError(msg, code, response)

        if code and code.startswith("51"):
            # 51xxx codes are order-related
            return OKXOrderError(msg, code, response)

        if code in OKX_RETRYABLE_CODES:
            return OKXSystemError(msg, code, response)

        return OKXError(msg, code, response)

    def is_retryable(self, error: OKXError) -> bool:
        """Check if error should be retried.

        Args:
            error: OKX error

        Returns:
            True if error is retryable
        """
        if error.code is None:
            return False

        return error.code in OKX_RETRYABLE_CODES

    def is_fatal(self, error: OKXError) -> bool:
        """Check if error is fatal (should not retry).

        Args:
            error: OKX error

        Returns:
            True if error is fatal
        """
        if error.code is None:
            return False

        return error.code in OKX_FATAL_CODES

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
            OKXError: If all retries exhausted or fatal error
        """
        if max_retries is None:
            max_retries = self.max_retries

        last_exception: Optional[Exception] = None
        retry_delay = Decimal(self.retry_delay_ms) / Decimal("1000")

        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info("okx_retry_succeeded", attempt=attempt)
                return result

            except OKXError as e:
                last_exception = e

                if self.is_fatal(e):
                    logger.error(
                        "okx_fatal_error_no_retry", error=str(e), code=e.code
                    )
                    raise

                if attempt >= max_retries:
                    logger.error(
                        "okx_max_retries_exceeded",
                        attempt=attempt,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    raise

                if self.is_retryable(e):
                    wait_time = float(
                        retry_delay * (self.backoff_multiplier**attempt)
                    )
                    logger.warning(
                        "okx_retrying_after_error",
                        attempt=attempt,
                        wait_seconds=wait_time,
                        error=str(e),
                        code=e.code,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("okx_non_retryable_error", error=str(e), code=e.code)
                    raise

            except Exception as e:
                last_exception = e
                logger.error(
                    "okx_unexpected_error",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(float(retry_delay))

        if last_exception:
            raise last_exception
        raise OKXSystemError("Retry loop completed without result")

    def get_retry_after(self, error: OKXError) -> Optional[int]:
        """Extract retry-after value from rate limit error.

        Args:
            error: OKX error

        Returns:
            Seconds to wait before retry, or None
        """
        if not isinstance(error, OKXRateLimitError):
            return None

        # OKX doesn't always provide retry-after, use default
        return int(os.getenv("OKX_RATE_LIMIT_BACKOFF", "30"))

    async def handle_rate_limit(self, error: OKXRateLimitError) -> None:
        """Handle rate limit error with appropriate backoff.

        Args:
            error: Rate limit error
        """
        retry_after = self.get_retry_after(error)
        if retry_after:
            logger.warning(
                "okx_rate_limit_backoff",
                retry_after_seconds=retry_after,
                error=str(error),
            )
            await asyncio.sleep(retry_after)
        else:
            default_wait = int(os.getenv("OKX_RATE_LIMIT_DEFAULT_WAIT", "10"))
            logger.warning("okx_rate_limit_default_wait", wait_seconds=default_wait)
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
            OKXError: If execution fails after all retries
        """
        async def wrapped() -> T:
            return await self.retry_with_backoff(func, *args, **kwargs)

        return await self.circuit_breaker.call(wrapped)


# Global error handler instance
error_handler = OKXErrorHandler()
