"""Bybit exchange rate limiting.

This module implements token bucket rate limiting for Bybit API to prevent
exceeding rate limits and ensure smooth operation.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime
import os
from structlog import get_logger

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(
        self, capacity: int, refill_rate: Decimal, refill_interval: Decimal
    ) -> None:
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Number of tokens to add per interval
            refill_interval: Interval in seconds for refill
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval

        self.tokens = Decimal(capacity)
        self.last_refill = datetime.utcnow()
        self._lock = asyncio.Lock()

        logger.info(
            "bybit_token_bucket_initialized",
            capacity=capacity,
            refill_rate=float(refill_rate),
            refill_interval=float(refill_interval),
        )

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= Decimal(tokens)
                    logger.debug(
                        "bybit_tokens_acquired",
                        tokens_consumed=tokens,
                        tokens_remaining=float(self.tokens),
                    )
                    return

                tokens_needed = Decimal(tokens) - self.tokens
                wait_time = float(
                    (tokens_needed / self.refill_rate) * self.refill_interval
                )

                logger.debug(
                    "bybit_waiting_for_tokens",
                    tokens_needed=float(tokens_needed),
                    wait_seconds=wait_time,
                )

                await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = datetime.utcnow()
        time_elapsed = (now - self.last_refill).total_seconds()

        if time_elapsed > 0:
            tokens_to_add = (
                Decimal(str(time_elapsed)) / self.refill_interval
            ) * self.refill_rate
            self.tokens = min(self.tokens + tokens_to_add, Decimal(self.capacity))
            self.last_refill = now

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= Decimal(tokens)
                return True

            return False

    def get_available_tokens(self) -> int:
        """Get current number of available tokens.

        Returns:
            Number of available tokens
        """
        self._refill()
        return int(self.tokens)


class BybitRateLimiter:
    """Rate limiter for Bybit API endpoints.

    Implements multiple rate limiters for different Bybit API limits:
    - General requests per second
    - Orders per second
    - WebSocket connections
    """

    def __init__(self) -> None:
        """Initialize Bybit rate limiter."""
        # Request rate limiter - Bybit has 50 requests/second for most endpoints
        requests_per_second = int(os.getenv("BYBIT_RATE_LIMIT_RPS", "50"))
        self.request_limiter = TokenBucket(
            capacity=requests_per_second * 2,  # Burst capacity
            refill_rate=Decimal(requests_per_second),
            refill_interval=Decimal("1"),  # 1 second
        )

        # Order rate limiter - Bybit allows 10 orders/second
        orders_per_second = int(os.getenv("BYBIT_RATE_LIMIT_OPS", "10"))
        self.order_limiter = TokenBucket(
            capacity=orders_per_second * 5,  # Burst capacity
            refill_rate=Decimal(orders_per_second),
            refill_interval=Decimal("1"),  # 1 second
        )

        # Account endpoint limiter (more restrictive)
        account_requests_per_second = int(os.getenv("BYBIT_ACCOUNT_RATE_LIMIT_RPS", "20"))
        self.account_limiter = TokenBucket(
            capacity=account_requests_per_second,
            refill_rate=Decimal(account_requests_per_second),
            refill_interval=Decimal("1"),  # 1 second
        )

        # Track statistics
        self.stats: Dict[str, int] = {
            "requests": 0,
            "orders": 0,
            "account_requests": 0,
            "rate_limit_waits": 0,
        }

        logger.info("bybit_rate_limiter_initialized")

    async def acquire_request(self, endpoint_type: str = "general") -> None:
        """Acquire permission for API request.

        Args:
            endpoint_type: Type of endpoint (general, account, order)
        """
        if endpoint_type == "account":
            await self.account_limiter.acquire(1)
            self.stats["account_requests"] += 1
        elif endpoint_type == "order":
            await self.order_limiter.acquire(1)
            self.stats["orders"] += 1
        else:
            await self.request_limiter.acquire(1)
            self.stats["requests"] += 1

        logger.debug(
            "bybit_request_acquired",
            endpoint_type=endpoint_type,
            total_requests=self.stats["requests"],
        )

    async def acquire_order(self) -> None:
        """Acquire permission for order placement."""
        await self.order_limiter.acquire(1)
        await self.request_limiter.acquire(1)
        self.stats["orders"] += 1
        self.stats["requests"] += 1

        logger.debug("bybit_order_acquired", total_orders=self.stats["orders"])

    async def acquire_with_backoff(
        self, endpoint_type: str = "general", max_attempts: Optional[int] = None
    ) -> None:
        """Acquire with automatic backoff on rate limit.

        Args:
            endpoint_type: Type of endpoint
            max_attempts: Maximum retry attempts
        """
        if max_attempts is None:
            max_attempts = int(os.getenv("BYBIT_RATE_LIMIT_MAX_ATTEMPTS", "5"))

        backoff_base = Decimal(os.getenv("BYBIT_RATE_LIMIT_BACKOFF_BASE", "0.5"))

        for attempt in range(max_attempts):
            try:
                timeout = float(backoff_base * (Decimal("2") ** attempt))
                await asyncio.wait_for(
                    self.acquire_request(endpoint_type), timeout=timeout
                )
                return

            except asyncio.TimeoutError:
                if attempt < max_attempts - 1:
                    wait_time = float(backoff_base * (Decimal("2") ** attempt))
                    logger.warning(
                        "bybit_rate_limit_backoff",
                        attempt=attempt,
                        wait_seconds=wait_time,
                        endpoint_type=endpoint_type,
                    )
                    self.stats["rate_limit_waits"] += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "bybit_rate_limit_max_attempts_exceeded",
                        attempts=max_attempts,
                        endpoint_type=endpoint_type,
                    )
                    raise

    async def batch_acquire(self, count: int, endpoint_type: str = "general") -> None:
        """Acquire multiple tokens for batch operations.

        Args:
            count: Number of requests to acquire
            endpoint_type: Type of endpoint
        """
        if endpoint_type == "order":
            await self.order_limiter.acquire(count)
            self.stats["orders"] += count
        elif endpoint_type == "account":
            await self.account_limiter.acquire(count)
            self.stats["account_requests"] += count
        else:
            await self.request_limiter.acquire(count)
            self.stats["requests"] += count

        logger.debug(
            "bybit_batch_acquired", count=count, endpoint_type=endpoint_type
        )

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "available_request_tokens": self.request_limiter.get_available_tokens(),
            "available_order_tokens": self.order_limiter.get_available_tokens(),
            "available_account_tokens": self.account_limiter.get_available_tokens(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "requests": 0,
            "orders": 0,
            "account_requests": 0,
            "rate_limit_waits": 0,
        }
        logger.info("bybit_rate_limiter_stats_reset")

    async def wait_for_capacity(
        self, tokens_needed: int = 1, endpoint_type: str = "general"
    ) -> None:
        """Wait until specified capacity is available.

        Args:
            tokens_needed: Number of tokens to wait for
            endpoint_type: Type of endpoint
        """
        limiter = self.request_limiter
        if endpoint_type == "order":
            limiter = self.order_limiter
        elif endpoint_type == "account":
            limiter = self.account_limiter

        while limiter.get_available_tokens() < tokens_needed:
            wait_time = float(
                Decimal(tokens_needed) / limiter.refill_rate * limiter.refill_interval
            )
            logger.debug(
                "bybit_waiting_for_capacity",
                tokens_needed=tokens_needed,
                endpoint_type=endpoint_type,
                wait_seconds=wait_time,
            )
            await asyncio.sleep(min(wait_time, 1.0))

    def check_capacity(self, tokens_needed: int = 1, endpoint_type: str = "general") -> bool:
        """Check if capacity is available without acquiring.

        Args:
            tokens_needed: Number of tokens to check
            endpoint_type: Type of endpoint

        Returns:
            True if capacity is available
        """
        limiter = self.request_limiter
        if endpoint_type == "order":
            limiter = self.order_limiter
        elif endpoint_type == "account":
            limiter = self.account_limiter

        return limiter.get_available_tokens() >= tokens_needed


# Global rate limiter instance
rate_limiter = BybitRateLimiter()
