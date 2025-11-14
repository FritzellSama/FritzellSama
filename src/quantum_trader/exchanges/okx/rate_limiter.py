"""OKX exchange rate limiting.

This module implements token bucket rate limiting for OKX API to prevent
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
            "okx_token_bucket_initialized",
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
                        "okx_tokens_acquired",
                        tokens_consumed=tokens,
                        tokens_remaining=float(self.tokens),
                    )
                    return

                tokens_needed = Decimal(tokens) - self.tokens
                wait_time = float(
                    (tokens_needed / self.refill_rate) * self.refill_interval
                )

                logger.debug(
                    "okx_waiting_for_tokens",
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


class OKXRateLimiter:
    """Rate limiter for OKX API endpoints.

    OKX uses a complex rate limiting system with different limits per endpoint:
    - Public endpoints: 20 requests/2 seconds
    - Private endpoints: 60 requests/2 seconds
    - Trading endpoints: 60 requests/2 seconds
    - Order placement: 60 orders/2 seconds
    """

    def __init__(self) -> None:
        """Initialize OKX rate limiter."""
        # Public endpoint limiter (20 requests per 2 seconds)
        public_requests = int(os.getenv("OKX_PUBLIC_RATE_LIMIT", "20"))
        self.public_limiter = TokenBucket(
            capacity=public_requests,
            refill_rate=Decimal(public_requests),
            refill_interval=Decimal("2"),  # 2 seconds
        )

        # Private endpoint limiter (60 requests per 2 seconds)
        private_requests = int(os.getenv("OKX_PRIVATE_RATE_LIMIT", "60"))
        self.private_limiter = TokenBucket(
            capacity=private_requests,
            refill_rate=Decimal(private_requests),
            refill_interval=Decimal("2"),  # 2 seconds
        )

        # Trading endpoint limiter (60 requests per 2 seconds)
        trading_requests = int(os.getenv("OKX_TRADING_RATE_LIMIT", "60"))
        self.trading_limiter = TokenBucket(
            capacity=trading_requests,
            refill_rate=Decimal(trading_requests),
            refill_interval=Decimal("2"),  # 2 seconds
        )

        # Order placement limiter (60 orders per 2 seconds)
        order_limit = int(os.getenv("OKX_ORDER_RATE_LIMIT", "60"))
        self.order_limiter = TokenBucket(
            capacity=order_limit,
            refill_rate=Decimal(order_limit),
            refill_interval=Decimal("2"),  # 2 seconds
        )

        # WebSocket connection limiter
        ws_limit = int(os.getenv("OKX_WS_RATE_LIMIT", "5"))
        self.ws_limiter = TokenBucket(
            capacity=ws_limit,
            refill_rate=Decimal(ws_limit),
            refill_interval=Decimal("1"),  # 1 second
        )

        # Track statistics
        self.stats: Dict[str, int] = {
            "public_requests": 0,
            "private_requests": 0,
            "trading_requests": 0,
            "orders": 0,
            "ws_requests": 0,
            "rate_limit_waits": 0,
        }

        logger.info("okx_rate_limiter_initialized")

    async def acquire_public(self, count: int = 1) -> None:
        """Acquire permission for public API request.

        Args:
            count: Number of requests to acquire
        """
        await self.public_limiter.acquire(count)
        self.stats["public_requests"] += count

        logger.debug(
            "okx_public_acquired",
            count=count,
            total_public=self.stats["public_requests"],
        )

    async def acquire_private(self, count: int = 1) -> None:
        """Acquire permission for private API request.

        Args:
            count: Number of requests to acquire
        """
        await self.private_limiter.acquire(count)
        self.stats["private_requests"] += count

        logger.debug(
            "okx_private_acquired",
            count=count,
            total_private=self.stats["private_requests"],
        )

    async def acquire_trading(self, count: int = 1) -> None:
        """Acquire permission for trading API request.

        Args:
            count: Number of requests to acquire
        """
        await self.trading_limiter.acquire(count)
        self.stats["trading_requests"] += count

        logger.debug(
            "okx_trading_acquired",
            count=count,
            total_trading=self.stats["trading_requests"],
        )

    async def acquire_order(self, count: int = 1) -> None:
        """Acquire permission for order placement.

        Args:
            count: Number of orders to acquire
        """
        await self.order_limiter.acquire(count)
        await self.trading_limiter.acquire(count)
        self.stats["orders"] += count

        logger.debug("okx_order_acquired", count=count, total_orders=self.stats["orders"])

    async def acquire_ws(self) -> None:
        """Acquire permission for WebSocket request."""
        await self.ws_limiter.acquire(1)
        self.stats["ws_requests"] += 1

        logger.debug("okx_ws_acquired", total_ws=self.stats["ws_requests"])

    async def acquire_with_backoff(
        self, endpoint_type: str = "public", count: int = 1, max_attempts: Optional[int] = None
    ) -> None:
        """Acquire with automatic backoff on rate limit.

        Args:
            endpoint_type: Type of endpoint (public, private, trading, order, ws)
            count: Number of requests to acquire
            max_attempts: Maximum retry attempts
        """
        if max_attempts is None:
            max_attempts = int(os.getenv("OKX_RATE_LIMIT_MAX_ATTEMPTS", "5"))

        backoff_base = Decimal(os.getenv("OKX_RATE_LIMIT_BACKOFF_BASE", "0.5"))

        for attempt in range(max_attempts):
            try:
                timeout = float(backoff_base * (Decimal("2") ** attempt))

                if endpoint_type == "public":
                    await asyncio.wait_for(self.acquire_public(count), timeout=timeout)
                elif endpoint_type == "private":
                    await asyncio.wait_for(self.acquire_private(count), timeout=timeout)
                elif endpoint_type == "trading":
                    await asyncio.wait_for(self.acquire_trading(count), timeout=timeout)
                elif endpoint_type == "order":
                    await asyncio.wait_for(self.acquire_order(count), timeout=timeout)
                elif endpoint_type == "ws":
                    await asyncio.wait_for(self.acquire_ws(), timeout=timeout)
                else:
                    await asyncio.wait_for(self.acquire_public(count), timeout=timeout)

                return

            except asyncio.TimeoutError:
                if attempt < max_attempts - 1:
                    wait_time = float(backoff_base * (Decimal("2") ** attempt))
                    logger.warning(
                        "okx_rate_limit_backoff",
                        attempt=attempt,
                        wait_seconds=wait_time,
                        endpoint_type=endpoint_type,
                    )
                    self.stats["rate_limit_waits"] += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "okx_rate_limit_max_attempts_exceeded",
                        attempts=max_attempts,
                        endpoint_type=endpoint_type,
                    )
                    raise

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "available_public_tokens": self.public_limiter.get_available_tokens(),
            "available_private_tokens": self.private_limiter.get_available_tokens(),
            "available_trading_tokens": self.trading_limiter.get_available_tokens(),
            "available_order_tokens": self.order_limiter.get_available_tokens(),
            "available_ws_tokens": self.ws_limiter.get_available_tokens(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "public_requests": 0,
            "private_requests": 0,
            "trading_requests": 0,
            "orders": 0,
            "ws_requests": 0,
            "rate_limit_waits": 0,
        }
        logger.info("okx_rate_limiter_stats_reset")

    async def wait_for_capacity(
        self, tokens_needed: int = 1, endpoint_type: str = "public"
    ) -> None:
        """Wait until specified capacity is available.

        Args:
            tokens_needed: Number of tokens to wait for
            endpoint_type: Type of endpoint
        """
        limiter_map = {
            "public": self.public_limiter,
            "private": self.private_limiter,
            "trading": self.trading_limiter,
            "order": self.order_limiter,
            "ws": self.ws_limiter,
        }

        limiter = limiter_map.get(endpoint_type, self.public_limiter)

        while limiter.get_available_tokens() < tokens_needed:
            wait_time = float(
                Decimal(tokens_needed) / limiter.refill_rate * limiter.refill_interval
            )
            logger.debug(
                "okx_waiting_for_capacity",
                tokens_needed=tokens_needed,
                endpoint_type=endpoint_type,
                wait_seconds=wait_time,
            )
            await asyncio.sleep(min(wait_time, 1.0))

    def check_capacity(self, tokens_needed: int = 1, endpoint_type: str = "public") -> bool:
        """Check if capacity is available without acquiring.

        Args:
            tokens_needed: Number of tokens to check
            endpoint_type: Type of endpoint

        Returns:
            True if capacity is available
        """
        limiter_map = {
            "public": self.public_limiter,
            "private": self.private_limiter,
            "trading": self.trading_limiter,
            "order": self.order_limiter,
            "ws": self.ws_limiter,
        }

        limiter = limiter_map.get(endpoint_type, self.public_limiter)
        return limiter.get_available_tokens() >= tokens_needed


# Global rate limiter instance
rate_limiter = OKXRateLimiter()
