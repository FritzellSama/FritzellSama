"""Binance exchange rate limiting.

This module implements token bucket rate limiting for Binance API to prevent
exceeding rate limits and ensure smooth operation.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Optional
from datetime import datetime, timedelta
import os
from structlog import get_logger

from .constants import (
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_ORDERS_PER_SECOND,
    RATE_LIMIT_ORDERS_PER_DAY,
)

logger = get_logger(__name__)


class TokenBucket:
    """Token bucket rate limiter implementation.

    This implements a token bucket algorithm for rate limiting with configurable
    capacity and refill rate.
    """

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
            "token_bucket_initialized",
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
                        "tokens_acquired",
                        tokens_consumed=tokens,
                        tokens_remaining=float(self.tokens),
                    )
                    return

                # Calculate wait time until enough tokens available
                tokens_needed = Decimal(tokens) - self.tokens
                wait_time = float(
                    (tokens_needed / self.refill_rate) * self.refill_interval
                )

                logger.debug(
                    "waiting_for_tokens",
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


class BinanceRateLimiter:
    """Rate limiter for Binance API endpoints.

    Implements multiple rate limiters for different Binance API limits:
    - General requests per minute
    - Orders per second
    - Orders per day
    """

    def __init__(self) -> None:
        """Initialize Binance rate limiter."""
        # Request rate limiter (1200 requests per minute)
        requests_per_minute = RATE_LIMIT_REQUESTS_PER_MINUTE
        self.request_limiter = TokenBucket(
            capacity=requests_per_minute,
            refill_rate=Decimal(requests_per_minute),
            refill_interval=Decimal("60"),  # 60 seconds
        )

        # Order rate limiter (10 orders per second)
        orders_per_second = RATE_LIMIT_ORDERS_PER_SECOND
        self.order_limiter = TokenBucket(
            capacity=orders_per_second * 10,  # Burst capacity
            refill_rate=Decimal(orders_per_second),
            refill_interval=Decimal("1"),  # 1 second
        )

        # Daily order limit (200k orders per day)
        orders_per_day = RATE_LIMIT_ORDERS_PER_DAY
        self.daily_order_limiter = TokenBucket(
            capacity=orders_per_day,
            refill_rate=Decimal(orders_per_day),
            refill_interval=Decimal("86400"),  # 24 hours
        )

        # Weight-based limiter for complex endpoints
        weight_limit = int(os.getenv("BINANCE_WEIGHT_LIMIT", "1200"))
        self.weight_limiter = TokenBucket(
            capacity=weight_limit,
            refill_rate=Decimal(weight_limit),
            refill_interval=Decimal("60"),  # 60 seconds
        )

        # Track statistics
        self.stats: Dict[str, int] = {
            "requests": 0,
            "orders": 0,
            "rate_limit_waits": 0,
        }

        logger.info("binance_rate_limiter_initialized")

    async def acquire_request(self, weight: int = 1) -> None:
        """Acquire permission for API request.

        Args:
            weight: Request weight (default 1)
        """
        await self.request_limiter.acquire(1)
        await self.weight_limiter.acquire(weight)
        self.stats["requests"] += 1

        logger.debug("request_acquired", weight=weight, total_requests=self.stats["requests"])

    async def acquire_order(self) -> None:
        """Acquire permission for order placement."""
        await self.order_limiter.acquire(1)
        await self.daily_order_limiter.acquire(1)
        self.stats["orders"] += 1

        logger.debug("order_acquired", total_orders=self.stats["orders"])

    async def acquire_with_backoff(self, weight: int = 1, is_order: bool = False) -> None:
        """Acquire with automatic backoff on rate limit.

        Args:
            weight: Request weight
            is_order: Whether this is an order request
        """
        max_attempts = int(os.getenv("BINANCE_RATE_LIMIT_MAX_ATTEMPTS", "5"))
        backoff_base = Decimal(os.getenv("BINANCE_RATE_LIMIT_BACKOFF_BASE", "1"))

        for attempt in range(max_attempts):
            try:
                if is_order:
                    await asyncio.wait_for(
                        self.acquire_order(), timeout=float(backoff_base * (2**attempt))
                    )
                else:
                    await asyncio.wait_for(
                        self.acquire_request(weight),
                        timeout=float(backoff_base * (2**attempt)),
                    )
                return

            except asyncio.TimeoutError:
                if attempt < max_attempts - 1:
                    wait_time = float(backoff_base * (Decimal("2") ** attempt))
                    logger.warning(
                        "rate_limit_backoff",
                        attempt=attempt,
                        wait_seconds=wait_time,
                    )
                    self.stats["rate_limit_waits"] += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("rate_limit_max_attempts_exceeded", attempts=max_attempts)
                    raise

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "available_request_tokens": self.request_limiter.get_available_tokens(),
            "available_order_tokens": self.order_limiter.get_available_tokens(),
            "available_daily_orders": self.daily_order_limiter.get_available_tokens(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "requests": 0,
            "orders": 0,
            "rate_limit_waits": 0,
        }
        logger.info("rate_limiter_stats_reset")

    async def wait_for_capacity(self, tokens_needed: int = 1) -> None:
        """Wait until specified capacity is available.

        Args:
            tokens_needed: Number of tokens to wait for
        """
        while self.request_limiter.get_available_tokens() < tokens_needed:
            wait_time = float(
                Decimal(tokens_needed)
                / self.request_limiter.refill_rate
                * self.request_limiter.refill_interval
            )
            logger.debug(
                "waiting_for_capacity",
                tokens_needed=tokens_needed,
                wait_seconds=wait_time,
            )
            await asyncio.sleep(min(wait_time, 1.0))


# Global rate limiter instance
rate_limiter = BinanceRateLimiter()
