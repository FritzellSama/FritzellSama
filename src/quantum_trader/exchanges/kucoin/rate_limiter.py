"""
KuCoin Rate Limiter - Production-grade rate limiting for KuCoin API.

Implements sliding window rate limiting with adaptive backoff and circuit breaker
patterns to prevent API rate limit violations.
"""

import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class RateLimitType(Enum):
    """Rate limit types for different API endpoints."""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    ORDERS = "ORDERS"
    WEBSOCKET = "WEBSOCKET"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    limit: int  # Max requests
    window_seconds: Decimal  # Time window in seconds
    burst_limit: Optional[int] = None  # Burst allowance
    weight: int = 1  # Request weight


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    rule: RateLimitRule
    tokens: Decimal
    last_refill: datetime
    request_times: deque = field(default_factory=lambda: deque(maxlen=10000))
    violations: int = 0


class KuCoinRateLimiter:
    """
    Production-grade rate limiter for KuCoin API.

    Implements sliding window algorithm with adaptive backoff and circuit breaker.
    Handles different rate limits for public/private/order endpoints.

    Attributes:
        config: Configuration dictionary
        buckets: Rate limit buckets by type
        circuit_breaker_threshold: Violations before circuit break
        circuit_breaker_cooldown: Cooldown period in seconds

    Example:
        >>> config = {
        ...     'rate_limits': {
        ...         'public': {'limit': 100, 'window_seconds': 10},
        ...         'private': {'limit': 200, 'window_seconds': 10},
        ...         'orders': {'limit': 45, 'window_seconds': 3}
        ...     }
        ... }
        >>> limiter = KuCoinRateLimiter(config)
        >>> async with limiter.acquire('PRIVATE'):
        ...     result = await api_call()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Configuration with rate_limits section

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.buckets: Dict[RateLimitType, RateLimitBucket] = {}
        self._initialize_buckets()

        self.circuit_breaker_threshold = int(config.get('circuit_breaker_threshold', 10))
        self.circuit_breaker_cooldown = Decimal(str(config.get('circuit_breaker_cooldown', 300)))
        self.circuit_open = False
        self.circuit_open_time: Optional[datetime] = None

        self._lock = asyncio.Lock()
        self._metrics: Dict[str, Any] = {
            'total_requests': 0,
            'throttled_requests': 0,
            'violations': 0,
            'circuit_breaks': 0
        }

        logger.info(
            "kucoin_rate_limiter_initialized",
            buckets=len(self.buckets),
            circuit_threshold=self.circuit_breaker_threshold
        )

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        if 'rate_limits' not in self.config:
            raise ValueError("Missing 'rate_limits' in configuration")

        required_types = ['public', 'private', 'orders']
        for limit_type in required_types:
            if limit_type not in self.config['rate_limits']:
                raise ValueError(f"Missing rate limit configuration for '{limit_type}'")

            limit_config = self.config['rate_limits'][limit_type]
            if 'limit' not in limit_config or 'window_seconds' not in limit_config:
                raise ValueError(f"Invalid rate limit config for '{limit_type}'")

    def _initialize_buckets(self) -> None:
        """Initialize rate limit buckets from configuration."""
        rate_limits = self.config['rate_limits']

        for limit_type_str, limit_config in rate_limits.items():
            try:
                limit_type = RateLimitType[limit_type_str.upper()]
            except KeyError:
                logger.warning(
                    "unknown_rate_limit_type",
                    type=limit_type_str
                )
                continue

            rule = RateLimitRule(
                limit=int(limit_config['limit']),
                window_seconds=Decimal(str(limit_config['window_seconds'])),
                burst_limit=limit_config.get('burst_limit'),
                weight=int(limit_config.get('weight', 1))
            )

            self.buckets[limit_type] = RateLimitBucket(
                rule=rule,
                tokens=Decimal(str(rule.limit)),
                last_refill=datetime.now(timezone.utc)
            )

    async def acquire(
        self,
        limit_type: str,
        weight: int = 1,
        max_wait: Optional[Decimal] = None
    ) -> 'RateLimitContext':
        """
        Acquire rate limit token.

        Args:
            limit_type: Type of rate limit (PUBLIC, PRIVATE, ORDERS)
            weight: Request weight (default 1)
            max_wait: Maximum wait time in seconds

        Returns:
            Context manager for rate-limited operation

        Raises:
            ValueError: If limit_type is invalid
            TimeoutError: If max_wait exceeded
            RuntimeError: If circuit breaker is open
        """
        try:
            rate_limit_type = RateLimitType[limit_type.upper()]
        except KeyError:
            raise ValueError(f"Invalid rate limit type: {limit_type}")

        return RateLimitContext(
            self,
            rate_limit_type,
            weight,
            max_wait or Decimal(str(self.config.get('max_wait_seconds', 30)))
        )

    async def _try_acquire(
        self,
        limit_type: RateLimitType,
        weight: int,
        max_wait: Decimal
    ) -> None:
        """
        Try to acquire rate limit token with waiting.

        Args:
            limit_type: Type of rate limit
            weight: Request weight
            max_wait: Maximum wait time

        Raises:
            TimeoutError: If cannot acquire within max_wait
            RuntimeError: If circuit breaker is open
        """
        start_time = datetime.now(timezone.utc)

        while True:
            async with self._lock:
                # Check circuit breaker
                if self.circuit_open:
                    if not self._check_circuit_recovery():
                        raise RuntimeError("Circuit breaker is open - rate limit protection active")

                # Try to consume tokens
                if await self._consume_tokens(limit_type, weight):
                    self._metrics['total_requests'] += 1
                    return

                self._metrics['throttled_requests'] += 1

            # Calculate wait time
            elapsed = Decimal(str((datetime.now(timezone.utc) - start_time).total_seconds()))
            if elapsed >= max_wait:
                logger.error(
                    "rate_limit_timeout",
                    limit_type=limit_type.value,
                    elapsed=float(elapsed)
                )
                raise TimeoutError(f"Could not acquire rate limit within {max_wait}s")

            # Wait before retry
            wait_time = min(
                Decimal('0.1'),
                max_wait - elapsed
            )
            await asyncio.sleep(float(wait_time))

    async def _consume_tokens(
        self,
        limit_type: RateLimitType,
        weight: int
    ) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            limit_type: Type of rate limit
            weight: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        bucket = self.buckets.get(limit_type)
        if not bucket:
            logger.warning(
                "unknown_rate_limit_bucket",
                limit_type=limit_type.value
            )
            return True  # Allow if bucket not configured

        # Refill tokens
        await self._refill_bucket(bucket)

        # Check sliding window
        if not self._check_sliding_window(bucket, weight):
            return False

        # Consume tokens
        required_tokens = Decimal(str(weight))
        if bucket.tokens >= required_tokens:
            bucket.tokens -= required_tokens
            bucket.request_times.append(datetime.now(timezone.utc))
            return True

        return False

    async def _refill_bucket(self, bucket: RateLimitBucket) -> None:
        """
        Refill bucket tokens based on time elapsed.

        Args:
            bucket: Bucket to refill
        """
        now = datetime.now(timezone.utc)
        elapsed = Decimal(str((now - bucket.last_refill).total_seconds()))

        # Calculate refill amount
        refill_rate = Decimal(str(bucket.rule.limit)) / bucket.rule.window_seconds
        refill_amount = refill_rate * elapsed

        # Refill tokens up to limit
        max_tokens = Decimal(str(bucket.rule.limit))
        bucket.tokens = min(
            max_tokens,
            bucket.tokens + refill_amount
        )
        bucket.last_refill = now

    def _check_sliding_window(
        self,
        bucket: RateLimitBucket,
        weight: int
    ) -> bool:
        """
        Check if request fits within sliding window.

        Args:
            bucket: Rate limit bucket
            weight: Request weight

        Returns:
            True if within limits
        """
        now = datetime.now(timezone.utc)
        window_start = now.timestamp() - float(bucket.rule.window_seconds)

        # Count requests in window
        requests_in_window = sum(
            1 for req_time in bucket.request_times
            if req_time.timestamp() >= window_start
        )

        # Check against limit
        return requests_in_window + weight <= bucket.rule.limit

    def _check_circuit_recovery(self) -> bool:
        """
        Check if circuit breaker can recover.

        Returns:
            True if recovered, False if still open
        """
        if not self.circuit_open or not self.circuit_open_time:
            return True

        elapsed = Decimal(str(
            (datetime.now(timezone.utc) - self.circuit_open_time).total_seconds()
        ))

        if elapsed >= self.circuit_breaker_cooldown:
            self.circuit_open = False
            self.circuit_open_time = None
            logger.info("circuit_breaker_recovered")
            return True

        return False

    def record_violation(self, limit_type: RateLimitType) -> None:
        """
        Record rate limit violation.

        Args:
            limit_type: Type that was violated
        """
        bucket = self.buckets.get(limit_type)
        if bucket:
            bucket.violations += 1
            self._metrics['violations'] += 1

            if bucket.violations >= self.circuit_breaker_threshold:
                self._open_circuit_breaker()

        logger.warning(
            "rate_limit_violation",
            limit_type=limit_type.value,
            total_violations=self._metrics['violations']
        )

    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker to prevent further requests."""
        self.circuit_open = True
        self.circuit_open_time = datetime.now(timezone.utc)
        self._metrics['circuit_breaks'] += 1

        logger.error(
            "circuit_breaker_opened",
            cooldown_seconds=float(self.circuit_breaker_cooldown)
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get rate limiter metrics.

        Returns:
            Dictionary of metrics
        """
        bucket_stats = {}
        for limit_type, bucket in self.buckets.items():
            bucket_stats[limit_type.value] = {
                'tokens': float(bucket.tokens),
                'limit': bucket.rule.limit,
                'violations': bucket.violations,
                'requests_in_window': len(bucket.request_times)
            }

        return {
            **self._metrics,
            'buckets': bucket_stats,
            'circuit_open': self.circuit_open
        }

    async def reset(self) -> None:
        """Reset all rate limit buckets and circuit breaker."""
        async with self._lock:
            for bucket in self.buckets.values():
                bucket.tokens = Decimal(str(bucket.rule.limit))
                bucket.request_times.clear()
                bucket.violations = 0

            self.circuit_open = False
            self.circuit_open_time = None

            logger.info("rate_limiter_reset")


class RateLimitContext:
    """Context manager for rate-limited operations."""

    def __init__(
        self,
        limiter: KuCoinRateLimiter,
        limit_type: RateLimitType,
        weight: int,
        max_wait: Decimal
    ) -> None:
        """Initialize context."""
        self.limiter = limiter
        self.limit_type = limit_type
        self.weight = weight
        self.max_wait = max_wait
        self.acquired = False

    async def __aenter__(self) -> 'RateLimitContext':
        """Acquire rate limit token."""
        await self.limiter._try_acquire(
            self.limit_type,
            self.weight,
            self.max_wait
        )
        self.acquired = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release context (no-op for token bucket)."""
        pass
