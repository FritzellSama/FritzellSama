"""
Rate Limiting Module - Token Bucket Algorithm
CRITICAL: Production-ready rate limiting for API endpoints
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from dataclasses import dataclass

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: Decimal
    tokens: Decimal
    refill_rate: Decimal  # tokens per second
    last_refill: datetime

    def __init__(self, capacity: Decimal, refill_rate: Decimal):
        """
        Initialize token bucket

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens to add per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = datetime.utcnow()


class RateLimiter:
    """Production token bucket rate limiter"""

    def __init__(self):
        """Initialize rate limiter from configuration"""
        self._config = get_config()
        self._enabled: bool = True
        self._buckets: Dict[str, TokenBucket] = {}
        self._endpoint_limits: Dict[str, Decimal] = {}
        self._default_limit_per_minute: Decimal = Decimal("100")
        self._burst_limit: Decimal = Decimal("150")
        self._violation_lockout_seconds: int = 300
        self._lockouts: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

        # Load configuration
        self._load_config()

        logger.info(
            f"Rate limiter initialized: enabled={self._enabled}, "
            f"default={self._default_limit_per_minute}/min, burst={self._burst_limit}"
        )

    def _load_config(self) -> None:
        """Load rate limiting configuration"""
        try:
            # Load rate limiting settings
            self._enabled = self._config.get_bool('security', 'rate_limiting.enabled', True)

            # Load default limit
            default_limit = self._config.get('security', 'rate_limiting.default_limit_per_minute', 100)
            self._default_limit_per_minute = Decimal(str(default_limit))

            # Load burst limit
            burst_limit = self._config.get('security', 'rate_limiting.burst_limit', 150)
            self._burst_limit = Decimal(str(burst_limit))

            # Load violation lockout
            self._violation_lockout_seconds = self._config.get_int(
                'security', 'rate_limiting.violation_lockout_seconds', 300
            )

            # Load endpoint-specific limits
            endpoints_config = self._config.get('security', 'rate_limiting.api_endpoints', {})
            if endpoints_config:
                for endpoint, limit in endpoints_config.items():
                    self._endpoint_limits[endpoint] = Decimal(str(limit))

            logger.info(
                f"Rate limiting configuration loaded: {len(self._endpoint_limits)} endpoint limits"
            )

        except Exception as e:
            logger.error(f"Failed to load rate limiting configuration: {e}")
            # Set safe defaults
            self._enabled = True
            self._default_limit_per_minute = Decimal("100")
            self._burst_limit = Decimal("150")

    def _get_bucket_key(self, user_id: str, endpoint: str) -> str:
        """
        Generate bucket key for user+endpoint combination

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Bucket key string
        """
        return f"{user_id}:{endpoint}"

    def _get_or_create_bucket(self, user_id: str, endpoint: str) -> TokenBucket:
        """
        Get or create token bucket for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            TokenBucket instance
        """
        bucket_key = self._get_bucket_key(user_id, endpoint)

        if bucket_key not in self._buckets:
            # Determine capacity based on endpoint-specific or default limit
            limit_per_minute = self._endpoint_limits.get(endpoint, self._default_limit_per_minute)

            # Calculate refill rate (tokens per second)
            refill_rate = limit_per_minute / Decimal("60")

            # Use burst limit as capacity
            capacity = self._burst_limit

            # Create new bucket
            self._buckets[bucket_key] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate
            )

            logger.debug(
                f"Created token bucket for {bucket_key}: "
                f"capacity={capacity}, rate={refill_rate}/sec"
            )

        return self._buckets[bucket_key]

    def _refill_bucket(self, bucket: TokenBucket) -> None:
        """
        Refill token bucket based on time elapsed

        Args:
            bucket: TokenBucket to refill
        """
        now = datetime.utcnow()
        time_elapsed = (now - bucket.last_refill).total_seconds()

        if time_elapsed > 0:
            # Calculate tokens to add
            tokens_to_add = Decimal(str(time_elapsed)) * bucket.refill_rate

            # Add tokens (cap at capacity)
            bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)

            # Update last refill time
            bucket.last_refill = now

    async def check_rate_limit(self, user_id: str, endpoint: str, tokens: int = 1) -> bool:
        """
        Check if request is within rate limit (without consuming tokens)

        Args:
            user_id: User identifier
            endpoint: Endpoint name
            tokens: Number of tokens required (default 1)

        Returns:
            True if within limit, False otherwise
        """
        try:
            # If rate limiting is disabled, allow all
            if not self._enabled:
                return True

            # Check if user is locked out
            if await self._is_locked_out(user_id, endpoint):
                logger.warning(f"User {user_id} locked out for endpoint {endpoint}")
                return False

            # Get bucket
            bucket = self._get_or_create_bucket(user_id, endpoint)

            # Refill bucket
            self._refill_bucket(bucket)

            # Check if enough tokens available
            required_tokens = Decimal(str(tokens))
            has_tokens = bucket.tokens >= required_tokens

            if has_tokens:
                logger.debug(
                    f"Rate limit OK for {user_id}:{endpoint} "
                    f"({bucket.tokens:.2f} tokens available)"
                )
            else:
                logger.warning(
                    f"Rate limit EXCEEDED for {user_id}:{endpoint} "
                    f"({bucket.tokens:.2f} < {required_tokens} required)"
                )

            return has_tokens

        except Exception as e:
            logger.error(f"Error checking rate limit for {user_id}:{endpoint}: {e}")
            # Fail secure - deny on error
            return False

    async def consume_tokens(self, user_id: str, endpoint: str, tokens: int = 1) -> bool:
        """
        Consume tokens from rate limit bucket

        Args:
            user_id: User identifier
            endpoint: Endpoint name
            tokens: Number of tokens to consume (default 1)

        Returns:
            True if tokens consumed successfully, False if rate limit exceeded
        """
        async with self._lock:
            try:
                # If rate limiting is disabled, allow all
                if not self._enabled:
                    return True

                # Check if user is locked out
                if await self._is_locked_out(user_id, endpoint):
                    logger.warning(f"User {user_id} locked out for endpoint {endpoint}")
                    return False

                # Get bucket
                bucket = self._get_or_create_bucket(user_id, endpoint)

                # Refill bucket
                self._refill_bucket(bucket)

                # Check if enough tokens available
                required_tokens = Decimal(str(tokens))

                if bucket.tokens >= required_tokens:
                    # Consume tokens
                    bucket.tokens -= required_tokens

                    logger.debug(
                        f"Consumed {required_tokens} tokens for {user_id}:{endpoint} "
                        f"({bucket.tokens:.2f} remaining)"
                    )
                    return True
                else:
                    # Rate limit exceeded - trigger lockout
                    await self._trigger_lockout(user_id, endpoint)

                    logger.warning(
                        f"Rate limit EXCEEDED for {user_id}:{endpoint} "
                        f"({bucket.tokens:.2f} < {required_tokens} required) - LOCKOUT triggered"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error consuming tokens for {user_id}:{endpoint}: {e}")
                # Fail secure - deny on error
                return False

    async def reset_limit(self, user_id: str, endpoint: str) -> bool:
        """
        Reset rate limit for user+endpoint (refill bucket to capacity)

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                bucket_key = self._get_bucket_key(user_id, endpoint)

                if bucket_key in self._buckets:
                    bucket = self._buckets[bucket_key]
                    bucket.tokens = bucket.capacity
                    bucket.last_refill = datetime.utcnow()

                    logger.info(f"Reset rate limit for {user_id}:{endpoint}")
                    return True
                else:
                    logger.warning(f"No bucket found for {user_id}:{endpoint}")
                    return False

            except Exception as e:
                logger.error(f"Failed to reset rate limit for {user_id}:{endpoint}: {e}")
                return False

    async def get_remaining(self, user_id: str, endpoint: str) -> Decimal:
        """
        Get remaining tokens for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Number of remaining tokens (Decimal)
        """
        try:
            # Get or create bucket
            bucket = self._get_or_create_bucket(user_id, endpoint)

            # Refill bucket
            self._refill_bucket(bucket)

            return bucket.tokens

        except Exception as e:
            logger.error(f"Error getting remaining tokens for {user_id}:{endpoint}: {e}")
            return Decimal("0")

    async def get_capacity(self, user_id: str, endpoint: str) -> Decimal:
        """
        Get bucket capacity for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Bucket capacity (Decimal)
        """
        try:
            bucket = self._get_or_create_bucket(user_id, endpoint)
            return bucket.capacity

        except Exception as e:
            logger.error(f"Error getting capacity for {user_id}:{endpoint}: {e}")
            return Decimal("0")

    async def get_refill_rate(self, user_id: str, endpoint: str) -> Decimal:
        """
        Get refill rate for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Refill rate in tokens per second (Decimal)
        """
        try:
            bucket = self._get_or_create_bucket(user_id, endpoint)
            return bucket.refill_rate

        except Exception as e:
            logger.error(f"Error getting refill rate for {user_id}:{endpoint}: {e}")
            return Decimal("0")

    async def _is_locked_out(self, user_id: str, endpoint: str) -> bool:
        """
        Check if user is locked out for endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            True if locked out, False otherwise
        """
        lockout_key = self._get_bucket_key(user_id, endpoint)

        if lockout_key in self._lockouts:
            lockout_time = self._lockouts[lockout_key]
            now = datetime.utcnow()

            # Check if lockout period has expired
            lockout_duration = timedelta(seconds=self._violation_lockout_seconds)
            if now < lockout_time + lockout_duration:
                return True
            else:
                # Lockout expired - remove from lockouts
                del self._lockouts[lockout_key]
                logger.info(f"Lockout expired for {user_id}:{endpoint}")
                return False

        return False

    async def _trigger_lockout(self, user_id: str, endpoint: str) -> None:
        """
        Trigger lockout for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name
        """
        lockout_key = self._get_bucket_key(user_id, endpoint)
        self._lockouts[lockout_key] = datetime.utcnow()

        logger.warning(
            f"Triggered lockout for {user_id}:{endpoint} "
            f"(duration: {self._violation_lockout_seconds}s)"
        )

    async def clear_lockout(self, user_id: str, endpoint: str) -> bool:
        """
        Clear lockout for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            True if lockout was cleared, False if no lockout existed
        """
        async with self._lock:
            try:
                lockout_key = self._get_bucket_key(user_id, endpoint)

                if lockout_key in self._lockouts:
                    del self._lockouts[lockout_key]
                    logger.info(f"Cleared lockout for {user_id}:{endpoint}")
                    return True
                else:
                    logger.debug(f"No lockout found for {user_id}:{endpoint}")
                    return False

            except Exception as e:
                logger.error(f"Failed to clear lockout for {user_id}:{endpoint}: {e}")
                return False

    async def get_lockout_remaining(self, user_id: str, endpoint: str) -> Optional[int]:
        """
        Get remaining lockout time in seconds

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Remaining seconds if locked out, None if not locked out
        """
        try:
            lockout_key = self._get_bucket_key(user_id, endpoint)

            if lockout_key in self._lockouts:
                lockout_time = self._lockouts[lockout_key]
                now = datetime.utcnow()

                lockout_duration = timedelta(seconds=self._violation_lockout_seconds)
                expiry_time = lockout_time + lockout_duration

                if now < expiry_time:
                    remaining = (expiry_time - now).total_seconds()
                    return int(remaining)

            return None

        except Exception as e:
            logger.error(f"Error getting lockout remaining for {user_id}:{endpoint}: {e}")
            return None

    async def set_endpoint_limit(self, endpoint: str, limit_per_minute: int) -> bool:
        """
        Set rate limit for specific endpoint

        Args:
            endpoint: Endpoint name
            limit_per_minute: Requests per minute limit

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                self._endpoint_limits[endpoint] = Decimal(str(limit_per_minute))

                logger.info(f"Set rate limit for endpoint {endpoint}: {limit_per_minute}/min")
                return True

            except Exception as e:
                logger.error(f"Failed to set endpoint limit for {endpoint}: {e}")
                return False

    async def remove_endpoint_limit(self, endpoint: str) -> bool:
        """
        Remove endpoint-specific limit (use default instead)

        Args:
            endpoint: Endpoint name

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                if endpoint in self._endpoint_limits:
                    del self._endpoint_limits[endpoint]
                    logger.info(f"Removed endpoint-specific limit for {endpoint}")
                    return True
                else:
                    logger.debug(f"No endpoint-specific limit for {endpoint}")
                    return False

            except Exception as e:
                logger.error(f"Failed to remove endpoint limit for {endpoint}: {e}")
                return False

    async def get_statistics(self, user_id: str, endpoint: str) -> Dict[str, any]:
        """
        Get rate limiting statistics for user+endpoint

        Args:
            user_id: User identifier
            endpoint: Endpoint name

        Returns:
            Dictionary with statistics
        """
        try:
            bucket = self._get_or_create_bucket(user_id, endpoint)
            self._refill_bucket(bucket)

            lockout_remaining = await self.get_lockout_remaining(user_id, endpoint)

            return {
                'tokens_remaining': float(bucket.tokens),
                'capacity': float(bucket.capacity),
                'refill_rate_per_second': float(bucket.refill_rate),
                'locked_out': lockout_remaining is not None,
                'lockout_remaining_seconds': lockout_remaining,
            }

        except Exception as e:
            logger.error(f"Error getting statistics for {user_id}:{endpoint}: {e}")
            return {}

    async def clear_all_buckets(self) -> bool:
        """
        Clear all rate limit buckets (for testing/maintenance)

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                bucket_count = len(self._buckets)
                lockout_count = len(self._lockouts)

                self._buckets.clear()
                self._lockouts.clear()

                logger.warning(
                    f"Cleared all rate limit buckets: {bucket_count} buckets, {lockout_count} lockouts"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to clear all buckets: {e}")
                return False
