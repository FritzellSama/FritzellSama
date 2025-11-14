"""
Rate Limiter - Notification rate limiting and throttling.

Prevents notification spam using token bucket and sliding window algorithms.
"""

import asyncio
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from collections import deque
import os
import time
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int  # Maximum requests per window
    window_seconds: float  # Time window in seconds
    burst_size: int  # Maximum burst size
    key_prefix: str = "rate_limit"


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    capacity: int
    refill_rate: float  # Tokens per second
    tokens: float
    last_refill: float = field(default_factory=time.time)


class RateLimiter:
    """
    Thread-safe rate limiter for notifications.

    Implements token bucket and sliding window algorithms to prevent
    notification spam and enforce rate limits per channel/user.

    Attributes:
        config: Configuration dictionary from environment
        buckets: Token buckets per key
        sliding_windows: Sliding window counters per key
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Optional configuration override. If None, loads from environment.
        """
        self.config = config or self._load_config()
        self._validate_config()

        # Token buckets for burst control
        self._buckets: Dict[str, TokenBucket] = {}

        # Sliding windows for rate tracking
        self._sliding_windows: Dict[str, deque] = {}

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("rate_limiter_initialized", config=self._get_safe_config())

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            # Global limits
            "global_max_per_minute": int(os.getenv("RATE_LIMIT_GLOBAL_MAX_PER_MINUTE", "100")),
            "global_max_per_hour": int(os.getenv("RATE_LIMIT_GLOBAL_MAX_PER_HOUR", "1000")),

            # Per-channel limits
            "telegram_max_per_minute": int(os.getenv("RATE_LIMIT_TELEGRAM_MAX_PER_MINUTE", "20")),
            "telegram_burst_size": int(os.getenv("RATE_LIMIT_TELEGRAM_BURST_SIZE", "5")),

            "email_max_per_minute": int(os.getenv("RATE_LIMIT_EMAIL_MAX_PER_MINUTE", "10")),
            "email_burst_size": int(os.getenv("RATE_LIMIT_EMAIL_BURST_SIZE", "3")),

            "slack_max_per_minute": int(os.getenv("RATE_LIMIT_SLACK_MAX_PER_MINUTE", "30")),
            "slack_burst_size": int(os.getenv("RATE_LIMIT_SLACK_BURST_SIZE", "10")),

            "webhook_max_per_minute": int(os.getenv("RATE_LIMIT_WEBHOOK_MAX_PER_MINUTE", "50")),
            "webhook_burst_size": int(os.getenv("RATE_LIMIT_WEBHOOK_BURST_SIZE", "15")),

            "sms_max_per_minute": int(os.getenv("RATE_LIMIT_SMS_MAX_PER_MINUTE", "5")),
            "sms_burst_size": int(os.getenv("RATE_LIMIT_SMS_BURST_SIZE", "2")),

            # Per-user limits
            "user_max_per_minute": int(os.getenv("RATE_LIMIT_USER_MAX_PER_MINUTE", "30")),
            "user_burst_size": int(os.getenv("RATE_LIMIT_USER_BURST_SIZE", "10")),

            # Cleanup settings
            "cleanup_interval_seconds": int(os.getenv("RATE_LIMIT_CLEANUP_INTERVAL", "300")),
            "bucket_ttl_seconds": int(os.getenv("RATE_LIMIT_BUCKET_TTL", "3600")),
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ["global_max_per_minute", "global_max_per_hour"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

            if self.config[key] < 1:
                raise ValueError(f"{key} must be >= 1")

    def _get_safe_config(self) -> Dict[str, Any]:
        """Get sanitized config for logging."""
        return {k: v for k, v in self.config.items() if not k.lower().endswith("_key")}

    async def start(self) -> None:
        """Start the rate limiter and cleanup task."""
        if self._running:
            logger.warning("rate_limiter_already_running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("rate_limiter_started")

    async def stop(self) -> None:
        """Stop the rate limiter."""
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("rate_limiter_stopped")

    async def check_rate_limit(
        self,
        key: str,
        channel: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if request is allowed under rate limits.

        Args:
            key: Unique key for this rate limit check
            channel: Optional channel name for channel-specific limits
            user_id: Optional user ID for user-specific limits

        Returns:
            Tuple of (allowed: bool, retry_after: Optional[float])
            retry_after is seconds to wait before retrying if not allowed
        """
        async with self._lock:
            try:
                # Check global limits
                global_allowed, global_retry = await self._check_global_limits(key)
                if not global_allowed:
                    logger.warning(
                        "rate_limit_exceeded_global",
                        key=key,
                        retry_after=global_retry
                    )
                    return False, global_retry

                # Check channel-specific limits
                if channel:
                    channel_allowed, channel_retry = await self._check_channel_limits(key, channel)
                    if not channel_allowed:
                        logger.warning(
                            "rate_limit_exceeded_channel",
                            key=key,
                            channel=channel,
                            retry_after=channel_retry
                        )
                        return False, channel_retry

                # Check user-specific limits
                if user_id:
                    user_allowed, user_retry = await self._check_user_limits(key, user_id)
                    if not user_allowed:
                        logger.warning(
                            "rate_limit_exceeded_user",
                            key=key,
                            user_id=user_id,
                            retry_after=user_retry
                        )
                        return False, user_retry

                logger.debug("rate_limit_passed", key=key, channel=channel, user_id=user_id)
                return True, None

            except Exception as e:
                logger.error("rate_limit_check_error", key=key, error=str(e))
                # Fail open to avoid blocking all notifications
                return True, None

    async def _check_global_limits(self, key: str) -> Tuple[bool, Optional[float]]:
        """Check global rate limits."""
        minute_key = f"global:minute:{key}"
        hour_key = f"global:hour:{key}"

        # Check per-minute limit
        minute_allowed = await self._check_sliding_window(
            minute_key,
            self.config["global_max_per_minute"],
            60.0
        )

        if not minute_allowed:
            return False, 60.0

        # Check per-hour limit
        hour_allowed = await self._check_sliding_window(
            hour_key,
            self.config["global_max_per_hour"],
            3600.0
        )

        if not hour_allowed:
            return False, 3600.0

        return True, None

    async def _check_channel_limits(self, key: str, channel: str) -> Tuple[bool, Optional[float]]:
        """Check channel-specific rate limits."""
        channel_lower = channel.lower()
        max_per_min_key = f"{channel_lower}_max_per_minute"
        burst_size_key = f"{channel_lower}_burst_size"

        if max_per_min_key not in self.config:
            logger.debug("no_channel_limit_configured", channel=channel)
            return True, None

        max_per_minute = self.config[max_per_min_key]
        burst_size = self.config.get(burst_size_key, max_per_minute)

        bucket_key = f"channel:{channel_lower}:{key}"

        # Check token bucket for burst control
        bucket_allowed = await self._check_token_bucket(
            bucket_key,
            burst_size,
            max_per_minute / 60.0  # Refill rate per second
        )

        if not bucket_allowed:
            # Calculate retry after based on refill rate
            retry_after = 60.0 / max_per_minute
            return False, retry_after

        # Check sliding window for sustained rate
        window_key = f"window:{channel_lower}:{key}"
        window_allowed = await self._check_sliding_window(
            window_key,
            max_per_minute,
            60.0
        )

        if not window_allowed:
            return False, 60.0

        return True, None

    async def _check_user_limits(self, key: str, user_id: str) -> Tuple[bool, Optional[float]]:
        """Check user-specific rate limits."""
        max_per_minute = self.config.get("user_max_per_minute", 30)
        burst_size = self.config.get("user_burst_size", 10)

        bucket_key = f"user:{user_id}:{key}"

        # Check token bucket
        bucket_allowed = await self._check_token_bucket(
            bucket_key,
            burst_size,
            max_per_minute / 60.0
        )

        if not bucket_allowed:
            retry_after = 60.0 / max_per_minute
            return False, retry_after

        # Check sliding window
        window_key = f"user_window:{user_id}:{key}"
        window_allowed = await self._check_sliding_window(
            window_key,
            max_per_minute,
            60.0
        )

        if not window_allowed:
            return False, 60.0

        return True, None

    async def _check_token_bucket(
        self,
        key: str,
        capacity: int,
        refill_rate: float
    ) -> bool:
        """
        Check and update token bucket.

        Args:
            key: Bucket key
            capacity: Maximum tokens
            refill_rate: Tokens added per second

        Returns:
            True if request allowed, False otherwise
        """
        now = time.time()

        # Get or create bucket
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate,
                tokens=float(capacity),
                last_refill=now
            )

        bucket = self._buckets[key]

        # Refill tokens based on elapsed time
        elapsed = now - bucket.last_refill
        tokens_to_add = elapsed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now

        # Check if token available
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True

        return False

    async def _check_sliding_window(
        self,
        key: str,
        max_requests: int,
        window_seconds: float
    ) -> bool:
        """
        Check and update sliding window counter.

        Args:
            key: Window key
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds

        Returns:
            True if request allowed, False otherwise
        """
        now = time.time()

        # Get or create window
        if key not in self._sliding_windows:
            self._sliding_windows[key] = deque()

        window = self._sliding_windows[key]

        # Remove old entries outside window
        cutoff = now - window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        # Check if under limit
        if len(window) < max_requests:
            window.append(now)
            return True

        return False

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old buckets and windows."""
        cleanup_interval = self.config.get("cleanup_interval_seconds", 300)

        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_loop_error", error=str(e))

    async def _cleanup(self) -> None:
        """Clean up old buckets and windows."""
        async with self._lock:
            try:
                now = time.time()
                ttl = self.config.get("bucket_ttl_seconds", 3600)

                # Cleanup old buckets
                buckets_to_remove = []
                for key, bucket in self._buckets.items():
                    if now - bucket.last_refill > ttl:
                        buckets_to_remove.append(key)

                for key in buckets_to_remove:
                    del self._buckets[key]

                # Cleanup old windows
                windows_to_remove = []
                for key, window in self._sliding_windows.items():
                    if not window or (now - window[-1]) > ttl:
                        windows_to_remove.append(key)

                for key in windows_to_remove:
                    del self._sliding_windows[key]

                if buckets_to_remove or windows_to_remove:
                    logger.info(
                        "rate_limiter_cleanup_completed",
                        buckets_removed=len(buckets_to_remove),
                        windows_removed=len(windows_to_remove)
                    )

            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    async def reset(self, key: Optional[str] = None) -> None:
        """
        Reset rate limits.

        Args:
            key: Optional specific key to reset. If None, resets all.
        """
        async with self._lock:
            if key:
                # Reset specific key
                keys_to_remove = [k for k in self._buckets.keys() if key in k]
                for k in keys_to_remove:
                    del self._buckets[k]

                keys_to_remove = [k for k in self._sliding_windows.keys() if key in k]
                for k in keys_to_remove:
                    del self._sliding_windows[k]

                logger.info("rate_limit_reset_for_key", key=key, count=len(keys_to_remove))
            else:
                # Reset all
                bucket_count = len(self._buckets)
                window_count = len(self._sliding_windows)

                self._buckets.clear()
                self._sliding_windows.clear()

                logger.info(
                    "rate_limit_reset_all",
                    buckets_cleared=bucket_count,
                    windows_cleared=window_count
                )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "active_buckets": len(self._buckets),
            "active_windows": len(self._sliding_windows),
            "running": self._running,
        }
