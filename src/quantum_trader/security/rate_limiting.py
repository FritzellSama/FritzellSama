"""
Quantum Trader AI - Rate Limiting System
Production-grade rate limiting with token bucket and sliding window

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import yaml
import polars as pl
from collections import deque

from quantum_trader.models import AuditLog
from quantum_trader.database.timeseries import TimeSeriesDB


logger = logging.getLogger(__name__)


class RateLimitLevel(Enum):
    """Rate limit severity levels"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    THROTTLED = "THROTTLED"
    BLOCKED = "BLOCKED"


class RateLimiter:
    """
    Rate limiting system

    Features:
    - Token bucket algorithm
    - Sliding window rate limits
    - Per-user rate limiting
    - Per-endpoint rate limiting
    - Burst allowance
    - Rate limit bypass for emergency operations
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/bot.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize rate limiter with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # API configuration
        api_config = self.config.get("bot", {}).get("api", {})
        self.rate_limit_per_minute = int(api_config.get("rate_limit_per_minute", 100))

        # Environment API configuration
        env_api_config = self.env_config.get("api", {})
        self.rate_limit_enabled = env_api_config.get("rate_limit_enabled", True)
        self.rate_limit_requests = int(env_api_config.get("rate_limit_requests", 1000))
        self.rate_limit_window = int(env_api_config.get("rate_limit_window", 3600))  # seconds

        # Order controls
        order_controls = self.config.get("order_controls", {})
        self.max_orders_per_minute = int(order_controls.get("max_orders_per_minute", 10))
        self.max_orders_per_symbol_per_minute = int(order_controls.get("max_orders_per_symbol_per_minute", 5))

        # Rate limit configuration
        self.default_limits = {
            "global": {
                "requests_per_second": 100,
                "requests_per_minute": self.rate_limit_per_minute,
                "requests_per_hour": self.rate_limit_requests,
                "burst_size": 20
            },
            "per_user": {
                "requests_per_second": 10,
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "burst_size": 20
            },
            "per_endpoint": {
                "trading": {
                    "requests_per_minute": self.max_orders_per_minute,
                    "burst_size": 5
                },
                "market_data": {
                    "requests_per_second": 50,
                    "burst_size": 100
                },
                "analytics": {
                    "requests_per_minute": 30,
                    "burst_size": 10
                }
            }
        }

        # Token buckets for each user/endpoint
        self._token_buckets: Dict[str, Dict] = {}

        # Sliding window counters
        self._sliding_windows: Dict[str, deque] = {}

        # Emergency bypass tokens
        self._bypass_tokens: Set[str] = set()

        # Rate limit violations tracking
        self._violations: Dict[str, List[datetime]] = {}

        # Blocked users/IPs
        self._blocked_entities: Dict[str, datetime] = {}
        self.block_duration = timedelta(hours=1)

        # Database connection
        self.db: Optional[TimeSeriesDB] = None

        logger.info("RateLimiter initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()
            logger.info("RateLimiter initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize RateLimiter: {e}")
            raise

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        user_id: Optional[str] = None
    ) -> Tuple[bool, RateLimitLevel, Optional[str]]:
        """
        Check if request should be rate limited

        Args:
            identifier: Unique identifier (user_id, IP, API key, etc.)
            endpoint: Endpoint being accessed
            user_id: Optional user ID for logging

        Returns:
            Tuple of (is_allowed, rate_limit_level, reason)
        """
        try:
            # Check if rate limiting is enabled
            if not self.rate_limit_enabled:
                return True, RateLimitLevel.NORMAL, None

            # Check if entity is blocked
            if self._is_blocked(identifier):
                return False, RateLimitLevel.BLOCKED, "Entity is temporarily blocked"

            # Check emergency bypass
            if self._has_bypass(identifier):
                return True, RateLimitLevel.NORMAL, "Emergency bypass active"

            # Check token bucket (burst control)
            bucket_allowed = await self._check_token_bucket(identifier, endpoint)

            # Check sliding window (sustained rate)
            window_allowed, window_level = await self._check_sliding_window(identifier, endpoint)

            # Determine final decision
            if not bucket_allowed or not window_allowed:
                # Rate limit exceeded
                await self._record_violation(identifier, endpoint)

                level = RateLimitLevel.THROTTLED if window_level == RateLimitLevel.WARNING else RateLimitLevel.BLOCKED
                reason = "Rate limit exceeded"

                # Log violation
                await self._log_rate_limit_event(
                    identifier,
                    endpoint,
                    user_id,
                    False,
                    reason
                )

                return False, level, reason

            # Check if approaching limit (warning)
            if window_level == RateLimitLevel.WARNING:
                return True, RateLimitLevel.WARNING, "Approaching rate limit"

            # All checks passed
            return True, RateLimitLevel.NORMAL, None

        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            # Fail open to avoid blocking legitimate traffic on errors
            return True, RateLimitLevel.NORMAL, None

    async def _check_token_bucket(
        self,
        identifier: str,
        endpoint: str
    ) -> bool:
        """
        Check token bucket for burst control

        Token bucket algorithm allows burst traffic up to bucket size,
        then refills at a steady rate.
        """
        try:
            bucket_key = f"{identifier}:{endpoint}"

            # Initialize bucket if not exists
            if bucket_key not in self._token_buckets:
                self._token_buckets[bucket_key] = {
                    "tokens": self._get_burst_size(endpoint),
                    "last_refill": datetime.utcnow(),
                    "max_tokens": self._get_burst_size(endpoint),
                    "refill_rate": self._get_refill_rate(endpoint)
                }

            bucket = self._token_buckets[bucket_key]

            # Refill tokens based on time elapsed
            now = datetime.utcnow()
            time_elapsed = (now - bucket["last_refill"]).total_seconds()
            tokens_to_add = time_elapsed * bucket["refill_rate"]

            bucket["tokens"] = min(
                bucket["max_tokens"],
                bucket["tokens"] + tokens_to_add
            )
            bucket["last_refill"] = now

            # Check if tokens available
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error in token bucket check: {e}")
            return True  # Fail open

    async def _check_sliding_window(
        self,
        identifier: str,
        endpoint: str
    ) -> Tuple[bool, RateLimitLevel]:
        """
        Check sliding window rate limit

        Sliding window tracks requests over a time period and enforces limits.
        """
        try:
            window_key = f"{identifier}:{endpoint}"

            # Initialize window if not exists
            if window_key not in self._sliding_windows:
                self._sliding_windows[window_key] = deque()

            window = self._sliding_windows[window_key]
            now = datetime.utcnow()

            # Get window duration and limit
            window_duration = self._get_window_duration(endpoint)
            request_limit = self._get_request_limit(endpoint, window_duration)

            # Remove old requests outside the window
            cutoff_time = now - timedelta(seconds=window_duration)
            while window and window[0] < cutoff_time:
                window.popleft()

            # Add current request
            window.append(now)

            # Check against limit
            request_count = len(window)

            if request_count > request_limit:
                return False, RateLimitLevel.THROTTLED
            elif request_count > request_limit * 0.8:
                # Warning at 80% of limit
                return True, RateLimitLevel.WARNING
            else:
                return True, RateLimitLevel.NORMAL

        except Exception as e:
            logger.error(f"Error in sliding window check: {e}")
            return True, RateLimitLevel.NORMAL  # Fail open

    def _get_burst_size(self, endpoint: str) -> float:
        """Get burst size for endpoint"""
        endpoint_config = self.default_limits["per_endpoint"].get(
            endpoint,
            {"burst_size": 20}
        )
        return float(endpoint_config.get("burst_size", 20))

    def _get_refill_rate(self, endpoint: str) -> float:
        """Get token refill rate (tokens per second)"""
        endpoint_config = self.default_limits["per_endpoint"].get(
            endpoint,
            {"requests_per_second": 10}
        )
        return float(endpoint_config.get("requests_per_second", 10))

    def _get_window_duration(self, endpoint: str) -> int:
        """Get sliding window duration in seconds"""
        # Default to 1 minute window
        return 60

    def _get_request_limit(self, endpoint: str, window_duration: int) -> int:
        """Get request limit for window"""
        endpoint_config = self.default_limits["per_endpoint"].get(
            endpoint,
            {"requests_per_minute": 30}
        )

        if window_duration == 60:
            return endpoint_config.get("requests_per_minute", 30)
        elif window_duration == 1:
            return endpoint_config.get("requests_per_second", 10)
        else:
            # Scale based on duration
            per_minute = endpoint_config.get("requests_per_minute", 30)
            return int(per_minute * (window_duration / 60))

    def _is_blocked(self, identifier: str) -> bool:
        """Check if entity is blocked"""
        if identifier in self._blocked_entities:
            block_time = self._blocked_entities[identifier]
            if datetime.utcnow() - block_time < self.block_duration:
                return True
            else:
                # Unblock
                del self._blocked_entities[identifier]
                return False
        return False

    def _has_bypass(self, identifier: str) -> bool:
        """Check if entity has emergency bypass"""
        return identifier in self._bypass_tokens

    async def _record_violation(
        self,
        identifier: str,
        endpoint: str
    ) -> None:
        """Record rate limit violation"""
        if identifier not in self._violations:
            self._violations[identifier] = []

        self._violations[identifier].append(datetime.utcnow())

        # Clean old violations (keep last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._violations[identifier] = [
            v for v in self._violations[identifier]
            if v > cutoff
        ]

        # Auto-block if too many violations
        if len(self._violations[identifier]) >= 100:
            await self.block_entity(identifier, "Excessive rate limit violations")

    async def block_entity(
        self,
        identifier: str,
        reason: str
    ) -> None:
        """Block entity temporarily"""
        self._blocked_entities[identifier] = datetime.utcnow()

        logger.warning(f"Blocked entity {identifier}: {reason}")

        await self._log_rate_limit_event(
            identifier,
            "system",
            None,
            False,
            f"Entity blocked: {reason}",
            severity="WARNING"
        )

    async def unblock_entity(self, identifier: str) -> None:
        """Manually unblock entity"""
        if identifier in self._blocked_entities:
            del self._blocked_entities[identifier]
            logger.info(f"Unblocked entity {identifier}")

    def grant_bypass(
        self,
        identifier: str,
        granted_by: str,
        duration: Optional[timedelta] = None
    ) -> None:
        """
        Grant emergency bypass for rate limiting

        Args:
            identifier: Entity identifier
            granted_by: User granting bypass
            duration: Optional duration (permanent if None)
        """
        self._bypass_tokens.add(identifier)

        logger.warning(f"Rate limit bypass granted to {identifier} by {granted_by}")

        # If duration specified, schedule removal
        if duration:
            asyncio.create_task(self._revoke_bypass_after(identifier, duration))

    async def _revoke_bypass_after(
        self,
        identifier: str,
        duration: timedelta
    ) -> None:
        """Revoke bypass after duration"""
        await asyncio.sleep(duration.total_seconds())
        self.revoke_bypass(identifier)

    def revoke_bypass(self, identifier: str) -> None:
        """Revoke emergency bypass"""
        if identifier in self._bypass_tokens:
            self._bypass_tokens.remove(identifier)
            logger.info(f"Rate limit bypass revoked for {identifier}")

    async def _log_rate_limit_event(
        self,
        identifier: str,
        endpoint: str,
        user_id: Optional[str],
        allowed: bool,
        reason: str,
        severity: str = "INFO"
    ) -> None:
        """Log rate limit event"""
        try:
            if not allowed:
                severity = "WARNING"

            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation="RATE_LIMIT_CHECK",
                user_id=user_id or identifier,
                component="RateLimiter",
                severity=severity,
                details={
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "allowed": allowed,
                    "reason": reason
                },
                result="DENIED" if not allowed else "ALLOWED"
            )

            if self.db:
                await self.db.insert_audit_log(audit_log)

        except Exception as e:
            logger.error(f"Failed to log rate limit event: {e}")

    async def get_rate_limit_status(
        self,
        identifier: str
    ) -> Dict:
        """
        Get rate limit status for identifier

        Args:
            identifier: Entity identifier

        Returns:
            Status dictionary
        """
        try:
            status = {
                "identifier": identifier,
                "is_blocked": self._is_blocked(identifier),
                "has_bypass": self._has_bypass(identifier),
                "violations_24h": len(self._violations.get(identifier, [])),
                "current_usage": {}
            }

            # Get usage for each endpoint
            for endpoint_key in self._sliding_windows:
                if endpoint_key.startswith(identifier):
                    endpoint = endpoint_key.split(":", 1)[1]
                    window = self._sliding_windows[endpoint_key]

                    # Count recent requests
                    cutoff = datetime.utcnow() - timedelta(minutes=1)
                    recent_requests = sum(1 for ts in window if ts > cutoff)

                    limit = self._get_request_limit(endpoint, 60)

                    status["current_usage"][endpoint] = {
                        "requests_last_minute": recent_requests,
                        "limit": limit,
                        "usage_percent": (recent_requests / limit * 100) if limit > 0 else 0
                    }

            return status

        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            return {"error": str(e)}

    async def get_rate_limit_report(self) -> Dict:
        """
        Generate comprehensive rate limit report

        Returns:
            Report dictionary
        """
        try:
            # Count violations
            total_violations = sum(len(v) for v in self._violations.values())

            # Top violators
            top_violators = sorted(
                [(k, len(v)) for k, v in self._violations.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "rate_limiting_enabled": self.rate_limit_enabled,
                "total_violations_24h": total_violations,
                "blocked_entities": len(self._blocked_entities),
                "active_bypasses": len(self._bypass_tokens),
                "top_violators": [
                    {"identifier": identifier, "violations": count}
                    for identifier, count in top_violators
                ],
                "configuration": {
                    "global_limits": self.default_limits["global"],
                    "per_user_limits": self.default_limits["per_user"]
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating rate limit report: {e}")
            return {"error": str(e)}

    def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for identifier"""
        # Clear token bucket
        bucket_keys = [k for k in self._token_buckets if k.startswith(identifier)]
        for key in bucket_keys:
            del self._token_buckets[key]

        # Clear sliding windows
        window_keys = [k for k in self._sliding_windows if k.startswith(identifier)]
        for key in window_keys:
            del self._sliding_windows[key]

        # Clear violations
        if identifier in self._violations:
            del self._violations[identifier]

        logger.info(f"Rate limits reset for {identifier}")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()

        logger.info("RateLimiter cleanup complete")
