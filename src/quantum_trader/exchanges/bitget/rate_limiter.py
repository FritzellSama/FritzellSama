"""
Bitget Rate Limiter - Production-grade rate limiting for Bitget API.

Implements sliding window rate limiting with IP-based and UID-based limits,
adaptive backoff, and circuit breaker patterns.
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


class RateLimitTier(Enum):
    """Rate limit tiers for Bitget API."""
    IP_PUBLIC = "IP_PUBLIC"
    IP_PRIVATE = "IP_PRIVATE"
    UID_TRADING = "UID_TRADING"
    UID_ACCOUNT = "UID_ACCOUNT"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for specific tier."""
    requests_per_second: Decimal
    requests_per_minute: Decimal
    burst_size: int
    weight_multiplier: Decimal = Decimal("1.0")


@dataclass
class RateLimitWindow:
    """Sliding window for rate limit tracking."""
    config: RateLimitConfig
    second_window: deque = field(default_factory=lambda: deque(maxlen=1000))
    minute_window: deque = field(default_factory=lambda: deque(maxlen=10000))
    tokens: Decimal = Decimal("0")
    last_refill: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    violation_count: int = 0
    consecutive_violations: int = 0


class BitgetRateLimiter:
    """
    Production-grade rate limiter for Bitget API.

    Handles both IP-based and UID-based rate limits with multiple time windows.
    Implements adaptive backoff and circuit breaker for protection.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Bitget rate limiter."""
        self.config = config
        self._validate_config()

        self.windows: Dict[RateLimitTier, RateLimitWindow] = {}
        self._initialize_windows()

        # Circuit breaker configuration
        circuit_config = self.config.get('circuit_breaker', {})
        self.circuit_breaker_threshold = int(circuit_config.get('threshold', 5))
        self.circuit_breaker_cooldown = Decimal(str(circuit_config.get('cooldown', 60)))
        self.circuit_breaker_enabled = circuit_config.get('enabled', True)
        self.circuit_open = False
        self.circuit_open_time: Optional[datetime] = None

        # Adaptive backoff configuration
        backoff_config = self.config.get('adaptive_backoff', {})
        self.adaptive_backoff_enabled = backoff_config.get('enabled', True)
        self.base_backoff = Decimal(str(backoff_config.get('base_seconds', '0.1')))
        self.max_backoff = Decimal(str(backoff_config.get('max_seconds', '10.0')))
        self.backoff_multiplier = Decimal(str(backoff_config.get('multiplier', '2.0')))

        self._lock = asyncio.Lock()
        self._metrics: Dict[str, Any] = {
            'total_requests': 0,
            'throttled_requests': 0,
            'violations': 0,
            'circuit_breaks': 0,
            'average_wait_time': Decimal("0")
        }

        logger.info(
            "bitget_rate_limiter_initialized",
            tiers=len(self.windows),
            circuit_breaker=self.circuit_breaker_enabled,
            adaptive_backoff=self.adaptive_backoff_enabled
        )

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        if 'rate_limits' not in self.config:
            raise ValueError("Missing 'rate_limits' in configuration")

        required_tiers = ['ip_public', 'uid_trading']
        for tier in required_tiers:
            if tier not in self.config['rate_limits']:
                logger.warning(
                    "missing_rate_limit_tier",
                    tier=tier
                )

    def _initialize_windows(self) -> None:
        """Initialize rate limit windows from configuration."""
        rate_limits = self.config['rate_limits']

        for tier_str, limit_config in rate_limits.items():
            try:
                tier = RateLimitTier[tier_str.upper()]
            except KeyError:
                logger.warning(
                    "unknown_rate_limit_tier",
                    tier=tier_str
                )
                continue

            config = RateLimitConfig(
                requests_per_second=Decimal(str(limit_config.get('rps', 10))),
                requests_per_minute=Decimal(str(limit_config.get('rpm', 500))),
                burst_size=int(limit_config.get('burst', 20)),
                weight_multiplier=Decimal(str(limit_config.get('weight_multiplier', 1.0)))
            )

            window = RateLimitWindow(
                config=config,
                tokens=Decimal(str(config.burst_size))
            )

            self.windows[tier] = window

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter metrics."""
        window_stats = {}
        for tier, window in self.windows.items():
            window_stats[tier.value] = {
                'tokens': float(window.tokens),
                'burst_size': window.config.burst_size,
                'rps_limit': float(window.config.requests_per_second),
                'rpm_limit': float(window.config.requests_per_minute),
                'violations': window.violation_count,
                'consecutive_violations': window.consecutive_violations
            }

        return {
            **self._metrics,
            'average_wait_time': float(self._metrics['average_wait_time']),
            'windows': window_stats,
            'circuit_open': self.circuit_open
        }
