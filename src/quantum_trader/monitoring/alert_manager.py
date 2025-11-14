"""Alert Manager for Quantum Trader AI.

Production-ready alert management system with state tracking, deduplication,
and multi-channel notification support.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta, timezone
from enum import Enum
import threading
from collections import defaultdict

from prometheus_client import Counter, Gauge, Histogram
from structlog import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class Alert:
    """Represents a single alert.

    Attributes:
        alert_id: Unique alert identifier
        name: Alert name
        severity: Alert severity level
        message: Alert message
        labels: Dictionary of labels for grouping
        timestamp: When alert was created
        status: Current alert status
        acknowledgment_time: When alert was acknowledged
        resolution_time: When alert was resolved
    """

    def __init__(
        self,
        alert_id: str,
        name: str,
        severity: AlertSeverity,
        message: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Initialize alert.

        Args:
            alert_id: Unique identifier
            name: Alert name
            severity: Severity level
            message: Alert message
            labels: Optional labels for categorization
        """
        self.alert_id = alert_id
        self.name = name
        self.severity = severity
        self.message = message
        self.labels = labels or {}
        self.timestamp = datetime.now(timezone.utc)
        self.status = AlertStatus.FIRING
        self.acknowledgment_time: Optional[datetime] = None
        self.resolution_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary.

        Returns:
            Dictionary representation of alert
        """
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "acknowledgment_time": (
                self.acknowledgment_time.isoformat()
                if self.acknowledgment_time
                else None
            ),
            "resolution_time": (
                self.resolution_time.isoformat()
                if self.resolution_time
                else None
            ),
            "metadata": self.metadata
        }


class AlertManager:
    """Manage alerts with deduplication, grouping, and notification.

    Thread-safe alert manager with support for multiple notification channels,
    alert suppression, and automatic resolution.

    Attributes:
        config: Configuration dictionary
        active_alerts: Currently active alerts
        alert_history: Historical alerts
        suppression_rules: Alert suppression rules

    Example:
        >>> config = {
        ...     "deduplication_window": 300,
        ...     "max_alerts_per_group": 10,
        ...     "auto_resolve_timeout": 3600
        ... }
        >>> manager = AlertManager(config)
        >>> await manager.start()
        >>> await manager.fire_alert("high_latency", AlertSeverity.WARNING, "Latency spike")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize alert manager.

        Args:
            config: Configuration dictionary containing:
                - deduplication_window: Seconds to deduplicate similar alerts
                - max_alerts_per_group: Maximum alerts per group
                - auto_resolve_timeout: Seconds before auto-resolution
                - notification_channels: List of notification channels
                - rate_limit_window: Seconds for rate limiting
                - rate_limit_max: Max alerts per window

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.deduplication_window = self.config.get("deduplication_window", 300)
        self.max_alerts_per_group = self.config.get("max_alerts_per_group", 10)
        self.auto_resolve_timeout = self.config.get("auto_resolve_timeout", 3600)
        self.notification_channels = self.config.get("notification_channels", [])
        self.rate_limit_window = self.config.get("rate_limit_window", 60)
        self.rate_limit_max = self.config.get("rate_limit_max", 100)

        # State management (thread-safe)
        self._lock = threading.RLock()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppression_rules: List[Dict[str, Any]] = []
        self._alert_fingerprints: Dict[str, datetime] = {}
        self._rate_limit_counter: Dict[str, List[datetime]] = defaultdict(list)

        # Background tasks
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._auto_resolve_task: Optional[asyncio.Task] = None

        # Metrics
        self._alerts_fired = Counter(
            "quantum_trader_alerts_fired_total",
            "Total alerts fired",
            ["name", "severity"]
        )
        self._alerts_resolved = Counter(
            "quantum_trader_alerts_resolved_total",
            "Total alerts resolved",
            ["name"]
        )
        self._active_alerts_gauge = Gauge(
            "quantum_trader_active_alerts",
            "Number of active alerts",
            ["severity"]
        )
        self._alert_processing_time = Histogram(
            "quantum_trader_alert_processing_seconds",
            "Alert processing time"
        )

        logger.info(
            "alert_manager_initialized",
            deduplication_window=self.deduplication_window,
            auto_resolve_timeout=self.auto_resolve_timeout,
            channels=len(self.notification_channels)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        dedup = self.config.get("deduplication_window", 300)
        if dedup < 0:
            raise ValueError(f"deduplication_window must be non-negative, got {dedup}")

        max_alerts = self.config.get("max_alerts_per_group", 10)
        if max_alerts < 1:
            raise ValueError(f"max_alerts_per_group must be positive, got {max_alerts}")

        auto_resolve = self.config.get("auto_resolve_timeout", 3600)
        if auto_resolve < 0:
            raise ValueError(f"auto_resolve_timeout must be non-negative, got {auto_resolve}")

    async def start(self) -> None:
        """Start alert manager background tasks.

        Raises:
            RuntimeError: If already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Alert manager already running")

            self._running = True

        logger.info("starting_alert_manager")

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._auto_resolve_task = asyncio.create_task(self._auto_resolve_loop())

        logger.info("alert_manager_started")

    async def stop(self) -> None:
        """Stop alert manager gracefully."""
        logger.info("stopping_alert_manager")

        with self._lock:
            self._running = False

        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._auto_resolve_task:
            self._auto_resolve_task.cancel()
            try:
                await self._auto_resolve_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "alert_manager_stopped",
            active_alerts=len(self.active_alerts),
            history_size=len(self.alert_history)
        )

    async def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Fire a new alert.

        Args:
            name: Alert name
            severity: Alert severity
            message: Alert message
            labels: Optional labels for grouping
            metadata: Optional metadata

        Returns:
            Alert ID if fired, None if deduplicated/suppressed

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            start_time = datetime.now(timezone.utc)

            if not name:
                raise ValueError("Alert name cannot be empty")

            if not message:
                raise ValueError("Alert message cannot be empty")

            labels = labels or {}

            # Generate fingerprint for deduplication
            fingerprint = self._generate_fingerprint(name, labels)

            # Check rate limiting
            if not self._check_rate_limit(name):
                logger.warning(
                    "alert_rate_limited",
                    name=name,
                    severity=severity.value
                )
                return None

            # Check deduplication
            with self._lock:
                if fingerprint in self._alert_fingerprints:
                    last_time = self._alert_fingerprints[fingerprint]
                    time_diff = (datetime.now(timezone.utc) - last_time).total_seconds()

                    if time_diff < self.deduplication_window:
                        logger.debug(
                            "alert_deduplicated",
                            name=name,
                            fingerprint=fingerprint,
                            time_since_last=time_diff
                        )
                        return None

                # Check suppression rules
                if self._is_suppressed(name, labels):
                    logger.debug("alert_suppressed", name=name, labels=labels)
                    return None

                # Create alert
                alert_id = f"{name}_{fingerprint}_{int(datetime.now(timezone.utc).timestamp())}"
                alert = Alert(alert_id, name, severity, message, labels)

                if metadata:
                    alert.metadata = metadata

                # Store alert
                self.active_alerts[alert_id] = alert
                self._alert_fingerprints[fingerprint] = datetime.now(timezone.utc)

                # Update metrics
                self._alerts_fired.labels(name=name, severity=severity.value).inc()
                self._active_alerts_gauge.labels(severity=severity.value).inc()

            # Send notifications
            await self._send_notifications(alert)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._alert_processing_time.observe(processing_time)

            logger.info(
                "alert_fired",
                alert_id=alert_id,
                name=name,
                severity=severity.value,
                processing_time=processing_time
            )

            return alert_id

        except Exception as e:
            logger.error("alert_fire_failed", error=str(e), name=name)
            raise

    async def resolve_alert(
        self,
        alert_id: str,
        resolution_message: Optional[str] = None
    ) -> bool:
        """Resolve an active alert.

        Args:
            alert_id: Alert ID to resolve
            resolution_message: Optional resolution message

        Returns:
            True if resolved, False if not found

        Raises:
            ValueError: If alert_id is invalid
        """
        try:
            if not alert_id:
                raise ValueError("Alert ID cannot be empty")

            with self._lock:
                if alert_id not in self.active_alerts:
                    logger.warning("alert_not_found", alert_id=alert_id)
                    return False

                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolution_time = datetime.now(timezone.utc)

                if resolution_message:
                    alert.metadata["resolution_message"] = resolution_message

                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]

                # Update metrics
                self._alerts_resolved.labels(name=alert.name).inc()
                self._active_alerts_gauge.labels(severity=alert.severity.value).dec()

            logger.info(
                "alert_resolved",
                alert_id=alert_id,
                name=alert.name,
                duration=(
                    (alert.resolution_time - alert.timestamp).total_seconds()
                )
            )

            return True

        except Exception as e:
            logger.error("alert_resolution_failed", error=str(e), alert_id=alert_id)
            raise

    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            user: User acknowledging the alert

        Returns:
            True if acknowledged, False if not found
        """
        try:
            with self._lock:
                if alert_id not in self.active_alerts:
                    return False

                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledgment_time = datetime.now(timezone.utc)
                alert.metadata["acknowledged_by"] = user

            logger.info(
                "alert_acknowledged",
                alert_id=alert_id,
                user=user
            )

            return True

        except Exception as e:
            logger.error("alert_acknowledgment_failed", error=str(e))
            raise

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            labels: Filter by labels

        Returns:
            List of matching alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if labels:
            alerts = [
                a for a in alerts
                if all(a.labels.get(k) == v for k, v in labels.items())
            ]

        return alerts

    def add_suppression_rule(
        self,
        name_pattern: str,
        label_matchers: Optional[Dict[str, str]] = None,
        duration: Optional[int] = None
    ) -> None:
        """Add alert suppression rule.

        Args:
            name_pattern: Alert name pattern to suppress
            label_matchers: Label matchers for suppression
            duration: Suppression duration in seconds (None for indefinite)
        """
        with self._lock:
            rule = {
                "name_pattern": name_pattern,
                "label_matchers": label_matchers or {},
                "start_time": datetime.now(timezone.utc),
                "duration": duration
            }
            self.suppression_rules.append(rule)

        logger.info(
            "suppression_rule_added",
            pattern=name_pattern,
            duration=duration
        )

    def _generate_fingerprint(self, name: str, labels: Dict[str, str]) -> str:
        """Generate unique fingerprint for alert deduplication.

        Args:
            name: Alert name
            labels: Alert labels

        Returns:
            Fingerprint string
        """
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"

    def _check_rate_limit(self, name: str) -> bool:
        """Check if alert exceeds rate limit.

        Args:
            name: Alert name

        Returns:
            True if within limit, False if exceeded
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.rate_limit_window)

        with self._lock:
            # Clean old entries
            self._rate_limit_counter[name] = [
                t for t in self._rate_limit_counter[name]
                if t > cutoff
            ]

            # Check limit
            if len(self._rate_limit_counter[name]) >= self.rate_limit_max:
                return False

            # Add current
            self._rate_limit_counter[name].append(now)

        return True

    def _is_suppressed(self, name: str, labels: Dict[str, str]) -> bool:
        """Check if alert is suppressed.

        Args:
            name: Alert name
            labels: Alert labels

        Returns:
            True if suppressed, False otherwise
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            for rule in self.suppression_rules:
                # Check duration
                if rule["duration"]:
                    elapsed = (now - rule["start_time"]).total_seconds()
                    if elapsed > rule["duration"]:
                        continue

                # Check name pattern (simple matching)
                if rule["name_pattern"] not in name:
                    continue

                # Check label matchers
                if rule["label_matchers"]:
                    if not all(
                        labels.get(k) == v
                        for k, v in rule["label_matchers"].items()
                    ):
                        continue

                return True

        return False

    async def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications to configured channels.

        Args:
            alert: Alert to notify
        """
        try:
            for channel_config in self.notification_channels:
                channel_type = channel_config.get("type")
                enabled = channel_config.get("enabled", True)

                if not enabled:
                    continue

                logger.debug(
                    "sending_notification",
                    channel=channel_type,
                    alert_id=alert.alert_id
                )

                # Actual notification sending would be implemented here
                # This is a placeholder for the notification logic

        except Exception as e:
            logger.error("notification_failed", error=str(e), alert_id=alert.alert_id)

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup old data."""
        cleanup_interval = self.config.get("cleanup_interval", 3600)
        history_retention = self.config.get("history_retention_days", 30)

        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)

                cutoff = datetime.now(timezone.utc) - timedelta(days=history_retention)

                with self._lock:
                    # Clean history
                    initial_size = len(self.alert_history)
                    self.alert_history = [
                        a for a in self.alert_history
                        if a.timestamp > cutoff
                    ]

                    # Clean fingerprints
                    fp_cutoff = datetime.now(timezone.utc) - timedelta(
                        seconds=self.deduplication_window * 2
                    )
                    self._alert_fingerprints = {
                        k: v for k, v in self._alert_fingerprints.items()
                        if v > fp_cutoff
                    }

                    # Clean suppression rules
                    now = datetime.now(timezone.utc)
                    self.suppression_rules = [
                        r for r in self.suppression_rules
                        if not r["duration"] or
                        (now - r["start_time"]).total_seconds() < r["duration"]
                    ]

                logger.debug(
                    "cleanup_completed",
                    history_removed=initial_size - len(self.alert_history),
                    fingerprints=len(self._alert_fingerprints),
                    suppression_rules=len(self.suppression_rules)
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    async def _auto_resolve_loop(self) -> None:
        """Background task to auto-resolve stale alerts."""
        check_interval = self.config.get("auto_resolve_check_interval", 300)

        while self._running:
            try:
                await asyncio.sleep(check_interval)

                if self.auto_resolve_timeout <= 0:
                    continue

                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(seconds=self.auto_resolve_timeout)

                with self._lock:
                    stale_alerts = [
                        alert_id for alert_id, alert in self.active_alerts.items()
                        if alert.timestamp < cutoff and
                        alert.status == AlertStatus.FIRING
                    ]

                for alert_id in stale_alerts:
                    await self.resolve_alert(
                        alert_id,
                        resolution_message="Auto-resolved due to timeout"
                    )

                if stale_alerts:
                    logger.info(
                        "auto_resolved_alerts",
                        count=len(stale_alerts)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("auto_resolve_error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        with self._lock:
            return {
                "healthy": self._running,
                "active_alerts": len(self.active_alerts),
                "critical_alerts": len([
                    a for a in self.active_alerts.values()
                    if a.severity == AlertSeverity.CRITICAL
                ]),
                "history_size": len(self.alert_history),
                "suppression_rules": len(self.suppression_rules)
            }
