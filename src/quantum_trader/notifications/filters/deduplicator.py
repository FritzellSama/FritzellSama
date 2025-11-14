"""Alert deduplication to prevent notification spam.

This module provides functionality to deduplicate alerts based on content,
time windows, and similarity to prevent overwhelming recipients with
duplicate notifications.
"""

import asyncio
import os
import hashlib
from decimal import Decimal
from typing import Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class AlertRecord:
    """Record of a sent alert.

    Attributes:
        alert_hash: Hash of alert content
        alert_type: Type of alert
        severity: Alert severity
        timestamp: When alert was sent
        count: Number of times this alert was triggered
        data: Alert data dictionary
    """
    alert_hash: str
    alert_type: str
    severity: str
    timestamp: datetime
    count: int = 1
    data: Dict[str, Any] = field(default_factory=dict)


class Deduplicator:
    """Alert deduplication system.

    Prevents duplicate alerts from being sent within configurable time windows
    based on alert content and type.

    Attributes:
        config: Configuration dictionary
        dedup_window: Time window for deduplication in seconds
        alert_records: Dictionary of alert hash to AlertRecord
        is_running: Whether deduplicator is active

    Example:
        >>> config = {
        ...     "dedup_window_seconds": 300,
        ...     "max_records": 10000
        ... }
        >>> dedup = Deduplicator(config)
        >>> await dedup.start()
        >>> alert = {"type": "order_error", "order_id": "123"}
        >>> if dedup.should_send_alert(alert):
        ...     # Send alert
        ...     dedup.record_alert(alert)
        >>> await dedup.stop()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize deduplicator.

        Args:
            config: Configuration dictionary containing:
                - dedup_window_seconds: Time window for deduplication
                - max_records: Maximum number of records to keep
                - cleanup_interval: How often to clean up old records
                - similarity_threshold: Similarity threshold for fuzzy matching

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.dedup_window = timedelta(
            seconds=int(
                self.config.get(
                    "dedup_window_seconds",
                    os.getenv("DEDUP_WINDOW_SECONDS", "300")
                )
            )
        )

        self.max_records = int(
            self.config.get("max_records", os.getenv("DEDUP_MAX_RECORDS", "10000"))
        )

        self.cleanup_interval = int(
            self.config.get(
                "cleanup_interval",
                os.getenv("DEDUP_CLEANUP_INTERVAL", "60")
            )
        )

        # Alert records storage
        self.alert_records: Dict[str, AlertRecord] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self._total_alerts = 0
        self._deduplicated_alerts = 0
        self._cleanup_count = 0

        # Background tasks
        self.is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            "Deduplicator initialized",
            dedup_window_seconds=self.dedup_window.total_seconds(),
            max_records=self.max_records
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    async def start(self) -> None:
        """Start the deduplicator.

        Raises:
            RuntimeError: If deduplicator is already running
        """
        if self.is_running:
            raise RuntimeError("Deduplicator is already running")

        self.is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Deduplicator started")

    async def stop(self) -> None:
        """Stop the deduplicator."""
        if not self.is_running:
            logger.warning("Deduplicator is not running")
            return

        self.is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "Deduplicator stopped",
            total_alerts=self._total_alerts,
            deduplicated_alerts=self._deduplicated_alerts,
            dedup_rate=f"{(self._deduplicated_alerts / max(self._total_alerts, 1)) * 100:.2f}%"
        )

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up old alert records."""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_records()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Error in cleanup loop",
                    error=str(e),
                    exc_info=True
                )

    async def _cleanup_old_records(self) -> None:
        """Remove alert records older than dedup window."""
        try:
            async with self._lock:
                now = datetime.now(timezone.utc)
                cutoff_time = now - self.dedup_window

                # Find expired records
                expired_hashes = [
                    alert_hash
                    for alert_hash, record in self.alert_records.items()
                    if record.timestamp < cutoff_time
                ]

                # Remove expired records
                for alert_hash in expired_hashes:
                    del self.alert_records[alert_hash]

                if expired_hashes:
                    self._cleanup_count += 1
                    logger.debug(
                        "Cleaned up old alert records",
                        removed_count=len(expired_hashes),
                        remaining_count=len(self.alert_records),
                        cleanup_count=self._cleanup_count
                    )

                # Enforce max records limit
                if len(self.alert_records) > self.max_records:
                    # Remove oldest records
                    sorted_records = sorted(
                        self.alert_records.items(),
                        key=lambda x: x[1].timestamp
                    )
                    to_remove = len(self.alert_records) - self.max_records
                    for alert_hash, _ in sorted_records[:to_remove]:
                        del self.alert_records[alert_hash]

                    logger.warning(
                        "Enforced max records limit",
                        removed_count=to_remove,
                        max_records=self.max_records
                    )

        except Exception as e:
            logger.error("Error cleaning up records", error=str(e), exc_info=True)

    def _compute_alert_hash(self, alert_data: Dict[str, Any]) -> str:
        """Compute hash for alert data.

        Args:
            alert_data: Alert data dictionary

        Returns:
            Hash string
        """
        # Create a canonical representation of the alert
        alert_type = alert_data.get("type", "unknown")
        severity = alert_data.get("severity", "unknown")

        # Include key fields that identify unique alerts
        key_fields = []

        # Always include type and severity
        key_fields.extend([alert_type, severity])

        # Add type-specific fields
        if alert_type == "order_error":
            key_fields.append(alert_data.get("order_id", ""))
            key_fields.append(alert_data.get("exchange", ""))
        elif alert_type == "risk_violation":
            key_fields.append(alert_data.get("violation_type", ""))
            key_fields.append(alert_data.get("strategy", ""))
        elif alert_type == "exchange_disconnect":
            key_fields.append(alert_data.get("exchange", ""))
        elif alert_type == "performance_alert":
            key_fields.append(alert_data.get("metric_name", ""))
            key_fields.append(alert_data.get("component", ""))
        else:
            # For unknown types, include all fields
            key_fields.extend(str(v) for v in sorted(alert_data.items()))

        # Compute hash
        hash_input = "|".join(key_fields)
        alert_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        return alert_hash

    def should_send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert should be sent based on deduplication rules.

        Args:
            alert_data: Alert data dictionary

        Returns:
            True if alert should be sent, False if duplicate
        """
        try:
            self._total_alerts += 1

            alert_hash = self._compute_alert_hash(alert_data)
            now = datetime.now(timezone.utc)

            # Check if we have a recent record of this alert
            if alert_hash in self.alert_records:
                record = self.alert_records[alert_hash]

                # Check if within dedup window
                if now - record.timestamp < self.dedup_window:
                    # Duplicate alert - don't send
                    record.count += 1
                    self._deduplicated_alerts += 1

                    logger.debug(
                        "Alert deduplicated",
                        alert_type=alert_data.get("type"),
                        alert_hash=alert_hash,
                        count=record.count,
                        age_seconds=(now - record.timestamp).total_seconds()
                    )

                    return False

            # Alert should be sent
            return True

        except Exception as e:
            logger.error(
                "Error checking alert deduplication",
                error=str(e),
                exc_info=True
            )
            # On error, allow alert through
            return True

    def record_alert(self, alert_data: Dict[str, Any]) -> None:
        """Record that an alert was sent.

        Args:
            alert_data: Alert data dictionary
        """
        try:
            alert_hash = self._compute_alert_hash(alert_data)
            now = datetime.now(timezone.utc)

            # Create or update record
            if alert_hash in self.alert_records:
                record = self.alert_records[alert_hash]
                record.timestamp = now
                record.count = 1  # Reset count
            else:
                self.alert_records[alert_hash] = AlertRecord(
                    alert_hash=alert_hash,
                    alert_type=alert_data.get("type", "unknown"),
                    severity=alert_data.get("severity", "unknown"),
                    timestamp=now,
                    data=alert_data.copy()
                )

            logger.debug(
                "Alert recorded",
                alert_type=alert_data.get("type"),
                alert_hash=alert_hash
            )

        except Exception as e:
            logger.error("Error recording alert", error=str(e), exc_info=True)

    def get_alert_count(self, alert_hash: str) -> int:
        """Get the count of how many times an alert was triggered.

        Args:
            alert_hash: Alert hash

        Returns:
            Count of alert occurrences
        """
        if alert_hash in self.alert_records:
            return self.alert_records[alert_hash].count
        return 0

    def get_recent_alerts(
        self,
        limit: Optional[int] = None,
        alert_type: Optional[str] = None
    ) -> list[AlertRecord]:
        """Get recent alert records.

        Args:
            limit: Maximum number of records to return
            alert_type: Filter by alert type (optional)

        Returns:
            List of AlertRecord objects
        """
        records = list(self.alert_records.values())

        # Filter by type if specified
        if alert_type:
            records = [r for r in records if r.alert_type == alert_type]

        # Sort by timestamp (newest first)
        records.sort(key=lambda r: r.timestamp, reverse=True)

        # Apply limit
        if limit:
            records = records[:limit]

        return records

    def clear_records(self, alert_type: Optional[str] = None) -> int:
        """Clear alert records.

        Args:
            alert_type: Clear only records of this type (optional)

        Returns:
            Number of records cleared
        """
        if alert_type:
            # Clear specific type
            to_remove = [
                h for h, r in self.alert_records.items()
                if r.alert_type == alert_type
            ]
            for alert_hash in to_remove:
                del self.alert_records[alert_hash]
            count = len(to_remove)
        else:
            # Clear all
            count = len(self.alert_records)
            self.alert_records.clear()

        logger.info("Alert records cleared", count=count, alert_type=alert_type)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics.

        Returns:
            Dictionary with deduplicator stats
        """
        dedup_rate = Decimal("0")
        if self._total_alerts > 0:
            dedup_rate = (
                Decimal(str(self._deduplicated_alerts)) /
                Decimal(str(self._total_alerts))
            ) * Decimal("100")

        return {
            "is_running": self.is_running,
            "total_alerts": self._total_alerts,
            "deduplicated_alerts": self._deduplicated_alerts,
            "deduplication_rate_percent": str(dedup_rate),
            "active_records": len(self.alert_records),
            "max_records": self.max_records,
            "dedup_window_seconds": int(self.dedup_window.total_seconds()),
            "cleanup_count": self._cleanup_count,
        }
