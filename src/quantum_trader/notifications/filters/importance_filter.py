"""Importance-based alert filtering to reduce notification noise.

This module provides filtering based on alert importance, severity levels,
business hours, and customizable rules to ensure only relevant alerts
are delivered to recipients.
"""

import asyncio
import os
from decimal import Decimal
from typing import Dict, Any, Optional, Set, List, Callable
from datetime import datetime, timezone, time
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    """Alert severity levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1


class ImportanceFilter:
    """Filter alerts based on importance and configurable rules.

    Filters alerts based on severity, time of day, alert type, and custom
    filtering rules to reduce notification noise.

    Attributes:
        config: Configuration dictionary
        min_severity: Minimum severity level to allow
        business_hours_only: Only send during business hours
        enabled_alert_types: Set of enabled alert types
        custom_filters: List of custom filter functions

    Example:
        >>> config = {
        ...     "min_severity": "MEDIUM",
        ...     "business_hours_only": False,
        ...     "enabled_alert_types": ["order_error", "risk_violation"]
        ... }
        >>> filter = ImportanceFilter(config)
        >>> alert = {
        ...     "type": "order_error",
        ...     "severity": "HIGH",
        ...     "order_id": "123"
        ... }
        >>> if filter.should_send_alert(alert):
        ...     # Send alert
        ...     pass
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize importance filter.

        Args:
            config: Configuration dictionary containing:
                - min_severity: Minimum severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)
                - business_hours_only: Only send during business hours
                - business_hours_start: Business hours start time (HH:MM)
                - business_hours_end: Business hours end time (HH:MM)
                - enabled_alert_types: List of enabled alert types
                - disabled_alert_types: List of disabled alert types
                - importance_threshold: Minimum importance score (0-100)

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        # Severity filtering
        min_severity_str = self.config.get(
            "min_severity",
            os.getenv("ALERT_MIN_SEVERITY", "INFO")
        ).upper()

        try:
            self.min_severity = Severity[min_severity_str]
        except KeyError:
            logger.warning(
                "Invalid min_severity, using INFO",
                provided=min_severity_str
            )
            self.min_severity = Severity.INFO

        # Business hours filtering
        self.business_hours_only = self.config.get(
            "business_hours_only",
            os.getenv("ALERT_BUSINESS_HOURS_ONLY", "false").lower() == "true"
        )

        self.business_hours_start = self._parse_time(
            self.config.get("business_hours_start", os.getenv("BUSINESS_HOURS_START", "09:00"))
        )
        self.business_hours_end = self._parse_time(
            self.config.get("business_hours_end", os.getenv("BUSINESS_HOURS_END", "17:00"))
        )

        # Alert type filtering
        enabled_types = self.config.get(
            "enabled_alert_types",
            os.getenv("ALERT_ENABLED_TYPES", "").split(",")
        )
        self.enabled_alert_types: Optional[Set[str]] = (
            set(t.strip() for t in enabled_types if t.strip())
            if enabled_types and enabled_types[0]
            else None
        )

        disabled_types = self.config.get(
            "disabled_alert_types",
            os.getenv("ALERT_DISABLED_TYPES", "").split(",")
        )
        self.disabled_alert_types: Set[str] = set(
            t.strip() for t in disabled_types if t.strip()
        )

        # Importance threshold
        self.importance_threshold = Decimal(
            str(self.config.get(
                "importance_threshold",
                os.getenv("ALERT_IMPORTANCE_THRESHOLD", "0")
            ))
        )

        # Custom filters
        self.custom_filters: List[Callable[[Dict[str, Any]], bool]] = []

        # Statistics
        self._total_alerts = 0
        self._filtered_alerts = 0
        self._filter_reasons: Dict[str, int] = {}

        logger.info(
            "ImportanceFilter initialized",
            min_severity=self.min_severity.name,
            business_hours_only=self.business_hours_only,
            enabled_types_count=len(self.enabled_alert_types) if self.enabled_alert_types else None
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    def _parse_time(self, time_str: str) -> time:
        """Parse time string in HH:MM format.

        Args:
            time_str: Time string in HH:MM format

        Returns:
            time object

        Raises:
            ValueError: If time format is invalid
        """
        try:
            hours, minutes = time_str.split(":")
            return time(int(hours), int(minutes))
        except Exception as e:
            logger.error(
                "Failed to parse time",
                time_str=time_str,
                error=str(e)
            )
            raise ValueError(f"Invalid time format: {time_str}") from e

    def should_send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert should be sent based on importance filters.

        Args:
            alert_data: Alert data dictionary

        Returns:
            True if alert should be sent, False if filtered
        """
        try:
            self._total_alerts += 1

            # Check severity filter
            if not self._check_severity(alert_data):
                self._record_filter("severity")
                return False

            # Check business hours filter
            if not self._check_business_hours(alert_data):
                self._record_filter("business_hours")
                return False

            # Check alert type filter
            if not self._check_alert_type(alert_data):
                self._record_filter("alert_type")
                return False

            # Check importance threshold
            if not self._check_importance(alert_data):
                self._record_filter("importance")
                return False

            # Check custom filters
            if not self._check_custom_filters(alert_data):
                self._record_filter("custom")
                return False

            # All checks passed
            return True

        except Exception as e:
            logger.error(
                "Error checking alert filters",
                error=str(e),
                exc_info=True
            )
            # On error, allow alert through (fail open)
            return True

    def _check_severity(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert meets minimum severity.

        Args:
            alert_data: Alert data

        Returns:
            True if alert meets severity requirement
        """
        severity_str = alert_data.get("severity", "INFO").upper()

        try:
            alert_severity = Severity[severity_str]
        except KeyError:
            logger.warning(
                "Unknown severity level, allowing alert",
                severity=severity_str
            )
            return True

        # CRITICAL alerts always pass
        if alert_severity == Severity.CRITICAL:
            return True

        passed = alert_severity.value >= self.min_severity.value

        if not passed:
            logger.debug(
                "Alert filtered by severity",
                alert_severity=alert_severity.name,
                min_severity=self.min_severity.name
            )

        return passed

    def _check_business_hours(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert is within business hours (if enabled).

        Args:
            alert_data: Alert data

        Returns:
            True if alert should be sent based on business hours
        """
        if not self.business_hours_only:
            return True

        # CRITICAL alerts bypass business hours check
        severity_str = alert_data.get("severity", "INFO").upper()
        try:
            if Severity[severity_str] == Severity.CRITICAL:
                return True
        except KeyError:
            pass

        now = datetime.now(timezone.utc)
        current_time = now.time()

        # Check if current time is within business hours
        if self.business_hours_start <= current_time <= self.business_hours_end:
            return True

        logger.debug(
            "Alert filtered by business hours",
            current_time=current_time.strftime("%H:%M"),
            business_hours=f"{self.business_hours_start}-{self.business_hours_end}"
        )

        return False

    def _check_alert_type(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert type is enabled.

        Args:
            alert_data: Alert data

        Returns:
            True if alert type is enabled
        """
        alert_type = alert_data.get("type", "unknown")

        # Check disabled types first
        if alert_type in self.disabled_alert_types:
            logger.debug(
                "Alert filtered by disabled type",
                alert_type=alert_type
            )
            return False

        # If enabled_alert_types is set, check if type is in the list
        if self.enabled_alert_types is not None:
            if alert_type not in self.enabled_alert_types:
                logger.debug(
                    "Alert filtered by enabled types",
                    alert_type=alert_type,
                    enabled_types=list(self.enabled_alert_types)
                )
                return False

        return True

    def _check_importance(self, alert_data: Dict[str, Any]) -> bool:
        """Check if alert meets importance threshold.

        Args:
            alert_data: Alert data

        Returns:
            True if alert meets importance threshold
        """
        if self.importance_threshold <= Decimal("0"):
            return True

        # Calculate importance score
        importance = self._calculate_importance(alert_data)

        passed = importance >= self.importance_threshold

        if not passed:
            logger.debug(
                "Alert filtered by importance",
                importance=str(importance),
                threshold=str(self.importance_threshold)
            )

        return passed

    def _calculate_importance(self, alert_data: Dict[str, Any]) -> Decimal:
        """Calculate importance score for alert.

        Args:
            alert_data: Alert data

        Returns:
            Importance score (0-100)
        """
        # Base score from severity
        severity_str = alert_data.get("severity", "INFO").upper()
        try:
            severity = Severity[severity_str]
            base_score = Decimal(str(severity.value * 20))  # 20, 40, 60, 80, 100
        except KeyError:
            base_score = Decimal("50")

        # Adjust based on alert type
        alert_type = alert_data.get("type", "unknown")
        type_multipliers = {
            "risk_violation": Decimal("1.5"),
            "exchange_disconnect": Decimal("1.3"),
            "authentication_error": Decimal("1.4"),
            "order_error": Decimal("1.2"),
            "performance_alert": Decimal("0.8"),
            "system_health": Decimal("0.9"),
        }

        multiplier = type_multipliers.get(alert_type, Decimal("1.0"))
        importance = base_score * multiplier

        # Cap at 100
        return min(importance, Decimal("100"))

    def _check_custom_filters(self, alert_data: Dict[str, Any]) -> bool:
        """Check custom filter functions.

        Args:
            alert_data: Alert data

        Returns:
            True if all custom filters pass
        """
        for custom_filter in self.custom_filters:
            try:
                if not custom_filter(alert_data):
                    logger.debug(
                        "Alert filtered by custom filter",
                        filter_name=custom_filter.__name__
                    )
                    return False
            except Exception as e:
                logger.error(
                    "Error in custom filter",
                    filter_name=custom_filter.__name__,
                    error=str(e),
                    exc_info=True
                )
                # Continue with other filters

        return True

    def _record_filter(self, reason: str) -> None:
        """Record that an alert was filtered.

        Args:
            reason: Reason for filtering
        """
        self._filtered_alerts += 1
        self._filter_reasons[reason] = self._filter_reasons.get(reason, 0) + 1

    def add_custom_filter(self, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a custom filter function.

        Args:
            filter_func: Function that takes alert_data and returns bool
        """
        self.custom_filters.append(filter_func)
        logger.info("Custom filter added", filter_name=filter_func.__name__)

    def remove_custom_filter(self, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Remove a custom filter function.

        Args:
            filter_func: Filter function to remove
        """
        if filter_func in self.custom_filters:
            self.custom_filters.remove(filter_func)
            logger.info("Custom filter removed", filter_name=filter_func.__name__)

    def clear_custom_filters(self) -> None:
        """Clear all custom filter functions."""
        count = len(self.custom_filters)
        self.custom_filters.clear()
        logger.info("Custom filters cleared", count=count)

    def set_min_severity(self, severity: str) -> None:
        """Set minimum severity level.

        Args:
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO)

        Raises:
            ValueError: If severity is invalid
        """
        try:
            self.min_severity = Severity[severity.upper()]
            logger.info("Min severity updated", severity=self.min_severity.name)
        except KeyError:
            raise ValueError(f"Invalid severity: {severity}")

    def enable_alert_type(self, alert_type: str) -> None:
        """Enable a specific alert type.

        Args:
            alert_type: Alert type to enable
        """
        if self.enabled_alert_types is None:
            self.enabled_alert_types = set()

        self.enabled_alert_types.add(alert_type)

        # Remove from disabled if present
        self.disabled_alert_types.discard(alert_type)

        logger.info("Alert type enabled", alert_type=alert_type)

    def disable_alert_type(self, alert_type: str) -> None:
        """Disable a specific alert type.

        Args:
            alert_type: Alert type to disable
        """
        self.disabled_alert_types.add(alert_type)

        # Remove from enabled if present
        if self.enabled_alert_types:
            self.enabled_alert_types.discard(alert_type)

        logger.info("Alert type disabled", alert_type=alert_type)

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics.

        Returns:
            Dictionary with filter stats
        """
        filter_rate = Decimal("0")
        if self._total_alerts > 0:
            filter_rate = (
                Decimal(str(self._filtered_alerts)) /
                Decimal(str(self._total_alerts))
            ) * Decimal("100")

        return {
            "min_severity": self.min_severity.name,
            "business_hours_only": self.business_hours_only,
            "enabled_alert_types": list(self.enabled_alert_types) if self.enabled_alert_types else None,
            "disabled_alert_types": list(self.disabled_alert_types),
            "importance_threshold": str(self.importance_threshold),
            "custom_filters_count": len(self.custom_filters),
            "total_alerts": self._total_alerts,
            "filtered_alerts": self._filtered_alerts,
            "filter_rate_percent": str(filter_rate),
            "filter_reasons": self._filter_reasons.copy(),
        }

    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self._total_alerts = 0
        self._filtered_alerts = 0
        self._filter_reasons.clear()
        logger.info("Filter statistics reset")
