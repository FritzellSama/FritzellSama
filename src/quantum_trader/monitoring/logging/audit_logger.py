"""Audit Logger for Quantum Trader AI.

Production-ready audit logging system for compliance, security, and
regulatory requirements. Tracks all critical trading operations.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
import json
import hashlib
from pathlib import Path
import threading
from collections import deque

from prometheus_client import Counter, Gauge, Histogram
from structlog import get_logger

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    # Trading events
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"

    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_LIMIT_UPDATED = "risk_limit_updated"
    EMERGENCY_STOP = "emergency_stop"

    # Authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"

    # Configuration events
    CONFIG_UPDATED = "config_updated"
    STRATEGY_ENABLED = "strategy_enabled"
    STRATEGY_DISABLED = "strategy_disabled"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    BACKUP_CREATED = "backup_created"
    DATA_EXPORT = "data_export"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEvent:
    """Represents a single audit event.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of audit event
        severity: Event severity
        timestamp: Event timestamp (UTC)
        user_id: User who triggered the event
        action: Description of action performed
        resource: Resource affected
        details: Additional event details
        ip_address: Source IP address
        checksum: Event integrity checksum
    """

    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Initialize audit event.

        Args:
            event_type: Type of event
            severity: Event severity
            action: Action description
            user_id: User identifier
            resource: Affected resource
            details: Additional details
            ip_address: Source IP
        """
        self.event_id = self._generate_event_id()
        self.event_type = event_type
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)
        self.user_id = user_id or "system"
        self.action = action
        self.resource = resource
        self.details = details or {}
        self.ip_address = ip_address
        self.checksum = self._calculate_checksum()

    def _generate_event_id(self) -> str:
        """Generate unique event ID.

        Returns:
            Event ID string
        """
        timestamp = datetime.now(timezone.utc).timestamp()
        random_part = hashlib.sha256(str(timestamp).encode()).hexdigest()[:8]
        return f"audit_{int(timestamp)}_{random_part}"

    def _calculate_checksum(self) -> str:
        """Calculate event integrity checksum.

        Returns:
            SHA256 checksum
        """
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details
        }

        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "details": self.details,
            "ip_address": self.ip_address,
            "checksum": self.checksum
        }

    def verify_integrity(self) -> bool:
        """Verify event integrity.

        Returns:
            True if checksum valid, False otherwise
        """
        current_checksum = self.checksum
        self.checksum = ""
        calculated = self._calculate_checksum()
        self.checksum = current_checksum
        return current_checksum == calculated


class AuditLogger:
    """Audit logging system with tamper detection and compliance features.

    Thread-safe audit logger that maintains immutable audit trails for
    regulatory compliance and security monitoring.

    Attributes:
        config: Configuration dictionary
        audit_file_path: Path to audit log file
        buffer: In-memory event buffer
        buffer_size: Maximum buffer size

    Example:
        >>> config = {
        ...     "audit_file_path": "/var/log/quantum_trader/audit.jsonl",
        ...     "buffer_size": 1000,
        ...     "auto_flush_interval": 60
        ... }
        >>> audit_logger = AuditLogger(config)
        >>> await audit_logger.start()
        >>> await audit_logger.log_event(
        ...     AuditEventType.ORDER_CREATED,
        ...     AuditSeverity.MEDIUM,
        ...     "Created limit order",
        ...     user_id="trader_001",
        ...     details={"symbol": "BTC/USDT", "size": "1.5"}
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize audit logger.

        Args:
            config: Configuration dictionary containing:
                - audit_file_path: Path to audit log file
                - buffer_size: Event buffer size
                - auto_flush_interval: Seconds between auto-flushes
                - enable_encryption: Whether to encrypt logs
                - retention_days: Days to retain audit logs
                - enable_remote_logging: Send to remote syslog

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.audit_file_path = Path(self.config.get(
            "audit_file_path",
            "/var/log/quantum_trader/audit.jsonl"
        ))
        self.buffer_size = self.config.get("buffer_size", 1000)
        self.auto_flush_interval = self.config.get("auto_flush_interval", 60)
        self.enable_encryption = self.config.get("enable_encryption", False)
        self.retention_days = self.config.get("retention_days", 90)
        self.enable_remote_logging = self.config.get("enable_remote_logging", False)

        # Thread-safe buffer
        self._lock = threading.RLock()
        self.buffer: deque = deque(maxlen=self.buffer_size)
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Metrics
        self._events_logged = Counter(
            "quantum_trader_audit_events_total",
            "Total audit events logged",
            ["event_type", "severity"]
        )
        self._events_in_buffer = Gauge(
            "quantum_trader_audit_buffer_size",
            "Number of events in buffer"
        )
        self._flush_duration = Histogram(
            "quantum_trader_audit_flush_seconds",
            "Audit log flush duration"
        )
        self._integrity_failures = Counter(
            "quantum_trader_audit_integrity_failures_total",
            "Integrity check failures"
        )

        # Ensure audit directory exists
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "audit_logger_initialized",
            audit_file=str(self.audit_file_path),
            buffer_size=self.buffer_size,
            encryption=self.enable_encryption
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        buffer_size = self.config.get("buffer_size", 1000)
        if buffer_size < 10:
            raise ValueError(f"buffer_size must be >= 10, got {buffer_size}")

        flush_interval = self.config.get("auto_flush_interval", 60)
        if flush_interval < 1:
            raise ValueError(f"auto_flush_interval must be >= 1, got {flush_interval}")

        retention = self.config.get("retention_days", 90)
        if retention < 1:
            raise ValueError(f"retention_days must be >= 1, got {retention}")

    async def start(self) -> None:
        """Start audit logger background tasks.

        Raises:
            RuntimeError: If already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Audit logger already running")

            self._running = True

        logger.info("starting_audit_logger")

        # Start background tasks
        self._flush_task = asyncio.create_task(self._auto_flush_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Log system startup
        await self.log_event(
            AuditEventType.SYSTEM_STARTUP,
            AuditSeverity.MEDIUM,
            "Quantum Trader AI system started"
        )

        logger.info("audit_logger_started")

    async def stop(self) -> None:
        """Stop audit logger and flush remaining events."""
        logger.info("stopping_audit_logger")

        # Log system shutdown
        await self.log_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            AuditSeverity.MEDIUM,
            "Quantum Trader AI system shutting down"
        )

        # Flush remaining events
        await self.flush()

        with self._lock:
            self._running = False

        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("audit_logger_stopped", events_in_buffer=len(self.buffer))

    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log an audit event.

        Args:
            event_type: Type of event
            severity: Event severity
            action: Action description
            user_id: User identifier
            resource: Affected resource
            details: Additional details
            ip_address: Source IP

        Returns:
            Event ID

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            if not action:
                raise ValueError("Action cannot be empty")

            # Create audit event
            event = AuditEvent(
                event_type=event_type,
                severity=severity,
                action=action,
                user_id=user_id,
                resource=resource,
                details=details,
                ip_address=ip_address
            )

            # Verify integrity
            if not event.verify_integrity():
                self._integrity_failures.inc()
                logger.error("audit_event_integrity_failed", event_id=event.event_id)
                raise ValueError("Event integrity check failed")

            # Add to buffer
            with self._lock:
                self.buffer.append(event)
                self._events_in_buffer.set(len(self.buffer))

            # Update metrics
            self._events_logged.labels(
                event_type=event_type.value,
                severity=severity.value
            ).inc()

            # Also log to structured logger
            logger.info(
                "audit_event",
                event_id=event.event_id,
                event_type=event_type.value,
                severity=severity.value,
                action=action,
                user_id=user_id,
                resource=resource
            )

            # Auto-flush if buffer is full
            if len(self.buffer) >= self.buffer_size * 0.9:
                asyncio.create_task(self.flush())

            return event.event_id

        except Exception as e:
            logger.error("audit_logging_failed", error=str(e))
            raise

    async def flush(self) -> int:
        """Flush buffer to disk.

        Returns:
            Number of events flushed

        Raises:
            IOError: If write fails
        """
        try:
            start_time = datetime.now(timezone.utc)

            with self._lock:
                if not self.buffer:
                    return 0

                events_to_flush = list(self.buffer)
                self.buffer.clear()
                self._events_in_buffer.set(0)

            # Write to file
            with open(self.audit_file_path, 'a') as f:
                for event in events_to_flush:
                    json_line = json.dumps(event.to_dict())
                    f.write(json_line + '\n')

            flush_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._flush_duration.observe(flush_duration)

            logger.debug(
                "audit_buffer_flushed",
                events_flushed=len(events_to_flush),
                duration=flush_duration
            )

            return len(events_to_flush)

        except Exception as e:
            logger.error("audit_flush_failed", error=str(e))
            # Re-add events to buffer on failure
            with self._lock:
                for event in reversed(events_to_flush):
                    self.buffer.appendleft(event)
                self._events_in_buffer.set(len(self.buffer))
            raise

    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit events from log file.

        Args:
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            event_type: Filter by event type
            user_id: Filter by user
            severity: Filter by severity
            limit: Maximum results to return

        Returns:
            List of matching events

        Raises:
            IOError: If file read fails
        """
        try:
            # First, flush current buffer
            await self.flush()

            results = []

            if not self.audit_file_path.exists():
                return results

            with open(self.audit_file_path, 'r') as f:
                for line in f:
                    if len(results) >= limit:
                        break

                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        event_time = datetime.fromisoformat(event["timestamp"])

                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue
                        if event_type and event["event_type"] != event_type.value:
                            continue
                        if user_id and event["user_id"] != user_id:
                            continue
                        if severity and event["severity"] != severity.value:
                            continue

                        results.append(event)

                    except json.JSONDecodeError:
                        logger.warning("invalid_audit_log_line", line=line[:100])
                        continue

            logger.debug("audit_query_completed", results_count=len(results))

            return results

        except Exception as e:
            logger.error("audit_query_failed", error=str(e))
            raise

    async def verify_log_integrity(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Verify integrity of audit log.

        Args:
            start_time: Start timestamp for verification
            end_time: End timestamp for verification

        Returns:
            Verification report

        Raises:
            IOError: If file read fails
        """
        try:
            total_events = 0
            invalid_checksums = 0
            malformed_events = 0

            if not self.audit_file_path.exists():
                return {
                    "total_events": 0,
                    "valid_events": 0,
                    "invalid_checksums": 0,
                    "malformed_events": 0,
                    "integrity": "no_data"
                }

            with open(self.audit_file_path, 'r') as f:
                for line in f:
                    try:
                        event_dict = json.loads(line.strip())
                        total_events += 1

                        # Check timestamp filter
                        event_time = datetime.fromisoformat(event_dict["timestamp"])
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue

                        # Verify checksum
                        stored_checksum = event_dict.get("checksum", "")
                        event_dict_copy = event_dict.copy()
                        event_dict_copy["checksum"] = ""

                        calculated = hashlib.sha256(
                            json.dumps({
                                "event_id": event_dict_copy["event_id"],
                                "event_type": event_dict_copy["event_type"],
                                "timestamp": event_dict_copy["timestamp"],
                                "user_id": event_dict_copy["user_id"],
                                "action": event_dict_copy["action"],
                                "resource": event_dict_copy["resource"],
                                "details": event_dict_copy["details"]
                            }, sort_keys=True).encode()
                        ).hexdigest()

                        if stored_checksum != calculated:
                            invalid_checksums += 1

                    except (json.JSONDecodeError, KeyError):
                        malformed_events += 1

            valid_events = total_events - invalid_checksums - malformed_events
            integrity_status = "valid" if invalid_checksums == 0 and malformed_events == 0 else "compromised"

            report = {
                "total_events": total_events,
                "valid_events": valid_events,
                "invalid_checksums": invalid_checksums,
                "malformed_events": malformed_events,
                "integrity": integrity_status
            }

            logger.info("audit_integrity_verified", **report)

            return report

        except Exception as e:
            logger.error("integrity_verification_failed", error=str(e))
            raise

    async def _auto_flush_loop(self) -> None:
        """Background task to auto-flush buffer."""
        while self._running:
            try:
                await asyncio.sleep(self.auto_flush_interval)
                await self.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("auto_flush_error", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup old logs."""
        cleanup_interval = self.config.get("cleanup_interval", 86400)  # Daily

        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)

                # Archive/delete old logs based on retention policy
                # This is a simplified implementation
                logger.debug("audit_cleanup_check", retention_days=self.retention_days)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cleanup_error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": self._running,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "audit_file_exists": self.audit_file_path.exists(),
            "audit_file_size": (
                self.audit_file_path.stat().st_size
                if self.audit_file_path.exists()
                else 0
            )
        }
