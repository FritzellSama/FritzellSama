"""Audit trail models for comprehensive tracking of all system actions.

Provides immutable audit logs for regulatory compliance and security monitoring.
All monetary values use Decimal, timestamps are UTC.
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class AuditEventType(Enum):
    """Types of auditable events in the system."""
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_MODIFIED = "POSITION_MODIFIED"
    BALANCE_DEPOSIT = "BALANCE_DEPOSIT"
    BALANCE_WITHDRAWAL = "BALANCE_WITHDRAWAL"
    BALANCE_TRANSFER = "BALANCE_TRANSFER"
    CONFIG_CHANGED = "CONFIG_CHANGED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    API_KEY_CREATED = "API_KEY_CREATED"
    API_KEY_REVOKED = "API_KEY_REVOKED"
    STRATEGY_STARTED = "STRATEGY_STARTED"
    STRATEGY_STOPPED = "STRATEGY_STOPPED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class AuditLog:
    """Immutable audit log entry.

    Attributes:
        event_type: Type of event being audited
        severity: Severity level of the event
        timestamp: UTC timestamp when event occurred
        user_id: Optional user identifier
        session_id: Optional session identifier
        exchange: Optional exchange name
        symbol: Optional trading pair
        order_id: Optional order identifier
        position_id: Optional position identifier
        amount: Optional monetary amount (always Decimal)
        currency: Optional currency code
        before_state: State before the event
        after_state: State after the event
        metadata: Additional context
        audit_id: Unique audit log identifier (set by database)
    """
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    amount: Optional[Decimal] = None
    currency: Optional[str] = None
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    audit_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate audit log after initialization."""
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            if self.timestamp.tzinfo is None:
                raise ValueError("Timestamp must have timezone (use UTC)")

        if self.amount is not None and not isinstance(self.amount, Decimal):
            raise ValueError("Amount must be Decimal, not float")

        if self.event_type not in AuditEventType:
            raise ValueError(f"Invalid event_type: {self.event_type}")

        if self.severity not in AuditSeverity:
            raise ValueError(f"Invalid severity: {self.severity}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary for serialization.

        Returns:
            Dictionary representation of audit log
        """
        return {
            'audit_id': self.audit_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'order_id': self.order_id,
            'position_id': self.position_id,
            'amount': str(self.amount) if self.amount else None,
            'currency': self.currency,
            'before_state': self.before_state,
            'after_state': self.after_state,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLog':
        """Create audit log from dictionary.

        Args:
            data: Dictionary with audit log data

        Returns:
            AuditLog instance
        """
        return cls(
            event_type=AuditEventType(data['event_type']),
            severity=AuditSeverity(data['severity']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            exchange=data.get('exchange'),
            symbol=data.get('symbol'),
            order_id=data.get('order_id'),
            position_id=data.get('position_id'),
            amount=Decimal(data['amount']) if data.get('amount') else None,
            currency=data.get('currency'),
            before_state=data.get('before_state', {}),
            after_state=data.get('after_state', {}),
            metadata=data.get('metadata', {}),
            audit_id=data.get('audit_id')
        )


class AuditLogger:
    """High-performance audit logger with async persistence.

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        _buffer: Buffered audit logs for batch insert
        _buffer_size: Maximum buffer size before flush
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize audit logger.

        Args:
            config: Must contain:
                - db_url: PostgreSQL connection string
                - buffer_size: Batch size for audit log writes
                - flush_interval_seconds: Max time between flushes
        """
        self.config = config
        self._validate_config()

        self.db_pool: Optional[Any] = None
        self._buffer: List[AuditLog] = []
        self._buffer_size = config['buffer_size']
        self._flush_interval = config['flush_interval_seconds']

        logger.info("AuditLogger initialized", buffer_size=self._buffer_size)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = ['db_url', 'buffer_size', 'flush_interval_seconds']
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

    async def connect(self) -> None:
        """Connect to PostgreSQL database."""
        import asyncpg

        try:
            self.db_pool = await asyncpg.create_pool(
                self.config['db_url'],
                min_size=self.config.get('db_pool_min', 2),
                max_size=self.config.get('db_pool_max', 10)
            )
            logger.info("AuditLogger connected to database")
        except Exception as e:
            logger.error("Failed to connect AuditLogger", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect and flush remaining logs."""
        if self._buffer:
            await self.flush()

        if self.db_pool:
            await self.db_pool.close()
            logger.info("AuditLogger disconnected")

    async def log(self, audit_log: AuditLog) -> None:
        """Log an audit event.

        Args:
            audit_log: AuditLog to persist
        """
        self._buffer.append(audit_log)

        if len(self._buffer) >= self._buffer_size:
            await self.flush()

    async def flush(self) -> None:
        """Flush buffered audit logs to database."""
        if not self._buffer:
            return

        logs_to_insert = self._buffer[:]
        self._buffer.clear()

        try:
            async with self.db_pool.acquire() as conn:
                records = [
                    (
                        log.event_type.value,
                        log.severity.value,
                        log.timestamp,
                        log.user_id,
                        log.session_id,
                        log.exchange,
                        log.symbol,
                        log.order_id,
                        log.position_id,
                        str(log.amount) if log.amount else None,
                        log.currency,
                        log.before_state,
                        log.after_state,
                        log.metadata
                    )
                    for log in logs_to_insert
                ]

                await conn.executemany(
                    """
                    INSERT INTO audit_logs (
                        event_type, severity, timestamp, user_id, session_id,
                        exchange, symbol, order_id, position_id, amount,
                        currency, before_state, after_state, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    records
                )

            logger.info("Audit logs flushed", count=len(logs_to_insert))

        except Exception as e:
            logger.error("Failed to flush audit logs", error=str(e))
            # Re-buffer on failure
            self._buffer.extend(logs_to_insert)
            raise

    async def query_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        user_id: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Query audit logs with filters.

        Args:
            start_time: Start of time range
            end_time: End of time range
            event_types: Optional filter by event types
            severity: Optional filter by severity
            user_id: Optional filter by user
            exchange: Optional filter by exchange
            limit: Maximum number of logs to return

        Returns:
            List of AuditLog objects
        """
        query_parts = [
            "SELECT * FROM audit_logs WHERE timestamp >= $1 AND timestamp <= $2"
        ]
        params = [start_time, end_time]
        param_idx = 3

        if event_types:
            placeholders = ', '.join(f'${i}' for i in range(param_idx, param_idx + len(event_types)))
            query_parts.append(f"AND event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])
            param_idx += len(event_types)

        if severity:
            query_parts.append(f"AND severity = ${param_idx}")
            params.append(severity.value)
            param_idx += 1

        if user_id:
            query_parts.append(f"AND user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if exchange:
            query_parts.append(f"AND exchange = ${param_idx}")
            params.append(exchange)
            param_idx += 1

        query_parts.append(f"ORDER BY timestamp DESC LIMIT ${param_idx}")
        params.append(limit)

        query = ' '.join(query_parts)

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            logs = [
                AuditLog.from_dict({
                    'audit_id': str(row['audit_id']),
                    'event_type': row['event_type'],
                    'severity': row['severity'],
                    'timestamp': row['timestamp'].isoformat(),
                    'user_id': row['user_id'],
                    'session_id': row['session_id'],
                    'exchange': row['exchange'],
                    'symbol': row['symbol'],
                    'order_id': row['order_id'],
                    'position_id': row['position_id'],
                    'amount': row['amount'],
                    'currency': row['currency'],
                    'before_state': row['before_state'],
                    'after_state': row['after_state'],
                    'metadata': row['metadata']
                })
                for row in rows
            ]

            logger.info("Audit logs queried", count=len(logs))
            return logs

        except Exception as e:
            logger.error("Failed to query audit logs", error=str(e))
            raise
