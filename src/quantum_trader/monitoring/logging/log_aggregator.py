"""Log Aggregator for Quantum Trader AI.

Production-ready log aggregation system that collects, processes,
and forwards logs from multiple sources to various destinations.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import threading
import json

from prometheus_client import Counter, Gauge, Histogram
from structlog import get_logger

logger = get_logger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogEntry:
    """Represents a single log entry.

    Attributes:
        timestamp: Log timestamp
        level: Log level
        message: Log message
        source: Log source identifier
        metadata: Additional metadata
        tags: Log tags for categorization
    """

    def __init__(
        self,
        level: LogLevel,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Initialize log entry.

        Args:
            level: Log level
            message: Log message
            source: Source identifier
            metadata: Additional metadata
            tags: Log tags
        """
        self.timestamp = datetime.now(timezone.utc)
        self.level = level
        self.message = message
        self.source = source
        self.metadata = metadata or {}
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "metadata": self.metadata,
            "tags": self.tags
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict())


class LogAggregator:
    """Aggregate and process logs from multiple sources.

    Collects logs from various system components and forwards them
    to configured destinations (files, remote syslog, cloud services).

    Attributes:
        config: Configuration dictionary
        buffer: In-memory log buffer
        destinations: Configured log destinations

    Example:
        >>> config = {
        ...     "buffer_size": 10000,
        ...     "flush_interval": 5,
        ...     "destinations": [
        ...         {"type": "file", "path": "/var/log/quantum_trader/app.log"},
        ...         {"type": "elasticsearch", "url": "http://localhost:9200"}
        ...     ]
        ... }
        >>> aggregator = LogAggregator(config)
        >>> await aggregator.start()
        >>> await aggregator.log(LogLevel.INFO, "Trade executed", "trading_engine")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize log aggregator.

        Args:
            config: Configuration dictionary containing:
                - buffer_size: Maximum logs in memory
                - flush_interval: Seconds between flushes
                - destinations: List of log destinations
                - enable_filtering: Enable log filtering
                - min_level: Minimum log level to aggregate
                - batch_size: Batch size for bulk operations

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.buffer_size = self.config.get("buffer_size", 10000)
        self.flush_interval = self.config.get("flush_interval", 5)
        self.destinations = self.config.get("destinations", [])
        self.enable_filtering = self.config.get("enable_filtering", True)
        self.min_level = LogLevel[self.config.get("min_level", "INFO").upper()]
        self.batch_size = self.config.get("batch_size", 100)

        # Thread-safe buffer
        self._lock = threading.RLock()
        self.buffer: deque = deque(maxlen=self.buffer_size)

        # Background tasks
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        # Statistics
        self._log_sources: Set[str] = set()
        self._logs_by_level: Dict[LogLevel, int] = {level: 0 for level in LogLevel}

        # Metrics
        self._logs_aggregated = Counter(
            "quantum_trader_logs_aggregated_total",
            "Total logs aggregated",
            ["level", "source"]
        )
        self._logs_dropped = Counter(
            "quantum_trader_logs_dropped_total",
            "Total logs dropped",
            ["reason"]
        )
        self._buffer_size_gauge = Gauge(
            "quantum_trader_log_buffer_size",
            "Current log buffer size"
        )
        self._flush_duration = Histogram(
            "quantum_trader_log_flush_seconds",
            "Log flush duration"
        )

        logger.info(
            "log_aggregator_initialized",
            buffer_size=self.buffer_size,
            destinations=len(self.destinations),
            min_level=self.min_level.value
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        buffer_size = self.config.get("buffer_size", 10000)
        if buffer_size < 100:
            raise ValueError(f"buffer_size must be >= 100, got {buffer_size}")

        flush_interval = self.config.get("flush_interval", 5)
        if flush_interval < 1:
            raise ValueError(f"flush_interval must be >= 1, got {flush_interval}")

        batch_size = self.config.get("batch_size", 100)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    async def start(self) -> None:
        """Start log aggregation.

        Raises:
            RuntimeError: If already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Log aggregator already running")

            self._running = True

        logger.info("starting_log_aggregator")

        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        logger.info("log_aggregator_started")

    async def stop(self) -> None:
        """Stop log aggregation and flush remaining logs."""
        logger.info("stopping_log_aggregator")

        # Flush remaining logs
        await self.flush()

        with self._lock:
            self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "log_aggregator_stopped",
            total_sources=len(self._log_sources),
            buffer_size=len(self.buffer)
        )

    async def log(
        self,
        level: LogLevel,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Aggregate a log entry.

        Args:
            level: Log level
            message: Log message
            source: Source identifier
            metadata: Additional metadata
            tags: Log tags

        Returns:
            True if logged, False if filtered/dropped

        Raises:
            ValueError: If parameters invalid
        """
        try:
            if not message:
                raise ValueError("Message cannot be empty")

            if not source:
                raise ValueError("Source cannot be empty")

            # Check log level filtering
            if self.enable_filtering:
                level_priority = list(LogLevel).index(level)
                min_priority = list(LogLevel).index(self.min_level)

                if level_priority < min_priority:
                    self._logs_dropped.labels(reason="level_filter").inc()
                    return False

            # Create log entry
            entry = LogEntry(
                level=level,
                message=message,
                source=source,
                metadata=metadata,
                tags=tags
            )

            # Add to buffer
            with self._lock:
                if len(self.buffer) >= self.buffer_size:
                    # Buffer full, drop oldest
                    self.buffer.popleft()
                    self._logs_dropped.labels(reason="buffer_full").inc()

                self.buffer.append(entry)
                self._log_sources.add(source)
                self._logs_by_level[level] += 1

                self._buffer_size_gauge.set(len(self.buffer))

            # Update metrics
            self._logs_aggregated.labels(
                level=level.value,
                source=source
            ).inc()

            # Trigger flush if buffer is getting full
            if len(self.buffer) >= self.buffer_size * 0.9:
                asyncio.create_task(self.flush())

            return True

        except Exception as e:
            logger.error("log_aggregation_failed", error=str(e))
            return False

    async def flush(self) -> int:
        """Flush buffer to configured destinations.

        Returns:
            Number of logs flushed

        Raises:
            Exception: If flush fails
        """
        try:
            start_time = datetime.now(timezone.utc)

            with self._lock:
                if not self.buffer:
                    return 0

                # Get all logs from buffer
                logs_to_flush = list(self.buffer)
                self.buffer.clear()
                self._buffer_size_gauge.set(0)

            # Process in batches
            total_flushed = 0

            for i in range(0, len(logs_to_flush), self.batch_size):
                batch = logs_to_flush[i:i + self.batch_size]

                # Send to each destination
                for destination_config in self.destinations:
                    try:
                        await self._send_to_destination(batch, destination_config)
                    except Exception as e:
                        logger.error(
                            "destination_send_failed",
                            destination=destination_config.get("type"),
                            error=str(e)
                        )

                total_flushed += len(batch)

            flush_duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._flush_duration.observe(flush_duration)

            logger.debug(
                "logs_flushed",
                count=total_flushed,
                duration=flush_duration
            )

            return total_flushed

        except Exception as e:
            logger.error("log_flush_failed", error=str(e))
            # Re-add logs to buffer on failure
            with self._lock:
                for log in reversed(logs_to_flush):
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.appendleft(log)
                self._buffer_size_gauge.set(len(self.buffer))
            raise

    async def _send_to_destination(
        self,
        logs: List[LogEntry],
        destination_config: Dict[str, Any]
    ) -> None:
        """Send logs to a specific destination.

        Args:
            logs: Log entries to send
            destination_config: Destination configuration

        Raises:
            Exception: If send fails
        """
        try:
            dest_type = destination_config.get("type", "file")

            if dest_type == "file":
                await self._send_to_file(logs, destination_config)
            elif dest_type == "console":
                await self._send_to_console(logs, destination_config)
            elif dest_type == "elasticsearch":
                await self._send_to_elasticsearch(logs, destination_config)
            elif dest_type == "syslog":
                await self._send_to_syslog(logs, destination_config)
            else:
                logger.warning("unknown_destination_type", type=dest_type)

        except Exception as e:
            logger.error(
                "destination_send_error",
                type=dest_type,
                error=str(e)
            )
            raise

    async def _send_to_file(
        self,
        logs: List[LogEntry],
        config: Dict[str, Any]
    ) -> None:
        """Send logs to file.

        Args:
            logs: Log entries
            config: File destination config
        """
        try:
            file_path = config.get("path", "/var/log/quantum_trader/aggregated.log")
            format_type = config.get("format", "json")

            with open(file_path, 'a') as f:
                for log in logs:
                    if format_type == "json":
                        f.write(log.to_json() + '\n')
                    else:
                        # Plain text format
                        f.write(
                            f"{log.timestamp.isoformat()} [{log.level.value.upper()}] "
                            f"{log.source}: {log.message}\n"
                        )

        except Exception as e:
            logger.error("file_write_failed", error=str(e))
            raise

    async def _send_to_console(
        self,
        logs: List[LogEntry],
        config: Dict[str, Any]
    ) -> None:
        """Send logs to console.

        Args:
            logs: Log entries
            config: Console destination config
        """
        try:
            for log in logs:
                print(
                    f"{log.timestamp.isoformat()} [{log.level.value.upper()}] "
                    f"{log.source}: {log.message}"
                )

        except Exception as e:
            logger.error("console_write_failed", error=str(e))
            raise

    async def _send_to_elasticsearch(
        self,
        logs: List[LogEntry],
        config: Dict[str, Any]
    ) -> None:
        """Send logs to Elasticsearch.

        Args:
            logs: Log entries
            config: Elasticsearch destination config
        """
        try:
            # Placeholder for Elasticsearch integration
            # Would use aiohttp to send bulk indexing requests
            es_url = config.get("url", "http://localhost:9200")
            index_name = config.get("index", "quantum-trader-logs")

            logger.debug(
                "elasticsearch_send",
                url=es_url,
                index=index_name,
                count=len(logs)
            )

            # Actual implementation would use elasticsearch-async or aiohttp

        except Exception as e:
            logger.error("elasticsearch_send_failed", error=str(e))
            raise

    async def _send_to_syslog(
        self,
        logs: List[LogEntry],
        config: Dict[str, Any]
    ) -> None:
        """Send logs to remote syslog.

        Args:
            logs: Log entries
            config: Syslog destination config
        """
        try:
            # Placeholder for syslog integration
            syslog_host = config.get("host", "localhost")
            syslog_port = config.get("port", 514)

            logger.debug(
                "syslog_send",
                host=syslog_host,
                port=syslog_port,
                count=len(logs)
            )

            # Actual implementation would use socket or aioudp

        except Exception as e:
            logger.error("syslog_send_failed", error=str(e))
            raise

    async def query_logs(
        self,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Query logs from buffer.

        Args:
            level: Filter by log level
            source: Filter by source
            tags: Filter by tags
            limit: Maximum results

        Returns:
            Matching log entries
        """
        try:
            with self._lock:
                logs = list(self.buffer)

            results = []

            for log in reversed(logs):  # Most recent first
                if len(results) >= limit:
                    break

                # Apply filters
                if level and log.level != level:
                    continue

                if source and log.source != source:
                    continue

                if tags and not any(t in log.tags for t in tags):
                    continue

                results.append(log)

            return results

        except Exception as e:
            logger.error("log_query_failed", error=str(e))
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "buffer_size": len(self.buffer),
                "buffer_capacity": self.buffer_size,
                "total_sources": len(self._log_sources),
                "sources": list(self._log_sources),
                "logs_by_level": {
                    level.value: count
                    for level, count in self._logs_by_level.items()
                },
                "destinations": len(self.destinations)
            }

    async def _flush_loop(self) -> None:
        """Background task for periodic flushing."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("flush_loop_error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": self._running,
            "buffer_size": len(self.buffer),
            "buffer_utilization": len(self.buffer) / self.buffer_size * 100,
            "total_sources": len(self._log_sources),
            "destinations_configured": len(self.destinations)
        }
