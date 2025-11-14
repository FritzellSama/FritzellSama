"""Log Handlers for Quantum Trader AI.

Production-ready log handlers with stream processing, batch optimization,
memory efficiency, and fault tolerance.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone
from enum import Enum
from collections import deque
import threading
from pathlib import Path
import gzip

from prometheus_client import Counter, Gauge, Histogram
from structlog import get_logger

logger = get_logger(__name__)


class HandlerType(Enum):
    """Log handler types."""
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    TIMED_ROTATING_FILE = "timed_rotating_file"
    STREAM = "stream"
    MEMORY = "memory"
    ASYNC_BATCH = "async_batch"


class LogRecord:
    """Represents a log record for processing.

    Attributes:
        timestamp: Record timestamp
        level: Log level
        message: Log message
        data: Additional data
    """

    def __init__(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize log record.

        Args:
            level: Log level
            message: Log message
            data: Additional data
        """
        self.timestamp = datetime.now(timezone.utc)
        self.level = level
        self.message = message
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            **self.data
        }


class BaseLogHandler:
    """Base class for log handlers.

    Attributes:
        config: Handler configuration
        enabled: Whether handler is enabled
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize base handler.

        Args:
            config: Handler configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self._lock = threading.RLock()

    async def handle(self, record: LogRecord) -> bool:
        """Handle a log record.

        Args:
            record: Log record to handle

        Returns:
            True if handled successfully

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement handle()")

    async def flush(self) -> None:
        """Flush any buffered records."""
        pass

    async def close(self) -> None:
        """Close handler and release resources."""
        pass


class FileLogHandler(BaseLogHandler):
    """File-based log handler with rotation support.

    Handles writing logs to files with optional rotation based on
    size or time.

    Example:
        >>> config = {
        ...     "file_path": "/var/log/quantum_trader/app.log",
        ...     "max_size_mb": 100,
        ...     "backup_count": 5,
        ...     "compress_rotated": True
        ... }
        >>> handler = FileLogHandler(config)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize file handler.

        Args:
            config: Configuration dictionary containing:
                - file_path: Path to log file
                - max_size_mb: Max file size before rotation (MB)
                - backup_count: Number of backup files to keep
                - compress_rotated: Compress rotated files
                - encoding: File encoding

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.file_path = Path(self.config["file_path"])
        self.max_size_bytes = self.config.get("max_size_mb", 100) * 1024 * 1024
        self.backup_count = self.config.get("backup_count", 5)
        self.compress_rotated = self.config.get("compress_rotated", True)
        self.encoding = self.config.get("encoding", "utf-8")

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_handle: Optional[Any] = None
        self._current_size = 0

        # Metrics
        self._bytes_written = Counter(
            "quantum_trader_log_bytes_written_total",
            "Total bytes written to log files",
            ["handler"]
        )
        self._rotations = Counter(
            "quantum_trader_log_rotations_total",
            "Total log file rotations",
            ["handler"]
        )

        logger.info(
            "file_handler_initialized",
            file_path=str(self.file_path),
            max_size_mb=self.max_size_bytes / 1024 / 1024
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if "file_path" not in self.config:
            raise ValueError("file_path is required")

        max_size = self.config.get("max_size_mb", 100)
        if max_size < 1:
            raise ValueError(f"max_size_mb must be >= 1, got {max_size}")

        backup_count = self.config.get("backup_count", 5)
        if backup_count < 0:
            raise ValueError(f"backup_count must be >= 0, got {backup_count}")

    async def handle(self, record: LogRecord) -> bool:
        """Handle log record by writing to file.

        Args:
            record: Log record

        Returns:
            True if handled successfully
        """
        try:
            with self._lock:
                # Open file if not already open
                if self._file_handle is None:
                    self._file_handle = open(
                        self.file_path,
                        'a',
                        encoding=self.encoding
                    )

                    # Get current file size
                    if self.file_path.exists():
                        self._current_size = self.file_path.stat().st_size

                # Format and write record
                import json
                log_line = json.dumps(record.to_dict()) + '\n'
                bytes_written = len(log_line.encode(self.encoding))

                self._file_handle.write(log_line)
                self._current_size += bytes_written

                self._bytes_written.labels(handler="file").inc(bytes_written)

                # Check if rotation needed
                if self._current_size >= self.max_size_bytes:
                    await self._rotate()

            return True

        except Exception as e:
            logger.error("file_handler_error", error=str(e))
            return False

    async def _rotate(self) -> None:
        """Rotate log file."""
        try:
            with self._lock:
                # Close current file
                if self._file_handle:
                    self._file_handle.close()
                    self._file_handle = None

                # Rotate existing backups
                for i in range(self.backup_count - 1, 0, -1):
                    source = self.file_path.with_suffix(f".{i}")
                    dest = self.file_path.with_suffix(f".{i + 1}")

                    if source.exists():
                        if dest.exists():
                            dest.unlink()
                        source.rename(dest)

                # Move current file to .1
                if self.file_path.exists():
                    backup_path = self.file_path.with_suffix(".1")
                    self.file_path.rename(backup_path)

                    # Compress if enabled
                    if self.compress_rotated:
                        await self._compress_file(backup_path)

                # Reset size counter
                self._current_size = 0

                self._rotations.labels(handler="file").inc()

                logger.info("log_file_rotated", file_path=str(self.file_path))

        except Exception as e:
            logger.error("log_rotation_failed", error=str(e))

    async def _compress_file(self, file_path: Path) -> None:
        """Compress a log file.

        Args:
            file_path: Path to file to compress
        """
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + ".gz")

            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)

            # Remove original
            file_path.unlink()

            logger.debug("log_file_compressed", file_path=str(compressed_path))

        except Exception as e:
            logger.error("log_compression_failed", error=str(e))

    async def flush(self) -> None:
        """Flush file buffer."""
        with self._lock:
            if self._file_handle:
                self._file_handle.flush()

    async def close(self) -> None:
        """Close file handler."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None


class AsyncBatchLogHandler(BaseLogHandler):
    """Async batch log handler with memory optimization.

    Batches log records and processes them asynchronously for
    improved performance and reduced I/O overhead.

    Example:
        >>> config = {
        ...     "batch_size": 100,
        ...     "flush_interval": 5,
        ...     "max_queue_size": 10000,
        ...     "processor": async_processor_func
        ... }
        >>> handler = AsyncBatchLogHandler(config)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize async batch handler.

        Args:
            config: Configuration dictionary containing:
                - batch_size: Number of records per batch
                - flush_interval: Seconds between flushes
                - max_queue_size: Maximum queue size
                - processor: Async function to process batches
                - enable_checkpointing: Enable batch checkpointing

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.batch_size = self.config.get("batch_size", 100)
        self.flush_interval = self.config.get("flush_interval", 5)
        self.max_queue_size = self.config.get("max_queue_size", 10000)
        self.processor: Callable = self.config["processor"]
        self.enable_checkpointing = self.config.get("enable_checkpointing", False)

        # Queue for records
        self.queue: deque = deque(maxlen=self.max_queue_size)
        self.checkpoint_data: List[Dict[str, Any]] = []

        # Processing state
        self._running = False
        self._process_task: Optional[asyncio.Task] = None

        # Metrics
        self._records_queued = Counter(
            "quantum_trader_log_records_queued_total",
            "Total records queued",
            ["handler"]
        )
        self._records_processed = Counter(
            "quantum_trader_log_records_processed_total",
            "Total records processed",
            ["handler"]
        )
        self._queue_size_gauge = Gauge(
            "quantum_trader_log_queue_size",
            "Current queue size",
            ["handler"]
        )
        self._batch_processing_time = Histogram(
            "quantum_trader_log_batch_processing_seconds",
            "Batch processing time",
            ["handler"]
        )

        logger.info(
            "async_batch_handler_initialized",
            batch_size=self.batch_size,
            flush_interval=self.flush_interval,
            max_queue_size=self.max_queue_size
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if "processor" not in self.config:
            raise ValueError("processor function is required")

        batch_size = self.config.get("batch_size", 100)
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        flush_interval = self.config.get("flush_interval", 5)
        if flush_interval < 1:
            raise ValueError(f"flush_interval must be >= 1, got {flush_interval}")

    async def start(self) -> None:
        """Start async processing."""
        with self._lock:
            if self._running:
                return

            self._running = True

        # Start processing task
        self._process_task = asyncio.create_task(self._process_loop())

        logger.info("async_batch_handler_started")

    async def stop(self) -> None:
        """Stop async processing and flush remaining records."""
        logger.info("stopping_async_batch_handler")

        # Flush remaining records
        await self.flush()

        with self._lock:
            self._running = False

        # Cancel processing task
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        logger.info("async_batch_handler_stopped", queue_size=len(self.queue))

    async def handle(self, record: LogRecord) -> bool:
        """Add record to batch queue.

        Args:
            record: Log record

        Returns:
            True if queued successfully
        """
        try:
            with self._lock:
                if len(self.queue) >= self.max_queue_size:
                    logger.warning("log_queue_full", dropping_record=True)
                    return False

                self.queue.append(record)
                self._queue_size_gauge.labels(handler="async_batch").set(len(self.queue))

            self._records_queued.labels(handler="async_batch").inc()

            # Trigger immediate flush if batch is full
            if len(self.queue) >= self.batch_size:
                asyncio.create_task(self.flush())

            return True

        except Exception as e:
            logger.error("async_handler_error", error=str(e))
            return False

    async def flush(self) -> None:
        """Flush queued records by processing batches."""
        try:
            while True:
                with self._lock:
                    if not self.queue:
                        break

                    # Extract batch
                    batch = []
                    for _ in range(min(self.batch_size, len(self.queue))):
                        if self.queue:
                            batch.append(self.queue.popleft())

                    self._queue_size_gauge.labels(handler="async_batch").set(len(self.queue))

                if not batch:
                    break

                # Process batch
                await self._process_batch(batch)

        except Exception as e:
            logger.error("flush_failed", error=str(e))

    async def _process_batch(self, batch: List[LogRecord]) -> None:
        """Process a batch of records.

        Args:
            batch: List of log records
        """
        try:
            start_time = datetime.now(timezone.utc)

            # Save checkpoint if enabled
            if self.enable_checkpointing:
                self.checkpoint_data = [r.to_dict() for r in batch]

            # Process batch using configured processor
            await self.processor(batch)

            # Clear checkpoint on success
            if self.enable_checkpointing:
                self.checkpoint_data = []

            # Update metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._batch_processing_time.labels(handler="async_batch").observe(processing_time)
            self._records_processed.labels(handler="async_batch").inc(len(batch))

            logger.debug(
                "batch_processed",
                count=len(batch),
                duration=processing_time
            )

        except Exception as e:
            logger.error("batch_processing_failed", error=str(e), count=len(batch))

            # Re-queue records on failure if checkpointing enabled
            if self.enable_checkpointing:
                with self._lock:
                    for record_dict in reversed(self.checkpoint_data):
                        if len(self.queue) < self.max_queue_size:
                            # Reconstruct record
                            record = LogRecord(
                                level=record_dict["level"],
                                message=record_dict["message"],
                                data=record_dict
                            )
                            self.queue.appendleft(record)

    async def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("process_loop_error", error=str(e))

    async def close(self) -> None:
        """Close handler."""
        await self.stop()


class MemoryLogHandler(BaseLogHandler):
    """In-memory log handler for testing and debugging.

    Stores logs in memory for fast access during development.

    Example:
        >>> config = {"max_records": 1000}
        >>> handler = MemoryLogHandler(config)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize memory handler.

        Args:
            config: Configuration dictionary containing:
                - max_records: Maximum records to keep in memory
        """
        super().__init__(config)

        self.max_records = self.config.get("max_records", 1000)
        self.records: deque = deque(maxlen=self.max_records)

        logger.info("memory_handler_initialized", max_records=self.max_records)

    async def handle(self, record: LogRecord) -> bool:
        """Store record in memory.

        Args:
            record: Log record

        Returns:
            True always
        """
        with self._lock:
            self.records.append(record)
        return True

    def get_records(
        self,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[LogRecord]:
        """Get records from memory.

        Args:
            level: Filter by log level
            limit: Maximum records to return

        Returns:
            List of log records
        """
        with self._lock:
            records = list(self.records)

        if level:
            records = [r for r in records if r.level == level]

        return records[-limit:]

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self.records.clear()


async def example_batch_processor(batch: List[LogRecord]) -> None:
    """Example batch processor function.

    Args:
        batch: Batch of log records to process
    """
    # Example: write to file, send to remote service, etc.
    for record in batch:
        # Process record
        pass
