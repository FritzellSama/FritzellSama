"""Structured logging configuration for Quantum Trader AI.

This module provides structured logging using structlog with JSON output,
contextual information, and integration with the monitoring system.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import structlog
from structlog.types import EventDict, WrappedLogger
from structlog.processors import JSONRenderer

logger = structlog.get_logger(__name__)


class StructuredLogger:
    """Structured logging configuration and management.

    Provides structured logging with JSON output, contextual fields, and
    integration with monitoring and alerting systems.

    Attributes:
        config: Configuration dictionary
        log_level: Logging level
        output_format: Output format (json/console)
        processors: List of log processors

    Example:
        >>> config = {
        ...     "log_level": "INFO",
        ...     "output_format": "json",
        ...     "log_file": "/var/log/quantum_trader.log"
        ... }
        >>> structured_logger = StructuredLogger(config)
        >>> structured_logger.configure()
        >>> log = structured_logger.get_logger("trading")
        >>> log.info("Order placed", order_id="123", symbol="BTC/USD")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize structured logger.

        Args:
            config: Configuration dictionary containing:
                - log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - output_format: Output format (json, console)
                - log_file: Optional log file path
                - enable_context: Enable contextual logging
                - additional_fields: Additional fields to include in logs

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.log_level = self.config.get(
            "log_level", os.getenv("LOG_LEVEL", "INFO")
        ).upper()

        self.output_format = self.config.get(
            "output_format", os.getenv("LOG_OUTPUT_FORMAT", "json")
        ).lower()

        self.log_file = self.config.get(
            "log_file", os.getenv("LOG_FILE_PATH")
        )

        self.enable_context = self.config.get(
            "enable_context",
            os.getenv("LOG_ENABLE_CONTEXT", "true").lower() == "true"
        )

        self.additional_fields = self.config.get("additional_fields", {})

        # Environment and service info
        self.service_name = self.config.get(
            "service_name", os.getenv("SERVICE_NAME", "quantum_trader_ai")
        )
        self.environment = self.config.get(
            "environment", os.getenv("ENVIRONMENT", "production")
        )

        self._configured = False

        logger.info(
            "StructuredLogger initialized",
            log_level=self.log_level,
            output_format=self.output_format
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Validation passed
        structlog.get_logger().debug("Config validation passed")

    def configure(self) -> None:
        """Configure structlog with processors and handlers.

        Raises:
            RuntimeError: If logger is already configured
        """
        if self._configured:
            raise RuntimeError("StructuredLogger is already configured")

        # Build processor chain
        processors: List[Any] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        # Add custom processors
        if self.enable_context:
            processors.append(self._add_context_processor)

        processors.append(self._add_service_info_processor)

        # Add format-specific processor
        if self.output_format == "json":
            processors.append(JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        self._configure_stdlib_logging()

        self._configured = True

        logger.info(
            "StructuredLogger configured",
            log_level=self.log_level,
            output_format=self.output_format,
            service_name=self.service_name
        )

    def _configure_stdlib_logging(self) -> None:
        """Configure standard library logging."""
        # Set log level
        numeric_level = getattr(logging, self.log_level, logging.INFO)
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=numeric_level,
        )

        # Add file handler if configured
        if self.log_file:
            try:
                # Ensure directory exists
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(numeric_level)
                logging.root.addHandler(file_handler)

                logger.info("File logging enabled", log_file=self.log_file)

            except Exception as e:
                logger.error(
                    "Failed to configure file logging",
                    log_file=self.log_file,
                    error=str(e),
                    exc_info=True
                )

    def _add_context_processor(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """Add contextual information to log entries.

        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Event dictionary

        Returns:
            Modified event dictionary
        """
        # Add timestamp if not present
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add additional fields
        event_dict.update(self.additional_fields)

        return event_dict

    def _add_service_info_processor(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """Add service information to log entries.

        Args:
            logger: Logger instance
            method_name: Log method name
            event_dict: Event dictionary

        Returns:
            Modified event dictionary
        """
        event_dict["service"] = self.service_name
        event_dict["environment"] = self.environment

        return event_dict

    def get_logger(self, name: Optional[str] = None) -> structlog.BoundLogger:
        """Get a logger instance.

        Args:
            name: Logger name (optional)

        Returns:
            Bound logger instance

        Raises:
            RuntimeError: If logger is not configured
        """
        if not self._configured:
            # Auto-configure if not already done
            self.configure()

        return structlog.get_logger(name)

    def set_log_level(self, level: str) -> None:
        """Set the logging level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Raises:
            ValueError: If level is invalid
        """
        level = level.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        if level not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

        self.log_level = level
        numeric_level = getattr(logging, level)
        logging.root.setLevel(numeric_level)

        logger.info("Log level changed", new_level=level)

    def add_context(self, **kwargs: Any) -> None:
        """Add context fields to all subsequent log entries.

        Args:
            **kwargs: Context fields to add
        """
        self.additional_fields.update(kwargs)
        logger.debug("Context added", fields=list(kwargs.keys()))

    def clear_context(self) -> None:
        """Clear all context fields."""
        self.additional_fields.clear()
        logger.debug("Context cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics.

        Returns:
            Dictionary with logger stats
        """
        return {
            "configured": self._configured,
            "log_level": self.log_level,
            "output_format": self.output_format,
            "log_file": self.log_file,
            "enable_context": self.enable_context,
            "service_name": self.service_name,
            "environment": self.environment,
            "context_fields": list(self.additional_fields.keys()),
        }


def create_structured_logger(config: Dict[str, Any]) -> StructuredLogger:
    """Create and configure a structured logger.

    Args:
        config: Configuration dictionary

    Returns:
        Configured StructuredLogger instance

    Example:
        >>> config = {"log_level": "INFO", "output_format": "json"}
        >>> structured_logger = create_structured_logger(config)
        >>> log = structured_logger.get_logger("trading")
    """
    structured_logger = StructuredLogger(config)
    structured_logger.configure()
    return structured_logger


class LogContext:
    """Context manager for temporary log context.

    Example:
        >>> log = structlog.get_logger()
        >>> with LogContext(order_id="123", strategy="momentum"):
        ...     log.info("Processing order")
    """

    def __init__(self, **context: Any) -> None:
        """Initialize log context.

        Args:
            **context: Context fields
        """
        self.context = context
        self.token: Optional[Any] = None

    def __enter__(self) -> "LogContext":
        """Enter context."""
        self.token = structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


class PerformanceLogger:
    """Logger for performance metrics.

    Example:
        >>> perf_log = PerformanceLogger("order_execution")
        >>> with perf_log:
        ...     # Execute order
        ...     pass
        >>> # Logs execution time automatically
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[structlog.BoundLogger] = None,
        **context: Any
    ) -> None:
        """Initialize performance logger.

        Args:
            operation_name: Name of operation being logged
            logger: Optional logger instance
            **context: Additional context fields
        """
        self.operation_name = operation_name
        self.logger = logger or structlog.get_logger()
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        """Enter context and start timing."""
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and log performance."""
        import time

        if self.start_time is None:
            return

        duration_ms = (time.perf_counter() - self.start_time) * 1000

        log_method = self.logger.info if exc_type is None else self.logger.error

        log_method(
            f"{self.operation_name} completed",
            operation=self.operation_name,
            duration_ms=f"{duration_ms:.2f}",
            success=exc_type is None,
            **self.context
        )
