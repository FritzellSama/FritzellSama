"""Log Formatter for Quantum Trader AI.

Production-ready log formatting system with multiple output formats
including JSON, structured text, and cloud-native formats.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
import json
import traceback

from structlog import get_logger

logger = get_logger(__name__)


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    STRUCTURED = "structured"
    PLAIN = "plain"
    LOGFMT = "logfmt"
    CEF = "cef"  # Common Event Format
    GELF = "gelf"  # Graylog Extended Log Format


class LogFormatter:
    """Format log records for various output formats.

    Provides consistent log formatting across the system with support
    for structured logging, JSON output, and cloud-native formats.

    Attributes:
        config: Configuration dictionary
        format_type: Output format type
        include_timestamp: Whether to include timestamps
        include_level: Whether to include log levels

    Example:
        >>> config = {
        ...     "format": "json",
        ...     "include_timestamp": True,
        ...     "timestamp_format": "iso8601",
        ...     "color_output": False
        ... }
        >>> formatter = LogFormatter(config)
        >>> formatted = formatter.format(
        ...     level="INFO",
        ...     message="Trade executed",
        ...     context={"symbol": "BTC/USDT", "pnl": 150.50}
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize log formatter.

        Args:
            config: Configuration dictionary containing:
                - format: Output format type
                - include_timestamp: Include timestamps
                - timestamp_format: Timestamp format (iso8601, unix)
                - include_level: Include log level
                - color_output: Use colored output
                - indent_json: Indent JSON output
                - max_message_length: Max message length before truncation

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.format_type = LogFormat[self.config.get("format", "JSON").upper()]
        self.include_timestamp = self.config.get("include_timestamp", True)
        self.timestamp_format = self.config.get("timestamp_format", "iso8601")
        self.include_level = self.config.get("include_level", True)
        self.color_output = self.config.get("color_output", False)
        self.indent_json = self.config.get("indent_json", False)
        self.max_message_length = self.config.get("max_message_length", 10000)

        # ANSI color codes
        self._colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[35m", # Magenta
            "RESET": "\033[0m"
        }

        logger.info(
            "log_formatter_initialized",
            format=self.format_type.value,
            timestamp_format=self.timestamp_format,
            color_output=self.color_output
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        format_str = self.config.get("format", "JSON").upper()
        if format_str not in [f.name for f in LogFormat]:
            raise ValueError(f"Invalid format: {format_str}")

        max_length = self.config.get("max_message_length", 10000)
        if max_length < 100:
            raise ValueError(f"max_message_length must be >= 100, got {max_length}")

    def format(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        exception: Optional[Exception] = None
    ) -> str:
        """Format a log record.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            context: Additional context data
            timestamp: Log timestamp (defaults to now)
            exception: Exception object if logging an error

        Returns:
            Formatted log string

        Raises:
            ValueError: If parameters are invalid
        """
        try:
            if not message:
                raise ValueError("Message cannot be empty")

            if not level:
                raise ValueError("Level cannot be empty")

            # Truncate message if too long
            if len(message) > self.max_message_length:
                message = message[:self.max_message_length] + "... [truncated]"

            # Default timestamp
            if timestamp is None:
                timestamp = datetime.now(timezone.utc)

            # Build log record
            record = {
                "level": level.upper(),
                "message": message,
            }

            if self.include_timestamp:
                record["timestamp"] = self._format_timestamp(timestamp)

            if context:
                record.update(context)

            if exception:
                record["exception"] = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc()
                }

            # Format based on type
            if self.format_type == LogFormat.JSON:
                return self._format_json(record)
            elif self.format_type == LogFormat.STRUCTURED:
                return self._format_structured(record)
            elif self.format_type == LogFormat.PLAIN:
                return self._format_plain(record)
            elif self.format_type == LogFormat.LOGFMT:
                return self._format_logfmt(record)
            elif self.format_type == LogFormat.CEF:
                return self._format_cef(record)
            elif self.format_type == LogFormat.GELF:
                return self._format_gelf(record)
            else:
                return self._format_json(record)

        except Exception as e:
            logger.error("log_formatting_failed", error=str(e))
            # Fallback to simple format
            return f"{timestamp.isoformat()} [{level}] {message}"

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp.

        Args:
            timestamp: Datetime object

        Returns:
            Formatted timestamp string
        """
        if self.timestamp_format == "unix":
            return str(int(timestamp.timestamp()))
        else:  # iso8601
            return timestamp.isoformat()

    def _format_json(self, record: Dict[str, Any]) -> str:
        """Format as JSON.

        Args:
            record: Log record

        Returns:
            JSON formatted string
        """
        try:
            if self.indent_json:
                return json.dumps(record, indent=2, default=str)
            else:
                return json.dumps(record, default=str)

        except Exception as e:
            logger.error("json_formatting_failed", error=str(e))
            return json.dumps({"level": "ERROR", "message": "JSON formatting failed"})

    def _format_structured(self, record: Dict[str, Any]) -> str:
        """Format as structured text.

        Args:
            record: Log record

        Returns:
            Structured text format
        """
        try:
            parts = []

            # Timestamp
            if "timestamp" in record:
                parts.append(f"time={record['timestamp']}")

            # Level with optional color
            level = record.get("level", "INFO")
            if self.color_output and level in self._colors:
                level_str = f"{self._colors[level]}{level}{self._colors['RESET']}"
            else:
                level_str = level
            parts.append(f"level={level_str}")

            # Message
            parts.append(f'msg="{record["message"]}"')

            # Context fields
            skip_fields = {"timestamp", "level", "message", "exception"}
            for key, value in record.items():
                if key not in skip_fields:
                    if isinstance(value, (str, int, float, bool)):
                        parts.append(f'{key}={value}')
                    else:
                        parts.append(f'{key}={json.dumps(value, default=str)}')

            # Exception
            if "exception" in record:
                exc = record["exception"]
                parts.append(f'exception={exc["type"]}: {exc["message"]}')

            return " ".join(parts)

        except Exception as e:
            logger.error("structured_formatting_failed", error=str(e))
            return f"level=ERROR msg=\"Structured formatting failed: {str(e)}\""

    def _format_plain(self, record: Dict[str, Any]) -> str:
        """Format as plain text.

        Args:
            record: Log record

        Returns:
            Plain text format
        """
        try:
            parts = []

            # Timestamp
            if "timestamp" in record:
                parts.append(record["timestamp"])

            # Level
            level = record.get("level", "INFO")
            if self.color_output and level in self._colors:
                parts.append(f"{self._colors[level]}[{level}]{self._colors['RESET']}")
            else:
                parts.append(f"[{level}]")

            # Message
            parts.append(record["message"])

            # Context
            skip_fields = {"timestamp", "level", "message", "exception"}
            context_parts = []
            for key, value in record.items():
                if key not in skip_fields:
                    context_parts.append(f"{key}={value}")

            if context_parts:
                parts.append("(" + ", ".join(context_parts) + ")")

            # Exception
            if "exception" in record:
                exc = record["exception"]
                parts.append(f"\nException: {exc['type']}: {exc['message']}")
                if exc.get("traceback"):
                    parts.append(f"\n{exc['traceback']}")

            return " ".join(parts)

        except Exception as e:
            logger.error("plain_formatting_failed", error=str(e))
            return f"{record.get('timestamp', '')} [ERROR] Plain formatting failed"

    def _format_logfmt(self, record: Dict[str, Any]) -> str:
        """Format as logfmt.

        Logfmt is a key=value format popularized by Heroku.

        Args:
            record: Log record

        Returns:
            Logfmt formatted string
        """
        try:
            parts = []

            for key, value in record.items():
                if isinstance(value, dict):
                    # Flatten nested dicts
                    for sub_key, sub_value in value.items():
                        parts.append(f'{key}_{sub_key}="{sub_value}"')
                elif isinstance(value, str):
                    # Quote strings with spaces
                    if ' ' in value:
                        parts.append(f'{key}="{value}"')
                    else:
                        parts.append(f'{key}={value}')
                else:
                    parts.append(f'{key}={value}')

            return " ".join(parts)

        except Exception as e:
            logger.error("logfmt_formatting_failed", error=str(e))
            return f'level=ERROR msg="Logfmt formatting failed"'

    def _format_cef(self, record: Dict[str, Any]) -> str:
        """Format as Common Event Format (CEF).

        CEF is used by security systems like ArcSight.

        Args:
            record: Log record

        Returns:
            CEF formatted string
        """
        try:
            # CEF Header
            # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
            cef_version = "0"
            device_vendor = "QuantumTrader"
            device_product = "TradingBot"
            device_version = "1.0"

            level = record.get("level", "INFO")
            severity_map = {
                "DEBUG": "1",
                "INFO": "3",
                "WARNING": "6",
                "ERROR": "8",
                "CRITICAL": "10"
            }
            severity = severity_map.get(level, "5")

            signature_id = record.get("event_id", "1000")
            name = record["message"][:50]  # Truncate name

            # Extensions
            extensions = []
            extensions.append(f'msg={record["message"]}')

            if "timestamp" in record:
                extensions.append(f'rt={record["timestamp"]}')

            skip_fields = {"level", "message", "timestamp"}
            for key, value in record.items():
                if key not in skip_fields:
                    extensions.append(f'{key}={value}')

            extension_str = " ".join(extensions)

            return (
                f"CEF:{cef_version}|{device_vendor}|{device_product}|{device_version}|"
                f"{signature_id}|{name}|{severity}|{extension_str}"
            )

        except Exception as e:
            logger.error("cef_formatting_failed", error=str(e))
            return f"CEF:0|QuantumTrader|TradingBot|1.0|9999|Formatting Error|10|msg=CEF formatting failed"

    def _format_gelf(self, record: Dict[str, Any]) -> str:
        """Format as Graylog Extended Log Format (GELF).

        GELF is JSON format with specific required fields.

        Args:
            record: Log record

        Returns:
            GELF formatted JSON string
        """
        try:
            import socket

            level_map = {
                "DEBUG": 7,
                "INFO": 6,
                "WARNING": 4,
                "ERROR": 3,
                "CRITICAL": 2
            }

            gelf = {
                "version": "1.1",
                "host": socket.gethostname(),
                "short_message": record["message"][:100],  # Max 100 chars
                "full_message": record["message"],
                "level": level_map.get(record.get("level", "INFO"), 6),
            }

            if "timestamp" in record:
                if self.timestamp_format == "unix":
                    gelf["timestamp"] = float(record["timestamp"])
                else:
                    # Convert ISO to unix timestamp
                    dt = datetime.fromisoformat(record["timestamp"])
                    gelf["timestamp"] = dt.timestamp()

            # Add custom fields with _ prefix
            skip_fields = {"level", "message", "timestamp"}
            for key, value in record.items():
                if key not in skip_fields:
                    gelf[f"_{key}"] = value

            return json.dumps(gelf, default=str)

        except Exception as e:
            logger.error("gelf_formatting_failed", error=str(e))
            return json.dumps({
                "version": "1.1",
                "host": "unknown",
                "short_message": "GELF formatting failed",
                "level": 3
            })

    def format_batch(
        self,
        records: List[Dict[str, Any]]
    ) -> List[str]:
        """Format multiple log records.

        Args:
            records: List of log records

        Returns:
            List of formatted log strings
        """
        try:
            formatted = []

            for record in records:
                try:
                    formatted_log = self.format(
                        level=record.get("level", "INFO"),
                        message=record.get("message", ""),
                        context=record.get("context"),
                        timestamp=record.get("timestamp"),
                        exception=record.get("exception")
                    )
                    formatted.append(formatted_log)
                except Exception as e:
                    logger.error("batch_record_formatting_failed", error=str(e))
                    continue

            return formatted

        except Exception as e:
            logger.error("batch_formatting_failed", error=str(e))
            return []

    def sanitize_sensitive_data(
        self,
        data: Dict[str, Any],
        sensitive_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Sanitize sensitive data from log records.

        Args:
            data: Data to sanitize
            sensitive_keys: List of sensitive key names

        Returns:
            Sanitized data dictionary
        """
        if sensitive_keys is None:
            sensitive_keys = [
                "password", "secret", "api_key", "token", "private_key",
                "access_token", "refresh_token", "authorization"
            ]

        try:
            sanitized = {}

            for key, value in data.items():
                # Check if key is sensitive (case-insensitive)
                if any(sk.lower() in key.lower() for sk in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = self.sanitize_sensitive_data(value, sensitive_keys)
                else:
                    sanitized[key] = value

            return sanitized

        except Exception as e:
            logger.error("sanitization_failed", error=str(e))
            return data

    def get_formatter_info(self) -> Dict[str, Any]:
        """Get formatter configuration info.

        Returns:
            Formatter info dictionary
        """
        return {
            "format": self.format_type.value,
            "timestamp_format": self.timestamp_format,
            "include_timestamp": self.include_timestamp,
            "include_level": self.include_level,
            "color_output": self.color_output,
            "max_message_length": self.max_message_length
        }
