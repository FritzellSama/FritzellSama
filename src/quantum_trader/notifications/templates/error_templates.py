"""Error message templates for notifications.

This module provides templates for formatting error messages and exceptions
for various notification channels with appropriate context and debugging info.
"""

import os
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from structlog import get_logger

logger = get_logger(__name__)


class ErrorTemplates:
    """Error message template generator.

    Generates formatted error messages with stack traces, context, and
    debugging information for notification channels.

    Attributes:
        config: Configuration dictionary
        service_name: Service name to include in errors
        include_stacktrace: Whether to include stack traces
        include_context: Whether to include context information

    Example:
        >>> config = {
        ...     "service_name": "Quantum Trader AI",
        ...     "include_stacktrace": True
        ... }
        >>> templates = ErrorTemplates(config)
        >>> try:
        ...     # Some operation
        ...     raise ValueError("Invalid order")
        ... except Exception as e:
        ...     message = templates.format_exception(e, {"order_id": "123"})
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize error templates.

        Args:
            config: Configuration dictionary containing:
                - service_name: Name of service
                - include_stacktrace: Include stack traces in errors
                - include_context: Include context information
                - stacktrace_limit: Maximum stack trace depth

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.service_name = self.config.get(
            "service_name", os.getenv("SERVICE_NAME", "Quantum Trader AI")
        )
        self.include_stacktrace = self.config.get(
            "include_stacktrace",
            os.getenv("ERROR_INCLUDE_STACKTRACE", "true").lower() == "true"
        )
        self.include_context = self.config.get(
            "include_context",
            os.getenv("ERROR_INCLUDE_CONTEXT", "true").lower() == "true"
        )
        self.stacktrace_limit = int(
            self.config.get(
                "stacktrace_limit",
                os.getenv("ERROR_STACKTRACE_LIMIT", "10")
            )
        )

        logger.info(
            "ErrorTemplates initialized",
            service_name=self.service_name,
            include_stacktrace=self.include_stacktrace
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    def format_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "HIGH"
    ) -> str:
        """Format an exception for notification.

        Args:
            exception: Exception to format
            context: Additional context information
            severity: Error severity level

        Returns:
            Formatted error message
        """
        message = f"🔴 **{severity.upper()}** - Exception Occurred\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"⚠️ Exception Type: {type(exception).__name__}\n"
        message += f"💬 Message: {str(exception)}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        # Add context if provided and enabled
        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        # Add stack trace if enabled
        if self.include_stacktrace:
            tb_lines = traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
                limit=self.stacktrace_limit
            )
            tb_str = "".join(tb_lines)
            message += f"\n📚 Stack Trace:\n```\n{tb_str}\n```"

        return message

    def format_connection_error(
        self,
        service: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a connection error.

        Args:
            service: Service/exchange name
            error_message: Error message
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🔴 **HIGH** - Connection Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🔌 Target: {service}\n"
        message += f"❌ Error: {error_message}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Connection Details:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_validation_error(
        self,
        field: str,
        value: Any,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a validation error.

        Args:
            field: Field that failed validation
            value: Invalid value
            reason: Reason for validation failure
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🟡 **MEDIUM** - Validation Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏷️ Field: {field}\n"
        message += f"📝 Value: {value}\n"
        message += f"❌ Reason: {reason}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, val in context.items():
                message += f"  • {key}: {val}\n"

        return message

    def format_timeout_error(
        self,
        operation: str,
        timeout_seconds: float,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a timeout error.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🟠 **HIGH** - Timeout Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"⚙️ Operation: {operation}\n"
        message += f"⏱️ Timeout: {timeout_seconds}s\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_rate_limit_error(
        self,
        service: str,
        limit_type: str,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a rate limit error.

        Args:
            service: Service that rate limited
            limit_type: Type of rate limit
            retry_after: Seconds until retry is allowed
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🟡 **MEDIUM** - Rate Limit Exceeded\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏦 Target: {service}\n"
        message += f"🚦 Limit Type: {limit_type}\n"

        if retry_after:
            message += f"⏳ Retry After: {retry_after}s\n"

        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_data_error(
        self,
        data_type: str,
        issue: str,
        affected_records: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a data error.

        Args:
            data_type: Type of data with error
            issue: Description of the issue
            affected_records: Number of affected records
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🟠 **HIGH** - Data Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"📊 Data Type: {data_type}\n"
        message += f"❌ Issue: {issue}\n"

        if affected_records is not None:
            message += f"📉 Affected Records: {affected_records}\n"

        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_authentication_error(
        self,
        service: str,
        auth_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format an authentication error.

        Args:
            service: Service with authentication issue
            auth_type: Type of authentication
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🔴 **CRITICAL** - Authentication Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏦 Target: {service}\n"
        message += f"🔐 Auth Type: {auth_type}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                # Don't include sensitive fields
                if key.lower() in ["password", "secret", "key", "token"]:
                    message += f"  • {key}: [REDACTED]\n"
                else:
                    message += f"  • {key}: {value}\n"

        return message

    def format_resource_error(
        self,
        resource_type: str,
        issue: str,
        current_value: Optional[str] = None,
        threshold: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a resource error (CPU, memory, disk, etc.).

        Args:
            resource_type: Type of resource
            issue: Description of the issue
            current_value: Current resource value
            threshold: Resource threshold
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🟠 **HIGH** - Resource Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"💾 Resource: {resource_type}\n"
        message += f"❌ Issue: {issue}\n"

        if current_value:
            message += f"📈 Current: {current_value}\n"

        if threshold:
            message += f"🎯 Threshold: {threshold}\n"

        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_configuration_error(
        self,
        config_key: str,
        issue: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a configuration error.

        Args:
            config_key: Configuration key with issue
            issue: Description of the issue
            context: Additional context

        Returns:
            Formatted error message
        """
        message = "🔴 **CRITICAL** - Configuration Error\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"⚙️ Config Key: {config_key}\n"
        message += f"❌ Issue: {issue}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def format_generic_error(
        self,
        error_type: str,
        message_text: str,
        severity: str = "HIGH",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a generic error message.

        Args:
            error_type: Type of error
            message_text: Error message
            severity: Error severity
            context: Additional context

        Returns:
            Formatted error message
        """
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }.get(severity.upper(), "⚪")

        message = f"{severity_emoji} **{severity.upper()}** - {error_type}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"💬 Message: {message_text}\n"
        message += f"⏰ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"

        if self.include_context and context:
            message += "\n📊 Context:\n"
            for key, value in context.items():
                message += f"  • {key}: {value}\n"

        return message

    def get_template_list(self) -> List[str]:
        """Get list of available error template types.

        Returns:
            List of error template type names
        """
        return [
            "exception",
            "connection_error",
            "validation_error",
            "timeout_error",
            "rate_limit_error",
            "data_error",
            "authentication_error",
            "resource_error",
            "configuration_error",
            "generic_error",
        ]
