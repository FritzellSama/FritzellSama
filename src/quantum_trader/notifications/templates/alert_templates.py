"""Alert message templates for notifications.

This module provides templates for formatting alert messages for various
notification channels (email, SMS, Telegram, etc.).
"""

import os
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from structlog import get_logger

logger = get_logger(__name__)


class AlertTemplates:
    """Alert message template generator.

    Generates formatted alert messages for various notification channels
    based on alert type and severity.

    Attributes:
        config: Configuration dictionary
        service_name: Service name to include in alerts
        templates: Dictionary of template functions

    Example:
        >>> config = {"service_name": "Quantum Trader AI"}
        >>> templates = AlertTemplates(config)
        >>> alert_data = {
        ...     "severity": "HIGH",
        ...     "order_id": "12345",
        ...     "error": "Connection timeout"
        ... }
        >>> message = templates.format_order_error(alert_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize alert templates.

        Args:
            config: Configuration dictionary containing:
                - service_name: Name of service
                - include_timestamp: Include timestamp in messages
                - timezone: Timezone for timestamps

        Raises:
            ValueError: If required config parameters are missing
        """
        self.config = config
        self._validate_config()

        self.service_name = self.config.get(
            "service_name", os.getenv("SERVICE_NAME", "Quantum Trader AI")
        )
        self.include_timestamp = self.config.get(
            "include_timestamp",
            os.getenv("ALERT_INCLUDE_TIMESTAMP", "true").lower() == "true"
        )

        logger.info("AlertTemplates initialized", service_name=self.service_name)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        logger.debug("Config validation passed")

    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level.

        Args:
            severity: Severity level

        Returns:
            Emoji string
        """
        severity_emojis = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "INFO": "ℹ️",
        }
        return severity_emojis.get(severity.upper(), "⚪")

    def _format_timestamp(self) -> str:
        """Format current timestamp.

        Returns:
            Formatted timestamp string
        """
        if not self.include_timestamp:
            return ""

        now = datetime.now(timezone.utc)
        return f"\n⏰ Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

    def _format_header(self, alert_type: str, severity: str) -> str:
        """Format alert header.

        Args:
            alert_type: Type of alert
            severity: Severity level

        Returns:
            Formatted header string
        """
        emoji = self._get_severity_emoji(severity)
        return f"{emoji} **{severity.upper()}** - {alert_type}"

    def format_order_error(self, data: Dict[str, Any]) -> str:
        """Format order error alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - order_id: Order ID
                - exchange: Exchange name
                - error: Error message
                - order_type: Order type (optional)
                - symbol: Trading symbol (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "HIGH")
        header = self._format_header("Order Error", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🆔 Order ID: {data.get('order_id', 'N/A')}\n"
        message += f"🏦 Exchange: {data.get('exchange', 'N/A')}\n"

        if "symbol" in data:
            message += f"💱 Symbol: {data['symbol']}\n"

        if "order_type" in data:
            message += f"📊 Type: {data['order_type']}\n"

        message += f"❌ Error: {data.get('error', 'Unknown error')}\n"
        message += self._format_timestamp()

        return message

    def format_risk_violation(self, data: Dict[str, Any]) -> str:
        """Format risk violation alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - violation_type: Type of violation
                - current_value: Current value
                - threshold: Threshold value
                - strategy: Strategy name (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "CRITICAL")
        header = self._format_header("Risk Violation", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"⚠️ Violation: {data.get('violation_type', 'N/A')}\n"
        message += f"📈 Current: {data.get('current_value', 'N/A')}\n"
        message += f"🎯 Threshold: {data.get('threshold', 'N/A')}\n"

        if "strategy" in data:
            message += f"🎲 Strategy: {data['strategy']}\n"

        message += self._format_timestamp()

        return message

    def format_performance_alert(self, data: Dict[str, Any]) -> str:
        """Format performance alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - metric_name: Name of metric
                - current_value: Current value
                - threshold: Threshold value
                - component: Component name (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "MEDIUM")
        header = self._format_header("Performance Alert", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"📊 Metric: {data.get('metric_name', 'N/A')}\n"
        message += f"📈 Value: {data.get('current_value', 'N/A')}\n"
        message += f"🎯 Threshold: {data.get('threshold', 'N/A')}\n"

        if "component" in data:
            message += f"🔧 Component: {data['component']}\n"

        message += self._format_timestamp()

        return message

    def format_exchange_disconnect(self, data: Dict[str, Any]) -> str:
        """Format exchange disconnection alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - exchange: Exchange name
                - reason: Disconnect reason (optional)
                - reconnect_attempts: Number of reconnect attempts (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "CRITICAL")
        header = self._format_header("Exchange Disconnected", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏦 Exchange: {data.get('exchange', 'N/A')}\n"

        if "reason" in data:
            message += f"❌ Reason: {data['reason']}\n"

        if "reconnect_attempts" in data:
            message += f"🔄 Reconnect Attempts: {data['reconnect_attempts']}\n"

        message += self._format_timestamp()

        return message

    def format_pnl_milestone(self, data: Dict[str, Any]) -> str:
        """Format PnL milestone alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - milestone_type: Type of milestone (profit/loss)
                - amount: PnL amount
                - strategy: Strategy name (optional)
                - period: Time period (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "INFO")
        header = self._format_header("PnL Milestone", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"💰 Type: {data.get('milestone_type', 'N/A')}\n"
        message += f"💵 Amount: ${data.get('amount', 'N/A')}\n"

        if "strategy" in data:
            message += f"🎲 Strategy: {data['strategy']}\n"

        if "period" in data:
            message += f"📅 Period: {data['period']}\n"

        message += self._format_timestamp()

        return message

    def format_system_health(self, data: Dict[str, Any]) -> str:
        """Format system health alert.

        Args:
            data: Alert data containing:
                - severity: Alert severity
                - health_status: Overall health status
                - issues: List of issues (optional)
                - metrics: Dictionary of health metrics (optional)

        Returns:
            Formatted alert message
        """
        severity = data.get("severity", "MEDIUM")
        header = self._format_header("System Health", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏥 Status: {data.get('health_status', 'N/A')}\n"

        if "issues" in data and data["issues"]:
            message += f"\n❗ Issues:\n"
            for issue in data["issues"]:
                message += f"  • {issue}\n"

        if "metrics" in data and data["metrics"]:
            message += f"\n📊 Metrics:\n"
            for key, value in data["metrics"].items():
                message += f"  • {key}: {value}\n"

        message += self._format_timestamp()

        return message

    def format_trade_execution(self, data: Dict[str, Any]) -> str:
        """Format trade execution notification.

        Args:
            data: Trade data containing:
                - severity: Alert severity
                - exchange: Exchange name
                - symbol: Trading symbol
                - side: Buy/Sell
                - quantity: Trade quantity
                - price: Execution price
                - order_id: Order ID (optional)

        Returns:
            Formatted notification message
        """
        severity = data.get("severity", "INFO")
        header = self._format_header("Trade Executed", severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"🏦 Exchange: {data.get('exchange', 'N/A')}\n"
        message += f"💱 Symbol: {data.get('symbol', 'N/A')}\n"

        side = data.get('side', 'N/A')
        side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
        message += f"{side_emoji} Side: {side}\n"

        message += f"📦 Quantity: {data.get('quantity', 'N/A')}\n"
        message += f"💵 Price: {data.get('price', 'N/A')}\n"

        if "order_id" in data:
            message += f"🆔 Order ID: {data['order_id']}\n"

        message += self._format_timestamp()

        return message

    def format_custom_alert(
        self,
        alert_type: str,
        severity: str,
        message_text: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format custom alert message.

        Args:
            alert_type: Type of alert
            severity: Severity level
            message_text: Main message text
            data: Optional additional data

        Returns:
            Formatted alert message
        """
        header = self._format_header(alert_type, severity)

        message = f"{header}\n\n"
        message += f"📋 Service: {self.service_name}\n"
        message += f"📝 Message: {message_text}\n"

        if data:
            message += "\n📊 Additional Data:\n"
            for key, value in data.items():
                message += f"  • {key}: {value}\n"

        message += self._format_timestamp()

        return message

    def format_email_subject(self, alert_type: str, severity: str) -> str:
        """Format email subject line.

        Args:
            alert_type: Type of alert
            severity: Severity level

        Returns:
            Formatted subject line
        """
        emoji = self._get_severity_emoji(severity)
        return f"{emoji} [{severity.upper()}] {self.service_name} - {alert_type}"

    def get_template_list(self) -> List[str]:
        """Get list of available template types.

        Returns:
            List of template type names
        """
        return [
            "order_error",
            "risk_violation",
            "performance_alert",
            "exchange_disconnect",
            "pnl_milestone",
            "system_health",
            "trade_execution",
            "custom_alert",
        ]
