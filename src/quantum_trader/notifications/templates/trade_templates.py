"""
Trade Templates - Trade execution and order notification formatting.

Generates formatted notifications for order submissions, executions,
fills, cancellations, and trade-related alerts.
"""

from typing import Dict, Optional, Any
from datetime import datetime
from decimal import Decimal
from enum import Enum
import os
from structlog import get_logger

logger = get_logger(__name__)


class TradeTemplates:
    """
    Trade notification formatting templates.

    Generates formatted messages for trade lifecycle events including
    order submissions, executions, fills, cancellations, and errors.

    Attributes:
        config: Configuration dictionary from environment
        currency_symbol: Currency symbol for formatting
        decimal_places: Decimal places for price formatting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize trade templates.

        Args:
            config: Optional configuration override. If None, loads from environment.
        """
        self.config = config or self._load_config()
        self._validate_config()

        self.currency_symbol = self.config.get(
            "currency_symbol",
            os.getenv("TRADE_CURRENCY_SYMBOL", "$")
        )
        self.decimal_places = int(self.config.get(
            "decimal_places",
            os.getenv("TRADE_DECIMAL_PLACES", "2")
        ))
        self.quantity_places = int(self.config.get(
            "quantity_places",
            os.getenv("TRADE_QUANTITY_PLACES", "8")
        ))

        logger.info("trade_templates_initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "currency_symbol": os.getenv("TRADE_CURRENCY_SYMBOL", "$"),
            "decimal_places": os.getenv("TRADE_DECIMAL_PLACES", "2"),
            "quantity_places": os.getenv("TRADE_QUANTITY_PLACES", "8"),
            "date_format": os.getenv("TRADE_DATE_FORMAT", "%Y-%m-%d %H:%M:%S UTC"),
            "include_footer": os.getenv("TRADE_INCLUDE_FOOTER", "true").lower() == "true",
            "footer_text": os.getenv("TRADE_FOOTER_TEXT", "Quantum Trader AI"),
            "include_exchange": os.getenv("TRADE_INCLUDE_EXCHANGE", "true").lower() == "true",
            "include_strategy": os.getenv("TRADE_INCLUDE_STRATEGY", "true").lower() == "true",
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if int(self.config.get("decimal_places", 2)) < 0:
            raise ValueError("decimal_places must be >= 0")
        if int(self.config.get("quantity_places", 8)) < 0:
            raise ValueError("quantity_places must be >= 0")

    def format_decimal(self, value: Decimal, places: Optional[int] = None) -> str:
        """
        Format Decimal value for display.

        Args:
            value: Decimal value to format
            places: Optional decimal places override

        Returns:
            Formatted string
        """
        if places is None:
            places = self.decimal_places

        return f"{value:.{places}f}"

    def format_currency(self, value: Decimal, places: Optional[int] = None) -> str:
        """
        Format Decimal value as currency.

        Args:
            value: Decimal value to format
            places: Optional decimal places override

        Returns:
            Formatted currency string
        """
        formatted = self.format_decimal(value, places)
        return f"{self.currency_symbol}{formatted}"

    def format_quantity(self, value: Decimal) -> str:
        """
        Format quantity with appropriate decimal places.

        Args:
            value: Quantity value

        Returns:
            Formatted quantity string
        """
        return self.format_decimal(value, self.quantity_places).rstrip('0').rstrip('.')

    def format_timestamp(self, timestamp: datetime) -> str:
        """
        Format timestamp for display.

        Args:
            timestamp: Datetime to format

        Returns:
            Formatted timestamp string
        """
        date_format = self.config.get("date_format", "%Y-%m-%d %H:%M:%S UTC")
        return timestamp.strftime(date_format)

    def order_submitted(self, order: Dict[str, Any]) -> str:
        """
        Generate order submission notification.

        Args:
            order: Order dictionary containing:
                - symbol: Trading pair
                - side: BUY or SELL
                - quantity: Order quantity (Decimal)
                - price: Order price (Decimal, optional for market orders)
                - order_type: Order type (MARKET, LIMIT, etc.)
                - order_id: Internal order ID
                - exchange: Exchange name (optional)
                - strategy: Strategy name (optional)
                - timestamp: Order timestamp

        Returns:
            Formatted notification string
        """
        try:
            symbol = order.get("symbol", "UNKNOWN")
            side = order.get("side", "UNKNOWN")
            quantity = Decimal(str(order.get("quantity", 0)))
            price = order.get("price")
            order_type = order.get("order_type", "UNKNOWN")
            order_id = order.get("order_id", "N/A")
            timestamp = order.get("timestamp", datetime.utcnow())

            side_emoji = "🟢" if side == "BUY" else "🔴"

            message = f"""
{side_emoji} ORDER SUBMITTED

📊 Symbol: {symbol}
⚡ Side: {side}
📈 Type: {order_type}
💼 Quantity: {self.format_quantity(quantity)}
"""

            if price is not None:
                price_decimal = Decimal(str(price))
                message += f"💰 Price: {self.format_currency(price_decimal)}\n"
                total = price_decimal * quantity
                message += f"💵 Total: {self.format_currency(total)}\n"

            message += f"🆔 Order ID: {order_id}\n"
            message += f"⏰ Time: {self.format_timestamp(timestamp)}\n"

            if self.config.get("include_exchange", True) and order.get("exchange"):
                message += f"🏛️ Exchange: {order['exchange']}\n"

            if self.config.get("include_strategy", True) and order.get("strategy"):
                message += f"🎯 Strategy: {order['strategy']}\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_order_submitted", error=str(e))
            raise

    def order_filled(self, execution: Dict[str, Any]) -> str:
        """
        Generate order filled notification.

        Args:
            execution: Execution dictionary containing:
                - order_id: Internal order ID
                - symbol: Trading pair
                - side: BUY or SELL
                - filled_quantity: Filled quantity (Decimal)
                - average_price: Average fill price (Decimal)
                - fees: Trading fees (Decimal)
                - exchange: Exchange name
                - exchange_order_id: Exchange order ID
                - timestamp: Execution timestamp

        Returns:
            Formatted notification string
        """
        try:
            symbol = execution.get("symbol", "UNKNOWN")
            side = execution.get("side", "UNKNOWN")
            filled_qty = Decimal(str(execution.get("filled_quantity", 0)))
            avg_price = Decimal(str(execution.get("average_price", 0)))
            fees = Decimal(str(execution.get("fees", 0)))
            order_id = execution.get("order_id", "N/A")
            exchange_order_id = execution.get("exchange_order_id", "N/A")
            timestamp = execution.get("timestamp", datetime.utcnow())

            side_emoji = "✅" if side == "BUY" else "💰"
            total_value = filled_qty * avg_price

            message = f"""
{side_emoji} ORDER FILLED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Filled: {self.format_quantity(filled_qty)}
💰 Avg Price: {self.format_currency(avg_price)}
💵 Total Value: {self.format_currency(total_value)}
💸 Fees: {self.format_currency(fees)}
🆔 Order ID: {order_id}
🏛️ Exchange ID: {exchange_order_id}
⏰ Time: {self.format_timestamp(timestamp)}
"""

            if execution.get("exchange"):
                message += f"🏛️ Exchange: {execution['exchange']}\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_order_filled", error=str(e))
            raise

    def order_partially_filled(self, execution: Dict[str, Any]) -> str:
        """
        Generate partial fill notification.

        Args:
            execution: Execution dictionary with partial fill info

        Returns:
            Formatted notification string
        """
        try:
            symbol = execution.get("symbol", "UNKNOWN")
            side = execution.get("side", "UNKNOWN")
            filled_qty = Decimal(str(execution.get("filled_quantity", 0)))
            total_qty = Decimal(str(execution.get("total_quantity", 0)))
            avg_price = Decimal(str(execution.get("average_price", 0)))
            remaining_qty = total_qty - filled_qty
            fill_pct = (filled_qty / total_qty * Decimal("100")) if total_qty > 0 else Decimal("0")

            side_emoji = "🟡"

            message = f"""
{side_emoji} PARTIAL FILL

📊 Symbol: {symbol}
⚡ Side: {side}
✅ Filled: {self.format_quantity(filled_qty)} / {self.format_quantity(total_qty)} ({self.format_decimal(fill_pct, 1)}%)
⏳ Remaining: {self.format_quantity(remaining_qty)}
💰 Avg Price: {self.format_currency(avg_price)}
⏰ Time: {self.format_timestamp(execution.get('timestamp', datetime.utcnow()))}
"""

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_partial_fill", error=str(e))
            raise

    def order_cancelled(self, order: Dict[str, Any]) -> str:
        """
        Generate order cancellation notification.

        Args:
            order: Order dictionary with cancellation info

        Returns:
            Formatted notification string
        """
        try:
            symbol = order.get("symbol", "UNKNOWN")
            side = order.get("side", "UNKNOWN")
            quantity = Decimal(str(order.get("quantity", 0)))
            order_id = order.get("order_id", "N/A")
            reason = order.get("reason", "User requested")
            timestamp = order.get("timestamp", datetime.utcnow())

            message = f"""
❌ ORDER CANCELLED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Quantity: {self.format_quantity(quantity)}
🆔 Order ID: {order_id}
📝 Reason: {reason}
⏰ Time: {self.format_timestamp(timestamp)}
"""

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_order_cancelled", error=str(e))
            raise

    def order_rejected(self, order: Dict[str, Any]) -> str:
        """
        Generate order rejection notification.

        Args:
            order: Order dictionary with rejection info

        Returns:
            Formatted notification string
        """
        try:
            symbol = order.get("symbol", "UNKNOWN")
            side = order.get("side", "UNKNOWN")
            quantity = Decimal(str(order.get("quantity", 0)))
            order_id = order.get("order_id", "N/A")
            reason = order.get("reason", "Unknown error")
            timestamp = order.get("timestamp", datetime.utcnow())

            message = f"""
⛔ ORDER REJECTED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Quantity: {self.format_quantity(quantity)}
🆔 Order ID: {order_id}
❗ Reason: {reason}
⏰ Time: {self.format_timestamp(timestamp)}
"""

            if order.get("exchange"):
                message += f"🏛️ Exchange: {order['exchange']}\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_order_rejected", error=str(e))
            raise

    def trade_alert(self, alert: Dict[str, Any]) -> str:
        """
        Generate general trade alert.

        Args:
            alert: Alert dictionary containing:
                - level: Alert level (INFO, WARNING, ERROR, CRITICAL)
                - title: Alert title
                - message: Alert message
                - symbol: Optional trading pair
                - timestamp: Alert timestamp

        Returns:
            Formatted notification string
        """
        try:
            level = alert.get("level", "INFO")
            title = alert.get("title", "Trade Alert")
            message_text = alert.get("message", "")
            symbol = alert.get("symbol")
            timestamp = alert.get("timestamp", datetime.utcnow())

            level_emoji = {
                "DEBUG": "🔍",
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🚨"
            }.get(level, "ℹ️")

            message = f"""
{level_emoji} {title.upper()}

{message_text}
"""

            if symbol:
                message += f"\n📊 Symbol: {symbol}\n"

            message += f"⏰ Time: {self.format_timestamp(timestamp)}\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_trade_alert", error=str(e))
            raise

    def position_opened(self, position: Dict[str, Any]) -> str:
        """
        Generate position opened notification.

        Args:
            position: Position dictionary

        Returns:
            Formatted notification string
        """
        try:
            symbol = position.get("symbol", "UNKNOWN")
            side = position.get("side", "UNKNOWN")
            size = Decimal(str(position.get("size", 0)))
            entry_price = Decimal(str(position.get("entry_price", 0)))
            leverage = Decimal(str(position.get("leverage", 1)))
            timestamp = position.get("timestamp", datetime.utcnow())

            side_emoji = "🟢" if side == "LONG" else "🔴"
            notional_value = size * entry_price

            message = f"""
{side_emoji} POSITION OPENED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Size: {self.format_quantity(size)}
💰 Entry Price: {self.format_currency(entry_price)}
📊 Notional: {self.format_currency(notional_value)}
⚙️ Leverage: {self.format_decimal(leverage, 1)}x
⏰ Time: {self.format_timestamp(timestamp)}
"""

            if position.get("stop_loss"):
                stop_loss = Decimal(str(position["stop_loss"]))
                message += f"🛡️ Stop Loss: {self.format_currency(stop_loss)}\n"

            if position.get("take_profit"):
                take_profit = Decimal(str(position["take_profit"]))
                message += f"🎯 Take Profit: {self.format_currency(take_profit)}\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_position_opened", error=str(e))
            raise

    def position_closed(self, position: Dict[str, Any]) -> str:
        """
        Generate position closed notification.

        Args:
            position: Position dictionary with close info

        Returns:
            Formatted notification string
        """
        try:
            symbol = position.get("symbol", "UNKNOWN")
            side = position.get("side", "UNKNOWN")
            size = Decimal(str(position.get("size", 0)))
            entry_price = Decimal(str(position.get("entry_price", 0)))
            exit_price = Decimal(str(position.get("exit_price", 0)))
            pnl = Decimal(str(position.get("pnl", 0)))
            pnl_pct = Decimal(str(position.get("pnl_percentage", 0)))
            timestamp = position.get("timestamp", datetime.utcnow())

            pnl_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""

            message = f"""
{pnl_emoji} POSITION CLOSED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Size: {self.format_quantity(size)}
💰 Entry: {self.format_currency(entry_price)}
💵 Exit: {self.format_currency(exit_price)}
📊 P&L: {pnl_sign}{self.format_currency(pnl)} ({pnl_sign}{self.format_decimal(pnl_pct, 2)}%)
⏰ Time: {self.format_timestamp(timestamp)}
"""

            if position.get("hold_time_hours"):
                hold_time = position["hold_time_hours"]
                message += f"⏱️ Hold Time: {hold_time:.1f} hours\n"

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_position_closed", error=str(e))
            raise

    def stop_loss_triggered(self, position: Dict[str, Any]) -> str:
        """
        Generate stop loss triggered notification.

        Args:
            position: Position dictionary

        Returns:
            Formatted notification string
        """
        try:
            symbol = position.get("symbol", "UNKNOWN")
            side = position.get("side", "UNKNOWN")
            size = Decimal(str(position.get("size", 0)))
            stop_price = Decimal(str(position.get("stop_price", 0)))
            pnl = Decimal(str(position.get("pnl", 0)))

            message = f"""
🛑 STOP LOSS TRIGGERED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Size: {self.format_quantity(size)}
🛡️ Stop Price: {self.format_currency(stop_price)}
📉 P&L: {self.format_currency(pnl)}
⏰ Time: {self.format_timestamp(position.get('timestamp', datetime.utcnow()))}
"""

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_stop_loss", error=str(e))
            raise

    def take_profit_triggered(self, position: Dict[str, Any]) -> str:
        """
        Generate take profit triggered notification.

        Args:
            position: Position dictionary

        Returns:
            Formatted notification string
        """
        try:
            symbol = position.get("symbol", "UNKNOWN")
            side = position.get("side", "UNKNOWN")
            size = Decimal(str(position.get("size", 0)))
            target_price = Decimal(str(position.get("target_price", 0)))
            pnl = Decimal(str(position.get("pnl", 0)))

            message = f"""
🎯 TAKE PROFIT TRIGGERED

📊 Symbol: {symbol}
⚡ Side: {side}
💼 Size: {self.format_quantity(size)}
🎯 Target Price: {self.format_currency(target_price)}
📈 P&L: {self.format_currency(pnl)}
⏰ Time: {self.format_timestamp(position.get('timestamp', datetime.utcnow()))}
"""

            return message.strip()

        except Exception as e:
            logger.error("failed_to_generate_take_profit", error=str(e))
            raise
