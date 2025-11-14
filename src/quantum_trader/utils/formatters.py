"""Formatting Utility Functions.

Production-ready formatters for displaying trading data, metrics,
and reports in human-readable formats.
"""

from decimal import Decimal
from typing import Optional, Union, Dict, Any
from datetime import datetime, timedelta
from structlog import get_logger

logger = get_logger(__name__)


def format_currency(
    amount: Union[Decimal, float, int],
    currency: str = "USD",
    decimals: int = 2,
    symbol: bool = True
) -> str:
    """Format amount as currency.

    Args:
        amount: Amount to format
        currency: Currency code
        decimals: Decimal places
        symbol: Include currency symbol

    Returns:
        Formatted string
    """
    try:
        if isinstance(amount, Decimal):
            value = float(amount)
        else:
            value = float(amount)

        # Currency symbols
        symbols = {
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "BTC": "₿",
            "ETH": "Ξ"
        }

        formatted = f"{abs(value):,.{decimals}f}"

        if symbol and currency in symbols:
            formatted = f"{symbols[currency]}{formatted}"
        else:
            formatted = f"{formatted} {currency}"

        if value < 0:
            formatted = f"-{formatted}"

        return formatted

    except Exception as e:
        logger.error("currency_format_failed", amount=amount, error=str(e))
        return str(amount)


def format_percentage(
    value: Union[Decimal, float],
    decimals: int = 2,
    show_sign: bool = True
) -> str:
    """Format value as percentage.

    Args:
        value: Decimal value (e.g., 0.1234 for 12.34%)
        decimals: Decimal places
        show_sign: Show + for positive values

    Returns:
        Formatted percentage string
    """
    try:
        if isinstance(value, Decimal):
            pct = float(value) * 100
        else:
            pct = float(value) * 100

        formatted = f"{abs(pct):.{decimals}f}%"

        if pct > 0 and show_sign:
            return f"+{formatted}"
        elif pct < 0:
            return f"-{formatted}"
        else:
            return formatted

    except Exception as e:
        logger.error("percentage_format_failed", value=value, error=str(e))
        return f"{value}%"


def format_price(
    price: Union[Decimal, float],
    precision: int = 8,
    remove_trailing_zeros: bool = True
) -> str:
    """Format price with appropriate precision.

    Args:
        price: Price value
        precision: Maximum decimal places
        remove_trailing_zeros: Remove trailing zeros

    Returns:
        Formatted price string
    """
    try:
        if isinstance(price, Decimal):
            value = float(price)
        else:
            value = float(price)

        formatted = f"{value:.{precision}f}"

        if remove_trailing_zeros:
            # Remove trailing zeros but keep at least 2 decimals
            formatted = formatted.rstrip('0')
            if '.' in formatted:
                parts = formatted.split('.')
                if len(parts[1]) < 2:
                    formatted = f"{parts[0]}.{parts[1]:0<2}"

        return formatted

    except Exception as e:
        logger.error("price_format_failed", price=price, error=str(e))
        return str(price)


def format_quantity(
    quantity: Union[Decimal, float],
    precision: int = 8,
    compact: bool = False
) -> str:
    """Format quantity/amount.

    Args:
        quantity: Quantity value
        precision: Maximum decimal places
        compact: Use compact notation (K, M, B)

    Returns:
        Formatted quantity string
    """
    try:
        if isinstance(quantity, Decimal):
            value = float(quantity)
        else:
            value = float(quantity)

        if compact and abs(value) >= 1000:
            return format_compact_number(value, precision)

        formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
        return formatted

    except Exception as e:
        logger.error("quantity_format_failed", quantity=quantity, error=str(e))
        return str(quantity)


def format_compact_number(
    value: Union[int, float, Decimal],
    decimals: int = 2
) -> str:
    """Format large number with K/M/B/T suffixes.

    Args:
        value: Number to format
        decimals: Decimal places

    Returns:
        Formatted string (e.g., "1.5M")
    """
    try:
        if isinstance(value, Decimal):
            num = float(value)
        else:
            num = float(value)

        abs_num = abs(num)

        if abs_num >= 1_000_000_000_000:
            formatted = f"{num / 1_000_000_000_000:.{decimals}f}T"
        elif abs_num >= 1_000_000_000:
            formatted = f"{num / 1_000_000_000:.{decimals}f}B"
        elif abs_num >= 1_000_000:
            formatted = f"{num / 1_000_000:.{decimals}f}M"
        elif abs_num >= 1_000:
            formatted = f"{num / 1_000:.{decimals}f}K"
        else:
            formatted = f"{num:.{decimals}f}"

        return formatted

    except Exception as e:
        logger.error("compact_number_failed", value=value, error=str(e))
        return str(value)


def format_pnl(
    pnl: Union[Decimal, float],
    currency: str = "USD",
    show_sign: bool = True
) -> str:
    """Format profit/loss.

    Args:
        pnl: PnL amount
        currency: Currency code
        show_sign: Show + for profits

    Returns:
        Formatted PnL string with color indicator
    """
    try:
        if isinstance(pnl, Decimal):
            value = float(pnl)
        else:
            value = float(pnl)

        formatted = format_currency(abs(value), currency, decimals=2, symbol=True)

        if value > 0:
            prefix = "+" if show_sign else ""
            return f"{prefix}{formatted}"
        elif value < 0:
            return f"-{formatted}"
        else:
            return formatted

    except Exception as e:
        logger.error("pnl_format_failed", pnl=pnl, error=str(e))
        return str(pnl)


def format_volume(
    volume: Union[Decimal, float, int],
    compact: bool = True
) -> str:
    """Format trading volume.

    Args:
        volume: Volume amount
        compact: Use compact notation

    Returns:
        Formatted volume string
    """
    try:
        if isinstance(volume, Decimal):
            value = float(volume)
        else:
            value = float(volume)

        if compact:
            return format_compact_number(value, decimals=2)
        else:
            return f"{value:,.2f}"

    except Exception as e:
        logger.error("volume_format_failed", volume=volume, error=str(e))
        return str(volume)


def format_trade_summary(trade: Dict[str, Any]) -> str:
    """Format trade information as summary string.

    Args:
        trade: Trade dictionary

    Returns:
        Formatted trade summary
    """
    try:
        symbol = trade.get("symbol", "UNKNOWN")
        side = trade.get("side", "UNKNOWN")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)
        pnl = trade.get("pnl", 0)

        formatted_qty = format_quantity(quantity, precision=8)
        formatted_price = format_price(price, precision=8)
        formatted_pnl = format_pnl(pnl) if pnl else "N/A"

        return (
            f"{side} {formatted_qty} {symbol} @ {formatted_price} | "
            f"PnL: {formatted_pnl}"
        )

    except Exception as e:
        logger.error("trade_summary_failed", error=str(e))
        return str(trade)


def format_order_summary(order: Dict[str, Any]) -> str:
    """Format order information as summary string.

    Args:
        order: Order dictionary

    Returns:
        Formatted order summary
    """
    try:
        symbol = order.get("symbol", "UNKNOWN")
        side = order.get("side", "UNKNOWN")
        order_type = order.get("order_type", "UNKNOWN")
        quantity = order.get("quantity", 0)
        price = order.get("price")
        status = order.get("status", "UNKNOWN")

        formatted_qty = format_quantity(quantity, precision=8)

        if price:
            formatted_price = format_price(price, precision=8)
            price_str = f"@ {formatted_price}"
        else:
            price_str = "MARKET"

        return (
            f"[{status}] {side} {formatted_qty} {symbol} "
            f"{price_str} ({order_type})"
        )

    except Exception as e:
        logger.error("order_summary_failed", error=str(e))
        return str(order)


def format_performance_metrics(metrics: Dict[str, Any]) -> str:
    """Format performance metrics as readable string.

    Args:
        metrics: Performance metrics dictionary

    Returns:
        Formatted metrics string
    """
    try:
        lines = []

        if "total_return" in metrics:
            lines.append(
                f"Total Return: {format_percentage(metrics['total_return'])}"
            )

        if "sharpe_ratio" in metrics:
            lines.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        if "max_drawdown" in metrics:
            lines.append(
                f"Max Drawdown: {format_percentage(metrics['max_drawdown'])}"
            )

        if "win_rate" in metrics:
            lines.append(f"Win Rate: {format_percentage(metrics['win_rate'])}")

        if "total_trades" in metrics:
            lines.append(f"Total Trades: {metrics['total_trades']:,}")

        return " | ".join(lines)

    except Exception as e:
        logger.error("metrics_format_failed", error=str(e))
        return str(metrics)


def format_latency(
    latency_ms: float,
    include_unit: bool = True
) -> str:
    """Format latency measurement.

    Args:
        latency_ms: Latency in milliseconds
        include_unit: Include unit suffix

    Returns:
        Formatted latency string
    """
    try:
        if latency_ms < 1:
            value = latency_ms * 1000
            unit = "μs" if include_unit else ""
            return f"{value:.0f}{unit}"
        elif latency_ms < 1000:
            unit = "ms" if include_unit else ""
            return f"{latency_ms:.2f}{unit}"
        else:
            value = latency_ms / 1000
            unit = "s" if include_unit else ""
            return f"{value:.2f}{unit}"

    except Exception as e:
        logger.error("latency_format_failed", latency=latency_ms, error=str(e))
        return f"{latency_ms}ms"


def format_table_row(
    values: list,
    widths: list,
    align: Optional[list] = None
) -> str:
    """Format row for text table.

    Args:
        values: Column values
        widths: Column widths
        align: Alignment for each column ('left', 'right', 'center')

    Returns:
        Formatted row string
    """
    try:
        if align is None:
            align = ['left'] * len(values)

        cells = []
        for value, width, alignment in zip(values, widths, align):
            str_value = str(value)

            if alignment == 'right':
                cell = str_value.rjust(width)
            elif alignment == 'center':
                cell = str_value.center(width)
            else:
                cell = str_value.ljust(width)

            cells.append(cell)

        return " | ".join(cells)

    except Exception as e:
        logger.error("table_row_format_failed", error=str(e))
        return str(values)


def format_timestamp(
    dt: datetime,
    include_ms: bool = False,
    format_type: str = "default"
) -> str:
    """Format timestamp in various formats.

    Args:
        dt: Datetime to format
        include_ms: Include milliseconds
        format_type: Format type ("default", "iso", "compact")

    Returns:
        Formatted timestamp string
    """
    try:
        if format_type == "iso":
            return dt.isoformat()
        elif format_type == "compact":
            return dt.strftime("%Y%m%d_%H%M%S")
        else:
            # Default format
            if include_ms:
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            else:
                return dt.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        logger.error("timestamp_format_failed", error=str(e))
        return str(dt)


def format_duration_human(seconds: float) -> str:
    """Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable string
    """
    try:
        if seconds < 0:
            return "0s"

        delta = timedelta(seconds=seconds)

        parts = []

        days = delta.days
        if days > 0:
            parts.append(f"{days}d")

        hours = delta.seconds // 3600
        if hours > 0:
            parts.append(f"{hours}h")

        minutes = (delta.seconds % 3600) // 60
        if minutes > 0:
            parts.append(f"{minutes}m")

        secs = delta.seconds % 60
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts[:2])  # Show max 2 units

    except Exception as e:
        logger.error("duration_format_failed", seconds=seconds, error=str(e))
        return f"{seconds:.0f}s"


def format_bytes(
    bytes_value: int,
    decimals: int = 2
) -> str:
    """Format bytes in human-readable format.

    Args:
        bytes_value: Bytes count
        decimals: Decimal places

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    try:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if bytes_value < 1024.0 or unit == 'PB':
                return f"{bytes_value:.{decimals}f} {unit}"
            bytes_value /= 1024.0

        return f"{bytes_value:.{decimals}f} PB"

    except Exception as e:
        logger.error("bytes_format_failed", error=str(e))
        return f"{bytes_value} B"


def colorize_pnl(pnl: Union[Decimal, float]) -> str:
    """Add ANSI color codes to PnL value.

    Args:
        pnl: PnL value

    Returns:
        Colorized string (green for profit, red for loss)
    """
    try:
        value = float(pnl) if isinstance(pnl, Decimal) else pnl

        # ANSI color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        formatted = format_pnl(pnl)

        if value > 0:
            return f"{GREEN}{formatted}{RESET}"
        elif value < 0:
            return f"{RED}{formatted}{RESET}"
        else:
            return formatted

    except Exception as e:
        logger.error("colorize_failed", error=str(e))
        return str(pnl)
