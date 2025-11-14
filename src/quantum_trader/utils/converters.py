"""Data Converter Utilities.

Production-ready converters for data type transformations, unit conversions,
and format standardization across the trading system.
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional, Dict, List, Any, Union
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


def to_decimal(
    value: Union[str, int, float, Decimal],
    precision: int = 8,
    rounding: str = ROUND_HALF_UP
) -> Decimal:
    """Convert value to Decimal with precision.

    Args:
        value: Value to convert
        precision: Decimal places
        rounding: Rounding mode

    Returns:
        Decimal value

    Raises:
        ValueError: If conversion fails
    """
    try:
        if isinstance(value, Decimal):
            dec = value
        elif isinstance(value, (int, float)):
            dec = Decimal(str(value))
        elif isinstance(value, str):
            # Remove whitespace and common currency symbols
            cleaned = value.strip().replace(',', '').replace('$', '').replace('€', '')
            dec = Decimal(cleaned)
        else:
            raise ValueError(f"Cannot convert {type(value)} to Decimal")

        # Apply precision
        quantize_format = Decimal('0.1') ** precision
        return dec.quantize(quantize_format, rounding=rounding)

    except (InvalidOperation, ValueError) as e:
        logger.error("decimal_conversion_failed", value=value, error=str(e))
        raise ValueError(f"Failed to convert to Decimal: {e}")


def decimal_to_float(value: Decimal) -> float:
    """Convert Decimal to float safely.

    Args:
        value: Decimal value

    Returns:
        Float value

    Raises:
        ValueError: If conversion fails
    """
    try:
        return float(value)
    except (ValueError, OverflowError) as e:
        logger.error("float_conversion_failed", value=str(value), error=str(e))
        raise ValueError(f"Failed to convert to float: {e}")


def satoshi_to_btc(satoshi: int) -> Decimal:
    """Convert satoshi to BTC.

    Args:
        satoshi: Satoshi amount

    Returns:
        BTC amount

    Raises:
        ValueError: If satoshi invalid
    """
    try:
        if satoshi < 0:
            raise ValueError("Satoshi amount cannot be negative")

        return Decimal(str(satoshi)) / Decimal("100000000")

    except Exception as e:
        logger.error("satoshi_conversion_failed", satoshi=satoshi, error=str(e))
        raise ValueError(f"Failed to convert satoshi: {e}")


def btc_to_satoshi(btc: Decimal) -> int:
    """Convert BTC to satoshi.

    Args:
        btc: BTC amount

    Returns:
        Satoshi amount

    Raises:
        ValueError: If BTC invalid
    """
    try:
        if btc < Decimal("0"):
            raise ValueError("BTC amount cannot be negative")

        satoshi = btc * Decimal("100000000")
        return int(satoshi)

    except Exception as e:
        logger.error("btc_conversion_failed", btc=str(btc), error=str(e))
        raise ValueError(f"Failed to convert BTC: {e}")


def wei_to_eth(wei: int) -> Decimal:
    """Convert Wei to ETH.

    Args:
        wei: Wei amount

    Returns:
        ETH amount

    Raises:
        ValueError: If wei invalid
    """
    try:
        if wei < 0:
            raise ValueError("Wei amount cannot be negative")

        return Decimal(str(wei)) / Decimal("1000000000000000000")

    except Exception as e:
        logger.error("wei_conversion_failed", wei=wei, error=str(e))
        raise ValueError(f"Failed to convert wei: {e}")


def eth_to_wei(eth: Decimal) -> int:
    """Convert ETH to Wei.

    Args:
        eth: ETH amount

    Returns:
        Wei amount

    Raises:
        ValueError: If ETH invalid
    """
    try:
        if eth < Decimal("0"):
            raise ValueError("ETH amount cannot be negative")

        wei = eth * Decimal("1000000000000000000")
        return int(wei)

    except Exception as e:
        logger.error("eth_conversion_failed", eth=str(eth), error=str(e))
        raise ValueError(f"Failed to convert ETH: {e}")


def percentage_to_decimal(percentage: Union[str, float, Decimal]) -> Decimal:
    """Convert percentage to decimal.

    Args:
        percentage: Percentage value (e.g., "15.5%" or 15.5)

    Returns:
        Decimal value (e.g., 0.155)

    Raises:
        ValueError: If conversion fails
    """
    try:
        if isinstance(percentage, str):
            # Remove % symbol and whitespace
            cleaned = percentage.strip().replace('%', '')
            pct = Decimal(cleaned)
        else:
            pct = Decimal(str(percentage))

        return pct / Decimal("100")

    except Exception as e:
        logger.error("percentage_conversion_failed", value=percentage, error=str(e))
        raise ValueError(f"Failed to convert percentage: {e}")


def decimal_to_percentage(value: Decimal, decimals: int = 2) -> str:
    """Convert decimal to percentage string.

    Args:
        value: Decimal value (e.g., 0.155)
        decimals: Number of decimal places

    Returns:
        Percentage string (e.g., "15.50%")
    """
    try:
        pct = value * Decimal("100")
        format_str = f"{{:.{decimals}f}}%"
        return format_str.format(pct)

    except Exception as e:
        logger.error("percentage_format_failed", value=str(value), error=str(e))
        raise ValueError(f"Failed to format percentage: {e}")


def normalize_symbol(symbol: str, exchange: Optional[str] = None) -> str:
    """Normalize trading symbol to standard format.

    Args:
        symbol: Trading symbol (various formats)
        exchange: Exchange name for specific normalization

    Returns:
        Normalized symbol (e.g., "BTC/USDT")

    Raises:
        ValueError: If symbol invalid
    """
    try:
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be non-empty string")

        # Remove whitespace
        symbol = symbol.strip().upper()

        # Exchange-specific normalization
        if exchange and exchange.upper() == "BINANCE":
            # Binance uses BTCUSDT, convert to BTC/USDT
            if "/" not in symbol and len(symbol) >= 6:
                # Common quote currencies
                for quote in ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"]:
                    if symbol.endswith(quote):
                        base = symbol[:-len(quote)]
                        return f"{base}/{quote}"

        # If already in standard format
        if "/" in symbol:
            parts = symbol.split("/")
            if len(parts) == 2 and all(parts):
                return symbol

        # Default: assume last 4 chars are quote currency
        if len(symbol) >= 6:
            base = symbol[:-4]
            quote = symbol[-4:]
            return f"{base}/{quote}"

        raise ValueError(f"Cannot normalize symbol: {symbol}")

    except Exception as e:
        logger.error("symbol_normalization_failed", symbol=symbol, error=str(e))
        raise ValueError(f"Failed to normalize symbol: {e}")


def denormalize_symbol(symbol: str, exchange: str) -> str:
    """Convert normalized symbol to exchange-specific format.

    Args:
        symbol: Normalized symbol (e.g., "BTC/USDT")
        exchange: Exchange name

    Returns:
        Exchange-specific symbol

    Raises:
        ValueError: If conversion fails
    """
    try:
        if "/" not in symbol:
            raise ValueError("Symbol must be in normalized format (BASE/QUOTE)")

        base, quote = symbol.split("/")

        exchange_upper = exchange.upper()

        # Exchange-specific formats
        if exchange_upper in ["BINANCE", "BYBIT", "OKX"]:
            # Remove slash: BTC/USDT -> BTCUSDT
            return f"{base}{quote}"
        elif exchange_upper == "KRAKEN":
            # Kraken uses X/Z prefixes for some pairs
            return f"{base}{quote}"
        else:
            # Default: keep normalized format
            return symbol

    except Exception as e:
        logger.error(
            "symbol_denormalization_failed",
            symbol=symbol,
            exchange=exchange,
            error=str(e)
        )
        raise ValueError(f"Failed to denormalize symbol: {e}")


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")

    Returns:
        Seconds

    Raises:
        ValueError: If timeframe invalid
    """
    try:
        if not timeframe:
            raise ValueError("Timeframe cannot be empty")

        # Parse timeframe
        timeframe = timeframe.strip().lower()

        if timeframe.endswith('s'):
            return int(timeframe[:-1])
        elif timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 604800
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

    except Exception as e:
        logger.error("timeframe_conversion_failed", timeframe=timeframe, error=str(e))
        raise ValueError(f"Failed to convert timeframe: {e}")


def seconds_to_timeframe(seconds: int) -> str:
    """Convert seconds to timeframe string.

    Args:
        seconds: Number of seconds

    Returns:
        Timeframe string

    Raises:
        ValueError: If seconds invalid
    """
    try:
        if seconds <= 0:
            raise ValueError("Seconds must be positive")

        # Find best unit
        if seconds % 604800 == 0:
            return f"{seconds // 604800}w"
        elif seconds % 86400 == 0:
            return f"{seconds // 86400}d"
        elif seconds % 3600 == 0:
            return f"{seconds // 3600}h"
        elif seconds % 60 == 0:
            return f"{seconds // 60}m"
        else:
            return f"{seconds}s"

    except Exception as e:
        logger.error("timeframe_format_failed", seconds=seconds, error=str(e))
        raise ValueError(f"Failed to format timeframe: {e}")


def price_to_tick_size(price: Decimal, tick_size: Decimal) -> Decimal:
    """Round price to tick size.

    Args:
        price: Original price
        tick_size: Minimum price increment

    Returns:
        Rounded price

    Raises:
        ValueError: If inputs invalid
    """
    try:
        if tick_size <= Decimal("0"):
            raise ValueError("Tick size must be positive")

        # Round to nearest tick
        ticks = (price / tick_size).to_integral_value(rounding=ROUND_HALF_UP)
        rounded_price = ticks * tick_size

        return rounded_price

    except Exception as e:
        logger.error(
            "tick_rounding_failed",
            price=str(price),
            tick_size=str(tick_size),
            error=str(e)
        )
        raise ValueError(f"Failed to round to tick size: {e}")


def quantity_to_lot_size(quantity: Decimal, lot_size: Decimal) -> Decimal:
    """Round quantity to lot size.

    Args:
        quantity: Original quantity
        lot_size: Minimum quantity increment

    Returns:
        Rounded quantity

    Raises:
        ValueError: If inputs invalid
    """
    try:
        if lot_size <= Decimal("0"):
            raise ValueError("Lot size must be positive")

        # Round down to lot size
        lots = (quantity / lot_size).to_integral_value(rounding='ROUND_DOWN')
        rounded_quantity = lots * lot_size

        return max(rounded_quantity, Decimal("0"))

    except Exception as e:
        logger.error(
            "lot_rounding_failed",
            quantity=str(quantity),
            lot_size=str(lot_size),
            error=str(e)
        )
        raise ValueError(f"Failed to round to lot size: {e}")


def dict_to_polars(
    data: Union[Dict[str, List], List[Dict]],
    schema: Optional[Dict[str, Any]] = None
) -> pl.DataFrame:
    """Convert dictionary to Polars DataFrame.

    Args:
        data: Dictionary with column names as keys or list of dicts
        schema: Optional schema specification

    Returns:
        Polars DataFrame

    Raises:
        ValueError: If conversion fails
    """
    try:
        if isinstance(data, list):
            # List of dicts
            df = pl.DataFrame(data, schema=schema)
        elif isinstance(data, dict):
            # Dict of lists
            df = pl.DataFrame(data, schema=schema)
        else:
            raise ValueError("Data must be dict or list of dicts")

        logger.debug("polars_df_created", rows=df.height, cols=df.width)
        return df

    except Exception as e:
        logger.error("polars_conversion_failed", error=str(e))
        raise ValueError(f"Failed to create Polars DataFrame: {e}")


def polars_to_dict(
    df: pl.DataFrame,
    orient: str = "dict"
) -> Union[Dict[str, List], List[Dict]]:
    """Convert Polars DataFrame to dictionary.

    Args:
        df: Polars DataFrame
        orient: Output format ("dict" or "records")

    Returns:
        Dictionary or list of dictionaries

    Raises:
        ValueError: If conversion fails
    """
    try:
        if orient == "dict":
            # {column: [values]}
            return {col: df[col].to_list() for col in df.columns}
        elif orient == "records":
            # [{col: val, ...}, ...]
            return df.to_dicts()
        else:
            raise ValueError(f"Invalid orient: {orient}")

    except Exception as e:
        logger.error("dict_conversion_failed", orient=orient, error=str(e))
        raise ValueError(f"Failed to convert to dict: {e}")


def format_number(
    value: Union[int, float, Decimal],
    precision: int = 2,
    use_thousands_separator: bool = True
) -> str:
    """Format number with thousands separator and precision.

    Args:
        value: Number to format
        precision: Decimal places
        use_thousands_separator: Add thousands separator

    Returns:
        Formatted string
    """
    try:
        if isinstance(value, Decimal):
            num = float(value)
        else:
            num = value

        if use_thousands_separator:
            return f"{num:,.{precision}f}"
        else:
            return f"{num:.{precision}f}"

    except Exception as e:
        logger.error("number_format_failed", value=value, error=str(e))
        return str(value)


def parse_number(value: str) -> Union[int, Decimal]:
    """Parse number from string.

    Args:
        value: Number string

    Returns:
        Parsed number

    Raises:
        ValueError: If parsing fails
    """
    try:
        # Remove common formatting
        cleaned = value.strip().replace(',', '').replace(' ', '')

        # Try integer first
        if '.' not in cleaned:
            return int(cleaned)

        # Otherwise decimal
        return Decimal(cleaned)

    except Exception as e:
        logger.error("number_parse_failed", value=value, error=str(e))
        raise ValueError(f"Failed to parse number: {e}")
