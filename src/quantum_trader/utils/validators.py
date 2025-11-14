"""Input Validation and Sanitization Utilities.

Production-ready validators for trading system inputs, data integrity,
and security validation with comprehensive error handling.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error exception.

    Attributes:
        field: Field name that failed validation
        message: Error message
        value: Invalid value
    """

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        """Initialize validation error.

        Args:
            field: Field name
            message: Error message
            value: Invalid value
        """
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


def validate_symbol(symbol: str, allow_slash: bool = True) -> bool:
    """Validate trading symbol format.

    Args:
        symbol: Trading symbol
        allow_slash: Allow slash in symbol (BTC/USDT format)

    Returns:
        True if valid

    Raises:
        ValidationError: If symbol invalid
    """
    try:
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("symbol", "Symbol must be non-empty string", symbol)

        # Check length
        if len(symbol) < 2 or len(symbol) > 20:
            raise ValidationError("symbol", "Symbol length must be 2-20 characters", symbol)

        # Check format
        if allow_slash:
            pattern = r'^[A-Z0-9]+(/[A-Z0-9]+)?$'
        else:
            pattern = r'^[A-Z0-9]+$'

        if not re.match(pattern, symbol.upper()):
            raise ValidationError("symbol", "Symbol contains invalid characters", symbol)

        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error("symbol_validation_failed", symbol=symbol, error=str(e))
        raise ValidationError("symbol", f"Validation failed: {e}", symbol)


def validate_price(
    price: Union[Decimal, float, str],
    min_price: Optional[Decimal] = None,
    max_price: Optional[Decimal] = None
) -> Decimal:
    """Validate and convert price value.

    Args:
        price: Price value
        min_price: Minimum allowed price
        max_price: Maximum allowed price

    Returns:
        Validated Decimal price

    Raises:
        ValidationError: If price invalid
    """
    try:
        # Convert to Decimal
        if isinstance(price, str):
            price_dec = Decimal(price)
        elif isinstance(price, (int, float)):
            price_dec = Decimal(str(price))
        elif isinstance(price, Decimal):
            price_dec = price
        else:
            raise ValidationError("price", "Price must be numeric", price)

        # Check positive
        if price_dec <= Decimal("0"):
            raise ValidationError("price", "Price must be positive", price)

        # Check min
        if min_price is not None and price_dec < min_price:
            raise ValidationError("price", f"Price below minimum: {min_price}", price)

        # Check max
        if max_price is not None and price_dec > max_price:
            raise ValidationError("price", f"Price above maximum: {max_price}", price)

        return price_dec

    except InvalidOperation as e:
        raise ValidationError("price", "Invalid price format", price)
    except ValidationError:
        raise
    except Exception as e:
        logger.error("price_validation_failed", price=price, error=str(e))
        raise ValidationError("price", f"Validation failed: {e}", price)


def validate_quantity(
    quantity: Union[Decimal, float, str],
    min_quantity: Optional[Decimal] = None,
    max_quantity: Optional[Decimal] = None,
    allow_zero: bool = False
) -> Decimal:
    """Validate and convert quantity value.

    Args:
        quantity: Quantity value
        min_quantity: Minimum allowed quantity
        max_quantity: Maximum allowed quantity
        allow_zero: Allow zero quantity

    Returns:
        Validated Decimal quantity

    Raises:
        ValidationError: If quantity invalid
    """
    try:
        # Convert to Decimal
        if isinstance(quantity, str):
            qty_dec = Decimal(quantity)
        elif isinstance(quantity, (int, float)):
            qty_dec = Decimal(str(quantity))
        elif isinstance(quantity, Decimal):
            qty_dec = quantity
        else:
            raise ValidationError("quantity", "Quantity must be numeric", quantity)

        # Check non-negative
        if qty_dec < Decimal("0"):
            raise ValidationError("quantity", "Quantity cannot be negative", quantity)

        # Check zero
        if not allow_zero and qty_dec == Decimal("0"):
            raise ValidationError("quantity", "Quantity cannot be zero", quantity)

        # Check min
        if min_quantity is not None and qty_dec < min_quantity:
            raise ValidationError("quantity", f"Quantity below minimum: {min_quantity}", quantity)

        # Check max
        if max_quantity is not None and qty_dec > max_quantity:
            raise ValidationError("quantity", f"Quantity above maximum: {max_quantity}", quantity)

        return qty_dec

    except InvalidOperation:
        raise ValidationError("quantity", "Invalid quantity format", quantity)
    except ValidationError:
        raise
    except Exception as e:
        logger.error("quantity_validation_failed", quantity=quantity, error=str(e))
        raise ValidationError("quantity", f"Validation failed: {e}", quantity)


def validate_order_side(side: str) -> str:
    """Validate order side.

    Args:
        side: Order side (BUY/SELL)

    Returns:
        Validated uppercase side

    Raises:
        ValidationError: If side invalid
    """
    try:
        if not side or not isinstance(side, str):
            raise ValidationError("side", "Side must be non-empty string", side)

        side_upper = side.upper()

        if side_upper not in ["BUY", "SELL"]:
            raise ValidationError("side", "Side must be BUY or SELL", side)

        return side_upper

    except ValidationError:
        raise
    except Exception as e:
        logger.error("side_validation_failed", side=side, error=str(e))
        raise ValidationError("side", f"Validation failed: {e}", side)


def validate_order_type(order_type: str) -> str:
    """Validate order type.

    Args:
        order_type: Order type

    Returns:
        Validated uppercase order type

    Raises:
        ValidationError: If order type invalid
    """
    try:
        if not order_type or not isinstance(order_type, str):
            raise ValidationError("order_type", "Order type must be non-empty string", order_type)

        type_upper = order_type.upper()

        valid_types = [
            "MARKET", "LIMIT", "STOP_LOSS", "STOP_LIMIT",
            "TAKE_PROFIT", "TRAILING_STOP"
        ]

        if type_upper not in valid_types:
            raise ValidationError(
                "order_type",
                f"Order type must be one of: {', '.join(valid_types)}",
                order_type
            )

        return type_upper

    except ValidationError:
        raise
    except Exception as e:
        logger.error("order_type_validation_failed", order_type=order_type, error=str(e))
        raise ValidationError("order_type", f"Validation failed: {e}", order_type)


def validate_exchange(exchange: str) -> str:
    """Validate exchange name.

    Args:
        exchange: Exchange name

    Returns:
        Validated uppercase exchange name

    Raises:
        ValidationError: If exchange invalid
    """
    try:
        if not exchange or not isinstance(exchange, str):
            raise ValidationError("exchange", "Exchange must be non-empty string", exchange)

        exchange_upper = exchange.upper()

        valid_exchanges = ["BINANCE", "BYBIT", "OKX", "KUCOIN", "BITGET"]

        if exchange_upper not in valid_exchanges:
            raise ValidationError(
                "exchange",
                f"Exchange must be one of: {', '.join(valid_exchanges)}",
                exchange
            )

        return exchange_upper

    except ValidationError:
        raise
    except Exception as e:
        logger.error("exchange_validation_failed", exchange=exchange, error=str(e))
        raise ValidationError("exchange", f"Validation failed: {e}", exchange)


def validate_timeframe(timeframe: str) -> str:
    """Validate timeframe string.

    Args:
        timeframe: Timeframe (e.g., "1m", "5m", "1h", "1d")

    Returns:
        Validated lowercase timeframe

    Raises:
        ValidationError: If timeframe invalid
    """
    try:
        if not timeframe or not isinstance(timeframe, str):
            raise ValidationError("timeframe", "Timeframe must be non-empty string", timeframe)

        timeframe_lower = timeframe.lower()

        # Check format: number followed by unit
        pattern = r'^\d+[smhdwM]$'
        if not re.match(pattern, timeframe_lower):
            raise ValidationError(
                "timeframe",
                "Timeframe must be number + unit (s/m/h/d/w/M)",
                timeframe
            )

        return timeframe_lower

    except ValidationError:
        raise
    except Exception as e:
        logger.error("timeframe_validation_failed", timeframe=timeframe, error=str(e))
        raise ValidationError("timeframe", f"Validation failed: {e}", timeframe)


def validate_timestamp(
    timestamp: Union[int, float, datetime],
    min_timestamp: Optional[Union[int, float, datetime]] = None,
    max_timestamp: Optional[Union[int, float, datetime]] = None
) -> datetime:
    """Validate and convert timestamp.

    Args:
        timestamp: Timestamp (unix or datetime)
        min_timestamp: Minimum allowed timestamp
        max_timestamp: Maximum allowed timestamp

    Returns:
        Validated datetime

    Raises:
        ValidationError: If timestamp invalid
    """
    try:
        from quantum_trader.utils.datetime_utils import timestamp_to_datetime, to_utc

        # Convert to datetime
        if isinstance(timestamp, datetime):
            dt = to_utc(timestamp)
        elif isinstance(timestamp, (int, float)):
            # Assume seconds, but check if milliseconds
            if timestamp > 1e12:  # Likely milliseconds
                dt = timestamp_to_datetime(timestamp, unit="ms")
            else:
                dt = timestamp_to_datetime(timestamp, unit="s")
        else:
            raise ValidationError("timestamp", "Timestamp must be int, float, or datetime", timestamp)

        # Check min
        if min_timestamp is not None:
            if isinstance(min_timestamp, datetime):
                min_dt = to_utc(min_timestamp)
            else:
                min_dt = timestamp_to_datetime(min_timestamp, unit="s")

            if dt < min_dt:
                raise ValidationError("timestamp", f"Timestamp before minimum: {min_dt}", timestamp)

        # Check max
        if max_timestamp is not None:
            if isinstance(max_timestamp, datetime):
                max_dt = to_utc(max_timestamp)
            else:
                max_dt = timestamp_to_datetime(max_timestamp, unit="s")

            if dt > max_dt:
                raise ValidationError("timestamp", f"Timestamp after maximum: {max_dt}", timestamp)

        return dt

    except ValidationError:
        raise
    except Exception as e:
        logger.error("timestamp_validation_failed", timestamp=timestamp, error=str(e))
        raise ValidationError("timestamp", f"Validation failed: {e}", timestamp)


def validate_api_key(api_key: str, min_length: int = 16) -> str:
    """Validate API key format.

    Args:
        api_key: API key string
        min_length: Minimum key length

    Returns:
        Validated API key

    Raises:
        ValidationError: If API key invalid
    """
    try:
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("api_key", "API key must be non-empty string", None)

        # Check length
        if len(api_key) < min_length:
            raise ValidationError("api_key", f"API key too short (min {min_length})", None)

        # Check for obvious patterns
        if api_key == "test" or api_key == "demo" or api_key.lower() == "apikey":
            raise ValidationError("api_key", "API key appears to be placeholder", None)

        return api_key

    except ValidationError:
        raise
    except Exception as e:
        logger.error("api_key_validation_failed", error=str(e))
        raise ValidationError("api_key", f"Validation failed: {e}", None)


def validate_percentage(
    value: Union[Decimal, float, str],
    min_pct: Decimal = Decimal("0"),
    max_pct: Decimal = Decimal("100")
) -> Decimal:
    """Validate percentage value.

    Args:
        value: Percentage value (0-100)
        min_pct: Minimum percentage
        max_pct: Maximum percentage

    Returns:
        Validated Decimal percentage

    Raises:
        ValidationError: If percentage invalid
    """
    try:
        # Convert to Decimal
        if isinstance(value, str):
            # Remove % if present
            value = value.strip().replace('%', '')
            pct_dec = Decimal(value)
        elif isinstance(value, (int, float)):
            pct_dec = Decimal(str(value))
        elif isinstance(value, Decimal):
            pct_dec = value
        else:
            raise ValidationError("percentage", "Percentage must be numeric", value)

        # Check range
        if pct_dec < min_pct:
            raise ValidationError("percentage", f"Percentage below minimum: {min_pct}", value)

        if pct_dec > max_pct:
            raise ValidationError("percentage", f"Percentage above maximum: {max_pct}", value)

        return pct_dec

    except InvalidOperation:
        raise ValidationError("percentage", "Invalid percentage format", value)
    except ValidationError:
        raise
    except Exception as e:
        logger.error("percentage_validation_failed", value=value, error=str(e))
        raise ValidationError("percentage", f"Validation failed: {e}", value)


def validate_config(
    config: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        optional_keys: List of optional keys

    Raises:
        ValidationError: If config invalid
    """
    try:
        if not isinstance(config, dict):
            raise ValidationError("config", "Config must be dictionary", config)

        # Check required keys
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValidationError(
                "config",
                f"Missing required keys: {', '.join(missing)}",
                None
            )

        # Check for unknown keys if optional_keys provided
        if optional_keys is not None:
            allowed = set(required_keys) | set(optional_keys)
            unknown = [key for key in config.keys() if key not in allowed]
            if unknown:
                logger.warning(
                    "config_unknown_keys",
                    keys=unknown
                )

    except ValidationError:
        raise
    except Exception as e:
        logger.error("config_validation_failed", error=str(e))
        raise ValidationError("config", f"Validation failed: {e}", None)


def validate_range(
    value: Union[Decimal, float, int],
    min_value: Optional[Union[Decimal, float, int]] = None,
    max_value: Optional[Union[Decimal, float, int]] = None,
    inclusive: bool = True
) -> Decimal:
    """Validate value is within range.

    Args:
        value: Value to validate
        min_value: Minimum value
        max_value: Maximum value
        inclusive: Include bounds in range

    Returns:
        Validated Decimal value

    Raises:
        ValidationError: If value out of range
    """
    try:
        value_dec = Decimal(str(value))

        if min_value is not None:
            min_dec = Decimal(str(min_value))
            if inclusive and value_dec < min_dec:
                raise ValidationError("value", f"Value below minimum: {min_value}", value)
            elif not inclusive and value_dec <= min_dec:
                raise ValidationError("value", f"Value must be greater than: {min_value}", value)

        if max_value is not None:
            max_dec = Decimal(str(max_value))
            if inclusive and value_dec > max_dec:
                raise ValidationError("value", f"Value above maximum: {max_value}", value)
            elif not inclusive and value_dec >= max_dec:
                raise ValidationError("value", f"Value must be less than: {max_value}", value)

        return value_dec

    except ValidationError:
        raise
    except Exception as e:
        logger.error("range_validation_failed", value=value, error=str(e))
        raise ValidationError("value", f"Validation failed: {e}", value)


def sanitize_input(
    text: str,
    max_length: int = 1000,
    allow_special_chars: bool = False
) -> str:
    """Sanitize user input for security.

    Args:
        text: Input text
        max_length: Maximum allowed length
        allow_special_chars: Allow special characters

    Returns:
        Sanitized string

    Raises:
        ValidationError: If input invalid
    """
    try:
        if not isinstance(text, str):
            raise ValidationError("input", "Input must be string", text)

        # Truncate
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning("input_truncated", original_length=len(text), max_length=max_length)

        # Remove dangerous characters
        if not allow_special_chars:
            # Keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\_]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text

    except ValidationError:
        raise
    except Exception as e:
        logger.error("input_sanitization_failed", error=str(e))
        raise ValidationError("input", f"Sanitization failed: {e}", None)


def is_valid_json(text: str) -> bool:
    """Check if string is valid JSON.

    Args:
        text: JSON string

    Returns:
        True if valid JSON
    """
    try:
        import json
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False
