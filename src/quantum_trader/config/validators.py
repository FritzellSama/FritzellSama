"""
Validators - Input validation and sanitization.

This module provides comprehensive validation for all inputs including
trading parameters, configuration values, and user inputs.
"""

from decimal import Decimal, InvalidOperation
from typing import Any, Optional, List, Dict, Union, Tuple
from datetime import datetime
import re
from structlog import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class ConfigValidator:
    """
    Validates configuration values and trading parameters.

    Provides type-safe validation with appropriate error messages
    for debugging and security.

    Example:
        >>> validator = ConfigValidator(config)
        >>> validator.validate_decimal('100.50', min_value=Decimal('0'))
        >>> validator.validate_symbol('BTC/USDT')
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize validator with configuration.

        Args:
            config: Configuration dict with validation rules
        """
        self.config = config

        # Load validation rules from config
        self.min_order_size = Decimal(str(config.get('min_order_size', '10')))
        self.max_order_size = Decimal(str(config.get('max_order_size', '1000000')))
        self.min_price = Decimal(str(config.get('min_price', '0.00000001')))
        self.max_price = Decimal(str(config.get('max_price', '1000000')))
        self.max_leverage = Decimal(str(config.get('max_leverage', '100')))

        # Allowed values
        self.valid_exchanges = set(config.get('valid_exchanges', [
            'binance', 'bybit', 'okx', 'kucoin', 'bitget'
        ]))
        self.valid_timeframes = set(config.get('valid_timeframes', [
            '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'
        ]))
        self.valid_order_types = set(config.get('valid_order_types', [
            'MARKET', 'LIMIT', 'STOP_LOSS', 'TAKE_PROFIT'
        ]))
        self.valid_order_sides = set(config.get('valid_order_sides', ['BUY', 'SELL']))

        logger.debug("ConfigValidator initialized")

    def validate_decimal(
        self,
        value: Any,
        min_value: Optional[Decimal] = None,
        max_value: Optional[Decimal] = None,
        allow_zero: bool = True,
        field_name: str = 'value'
    ) -> Decimal:
        """
        Validate and convert value to Decimal.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_zero: Whether zero is allowed
            field_name: Field name for error messages

        Returns:
            Validated Decimal value

        Raises:
            ValidationError: If validation fails

        Example:
            >>> price = validator.validate_decimal('99.50', min_value=Decimal('0'))
        """
        try:
            if value is None:
                raise ValidationError(f"{field_name} cannot be None")

            # Convert to Decimal
            if isinstance(value, Decimal):
                decimal_value = value
            else:
                decimal_value = Decimal(str(value))

            # Check for NaN or Infinity
            if decimal_value.is_nan() or decimal_value.is_infinite():
                raise ValidationError(f"{field_name} must be a finite number")

            # Check zero
            if not allow_zero and decimal_value == Decimal('0'):
                raise ValidationError(f"{field_name} cannot be zero")

            # Check min/max
            if min_value is not None and decimal_value < min_value:
                raise ValidationError(
                    f"{field_name} must be >= {min_value}, got {decimal_value}"
                )

            if max_value is not None and decimal_value > max_value:
                raise ValidationError(
                    f"{field_name} must be <= {max_value}, got {decimal_value}"
                )

            return decimal_value

        except InvalidOperation as e:
            raise ValidationError(f"Invalid decimal value for {field_name}: {value}")
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Error validating {field_name}: {str(e)}")

    def validate_price(self, price: Any, symbol: Optional[str] = None) -> Decimal:
        """
        Validate trading price.

        Args:
            price: Price value
            symbol: Trading symbol (for symbol-specific rules)

        Returns:
            Validated price as Decimal

        Raises:
            ValidationError: If price invalid
        """
        return self.validate_decimal(
            price,
            min_value=self.min_price,
            max_value=self.max_price,
            allow_zero=False,
            field_name='price'
        )

    def validate_quantity(self, quantity: Any, symbol: Optional[str] = None) -> Decimal:
        """
        Validate order quantity.

        Args:
            quantity: Quantity value
            symbol: Trading symbol

        Returns:
            Validated quantity as Decimal

        Raises:
            ValidationError: If quantity invalid
        """
        return self.validate_decimal(
            quantity,
            min_value=self.min_order_size,
            max_value=self.max_order_size,
            allow_zero=False,
            field_name='quantity'
        )

    def validate_symbol(self, symbol: str) -> str:
        """
        Validate trading symbol format.

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')

        Returns:
            Validated symbol string

        Raises:
            ValidationError: If symbol invalid

        Example:
            >>> symbol = validator.validate_symbol('BTC/USDT')
        """
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")

        # Check format: BASE/QUOTE
        if '/' not in symbol:
            raise ValidationError(f"Invalid symbol format: {symbol}. Expected format: BASE/QUOTE")

        parts = symbol.split('/')
        if len(parts) != 2:
            raise ValidationError(f"Invalid symbol format: {symbol}")

        base, quote = parts

        # Validate base and quote currencies
        if not base or not quote:
            raise ValidationError(f"Invalid symbol: {symbol}")

        # Check allowed characters (alphanumeric only)
        if not re.match(r'^[A-Z0-9]+/[A-Z0-9]+$', symbol):
            raise ValidationError(
                f"Symbol must contain only uppercase letters and numbers: {symbol}"
            )

        return symbol

    def validate_exchange(self, exchange: str) -> str:
        """
        Validate exchange name.

        Args:
            exchange: Exchange name

        Returns:
            Validated exchange name (lowercase)

        Raises:
            ValidationError: If exchange invalid
        """
        if not exchange or not isinstance(exchange, str):
            raise ValidationError("Exchange must be a non-empty string")

        exchange_lower = exchange.lower()

        if exchange_lower not in self.valid_exchanges:
            raise ValidationError(
                f"Invalid exchange: {exchange}. "
                f"Valid exchanges: {', '.join(sorted(self.valid_exchanges))}"
            )

        return exchange_lower

    def validate_timeframe(self, timeframe: str) -> str:
        """
        Validate timeframe string.

        Args:
            timeframe: Timeframe (e.g., '1h', '5m')

        Returns:
            Validated timeframe string

        Raises:
            ValidationError: If timeframe invalid
        """
        if not timeframe or not isinstance(timeframe, str):
            raise ValidationError("Timeframe must be a non-empty string")

        if timeframe not in self.valid_timeframes:
            raise ValidationError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid timeframes: {', '.join(sorted(self.valid_timeframes))}"
            )

        return timeframe

    def validate_order_type(self, order_type: str) -> str:
        """
        Validate order type.

        Args:
            order_type: Order type (e.g., 'MARKET', 'LIMIT')

        Returns:
            Validated order type string

        Raises:
            ValidationError: If order type invalid
        """
        if not order_type or not isinstance(order_type, str):
            raise ValidationError("Order type must be a non-empty string")

        order_type_upper = order_type.upper()

        if order_type_upper not in self.valid_order_types:
            raise ValidationError(
                f"Invalid order type: {order_type}. "
                f"Valid types: {', '.join(sorted(self.valid_order_types))}"
            )

        return order_type_upper

    def validate_order_side(self, side: str) -> str:
        """
        Validate order side.

        Args:
            side: Order side ('BUY' or 'SELL')

        Returns:
            Validated order side string

        Raises:
            ValidationError: If order side invalid
        """
        if not side or not isinstance(side, str):
            raise ValidationError("Order side must be a non-empty string")

        side_upper = side.upper()

        if side_upper not in self.valid_order_sides:
            raise ValidationError(
                f"Invalid order side: {side}. "
                f"Valid sides: {', '.join(sorted(self.valid_order_sides))}"
            )

        return side_upper

    def validate_leverage(self, leverage: Any) -> Decimal:
        """
        Validate leverage value.

        Args:
            leverage: Leverage value

        Returns:
            Validated leverage as Decimal

        Raises:
            ValidationError: If leverage invalid
        """
        return self.validate_decimal(
            leverage,
            min_value=Decimal('1'),
            max_value=self.max_leverage,
            allow_zero=False,
            field_name='leverage'
        )

    def validate_percentage(self, percentage: Any, field_name: str = 'percentage') -> Decimal:
        """
        Validate percentage value (0-100).

        Args:
            percentage: Percentage value
            field_name: Field name for error messages

        Returns:
            Validated percentage as Decimal

        Raises:
            ValidationError: If percentage invalid
        """
        return self.validate_decimal(
            percentage,
            min_value=Decimal('0'),
            max_value=Decimal('100'),
            allow_zero=True,
            field_name=field_name
        )

    def validate_timestamp(self, timestamp: Any) -> datetime:
        """
        Validate timestamp.

        Args:
            timestamp: Timestamp (datetime, int milliseconds, or ISO string)

        Returns:
            Validated datetime object (UTC)

        Raises:
            ValidationError: If timestamp invalid
        """
        try:
            if isinstance(timestamp, datetime):
                return timestamp

            if isinstance(timestamp, int):
                # Assume milliseconds
                return datetime.fromtimestamp(timestamp / 1000)

            if isinstance(timestamp, str):
                # Try ISO format
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            raise ValidationError(f"Invalid timestamp type: {type(timestamp)}")

        except Exception as e:
            raise ValidationError(f"Invalid timestamp: {timestamp}, error: {str(e)}")

    def validate_string(
        self,
        value: Any,
        min_length: int = 1,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        field_name: str = 'value'
    ) -> str:
        """
        Validate string value.

        Args:
            value: String value
            min_length: Minimum length
            max_length: Maximum length
            pattern: Regex pattern to match
            field_name: Field name for error messages

        Returns:
            Validated string

        Raises:
            ValidationError: If string invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string, got {type(value)}")

        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters, got {len(value)}"
            )

        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters, got {len(value)}"
            )

        if pattern is not None and not re.match(pattern, value):
            raise ValidationError(f"{field_name} does not match required pattern")

        return value

    def validate_integer(
        self,
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        field_name: str = 'value'
    ) -> int:
        """
        Validate integer value.

        Args:
            value: Integer value
            min_value: Minimum value
            max_value: Maximum value
            field_name: Field name for error messages

        Returns:
            Validated integer

        Raises:
            ValidationError: If integer invalid
        """
        try:
            int_value = int(value)

            if min_value is not None and int_value < min_value:
                raise ValidationError(
                    f"{field_name} must be >= {min_value}, got {int_value}"
                )

            if max_value is not None and int_value > max_value:
                raise ValidationError(
                    f"{field_name} must be <= {max_value}, got {int_value}"
                )

            return int_value

        except (ValueError, TypeError):
            raise ValidationError(f"Invalid integer value for {field_name}: {value}")

    def validate_list(
        self,
        value: Any,
        min_length: int = 0,
        max_length: Optional[int] = None,
        item_type: Optional[type] = None,
        field_name: str = 'value'
    ) -> List[Any]:
        """
        Validate list value.

        Args:
            value: List value
            min_length: Minimum list length
            max_length: Maximum list length
            item_type: Expected type of list items
            field_name: Field name for error messages

        Returns:
            Validated list

        Raises:
            ValidationError: If list invalid
        """
        if not isinstance(value, list):
            raise ValidationError(f"{field_name} must be a list, got {type(value)}")

        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must have at least {min_length} items, got {len(value)}"
            )

        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must have at most {max_length} items, got {len(value)}"
            )

        if item_type is not None:
            for i, item in enumerate(value):
                if not isinstance(item, item_type):
                    raise ValidationError(
                        f"{field_name}[{i}] must be {item_type.__name__}, got {type(item).__name__}"
                    )

        return value

    def validate_dict(
        self,
        value: Any,
        required_keys: Optional[List[str]] = None,
        field_name: str = 'value'
    ) -> Dict[str, Any]:
        """
        Validate dictionary value.

        Args:
            value: Dictionary value
            required_keys: List of required keys
            field_name: Field name for error messages

        Returns:
            Validated dictionary

        Raises:
            ValidationError: If dictionary invalid
        """
        if not isinstance(value, dict):
            raise ValidationError(f"{field_name} must be a dict, got {type(value)}")

        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                raise ValidationError(
                    f"{field_name} missing required keys: {', '.join(missing_keys)}"
                )

        return value

    def sanitize_string(self, value: str) -> str:
        """
        Sanitize string for safe storage/display.

        Args:
            value: String to sanitize

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            return str(value)

        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)

        # Trim whitespace
        sanitized = sanitized.strip()

        return sanitized
