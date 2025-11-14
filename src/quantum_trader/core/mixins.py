"""
Mixins - Reusable mixin classes for common functionality.

This module provides mixin classes that can be combined with other classes
to add specific capabilities like logging, caching, and event handling.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from structlog import get_logger

logger = get_logger(__name__)


class LoggerMixin:
    """
    Mixin to add structured logging capabilities.

    Provides convenient logging methods with automatic context.

    Example:
        >>> class MyClass(LoggerMixin):
        ...     def process(self):
        ...         self.log_info("Processing started")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__name__)

    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, **kwargs)

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, **kwargs)

    def log_critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, **kwargs)


class CacheMixin:
    """
    Mixin to add simple in-memory caching.

    Provides TTL-based caching for method results.

    Example:
        >>> class DataFetcher(CacheMixin):
        ...     def get_data(self, key):
        ...         cached = self.cache_get(key)
        ...         if cached:
        ...             return cached
        ...         data = self.fetch_from_api(key)
        ...         self.cache_set(key, data, ttl=60)
        ...         return data
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)

    def cache_get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                # Expired, remove from cache
                del self._cache[key]
        return None

    def cache_set(self, key: str, value: Any, ttl: int = 300) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = (value, expiry)

    def cache_delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def cache_clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def cache_cleanup(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, expiry) in self._cache.items()
            if now >= expiry
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)


class TimestampMixin:
    """
    Mixin to add created_at and updated_at timestamps.

    Example:
        >>> class Order(TimestampMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.order_id = generate_id()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class ValidatorMixin:
    """
    Mixin to add validation capabilities.

    Example:
        >>> class TradingOrder(ValidatorMixin):
        ...     def __init__(self, price, quantity):
        ...         super().__init__()
        ...         self.validate_positive(price, "price")
        ...         self.validate_positive(quantity, "quantity")
        ...         self.price = price
        ...         self.quantity = quantity
    """

    def validate_required(self, value: Any, field_name: str) -> None:
        """
        Validate that value is not None.

        Args:
            value: Value to validate
            field_name: Field name for error message

        Raises:
            ValueError: If value is None
        """
        if value is None:
            raise ValueError(f"{field_name} is required")

    def validate_positive(self, value: Decimal, field_name: str) -> None:
        """
        Validate that Decimal value is positive.

        Args:
            value: Value to validate
            field_name: Field name for error message

        Raises:
            ValueError: If value is not positive
        """
        if value <= Decimal('0'):
            raise ValueError(f"{field_name} must be positive, got {value}")

    def validate_range(
        self,
        value: Decimal,
        min_value: Decimal,
        max_value: Decimal,
        field_name: str
    ) -> None:
        """
        Validate that value is within range.

        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            field_name: Field name for error message

        Raises:
            ValueError: If value out of range
        """
        if not (min_value <= value <= max_value):
            raise ValueError(
                f"{field_name} must be between {min_value} and {max_value}, got {value}"
            )

    def validate_type(self, value: Any, expected_type: type, field_name: str) -> None:
        """
        Validate value type.

        Args:
            value: Value to validate
            expected_type: Expected type
            field_name: Field name for error message

        Raises:
            TypeError: If value has wrong type
        """
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}"
            )


class EventEmitterMixin:
    """
    Mixin to add event emission capabilities.

    Example:
        >>> class TradingEngine(EventEmitterMixin):
        ...     def place_order(self, order):
        ...         # Place order logic
        ...         self.emit_event('order_placed', {'order_id': order.id})
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._event_handlers: Dict[str, List[Callable]] = {}

    def on(self, event_type: str, handler: Callable) -> None:
        """
        Register event handler.

        Args:
            event_type: Type of event
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def off(self, event_type: str, handler: Callable) -> None:
        """
        Unregister event handler.

        Args:
            event_type: Type of event
            handler: Handler function to remove
        """
        if event_type in self._event_handlers:
            self._event_handlers[event_type] = [
                h for h in self._event_handlers[event_type] if h != handler
            ]

    def emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit event to all registered handlers.

        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(data or {})
                except Exception as e:
                    logger.error(
                        "Event handler failed",
                        event_type=event_type,
                        error=str(e)
                    )

    async def emit_event_async(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit event asynchronously.

        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self._event_handlers:
            tasks = []
            for handler in self._event_handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(data or {}))
                else:
                    try:
                        handler(data or {})
                    except Exception as e:
                        logger.error(
                            "Event handler failed",
                            event_type=event_type,
                            error=str(e)
                        )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class StateMachineMixin:
    """
    Mixin to add state machine capabilities.

    Example:
        >>> class Order(StateMachineMixin):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.define_states(['PENDING', 'FILLED', 'CANCELLED'])
        ...         self.set_state('PENDING')
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state: Optional[str] = None
        self._valid_states: set = set()
        self._state_transitions: Dict[str, List[str]] = {}
        self._state_history: List[tuple] = []

    def define_states(self, states: List[str]) -> None:
        """
        Define valid states.

        Args:
            states: List of valid state names
        """
        self._valid_states = set(states)

    def define_transition(self, from_state: str, to_state: str) -> None:
        """
        Define allowed state transition.

        Args:
            from_state: Source state
            to_state: Target state
        """
        if from_state not in self._state_transitions:
            self._state_transitions[from_state] = []
        self._state_transitions[from_state].append(to_state)

    def set_state(self, new_state: str, force: bool = False) -> None:
        """
        Set state.

        Args:
            new_state: New state
            force: Skip transition validation

        Raises:
            ValueError: If state invalid or transition not allowed
        """
        if new_state not in self._valid_states:
            raise ValueError(f"Invalid state: {new_state}")

        if self._state is not None and not force:
            # Check if transition is allowed
            allowed_transitions = self._state_transitions.get(self._state, [])
            if new_state not in allowed_transitions:
                raise ValueError(
                    f"Invalid state transition from {self._state} to {new_state}"
                )

        # Record transition
        old_state = self._state
        self._state = new_state
        self._state_history.append((old_state, new_state, datetime.utcnow()))

    def get_state(self) -> Optional[str]:
        """
        Get current state.

        Returns:
            Current state name
        """
        return self._state

    def is_state(self, state: str) -> bool:
        """
        Check if in specific state.

        Args:
            state: State to check

        Returns:
            True if in specified state
        """
        return self._state == state

    def get_state_history(self) -> List[tuple]:
        """
        Get state transition history.

        Returns:
            List of (from_state, to_state, timestamp) tuples
        """
        return self._state_history.copy()


class SerializableMixin:
    """
    Mixin to add serialization capabilities.

    Example:
        >>> class Order(SerializableMixin):
        ...     def __init__(self, order_id, price):
        ...         super().__init__()
        ...         self.order_id = order_id
        ...         self.price = price
        >>> order = Order('123', Decimal('100.50'))
        >>> data = order.to_dict()
    """

    def to_dict(self, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Convert object to dictionary.

        Args:
            exclude: List of attribute names to exclude

        Returns:
            Dictionary representation
        """
        exclude = exclude or []
        result = {}

        for key, value in self.__dict__.items():
            if key.startswith('_') or key in exclude:
                continue

            # Convert Decimal to string
            if isinstance(value, Decimal):
                result[key] = str(value)
            # Convert datetime to ISO format
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            # Handle nested serializable objects
            elif hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            # Handle lists
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SerializableMixin':
        """
        Create object from dictionary.

        Args:
            data: Dictionary with object data

        Returns:
            New object instance
        """
        return cls(**data)
