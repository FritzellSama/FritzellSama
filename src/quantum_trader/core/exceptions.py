"""
Exceptions - Custom exception hierarchy for the trading system.

This module defines all custom exceptions used throughout the system
for precise error handling and debugging.
"""

from typing import Optional, Dict, Any
from decimal import Decimal


class QuantumTraderError(Exception):
    """Base exception for all Quantum Trader errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize base exception.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """String representation."""
        if self.details:
            details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# Configuration Errors


class ConfigurationError(QuantumTraderError):
    """Raised when configuration is invalid or missing."""

    pass


class ValidationError(QuantumTraderError):
    """Raised when input validation fails."""

    pass


# Exchange Errors


class ExchangeError(QuantumTraderError):
    """Base exception for exchange-related errors."""

    pass


class ExchangeConnectionError(ExchangeError):
    """Raised when exchange connection fails."""

    pass


class ExchangeAuthenticationError(ExchangeError):
    """Raised when exchange authentication fails."""

    pass


class ExchangeAPIError(ExchangeError):
    """Raised when exchange API returns an error."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Exchange-specific error code
            details: Additional error details
        """
        super().__init__(message, details)
        self.status_code = status_code
        self.error_code = error_code


class RateLimitError(ExchangeError):
    """Raised when exchange rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            details: Additional details
        """
        super().__init__(message, details)
        self.retry_after = retry_after


class InsufficientBalanceError(ExchangeError):
    """Raised when account balance is insufficient for operation."""

    def __init__(
        self,
        message: str,
        required: Optional[Decimal] = None,
        available: Optional[Decimal] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize insufficient balance error.

        Args:
            message: Error message
            required: Required balance
            available: Available balance
            details: Additional details
        """
        super().__init__(message, details)
        self.required = required
        self.available = available


# Order Errors


class OrderError(QuantumTraderError):
    """Base exception for order-related errors."""

    pass


class InvalidOrderError(OrderError):
    """Raised when order parameters are invalid."""

    pass


class OrderRejectedError(OrderError):
    """Raised when order is rejected by exchange."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize order rejected error.

        Args:
            message: Error message
            order_id: Order identifier
            reason: Rejection reason
            details: Additional details
        """
        super().__init__(message, details)
        self.order_id = order_id
        self.reason = reason


class OrderNotFoundError(OrderError):
    """Raised when order cannot be found."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize order not found error.

        Args:
            message: Error message
            order_id: Order identifier
            details: Additional details
        """
        super().__init__(message, details)
        self.order_id = order_id


class OrderTimeoutError(OrderError):
    """Raised when order operation times out."""

    pass


# Risk Management Errors


class RiskError(QuantumTraderError):
    """Base exception for risk management errors."""

    pass


class RiskLimitExceededError(RiskError):
    """Raised when risk limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,
        current_value: Optional[Decimal] = None,
        limit_value: Optional[Decimal] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize risk limit exceeded error.

        Args:
            message: Error message
            limit_type: Type of limit exceeded
            current_value: Current value
            limit_value: Limit value
            details: Additional details
        """
        super().__init__(message, details)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class PositionLimitError(RiskError):
    """Raised when position limit is exceeded."""

    pass


class DrawdownLimitError(RiskError):
    """Raised when drawdown limit is exceeded."""

    def __init__(
        self,
        message: str,
        current_drawdown: Optional[Decimal] = None,
        max_drawdown: Optional[Decimal] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize drawdown limit error.

        Args:
            message: Error message
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum allowed drawdown
            details: Additional details
        """
        super().__init__(message, details)
        self.current_drawdown = current_drawdown
        self.max_drawdown = max_drawdown


class ExposureLimit Error(RiskError):
    """Raised when exposure limit is exceeded."""

    pass


# Strategy Errors


class StrategyError(QuantumTraderError):
    """Base exception for strategy errors."""

    pass


class StrategyInitializationError(StrategyError):
    """Raised when strategy initialization fails."""

    pass


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""

    pass


class InvalidSignalError(StrategyError):
    """Raised when trading signal is invalid."""

    pass


# Data Errors


class DataError(QuantumTraderError):
    """Base exception for data-related errors."""

    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


class DataQualityError(DataError):
    """Raised when data quality is insufficient."""

    def __init__(
        self,
        message: str,
        missing_count: Optional[int] = None,
        total_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize data quality error.

        Args:
            message: Error message
            missing_count: Number of missing data points
            total_count: Total expected data points
            details: Additional details
        """
        super().__init__(message, details)
        self.missing_count = missing_count
        self.total_count = total_count


class InsufficientDataError(DataError):
    """Raised when insufficient data for operation."""

    def __init__(
        self,
        message: str,
        required: Optional[int] = None,
        available: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize insufficient data error.

        Args:
            message: Error message
            required: Required data points
            available: Available data points
            details: Additional details
        """
        super().__init__(message, details)
        self.required = required
        self.available = available


# Database Errors


class DatabaseError(QuantumTraderError):
    """Base exception for database errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""

    pass


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails."""

    pass


# ML Model Errors


class ModelError(QuantumTraderError):
    """Base exception for ML model errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when ML model is not found."""

    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""

    pass


# WebSocket Errors


class WebSocketError(QuantumTraderError):
    """Base exception for WebSocket errors."""

    pass


class WebSocketConnectionError(WebSocketError):
    """Raised when WebSocket connection fails."""

    pass


class WebSocketSubscriptionError(WebSocketError):
    """Raised when WebSocket subscription fails."""

    pass


class WebSocketMessageError(WebSocketError):
    """Raised when WebSocket message parsing fails."""

    pass


# Authentication Errors


class AuthenticationError(QuantumTraderError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(QuantumTraderError):
    """Raised when authorization fails."""

    pass


# System Errors


class SystemError(QuantumTraderError):
    """Base exception for system-level errors."""

    pass


class ComponentError(SystemError):
    """Raised when system component fails."""

    def __init__(
        self,
        message: str,
        component_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize component error.

        Args:
            message: Error message
            component_name: Name of failed component
            details: Additional details
        """
        super().__init__(message, details)
        self.component_name = component_name


class ServiceUnavailableError(SystemError):
    """Raised when service is unavailable."""

    pass


class TimeoutError(SystemError):
    """Raised when operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration
            details: Additional details
        """
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


# Backtesting Errors


class BacktestError(QuantumTraderError):
    """Base exception for backtesting errors."""

    pass


class BacktestConfigurationError(BacktestError):
    """Raised when backtest configuration is invalid."""

    pass


class BacktestExecutionError(BacktestError):
    """Raised when backtest execution fails."""

    pass


# Cache Errors


class CacheError(QuantumTraderError):
    """Base exception for cache errors."""

    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""

    pass


class CacheMissError(CacheError):
    """Raised when cache key is not found."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize cache miss error.

        Args:
            message: Error message
            cache_key: Missing cache key
            details: Additional details
        """
        super().__init__(message, details)
        self.cache_key = cache_key


# Notification Errors


class NotificationError(QuantumTraderError):
    """Base exception for notification errors."""

    pass


class NotificationDeliveryError(NotificationError):
    """Raised when notification delivery fails."""

    pass
