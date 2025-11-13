"""
Custom exceptions for Quantum Trader AI API.

Defines hierarchy of exceptions with error codes and HTTP status codes.
"""

from typing import Optional, Dict, Any


class QuantumTraderException(Exception):
    """Base exception for Quantum Trader AI.

    All custom exceptions inherit from this base class.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        status_code: HTTP status code
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        error_code: str = "QUANTUM_TRADER_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exception.

        Args:
            message: Error message
            error_code: Error code
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# Authentication and Authorization Exceptions


class AuthenticationException(QuantumTraderException):
    """Authentication failed exception."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize authentication exception."""
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            details=details
        )


class AuthorizationException(QuantumTraderException):
    """Authorization failed exception."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize authorization exception."""
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_FAILED",
            status_code=403,
            details=details
        )


class TokenExpiredException(AuthenticationException):
    """Token expired exception."""

    def __init__(
        self,
        message: str = "Token has expired",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize token expired exception."""
        super().__init__(message=message, details=details)
        self.error_code = "TOKEN_EXPIRED"


class InvalidTokenException(AuthenticationException):
    """Invalid token exception."""

    def __init__(
        self,
        message: str = "Invalid token",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize invalid token exception."""
        super().__init__(message=message, details=details)
        self.error_code = "INVALID_TOKEN"


# Validation Exceptions


class ValidationException(QuantumTraderException):
    """Validation failed exception."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize validation exception."""
        super().__init__(
            message=message,
            error_code="VALIDATION_FAILED",
            status_code=400,
            details=details
        )


class InvalidParameterException(ValidationException):
    """Invalid parameter exception."""

    def __init__(
        self,
        parameter: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize invalid parameter exception."""
        msg = message or f"Invalid parameter: {parameter}"
        super().__init__(message=msg, details=details)
        self.error_code = "INVALID_PARAMETER"


# Resource Exceptions


class NotFoundException(QuantumTraderException):
    """Resource not found exception."""

    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize not found exception."""
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"

        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=404,
            details=details
        )


class AlreadyExistsException(QuantumTraderException):
    """Resource already exists exception."""

    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize already exists exception."""
        message = f"{resource} already exists"
        if resource_id:
            message += f": {resource_id}"

        super().__init__(
            message=message,
            error_code="RESOURCE_ALREADY_EXISTS",
            status_code=409,
            details=details
        )


class ConflictException(QuantumTraderException):
    """Conflict exception."""

    def __init__(
        self,
        message: str = "Conflict occurred",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize conflict exception."""
        super().__init__(
            message=message,
            error_code="CONFLICT",
            status_code=409,
            details=details
        )


# Trading Exceptions


class TradingException(QuantumTraderException):
    """Base trading exception."""

    def __init__(
        self,
        message: str,
        error_code: str = "TRADING_ERROR",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize trading exception."""
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )


class OrderException(TradingException):
    """Order-related exception."""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize order exception."""
        super().__init__(
            message=message,
            error_code="ORDER_ERROR",
            status_code=400,
            details=details
        )
        if order_id:
            self.details["order_id"] = order_id


class OrderRejected(OrderException):
    """Order rejected exception."""

    def __init__(
        self,
        message: str = "Order rejected",
        order_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize order rejected exception."""
        super().__init__(message=message, order_id=order_id, details=details)
        self.error_code = "ORDER_REJECTED"


class InsufficientBalanceException(TradingException):
    """Insufficient balance exception."""

    def __init__(
        self,
        required: str,
        available: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize insufficient balance exception."""
        message = f"Insufficient balance. Required: {required}, Available: {available}"
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_BALANCE",
            status_code=400,
            details=details
        )


class RiskLimitExceededException(TradingException):
    """Risk limit exceeded exception."""

    def __init__(
        self,
        message: str = "Risk limit exceeded",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize risk limit exceeded exception."""
        super().__init__(
            message=message,
            error_code="RISK_LIMIT_EXCEEDED",
            status_code=400,
            details=details
        )


class PositionLimitExceededException(TradingException):
    """Position limit exceeded exception."""

    def __init__(
        self,
        message: str = "Position limit exceeded",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize position limit exceeded exception."""
        super().__init__(
            message=message,
            error_code="POSITION_LIMIT_EXCEEDED",
            status_code=400,
            details=details
        )


# Exchange Exceptions


class ExchangeException(QuantumTraderException):
    """Base exchange exception."""

    def __init__(
        self,
        message: str,
        exchange: Optional[str] = None,
        error_code: str = "EXCHANGE_ERROR",
        status_code: int = 502,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exchange exception."""
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )
        if exchange:
            self.details["exchange"] = exchange


class ExchangeConnectionException(ExchangeException):
    """Exchange connection failed exception."""

    def __init__(
        self,
        exchange: str,
        message: str = "Exchange connection failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exchange connection exception."""
        super().__init__(
            message=message,
            exchange=exchange,
            error_code="EXCHANGE_CONNECTION_FAILED",
            status_code=502,
            details=details
        )


class ExchangeTimeoutException(ExchangeException):
    """Exchange request timeout exception."""

    def __init__(
        self,
        exchange: str,
        message: str = "Exchange request timeout",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exchange timeout exception."""
        super().__init__(
            message=message,
            exchange=exchange,
            error_code="EXCHANGE_TIMEOUT",
            status_code=504,
            details=details
        )


class ExchangeAPIException(ExchangeException):
    """Exchange API error exception."""

    def __init__(
        self,
        exchange: str,
        message: str,
        api_error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize exchange API exception."""
        super().__init__(
            message=message,
            exchange=exchange,
            error_code="EXCHANGE_API_ERROR",
            status_code=502,
            details=details
        )
        if api_error_code:
            self.details["api_error_code"] = api_error_code


# Rate Limiting Exceptions


class RateLimitException(QuantumTraderException):
    """Rate limit exceeded exception."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize rate limit exception."""
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details
        )
        if retry_after:
            self.details["retry_after"] = retry_after


# Data Exceptions


class DataException(QuantumTraderException):
    """Base data exception."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATA_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize data exception."""
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )


class DataNotFoundException(DataException):
    """Data not found exception."""

    def __init__(
        self,
        message: str = "Data not found",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize data not found exception."""
        super().__init__(
            message=message,
            error_code="DATA_NOT_FOUND",
            status_code=404,
            details=details
        )


class DataValidationException(DataException):
    """Data validation failed exception."""

    def __init__(
        self,
        message: str = "Data validation failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize data validation exception."""
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_FAILED",
            status_code=400,
            details=details
        )


# System Exceptions


class SystemException(QuantumTraderException):
    """Base system exception."""

    def __init__(
        self,
        message: str = "System error occurred",
        error_code: str = "SYSTEM_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize system exception."""
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )


class DatabaseException(SystemException):
    """Database error exception."""

    def __init__(
        self,
        message: str = "Database error occurred",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize database exception."""
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details
        )


class CacheException(SystemException):
    """Cache error exception."""

    def __init__(
        self,
        message: str = "Cache error occurred",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize cache exception."""
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=500,
            details=details
        )


class ConfigurationException(SystemException):
    """Configuration error exception."""

    def __init__(
        self,
        message: str = "Configuration error",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize configuration exception."""
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details
        )


class ServiceUnavailableException(SystemException):
    """Service unavailable exception."""

    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize service unavailable exception."""
        msg = message or f"Service unavailable: {service}"
        super().__init__(
            message=msg,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details
        )
