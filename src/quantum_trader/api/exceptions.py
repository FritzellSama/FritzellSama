"""
Custom exceptions for API layer.

Defines exception hierarchy for API-specific errors including validation,
authentication, rate limiting, and external service errors.
"""

from decimal import Decimal
from typing import Dict, Optional, Any
from datetime import datetime

from fastapi import status
from structlog import get_logger

logger = get_logger(__name__)


class APIException(Exception):
    """
    Base exception for all API errors.

    Attributes:
        message: Error message
        status_code: HTTP status code
        error_code: Application-specific error code
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize API exception.

        Args:
            message: Human-readable error message
            status_code: HTTP status code
            error_code: Application error code
            details: Additional error context
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow()

        super().__init__(self.message)

        logger.error(
            "api_exception",
            exception=self.error_code,
            message=message,
            status_code=status_code,
            details=details
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format.

        Returns:
            Dictionary representation of exception
        """
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# Authentication and Authorization Exceptions

class AuthenticationError(APIException):
    """Authentication failed - invalid or missing credentials."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(APIException):
    """Authorization failed - insufficient permissions."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid or expired API key."""

    def __init__(
        self,
        message: str = "Invalid or expired API key",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "INVALID_API_KEY"


class InvalidTokenError(AuthenticationError):
    """Invalid or expired JWT token."""

    def __init__(
        self,
        message: str = "Invalid or expired token",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "INVALID_TOKEN"


# Validation Exceptions

class ValidationError(APIException):
    """Request validation failed."""

    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            details: Additional context
        """
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = str(value)

        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=error_details
        )


class InvalidParameterError(ValidationError):
    """Invalid parameter value."""

    def __init__(
        self,
        parameter: str,
        value: Any,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize invalid parameter error.

        Args:
            parameter: Parameter name
            value: Invalid value
            reason: Why it's invalid
            details: Additional context
        """
        super().__init__(
            message=f"Invalid parameter '{parameter}': {reason}",
            field=parameter,
            value=value,
            details=details
        )
        self.error_code = "INVALID_PARAMETER"


class MissingParameterError(ValidationError):
    """Required parameter missing."""

    def __init__(
        self,
        parameter: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize missing parameter error.

        Args:
            parameter: Missing parameter name
            details: Additional context
        """
        super().__init__(
            message=f"Required parameter missing: {parameter}",
            field=parameter,
            details=details
        )
        self.error_code = "MISSING_PARAMETER"


class InvalidSymbolError(ValidationError):
    """Invalid trading symbol."""

    def __init__(
        self,
        symbol: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=f"Invalid symbol: {symbol}",
            field="symbol",
            value=symbol,
            details=details
        )
        self.error_code = "INVALID_SYMBOL"


class InvalidExchangeError(ValidationError):
    """Invalid exchange name."""

    def __init__(
        self,
        exchange: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=f"Invalid exchange: {exchange}",
            field="exchange",
            value=exchange,
            details=details
        )
        self.error_code = "INVALID_EXCHANGE"


# Resource Exceptions

class ResourceNotFoundError(APIException):
    """Requested resource not found."""

    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize resource not found error.

        Args:
            resource: Resource type
            resource_id: Resource identifier
            details: Additional context
        """
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"

        error_details = details or {}
        error_details["resource"] = resource
        if resource_id:
            error_details["resource_id"] = resource_id

        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details=error_details
        )


class ResourceAlreadyExistsError(APIException):
    """Resource already exists."""

    def __init__(
        self,
        resource: str,
        resource_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        error_details["resource"] = resource
        error_details["resource_id"] = resource_id

        super().__init__(
            message=f"{resource} already exists: {resource_id}",
            status_code=status.HTTP_409_CONFLICT,
            error_code="RESOURCE_ALREADY_EXISTS",
            details=error_details
        )


# Rate Limiting Exceptions

class RateLimitExceededError(APIException):
    """Rate limit exceeded."""

    def __init__(
        self,
        limit: int,
        window: int,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize rate limit error.

        Args:
            limit: Request limit
            window: Time window in seconds
            retry_after: Seconds until retry allowed
            details: Additional context
        """
        error_details = details or {}
        error_details["limit"] = limit
        error_details["window"] = window
        if retry_after:
            error_details["retry_after"] = retry_after

        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}s",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=error_details
        )


# External Service Exceptions

class ExternalServiceError(APIException):
    """External service error."""

    def __init__(
        self,
        service: str,
        message: str = "External service error",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize external service error.

        Args:
            service: Service name
            message: Error message
            details: Additional context
        """
        error_details = details or {}
        error_details["service"] = service

        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code="EXTERNAL_SERVICE_ERROR",
            details=error_details
        )


class ExchangeError(ExternalServiceError):
    """Exchange API error."""

    def __init__(
        self,
        exchange: str,
        message: str = "Exchange API error",
        original_error: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize exchange error.

        Args:
            exchange: Exchange name
            message: Error message
            original_error: Original error from exchange
            details: Additional context
        """
        error_details = details or {}
        error_details["exchange"] = exchange
        if original_error:
            error_details["original_error"] = original_error

        super().__init__(
            service=exchange,
            message=message,
            details=error_details
        )
        self.error_code = "EXCHANGE_ERROR"


class ExchangeTimeoutError(ExchangeError):
    """Exchange request timeout."""

    def __init__(
        self,
        exchange: str,
        timeout: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        error_details["timeout"] = timeout

        super().__init__(
            exchange=exchange,
            message=f"Exchange request timeout: {timeout}s",
            details=error_details
        )
        self.error_code = "EXCHANGE_TIMEOUT"


class DatabaseError(ExternalServiceError):
    """Database error."""

    def __init__(
        self,
        message: str = "Database error",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        if operation:
            error_details["operation"] = operation

        super().__init__(
            service="database",
            message=message,
            details=error_details
        )
        self.error_code = "DATABASE_ERROR"


class CacheError(ExternalServiceError):
    """Cache service error."""

    def __init__(
        self,
        message: str = "Cache error",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            service="cache",
            message=message,
            details=details
        )
        self.error_code = "CACHE_ERROR"


# Trading-Specific Exceptions

class TradingError(APIException):
    """Base trading operation error."""

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        if symbol:
            error_details["symbol"] = symbol
        if exchange:
            error_details["exchange"] = exchange

        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="TRADING_ERROR",
            details=error_details
        )


class OrderRejectedError(TradingError):
    """Order rejected."""

    def __init__(
        self,
        reason: str,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=f"Order rejected: {reason}",
            symbol=symbol,
            exchange=exchange,
            details=details
        )
        self.error_code = "ORDER_REJECTED"


class InsufficientBalanceError(TradingError):
    """Insufficient balance for order."""

    def __init__(
        self,
        required: Decimal,
        available: Decimal,
        currency: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        error_details["required"] = str(required)
        error_details["available"] = str(available)
        error_details["currency"] = currency

        super().__init__(
            message=f"Insufficient {currency}: required {required}, available {available}",
            details=error_details
        )
        self.error_code = "INSUFFICIENT_BALANCE"


class RiskLimitExceededError(TradingError):
    """Risk limit exceeded."""

    def __init__(
        self,
        limit_type: str,
        limit_value: Decimal,
        current_value: Decimal,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        error_details["limit_type"] = limit_type
        error_details["limit_value"] = str(limit_value)
        error_details["current_value"] = str(current_value)

        super().__init__(
            message=f"Risk limit exceeded: {limit_type}",
            details=error_details
        )
        self.error_code = "RISK_LIMIT_EXCEEDED"


class InvalidOrderError(TradingError):
    """Invalid order parameters."""

    def __init__(
        self,
        reason: str,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=f"Invalid order: {reason}",
            symbol=symbol,
            details=details
        )
        self.error_code = "INVALID_ORDER"


# System Exceptions

class ServiceUnavailableError(APIException):
    """Service temporarily unavailable."""

    def __init__(
        self,
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        if service:
            error_details["service"] = service
        if retry_after:
            error_details["retry_after"] = retry_after

        message = "Service unavailable"
        if service:
            message = f"{service} unavailable"

        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            details=error_details
        )


class MaintenanceModeError(APIException):
    """System in maintenance mode."""

    def __init__(
        self,
        message: str = "System under maintenance",
        estimated_end: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        error_details = details or {}
        if estimated_end:
            error_details["estimated_end"] = estimated_end.isoformat()

        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MAINTENANCE_MODE",
            details=error_details
        )


class InternalServerError(APIException):
    """Internal server error."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            details=details
        )
