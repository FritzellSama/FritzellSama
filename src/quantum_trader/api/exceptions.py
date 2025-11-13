"""Custom API exceptions."""
from fastapi import HTTPException, status


class TradingException(HTTPException):
    """Base trading exception."""
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class InsufficientFundsException(TradingException):
    """Insufficient funds for trade."""
    def __init__(self):
        super().__init__("Insufficient funds for this trade")


class InvalidOrderException(TradingException):
    """Invalid order parameters."""
    def __init__(self, reason: str):
        super().__init__(f"Invalid order: {reason}")


class RiskLimitExceededException(TradingException):
    """Risk limit exceeded."""
    def __init__(self, limit_type: str):
        super().__init__(f"Risk limit exceeded: {limit_type}")


class MarketDataException(HTTPException):
    """Market data error."""
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


class AuthenticationException(HTTPException):
    """Authentication error."""
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationException(HTTPException):
    """Authorization error."""
    def __init__(self, detail: str = "Not authorized"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
