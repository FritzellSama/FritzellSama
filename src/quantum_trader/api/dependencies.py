"""
FastAPI dependency injection providers.

Provides dependency injection for API endpoints including authentication,
rate limiting, database connections, and service access.
"""

from decimal import Decimal
from typing import Dict, Optional, Any, AsyncGenerator
from datetime import datetime
import os

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

logger = structlog.get_logger(__name__)

# Security scheme for JWT bearer tokens
security = HTTPBearer()


class DependencyContainer:
    """
    Centralized dependency container for API services.

    Manages singleton instances of services and provides them
    as FastAPI dependencies. Ensures proper initialization and
    shutdown of resources.

    Attributes:
        config: Application configuration
        _instances: Service instance cache
        _initialized: Initialization state
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize dependency container.

        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self._instances: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all services."""
        if self._initialized:
            logger.warning("dependency_container_already_initialized")
            return

        logger.info("initializing_dependency_container")

        # Initialize services as needed
        # This would typically create database pools, Redis connections, etc.

        self._initialized = True
        logger.info("dependency_container_initialized")

    async def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        if not self._initialized:
            return

        logger.info("shutting_down_dependency_container")

        # Shutdown services in reverse order
        for service_name, service in reversed(list(self._instances.items())):
            try:
                if hasattr(service, "stop"):
                    await service.stop()
                logger.debug("service_stopped", service=service_name)
            except Exception as e:
                logger.error(
                    "service_shutdown_error",
                    service=service_name,
                    error=str(e)
                )

        self._instances.clear()
        self._initialized = False
        logger.info("dependency_container_shutdown_complete")

    def get_service(self, service_name: str) -> Any:
        """
        Get service instance.

        Args:
            service_name: Name of service to retrieve

        Returns:
            Service instance

        Raises:
            RuntimeError: If container not initialized
            KeyError: If service not found
        """
        if not self._initialized:
            raise RuntimeError("Dependency container not initialized")

        if service_name not in self._instances:
            raise KeyError(f"Service not found: {service_name}")

        return self._instances[service_name]

    def register_service(self, service_name: str, service: Any) -> None:
        """
        Register service instance.

        Args:
            service_name: Name to register service under
            service: Service instance
        """
        self._instances[service_name] = service
        logger.debug("service_registered", service=service_name)


# Global container instance (initialized at startup)
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """
    Get global dependency container.

    Returns:
        DependencyContainer instance

    Raises:
        RuntimeError: If container not initialized
    """
    if _container is None:
        raise RuntimeError("Dependency container not initialized")
    return _container


def set_container(container: DependencyContainer) -> None:
    """
    Set global dependency container.

    Args:
        container: Container instance to set
    """
    global _container
    _container = container


# Authentication dependencies

async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> str:
    """
    Verify API key from header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid or missing

    Example:
        >>> @router.get("/protected")
        >>> async def protected_endpoint(api_key: str = Depends(verify_api_key)):
        >>>     return {"status": "authorized"}
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Get valid API keys from environment
    valid_keys = os.getenv("VALID_API_KEYS", "").split(",")

    if not valid_keys or x_api_key not in valid_keys:
        logger.warning("invalid_api_key_attempt", key_prefix=x_api_key[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


async def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Verify JWT bearer token.

    Args:
        credentials: Bearer token credentials

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired

    Example:
        >>> @router.get("/protected")
        >>> async def protected_endpoint(token: Dict = Depends(verify_jwt_token)):
        >>>     user_id = token["user_id"]
        >>>     return {"user_id": user_id}
    """
    try:
        token = credentials.credentials

        # In production, decode and verify JWT
        # This is a placeholder for actual JWT verification
        # import jwt
        # secret = os.getenv("JWT_SECRET")
        # payload = jwt.decode(token, secret, algorithms=["HS256"])

        # For now, return mock payload
        payload = {
            "user_id": "mock_user",
            "exp": datetime.utcnow().timestamp() + 3600
        }

        return payload

    except Exception as e:
        logger.warning("jwt_verification_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token_payload: Dict[str, Any] = Depends(verify_jwt_token)
) -> str:
    """
    Get current authenticated user ID.

    Args:
        token_payload: Decoded JWT payload

    Returns:
        User ID string

    Example:
        >>> @router.get("/me")
        >>> async def get_user_info(user_id: str = Depends(get_current_user)):
        >>>     return {"user_id": user_id}
    """
    return token_payload.get("user_id", "unknown")


# Rate limiting dependency

class RateLimiter:
    """
    Rate limiting dependency.

    Tracks request counts and enforces rate limits per user/API key.

    Attributes:
        config: Rate limit configuration
        _requests: Request count tracking
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Configuration with rate limits
        """
        self.config = config
        self._requests: Dict[str, list] = {}

    async def check_rate_limit(
        self,
        identifier: str,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None
    ) -> None:
        """
        Check if request is within rate limit.

        Args:
            identifier: User/API key identifier
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds

        Raises:
            HTTPException: If rate limit exceeded
        """
        max_requests = max_requests or self.config.get("max_requests_per_minute", 60)
        window_seconds = window_seconds or self.config.get("rate_limit_window", 60)

        now = datetime.utcnow()

        if identifier not in self._requests:
            self._requests[identifier] = []

        # Remove old requests outside window
        cutoff = now.timestamp() - window_seconds
        self._requests[identifier] = [
            ts for ts in self._requests[identifier]
            if ts > cutoff
        ]

        # Check limit
        if len(self._requests[identifier]) >= max_requests:
            logger.warning(
                "rate_limit_exceeded",
                identifier=identifier,
                count=len(self._requests[identifier])
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
            )

        # Add current request
        self._requests[identifier].append(now.timestamp())


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized")
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = limiter


async def check_rate_limit(
    api_key: str = Depends(verify_api_key)
) -> None:
    """
    Dependency to check rate limit.

    Args:
        api_key: API key identifier

    Raises:
        HTTPException: If rate limit exceeded

    Example:
        >>> @router.get("/limited", dependencies=[Depends(check_rate_limit)])
        >>> async def limited_endpoint():
        >>>     return {"status": "ok"}
    """
    limiter = get_rate_limiter()
    await limiter.check_rate_limit(api_key)


# Configuration dependencies

async def get_config() -> Dict[str, Any]:
    """
    Get application configuration.

    Returns:
        Configuration dictionary

    Example:
        >>> @router.get("/config")
        >>> async def get_settings(config: Dict = Depends(get_config)):
        >>>     return {"env": config.get("environment")}
    """
    container = get_container()
    return container.config


# Pagination dependency

class PaginationParams:
    """
    Pagination parameters.

    Attributes:
        skip: Number of records to skip
        limit: Maximum number of records to return
    """

    def __init__(
        self,
        skip: int = 0,
        limit: int = 100
    ) -> None:
        """
        Initialize pagination parameters.

        Args:
            skip: Number of records to skip (default: 0)
            limit: Maximum records to return (default: 100)

        Raises:
            HTTPException: If parameters are invalid
        """
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="skip must be >= 0"
            )

        if limit < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be >= 1"
            )

        if limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be <= 1000"
            )

        self.skip = skip
        self.limit = limit


def get_pagination(
    skip: int = 0,
    limit: int = 100
) -> PaginationParams:
    """
    Get pagination parameters dependency.

    Args:
        skip: Number of records to skip
        limit: Maximum records to return

    Returns:
        PaginationParams instance

    Example:
        >>> @router.get("/items")
        >>> async def get_items(page: PaginationParams = Depends(get_pagination)):
        >>>     return {"skip": page.skip, "limit": page.limit}
    """
    return PaginationParams(skip=skip, limit=limit)


# Database session dependency

async def get_db_session() -> AsyncGenerator:
    """
    Get database session.

    Yields:
        Database session

    Example:
        >>> @router.get("/data")
        >>> async def get_data(db = Depends(get_db_session)):
        >>>     result = await db.execute(query)
        >>>     return result
    """
    # In production, this would create a database session
    # from a connection pool managed by the container

    # Placeholder implementation
    session = None

    try:
        # session = SessionLocal()
        yield session
    finally:
        if session:
            # await session.close()
            pass


# Cache dependency

async def get_cache() -> Any:
    """
    Get cache instance.

    Returns:
        Cache instance (Redis, etc.)

    Example:
        >>> @router.get("/cached")
        >>> async def get_cached_data(cache = Depends(get_cache)):
        >>>     value = await cache.get("key")
        >>>     return {"value": value}
    """
    # In production, return Redis or other cache instance
    # managed by the container
    container = get_container()

    try:
        return container.get_service("cache")
    except KeyError:
        # Return None if cache not configured
        return None


# Service dependencies

async def get_data_service():
    """
    Get data service instance.

    Returns:
        DataService instance
    """
    container = get_container()
    return container.get_service("data_service")


async def get_trading_engine():
    """
    Get trading engine instance.

    Returns:
        TradingEngine instance
    """
    container = get_container()
    return container.get_service("trading_engine")


async def get_risk_manager():
    """
    Get risk manager instance.

    Returns:
        RiskManager instance
    """
    container = get_container()
    return container.get_service("risk_manager")


async def get_portfolio():
    """
    Get portfolio instance.

    Returns:
        Portfolio instance
    """
    container = get_container()
    return container.get_service("portfolio")
