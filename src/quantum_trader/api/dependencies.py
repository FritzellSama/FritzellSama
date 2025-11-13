"""
FastAPI dependencies for Quantum Trader AI.

Provides dependency injection for services, authentication, and authorization.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable
from functools import wraps
import yaml
import asyncpg
import redis.asyncio as redis
from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from structlog import get_logger

from quantum_trader.api.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ConfigurationException
)

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer()

# Global state (initialized on startup)
_config: Optional[Dict[str, Any]] = None
_db_pool: Optional[asyncpg.Pool] = None
_redis_client: Optional[redis.Redis] = None


async def get_config() -> Dict[str, Any]:
    """Get application configuration.

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationException: If config not loaded
    """
    global _config

    if _config is None:
        # Load configuration files
        try:
            import os

            # Get config path from environment or use default
            config_path = os.getenv("QUANTUM_TRADER_CONFIG", "/app/config")

            # Load main config
            with open(f"{config_path}/bot/config.yaml", "r") as f:
                bot_config = yaml.safe_load(f)

            # Load environment-specific config
            env = os.getenv("ENVIRONMENT", "development")
            with open(f"{config_path}/environments/{env}.yaml", "r") as f:
                env_config = yaml.safe_load(f)

            # Merge configurations
            _config = {**bot_config, **env_config}

            logger.info("Configuration loaded", environment=env)

        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise ConfigurationException(f"Failed to load configuration: {str(e)}")

    return _config


async def get_db_pool(config: Dict[str, Any] = Depends(get_config)) -> asyncpg.Pool:
    """Get database connection pool.

    Args:
        config: Application configuration

    Returns:
        AsyncPG connection pool

    Raises:
        ConfigurationException: If database connection fails
    """
    global _db_pool

    if _db_pool is None:
        try:
            db_config = config.get("database", {})

            _db_pool = await asyncpg.create_pool(
                host=db_config.get("host"),
                port=db_config.get("port", 5432),
                database=db_config.get("database"),
                user=db_config.get("user"),
                password=db_config.get("password"),
                min_size=db_config.get("pool_min_size", 10),
                max_size=db_config.get("pool_max_size", 50),
                command_timeout=db_config.get("command_timeout", 60),
            )

            logger.info("Database pool created")

        except Exception as e:
            logger.error("Failed to create database pool", error=str(e))
            raise ConfigurationException(f"Database connection failed: {str(e)}")

    return _db_pool


async def get_redis_client(config: Dict[str, Any] = Depends(get_config)) -> redis.Redis:
    """Get Redis client.

    Args:
        config: Application configuration

    Returns:
        Redis client

    Raises:
        ConfigurationException: If Redis connection fails
    """
    global _redis_client

    if _redis_client is None:
        try:
            redis_config = config.get("redis", {})

            _redis_client = redis.Redis(
                host=redis_config.get("host"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password"),
                db=redis_config.get("db", 0),
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=redis_config.get("socket_timeout", 5),
                socket_connect_timeout=redis_config.get("socket_connect_timeout", 5),
            )

            # Test connection
            await _redis_client.ping()

            logger.info("Redis client created")

        except Exception as e:
            logger.error("Failed to create Redis client", error=str(e))
            raise ConfigurationException(f"Redis connection failed: {str(e)}")

    return _redis_client


async def get_auth_service(
    config: Dict[str, Any] = Depends(get_config),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get authentication service.

    Args:
        config: Configuration
        db_pool: Database pool
        redis_client: Redis client

    Returns:
        AuthService instance
    """
    from quantum_trader.api.services.auth_service import AuthService
    return AuthService(config, db_pool, redis_client)


async def get_data_service(
    config: Dict[str, Any] = Depends(get_config),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get data service.

    Args:
        config: Configuration
        db_pool: Database pool
        redis_client: Redis client

    Returns:
        DataService instance
    """
    from quantum_trader.api.services.data_service import DataService
    return DataService(config, db_pool, redis_client)


async def get_analytics_service(
    config: Dict[str, Any] = Depends(get_config),
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get analytics service.

    Args:
        config: Configuration
        db_pool: Database pool
        redis_client: Redis client

    Returns:
        AnalyticsService instance
    """
    from quantum_trader.api.services.analytics_service import AnalyticsService
    return AnalyticsService(config, db_pool, redis_client)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials
        auth_service: Authentication service

    Returns:
        User information dictionary

    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials

        # Verify token
        payload = auth_service._verify_token(token)

        user_id = payload.get("sub")
        if not user_id:
            raise AuthenticationException("Invalid token payload")

        # Get user from database
        async with auth_service.db_pool.acquire() as conn:
            user = await conn.fetchrow(
                """
                SELECT user_id, email, username, full_name, role, is_active
                FROM users
                WHERE user_id = $1
                """,
                user_id
            )

            if not user:
                raise AuthenticationException("User not found")

            if not user["is_active"]:
                raise AuthenticationException("User account is inactive")

        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "username": user["username"],
            "full_name": user["full_name"],
            "role": user["role"],
            "is_active": user["is_active"]
        }

    except AuthenticationException as e:
        logger.warning("Authentication failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_optional_user(
    authorization: Optional[str] = Header(None),
    auth_service = Depends(get_auth_service)
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise.

    Args:
        authorization: Authorization header
        auth_service: Authentication service

    Returns:
        User information or None
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    try:
        token = authorization.replace("Bearer ", "")
        payload = auth_service._verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            return None

        async with auth_service.db_pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT user_id, email, username, role, is_active FROM users WHERE user_id = $1",
                user_id
            )

            if not user or not user["is_active"]:
                return None

            return dict(user)

    except Exception:
        return None


def require_permissions(required_permissions: List[str]) -> Callable:
    """Dependency to require specific permissions.

    Args:
        required_permissions: List of required permissions

    Returns:
        Dependency function

    Example:
        @router.post("/admin", dependencies=[Depends(require_permissions(["admin"]))])
    """
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> None:
        """Check if user has required permissions.

        Args:
            current_user: Current authenticated user

        Raises:
            HTTPException: If user lacks permissions
        """
        user_role = current_user.get("role", "")

        # Admin has all permissions
        if user_role == "ADMIN":
            return

        # Check specific permissions based on role
        role_permissions = {
            "TRADER": ["trade", "read_market_data", "manage_orders", "view_positions"],
            "VIEWER": ["read_market_data", "view_positions"],
            "API_USER": ["trade", "read_market_data"]
        }

        user_permissions = role_permissions.get(user_role, [])

        for required_permission in required_permissions:
            if required_permission not in user_permissions:
                logger.warning(
                    "Permission denied",
                    user_id=current_user.get("user_id"),
                    required_permission=required_permission
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {required_permission}"
                )

    return permission_checker


def require_role(required_role: str) -> Callable:
    """Dependency to require specific role.

    Args:
        required_role: Required user role

    Returns:
        Dependency function

    Example:
        @router.post("/admin", dependencies=[Depends(require_role("ADMIN"))])
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> None:
        """Check if user has required role.

        Args:
            current_user: Current authenticated user

        Raises:
            HTTPException: If user lacks role
        """
        user_role = current_user.get("role", "")

        if user_role != required_role:
            logger.warning(
                "Role check failed",
                user_id=current_user.get("user_id"),
                user_role=user_role,
                required_role=required_role
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}"
            )

    return role_checker


async def get_rate_limit_key(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> str:
    """Get rate limit key for request.

    Args:
        request: FastAPI request
        current_user: Current user (optional)

    Returns:
        Rate limit key
    """
    if current_user:
        return f"rate_limit:user:{current_user['user_id']}"
    else:
        # Use IP address for unauthenticated requests
        client_ip = request.client.host if request.client else "unknown"
        return f"rate_limit:ip:{client_ip}"


async def check_rate_limit(
    rate_limit_key: str = Depends(get_rate_limit_key),
    redis_client: redis.Redis = Depends(get_redis_client),
    config: Dict[str, Any] = Depends(get_config)
) -> None:
    """Check rate limit for request.

    Args:
        rate_limit_key: Rate limit key
        redis_client: Redis client
        config: Configuration

    Raises:
        HTTPException: If rate limit exceeded
    """
    try:
        rate_limit_config = config.get("api", {}).get("rate_limit", {})
        max_requests = rate_limit_config.get("max_requests_per_minute", 60)
        window_seconds = rate_limit_config.get("window_seconds", 60)

        # Increment counter
        current = await redis_client.incr(rate_limit_key)

        # Set expiry on first request
        if current == 1:
            await redis_client.expire(rate_limit_key, window_seconds)

        # Check limit
        if current > max_requests:
            logger.warning("Rate limit exceeded", key=rate_limit_key, count=current)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rate limit check failed", error=str(e))
        # Don't block on rate limit check failure
