"""
Authentication service for Quantum Trader AI.

Handles user authentication, token management, and API keys.
"""

import asyncio
import secrets
import hashlib
from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from structlog import get_logger

from quantum_trader.api.exceptions import (
    AuthenticationException,
    ValidationException
)

logger = get_logger(__name__)


class AuthService:
    """Production-ready authentication service.

    Provides secure authentication and authorization:
    - User registration and login
    - JWT token generation and validation
    - API key management
    - Password reset flows
    - Session management

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        redis_client: Redis client for session storage
        pwd_context: Password hashing context
    """

    def __init__(
        self,
        config: Dict[str, Any],
        db_pool: Any,
        redis_client: Any
    ) -> None:
        """Initialize authentication service.

        Args:
            config: Configuration from config files
            db_pool: Database connection pool
            redis_client: Redis client

        Raises:
            ValueError: If config validation fails
        """
        self.config = config
        self.db_pool = db_pool
        self.redis_client = redis_client

        # Load from config
        auth_config = config.get("auth", {})
        self.secret_key = config.get("security", {}).get("jwt_secret_key")
        self.algorithm = auth_config.get("jwt_algorithm", "HS256")
        self.access_token_expire_minutes = auth_config.get("access_token_expire_minutes", 60)
        self.refresh_token_expire_days = auth_config.get("refresh_token_expire_days", 30)
        self.password_reset_expire_minutes = auth_config.get("password_reset_expire_minutes", 60)

        # Password hashing
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto"
        )

        self._validate_config()
        logger.info("Auth service initialized")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.secret_key:
            raise ValueError("JWT secret key must be configured")
        if self.access_token_expire_minutes <= 0:
            raise ValueError("access_token_expire_minutes must be positive")
        if self.refresh_token_expire_days <= 0:
            raise ValueError("refresh_token_expire_days must be positive")

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    def _generate_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Generate JWT token.

        Args:
            data: Data to encode in token
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )

        return encoded_jwt

    def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token.

        Args:
            token: JWT token to verify

        Returns:
            Decoded token data

        Raises:
            AuthenticationException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.JWTError as e:
            raise AuthenticationException(f"Invalid token: {str(e)}")

    async def register_user(
        self,
        email: str,
        password: str,
        username: Optional[str] = None,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register new user.

        Args:
            email: User email
            password: User password
            username: Username (optional)
            full_name: Full name (optional)

        Returns:
            User information dictionary

        Raises:
            ValidationException: If user already exists
            Exception: If registration fails
        """
        try:
            # Check if user exists
            async with self.db_pool.acquire() as conn:
                existing = await conn.fetchrow(
                    "SELECT user_id FROM users WHERE email = $1",
                    email
                )

                if existing:
                    raise ValidationException("User with this email already exists")

                # Hash password
                hashed_password = self._hash_password(password)

                # Generate user ID
                user_id = f"usr_{secrets.token_hex(16)}"

                # Insert user
                now = datetime.utcnow()
                user = await conn.fetchrow(
                    """
                    INSERT INTO users (
                        user_id, email, password_hash, username, full_name,
                        role, is_active, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING user_id, email, username, full_name, role, is_active, created_at
                    """,
                    user_id, email, hashed_password, username, full_name,
                    "TRADER", True, now
                )

            logger.info("User registered", user_id=user_id, email=email)

            return {
                "user_id": user["user_id"],
                "email": user["email"],
                "username": user["username"],
                "full_name": user["full_name"],
                "role": user["role"],
                "is_active": user["is_active"],
                "created_at": user["created_at"],
                "last_login": None
            }

        except ValidationException:
            raise
        except Exception as e:
            logger.error("User registration failed", error=str(e))
            raise

    async def authenticate_user(
        self,
        email: str,
        password: str,
        client_ip: str = "unknown",
        user_agent: str = "unknown"
    ) -> Dict[str, Any]:
        """Authenticate user and generate tokens.

        Args:
            email: User email
            password: User password
            client_ip: Client IP address
            user_agent: User agent string

        Returns:
            Dictionary with access and refresh tokens

        Raises:
            AuthenticationException: If authentication fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                user = await conn.fetchrow(
                    """
                    SELECT user_id, email, password_hash, role, is_active
                    FROM users
                    WHERE email = $1
                    """,
                    email
                )

                if not user:
                    raise AuthenticationException("Invalid credentials")

                if not user["is_active"]:
                    raise AuthenticationException("Account is inactive")

                # Verify password
                if not self._verify_password(password, user["password_hash"]):
                    # Log failed attempt
                    await self._log_auth_attempt(
                        user["user_id"],
                        False,
                        client_ip,
                        user_agent
                    )
                    raise AuthenticationException("Invalid credentials")

                # Update last login
                await conn.execute(
                    "UPDATE users SET last_login = $1 WHERE user_id = $2",
                    datetime.utcnow(),
                    user["user_id"]
                )

                # Log successful attempt
                await self._log_auth_attempt(
                    user["user_id"],
                    True,
                    client_ip,
                    user_agent
                )

            # Generate tokens
            access_token = self._generate_token(
                {"sub": user["user_id"], "email": user["email"], "role": user["role"]},
                expires_delta=timedelta(minutes=self.access_token_expire_minutes)
            )

            refresh_token = self._generate_token(
                {"sub": user["user_id"], "type": "refresh"},
                expires_delta=timedelta(days=self.refresh_token_expire_days)
            )

            # Store refresh token in Redis
            await self.redis_client.setex(
                f"refresh_token:{user['user_id']}:{hashlib.sha256(refresh_token.encode()).hexdigest()}",
                self.refresh_token_expire_days * 86400,
                "1"
            )

            logger.info("User authenticated", user_id=user["user_id"])

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }

        except AuthenticationException:
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise AuthenticationException("Authentication failed")

    async def _log_auth_attempt(
        self,
        user_id: str,
        success: bool,
        client_ip: str,
        user_agent: str
    ) -> None:
        """Log authentication attempt.

        Args:
            user_id: User ID
            success: Whether attempt was successful
            client_ip: Client IP address
            user_agent: User agent string
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO auth_logs (
                        user_id, success, client_ip, user_agent, timestamp
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    user_id, success, client_ip, user_agent, datetime.utcnow()
                )
        except Exception as e:
            logger.warning("Failed to log auth attempt", error=str(e))

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New access and refresh tokens

        Raises:
            AuthenticationException: If refresh fails
        """
        try:
            # Verify refresh token
            payload = self._verify_token(refresh_token)

            if payload.get("type") != "refresh":
                raise AuthenticationException("Invalid token type")

            user_id = payload.get("sub")

            # Check if refresh token is valid in Redis
            token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            is_valid = await self.redis_client.get(f"refresh_token:{user_id}:{token_hash}")

            if not is_valid:
                raise AuthenticationException("Invalid or revoked refresh token")

            # Get user info
            async with self.db_pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT user_id, email, role, is_active FROM users WHERE user_id = $1",
                    user_id
                )

                if not user or not user["is_active"]:
                    raise AuthenticationException("User not found or inactive")

            # Generate new tokens
            access_token = self._generate_token(
                {"sub": user["user_id"], "email": user["email"], "role": user["role"]},
                expires_delta=timedelta(minutes=self.access_token_expire_minutes)
            )

            new_refresh_token = self._generate_token(
                {"sub": user["user_id"], "type": "refresh"},
                expires_delta=timedelta(days=self.refresh_token_expire_days)
            )

            # Invalidate old refresh token
            await self.redis_client.delete(f"refresh_token:{user_id}:{token_hash}")

            # Store new refresh token
            new_token_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()
            await self.redis_client.setex(
                f"refresh_token:{user_id}:{new_token_hash}",
                self.refresh_token_expire_days * 86400,
                "1"
            )

            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "Bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }

        except AuthenticationException:
            raise
        except Exception as e:
            logger.error("Token refresh failed", error=str(e))
            raise AuthenticationException("Token refresh failed")

    async def logout_user(self, user_id: str) -> None:
        """Logout user by invalidating all refresh tokens.

        Args:
            user_id: User ID
        """
        try:
            # Delete all refresh tokens for user
            pattern = f"refresh_token:{user_id}:*"
            keys = await self.redis_client.keys(pattern)

            if keys:
                await self.redis_client.delete(*keys)

            logger.info("User logged out", user_id=user_id)

        except Exception as e:
            logger.error("Logout failed", error=str(e))
            raise

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str
    ) -> None:
        """Change user password.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Raises:
            AuthenticationException: If current password is incorrect
        """
        try:
            async with self.db_pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT password_hash FROM users WHERE user_id = $1",
                    user_id
                )

                if not user:
                    raise AuthenticationException("User not found")

                # Verify current password
                if not self._verify_password(current_password, user["password_hash"]):
                    raise AuthenticationException("Current password is incorrect")

                # Hash new password
                new_hash = self._hash_password(new_password)

                # Update password
                await conn.execute(
                    "UPDATE users SET password_hash = $1 WHERE user_id = $2",
                    new_hash,
                    user_id
                )

            # Invalidate all refresh tokens
            await self.logout_user(user_id)

            logger.info("Password changed", user_id=user_id)

        except AuthenticationException:
            raise
        except Exception as e:
            logger.error("Password change failed", error=str(e))
            raise

    async def request_password_reset(self, email: str) -> None:
        """Request password reset.

        Args:
            email: User email
        """
        try:
            async with self.db_pool.acquire() as conn:
                user = await conn.fetchrow(
                    "SELECT user_id FROM users WHERE email = $1",
                    email
                )

                if not user:
                    # Don't reveal if email exists
                    logger.info("Password reset requested for non-existent email")
                    return

                # Generate reset token
                reset_token = secrets.token_urlsafe(32)
                token_hash = hashlib.sha256(reset_token.encode()).hexdigest()

                # Store token in Redis
                await self.redis_client.setex(
                    f"password_reset:{token_hash}",
                    self.password_reset_expire_minutes * 60,
                    user["user_id"]
                )

                # TODO: Send email with reset_token
                logger.info("Password reset token generated", user_id=user["user_id"])

        except Exception as e:
            logger.error("Password reset request failed", error=str(e))

    async def confirm_password_reset(self, token: str, new_password: str) -> None:
        """Confirm password reset.

        Args:
            token: Reset token
            new_password: New password

        Raises:
            AuthenticationException: If token is invalid
        """
        try:
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            user_id = await self.redis_client.get(f"password_reset:{token_hash}")

            if not user_id:
                raise AuthenticationException("Invalid or expired reset token")

            # Hash new password
            new_hash = self._hash_password(new_password)

            # Update password
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE users SET password_hash = $1 WHERE user_id = $2",
                    new_hash,
                    user_id
                )

            # Delete reset token
            await self.redis_client.delete(f"password_reset:{token_hash}")

            # Invalidate all refresh tokens
            await self.logout_user(user_id)

            logger.info("Password reset successful", user_id=user_id)

        except AuthenticationException:
            raise
        except Exception as e:
            logger.error("Password reset confirmation failed", error=str(e))
            raise

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create API key.

        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions
            expires_in_days: Expiration in days

        Returns:
            API key information

        Raises:
            ValidationException: If validation fails
        """
        try:
            # Generate API key
            api_key = f"qtai_{secrets.token_urlsafe(32)}"
            api_key_id = f"key_{secrets.token_hex(16)}"

            # Hash API key for storage
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO api_keys (
                        api_key_id, user_id, api_key_hash, name, permissions,
                        created_at, expires_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    api_key_id, user_id, api_key_hash, name, permissions,
                    datetime.utcnow(), expires_at
                )

            logger.info("API key created", api_key_id=api_key_id, user_id=user_id)

            return {
                "api_key_id": api_key_id,
                "api_key": api_key,
                "name": name,
                "permissions": permissions,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            }

        except Exception as e:
            logger.error("API key creation failed", error=str(e))
            raise

    async def revoke_api_key(self, user_id: str, api_key_id: str) -> None:
        """Revoke API key.

        Args:
            user_id: User ID
            api_key_id: API key ID
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    DELETE FROM api_keys
                    WHERE api_key_id = $1 AND user_id = $2
                    """,
                    api_key_id, user_id
                )

            logger.info("API key revoked", api_key_id=api_key_id)

        except Exception as e:
            logger.error("API key revocation failed", error=str(e))
            raise
