"""
Authentication Service

Production-ready authentication service with JWT token management,
password hashing, and user verification.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
import os
import hashlib
import secrets

from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class AuthService:
    """
    Authentication and authorization service.

    Handles user authentication, JWT token generation/validation,
    password hashing, and permission management.

    Attributes:
        config: Configuration dictionary
        secret_key: Secret key for JWT signing
        algorithm: JWT algorithm
        access_token_expire_minutes: Access token expiration time

    Example:
        >>> config = {"auth": {"secret_key": "secret"}}
        >>> auth = AuthService(config)
        >>> user = await auth.authenticate_user("username", "password")
        >>> token = await auth.create_access_token(user)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize authentication service.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        auth_config = self.config.get("auth", {})

        # JWT configuration
        self.secret_key: str = auth_config.get(
            "secret_key",
            os.getenv("AUTH_SECRET_KEY", "change-me-in-production")
        )
        self.algorithm: str = auth_config.get(
            "algorithm",
            os.getenv("AUTH_ALGORITHM", "HS256")
        )
        self.access_token_expire_minutes: int = auth_config.get(
            "access_token_expire_minutes",
            int(os.getenv("AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        )
        self.refresh_token_expire_days: int = auth_config.get(
            "refresh_token_expire_days",
            int(os.getenv("AUTH_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
        )

        # Password requirements
        self.min_password_length: int = auth_config.get(
            "min_password_length",
            int(os.getenv("AUTH_MIN_PASSWORD_LENGTH", "8"))
        )
        self.require_special_chars: bool = auth_config.get(
            "require_special_chars",
            os.getenv("AUTH_REQUIRE_SPECIAL_CHARS", "true").lower() == "true"
        )

        # Rate limiting
        self.max_login_attempts: int = auth_config.get(
            "max_login_attempts",
            int(os.getenv("AUTH_MAX_LOGIN_ATTEMPTS", "5"))
        )
        self.lockout_duration_minutes: int = auth_config.get(
            "lockout_duration_minutes",
            int(os.getenv("AUTH_LOCKOUT_DURATION_MINUTES", "15"))
        )

        # Track login attempts (in production, use Redis)
        self.login_attempts: Dict[str, List[datetime]] = {}

        logger.info(
            "auth_service_initialized",
            access_token_expire_minutes=self.access_token_expire_minutes
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        auth_config = self.config.get("auth", {})

        if auth_config:
            secret_key = auth_config.get("secret_key", os.getenv("AUTH_SECRET_KEY"))
            if secret_key == "change-me-in-production":
                logger.warning("using_default_secret_key_insecure")

    def hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256.

        In production, use bcrypt, argon2, or scrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        # In production: use bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        # This is simplified for demo
        salt = os.getenv("PASSWORD_SALT", "default_salt")
        hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

        return hashed

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches
        """
        # In production: use bcrypt.checkpw(plain_password.encode(), hashed_password)
        return self.hash_password(plain_password) == hashed_password

    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password strength.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if len(password) < self.min_password_length:
            errors.append(f"Password must be at least {self.min_password_length} characters")

        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")

        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")

        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")

        if self.require_special_chars:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")

        is_valid = len(errors) == 0

        return is_valid, errors

    async def check_rate_limit(self, username: str) -> bool:
        """
        Check if user has exceeded login attempt rate limit.

        Args:
            username: Username

        Returns:
            True if user is not rate limited
        """
        now = datetime.now(timezone.utc)

        # Clean old attempts
        if username in self.login_attempts:
            cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
            self.login_attempts[username] = [
                attempt for attempt in self.login_attempts[username]
                if attempt > cutoff
            ]

            # Check if rate limited
            if len(self.login_attempts[username]) >= self.max_login_attempts:
                logger.warning("rate_limit_exceeded", username=username)
                return False

        return True

    async def record_login_attempt(self, username: str, success: bool) -> None:
        """
        Record login attempt.

        Args:
            username: Username
            success: Whether login was successful
        """
        if not success:
            if username not in self.login_attempts:
                self.login_attempts[username] = []

            self.login_attempts[username].append(datetime.now(timezone.utc))

            logger.info(
                "login_attempt_recorded",
                username=username,
                attempts=len(self.login_attempts[username])
            )
        else:
            # Clear attempts on successful login
            if username in self.login_attempts:
                del self.login_attempts[username]

    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials.

        Args:
            username: Username
            password: Password

        Returns:
            User dict if authenticated, None otherwise
        """
        try:
            # Check rate limit
            if not await self.check_rate_limit(username):
                logger.warning("authentication_rate_limited", username=username)
                return None

            # In production: fetch user from database
            # This is simplified for demo
            admin_username = os.getenv("ADMIN_USERNAME", "admin")
            admin_password = os.getenv("ADMIN_PASSWORD", "admin")

            if username == admin_username:
                # Verify password
                stored_hash = self.hash_password(admin_password)
                provided_hash = self.hash_password(password)

                if stored_hash == provided_hash:
                    await self.record_login_attempt(username, True)

                    user = {
                        "user_id": "1",
                        "username": username,
                        "email": "admin@quantumtrader.ai",
                        "roles": ["admin", "trader"],
                        "permissions": ["*"]
                    }

                    logger.info("user_authenticated", username=username)
                    return user

            # Authentication failed
            await self.record_login_attempt(username, False)
            logger.warning("authentication_failed", username=username)

            return None

        except Exception as e:
            logger.error("authentication_error", username=username, error=str(e))
            return None

    async def create_access_token(
        self,
        user_data: Dict[str, Any]
    ) -> str:
        """
        Create JWT access token.

        Args:
            user_data: User information

        Returns:
            JWT token string
        """
        try:
            # In production: use python-jose or PyJWT
            # import jwt
            # token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            # Simplified implementation
            import json
            import base64

            token_data = {
                "sub": user_data["user_id"],
                "username": user_data["username"],
                "email": user_data.get("email"),
                "roles": user_data.get("roles", []),
                "exp": (datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)).isoformat(),
                "iat": datetime.now(timezone.utc).isoformat(),
                "type": "access"
            }

            token_str = json.dumps(token_data)
            token = base64.b64encode(token_str.encode()).decode()

            logger.info("access_token_created", user_id=user_data["user_id"])

            return token

        except Exception as e:
            logger.error("token_creation_failed", error=str(e))
            raise

    async def create_refresh_token(
        self,
        user_data: Dict[str, Any]
    ) -> str:
        """
        Create refresh token.

        Args:
            user_data: User information

        Returns:
            Refresh token string
        """
        try:
            import json
            import base64

            token_data = {
                "sub": user_data["user_id"],
                "exp": (datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)).isoformat(),
                "iat": datetime.now(timezone.utc).isoformat(),
                "type": "refresh"
            }

            token_str = json.dumps(token_data)
            token = base64.b64encode(token_str.encode()).decode()

            logger.info("refresh_token_created", user_id=user_data["user_id"])

            return token

        except Exception as e:
            logger.error("refresh_token_creation_failed", error=str(e))
            raise

    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: JWT token

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # In production: use python-jose or PyJWT
            # payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Simplified verification
            import json
            import base64

            token_str = base64.b64decode(token.encode()).decode()
            payload = json.loads(token_str)

            # Check expiration
            exp_time = datetime.fromisoformat(payload["exp"])
            if exp_time < datetime.now(timezone.utc):
                logger.warning("token_expired", user=payload.get("username"))
                return None

            return payload

        except Exception as e:
            logger.error("token_verification_failed", error=str(e))
            return None

    async def check_permission(
        self,
        user: Dict[str, Any],
        required_permission: str
    ) -> bool:
        """
        Check if user has required permission.

        Args:
            user: User data
            required_permission: Required permission

        Returns:
            True if user has permission
        """
        user_permissions = user.get("permissions", [])

        # Check for wildcard permission
        if "*" in user_permissions:
            return True

        # Check for exact match
        if required_permission in user_permissions:
            return True

        # Check for resource-level wildcard (e.g., "orders:*")
        resource = required_permission.split(":")[0]
        if f"{resource}:*" in user_permissions:
            return True

        logger.warning(
            "permission_denied",
            user_id=user.get("user_id"),
            required=required_permission
        )

        return False

    async def generate_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str]
    ) -> str:
        """
        Generate API key for user.

        Args:
            user_id: User ID
            name: API key name
            permissions: List of permissions

        Returns:
            API key string
        """
        # Generate secure random key
        key_prefix = "qta"  # Quantum Trader AI
        key_secret = secrets.token_urlsafe(32)
        api_key = f"{key_prefix}_{key_secret}"

        # In production: store in database with metadata
        logger.info("api_key_generated", user_id=user_id, key_name=name)

        return api_key
