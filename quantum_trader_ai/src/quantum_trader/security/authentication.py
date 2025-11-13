"""
Authentication System - CRITICAL SECURITY SYSTEM
Multi-factor authentication, session management, and access control
"""

import logging
import secrets
import hashlib
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import jwt
import pyotp

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class AuthStatus(Enum):
    """Authentication status"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    MFA_REQUIRED = "MFA_REQUIRED"
    LOCKED = "LOCKED"
    EXPIRED = "EXPIRED"


@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    ip_address: str
    created_at: datetime
    expires_at: datetime
    mfa_verified: bool = False


@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    password_hash: str
    mfa_secret: Optional[str]
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None


class AuthenticationManager:
    """Production authentication and session management"""

    def __init__(self) -> None:
        self.config = get_config()
        self._load_config()

        # Session storage
        self._sessions: Dict[str, Session] = {}
        self._users: Dict[str, User] = {}  # In production: use database

        # Start cleanup task
        asyncio.create_task(self._cleanup_sessions_periodically())

        logger.info("AuthenticationManager initialized")

    def _load_config(self) -> None:
        """Load authentication configuration"""
        self.session_timeout = self.config.get_int('security', 'authentication.session_timeout_seconds', 3600)
        self.max_failed_attempts = self.config.get_int('security', 'authentication.max_failed_attempts', 3)
        self.lockout_duration = self.config.get_int('security', 'authentication.lockout_duration_seconds', 900)
        self.password_min_length = self.config.get_int('security', 'authentication.password_min_length', 16)
        self.require_mfa = self.config.get_bool('security', 'authentication.require_mfa', True)
        self.jwt_algorithm = self.config.get('security', 'authentication.jwt_algorithm', 'RS256')
        self.jwt_expiry = self.config.get_int('security', 'authentication.jwt_expiry_seconds', 1800)

        # MFA configuration
        self.mfa_enabled = self.config.get_bool('security', 'mfa.enabled', True)
        self.mfa_issuer = self.config.get('security', 'mfa.issuer', 'QuantumTrader')
        self.mfa_token_validity = self.config.get_int('security', 'mfa.token_validity_seconds', 30)

    async def authenticate(self, username: str, password: str,
                          ip_address: str, mfa_token: Optional[str] = None) -> Tuple[AuthStatus, Optional[str]]:
        """
        Authenticate user

        Returns:
            Tuple of (AuthStatus, session_id or error_message)
        """
        try:
            # Get user (in production: from database)
            user = self._users.get(username)
            if not user:
                logger.warning(f"Authentication failed: user not found - {username}")
                return (AuthStatus.FAILED, "Invalid credentials")

            # Check if account is locked
            if user.locked_until and datetime.now(timezone.utc) < user.locked_until:
                remaining = (user.locked_until - datetime.now(timezone.utc)).total_seconds()
                return (AuthStatus.LOCKED, f"Account locked for {int(remaining)} seconds")

            # Verify password
            if not self._verify_password(password, user.password_hash):
                user.failed_attempts += 1

                # Lock account if too many failed attempts
                if user.failed_attempts >= self.max_failed_attempts:
                    user.locked_until = datetime.now(timezone.utc) + timedelta(seconds=self.lockout_duration)
                    logger.warning(f"Account locked due to failed attempts: {username}")
                    return (AuthStatus.LOCKED, "Account locked due to multiple failed attempts")

                return (AuthStatus.FAILED, "Invalid credentials")

            # Check MFA if required
            if self.require_mfa and user.mfa_secret:
                if not mfa_token:
                    return (AuthStatus.MFA_REQUIRED, "MFA token required")

                if not self._verify_mfa_token(user.mfa_secret, mfa_token):
                    return (AuthStatus.FAILED, "Invalid MFA token")

            # Reset failed attempts on successful auth
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now(timezone.utc)

            # Create session
            session = await self._create_session(user.user_id, ip_address)

            logger.info(f"User authenticated successfully: {username}")
            return (AuthStatus.SUCCESS, session.session_id)

        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            return (AuthStatus.FAILED, "Authentication error")

    async def _create_session(self, user_id: str, ip_address: str) -> Session:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout),
            mfa_verified=True
        )

        self._sessions[session_id] = session
        return session

    async def validate_session(self, session_id: str) -> Tuple[bool, Optional[str]]:
        """Validate session"""
        session = self._sessions.get(session_id)

        if not session:
            return (False, "Invalid session")

        if datetime.now(timezone.utc) > session.expires_at:
            del self._sessions[session_id]
            return (False, "Session expired")

        return (True, session.user_id)

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate (logout) session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session invalidated: {session_id}")
            return True
        return False

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        # In production: use bcrypt or argon2
        computed_hash = hashlib.sha256(password.encode()).hexdigest()
        return secrets.compare_digest(computed_hash, password_hash)

    def _verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify TOTP MFA token"""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False

    def generate_mfa_secret(self, username: str) -> Tuple[str, str]:
        """Generate MFA secret and QR code URI"""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        uri = totp.provisioning_uri(name=username, issuer_name=self.mfa_issuer)
        return (secret, uri)

    async def _cleanup_sessions_periodically(self) -> None:
        """Cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                now = datetime.now(timezone.utc)
                expired = [sid for sid, session in self._sessions.items()
                          if session.expires_at < now]

                for session_id in expired:
                    del self._sessions[session_id]

                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired sessions")

            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
