"""
Session Manager

Manages user sessions, authentication tokens, and session lifecycle.
Implements thread-safe session storage with expiration and cleanup.

This module provides:
- Session creation and validation
- Token generation and verification
- Automatic session expiration
- Session cleanup and garbage collection
- Thread-safe session operations
"""

import asyncio
import secrets
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import hmac

import structlog
from threading import Lock

logger = structlog.get_logger(__name__)


@dataclass
class Session:
    """Session data container.

    Attributes:
        token: Unique session token
        user_id: User identifier
        created_at: Session creation timestamp
        expires_at: Session expiration timestamp
        last_accessed: Last access timestamp
        data: Additional session data
    """

    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_accessed: datetime
    data: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if session is expired.

        Returns:
            True if session is expired, False otherwise
        """
        return datetime.utcnow() >= self.expires_at

    def is_active(self) -> bool:
        """Check if session is active.

        Returns:
            True if session is active, False otherwise
        """
        return not self.is_expired()

    def refresh(self, extend_minutes: int) -> None:
        """Refresh session expiration.

        Args:
            extend_minutes: Minutes to extend session
        """
        self.expires_at = datetime.utcnow() + timedelta(minutes=extend_minutes)
        self.last_accessed = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary.

        Returns:
            Session data as dictionary
        """
        return {
            "token": self.token,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "data": self.data,
        }


class SessionManager:
    """Thread-safe session manager with automatic cleanup.

    Manages user sessions, handles token generation and validation,
    and implements automatic session expiration and cleanup.

    Attributes:
        secret_key: Secret key for token signing
        default_expire_minutes: Default session expiration in minutes
        sessions: Active sessions storage
        cleanup_interval: Cleanup task interval in seconds
    """

    def __init__(
        self,
        secret_key: str,
        default_expire_minutes: int = 60,
        cleanup_interval: int = 300,
    ) -> None:
        """Initialize session manager.

        Args:
            secret_key: Secret key for token signing
            default_expire_minutes: Default session expiration in minutes
            cleanup_interval: Cleanup task interval in seconds

        Raises:
            ValueError: If secret_key is empty
        """
        if not secret_key:
            raise ValueError("secret_key cannot be empty")

        self.secret_key = secret_key.encode()
        self.default_expire_minutes = default_expire_minutes
        self.cleanup_interval = cleanup_interval

        self.sessions: Dict[str, Session] = {}
        self._lock = Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "SessionManager initialized",
            default_expire_minutes=default_expire_minutes,
            cleanup_interval=cleanup_interval,
        )

        # Start cleanup task
        asyncio.create_task(self._start_cleanup())

    async def _start_cleanup(self) -> None:
        """Start the cleanup background task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Session cleanup task started")

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e), exc_info=True)

    def _generate_token(self) -> str:
        """Generate a secure random token.

        Returns:
            Secure random token string
        """
        random_part = secrets.token_urlsafe(32)
        timestamp = str(int(time.time()))

        # Create HMAC signature
        message = f"{random_part}:{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()

        token = f"{random_part}:{timestamp}:{signature}"

        return token

    def _verify_token(self, token: str) -> bool:
        """Verify token signature.

        Args:
            token: Token to verify

        Returns:
            True if token is valid, False otherwise
        """
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False

            random_part, timestamp, signature = parts

            # Verify signature
            message = f"{random_part}:{timestamp}".encode()
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()

            return hmac.compare_digest(signature, expected_signature)

        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            return False

    async def create_session(
        self,
        user_id: str,
        data: Optional[Dict[str, Any]] = None,
        expire_minutes: Optional[int] = None,
    ) -> str:
        """Create a new session.

        Args:
            user_id: User identifier
            data: Optional session data
            expire_minutes: Optional custom expiration time

        Returns:
            Session token

        Raises:
            ValueError: If user_id is empty
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")

        token = self._generate_token()
        expires_in = expire_minutes or self.default_expire_minutes
        now = datetime.utcnow()

        session = Session(
            token=token,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(minutes=expires_in),
            last_accessed=now,
            data=data or {},
        )

        with self._lock:
            self.sessions[token] = session

        logger.info(
            "Session created",
            user_id=user_id,
            token_prefix=token[:16],
            expires_in_minutes=expires_in,
        )

        return token

    async def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Get session data by token.

        Args:
            token: Session token

        Returns:
            Session data dictionary or None if not found/expired
        """
        if not token or not self._verify_token(token):
            return None

        with self._lock:
            session = self.sessions.get(token)

            if not session:
                return None

            if session.is_expired():
                # Remove expired session
                del self.sessions[token]
                logger.debug("Session expired", token_prefix=token[:16])
                return None

            # Update last accessed time
            session.last_accessed = datetime.utcnow()

            return session.data

    async def refresh_session(
        self,
        token: str,
        extend_minutes: Optional[int] = None,
    ) -> bool:
        """Refresh session expiration.

        Args:
            token: Session token
            extend_minutes: Minutes to extend session (uses default if not provided)

        Returns:
            True if session was refreshed, False otherwise
        """
        if not token or not self._verify_token(token):
            return False

        with self._lock:
            session = self.sessions.get(token)

            if not session:
                return False

            if session.is_expired():
                del self.sessions[token]
                return False

            extend = extend_minutes or self.default_expire_minutes
            session.refresh(extend)

            logger.debug(
                "Session refreshed",
                token_prefix=token[:16],
                new_expiry=session.expires_at.isoformat(),
            )

            return True

    async def delete_session(self, token: str) -> bool:
        """Delete a session.

        Args:
            token: Session token

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if token in self.sessions:
                del self.sessions[token]
                logger.info("Session deleted", token_prefix=token[:16])
                return True

            return False

    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions deleted
        """
        count = 0

        with self._lock:
            tokens_to_delete = [
                token for token, session in self.sessions.items()
                if session.user_id == user_id
            ]

            for token in tokens_to_delete:
                del self.sessions[token]
                count += 1

        if count > 0:
            logger.info("User sessions deleted", user_id=user_id, count=count)

        return count

    async def cleanup_expired_sessions(self) -> int:
        """Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        count = 0

        with self._lock:
            tokens_to_delete = [
                token for token, session in self.sessions.items()
                if session.is_expired()
            ]

            for token in tokens_to_delete:
                del self.sessions[token]
                count += 1

        if count > 0:
            logger.info("Expired sessions cleaned up", count=count)

        return count

    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions.

        Returns:
            Number of active sessions
        """
        with self._lock:
            return len([s for s in self.sessions.values() if s.is_active()])

    async def get_user_sessions_count(self, user_id: str) -> int:
        """Get count of active sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of active sessions for user
        """
        with self._lock:
            return len([
                s for s in self.sessions.values()
                if s.user_id == user_id and s.is_active()
            ])

    async def get_session_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get full session information.

        Args:
            token: Session token

        Returns:
            Session information or None if not found
        """
        if not token or not self._verify_token(token):
            return None

        with self._lock:
            session = self.sessions.get(token)

            if not session:
                return None

            if session.is_expired():
                del self.sessions[token]
                return None

            return session.to_dict()

    async def cleanup(self) -> None:
        """Cleanup session manager resources.

        Stops the cleanup task and removes all sessions.
        """
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        with self._lock:
            session_count = len(self.sessions)
            self.sessions.clear()

        logger.info("SessionManager cleanup complete", sessions_removed=session_count)

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("SessionManager destroyed without cleanup")
