"""
Quantum Trader AI - Authorization and Permissions System
Production-grade RBAC with audit logging

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from enum import Enum
import yaml
import polars as pl
import hashlib
import secrets

from quantum_trader.models import AuditLog
from quantum_trader.database.timeseries import TimeSeriesDB


logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles in the system"""
    ADMIN = "ADMIN"
    TRADER = "TRADER"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"
    API_USER = "API_USER"
    SYSTEM = "SYSTEM"


class Permission(Enum):
    """System permissions"""
    # Trading permissions
    EXECUTE_TRADE = "EXECUTE_TRADE"
    CANCEL_ORDER = "CANCEL_ORDER"
    MODIFY_ORDER = "MODIFY_ORDER"
    VIEW_POSITIONS = "VIEW_POSITIONS"
    CLOSE_POSITION = "CLOSE_POSITION"

    # Risk management permissions
    VIEW_RISK_METRICS = "VIEW_RISK_METRICS"
    MODIFY_RISK_LIMITS = "MODIFY_RISK_LIMITS"
    TRIGGER_CIRCUIT_BREAKER = "TRIGGER_CIRCUIT_BREAKER"
    OVERRIDE_RISK_CHECK = "OVERRIDE_RISK_CHECK"

    # Strategy permissions
    VIEW_STRATEGIES = "VIEW_STRATEGIES"
    MODIFY_STRATEGIES = "MODIFY_STRATEGIES"
    DEPLOY_STRATEGY = "DEPLOY_STRATEGY"
    STOP_STRATEGY = "STOP_STRATEGY"

    # Data permissions
    VIEW_MARKET_DATA = "VIEW_MARKET_DATA"
    EXPORT_DATA = "EXPORT_DATA"
    VIEW_ANALYTICS = "VIEW_ANALYTICS"

    # System permissions
    VIEW_LOGS = "VIEW_LOGS"
    MODIFY_CONFIG = "MODIFY_CONFIG"
    MANAGE_USERS = "MANAGE_USERS"
    VIEW_AUDIT_TRAIL = "VIEW_AUDIT_TRAIL"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"

    # API permissions
    API_READ = "API_READ"
    API_WRITE = "API_WRITE"
    API_ADMIN = "API_ADMIN"


class AuthorizationManager:
    """
    Authorization and permissions management system

    Features:
    - Role-based access control (RBAC)
    - Permission checking
    - Resource access control
    - Audit logging of all access
    - Privilege escalation detection
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize authorization manager with configuration"""
        self.config = self._load_config(config_path)

        # Security configuration
        security_config = self.config.get("security", {})
        self.enable_api_keys = security_config.get("enable_api_keys", True)

        # Define role-permission mappings
        self._role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: set(Permission),  # All permissions

            Role.TRADER: {
                Permission.EXECUTE_TRADE,
                Permission.CANCEL_ORDER,
                Permission.MODIFY_ORDER,
                Permission.VIEW_POSITIONS,
                Permission.CLOSE_POSITION,
                Permission.VIEW_RISK_METRICS,
                Permission.VIEW_STRATEGIES,
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.API_READ,
                Permission.API_WRITE
            },

            Role.ANALYST: {
                Permission.VIEW_POSITIONS,
                Permission.VIEW_RISK_METRICS,
                Permission.VIEW_STRATEGIES,
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.EXPORT_DATA,
                Permission.VIEW_LOGS,
                Permission.VIEW_AUDIT_TRAIL,
                Permission.API_READ
            },

            Role.VIEWER: {
                Permission.VIEW_POSITIONS,
                Permission.VIEW_RISK_METRICS,
                Permission.VIEW_STRATEGIES,
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_ANALYTICS,
                Permission.API_READ
            },

            Role.API_USER: {
                Permission.API_READ,
                Permission.API_WRITE,
                Permission.VIEW_POSITIONS,
                Permission.VIEW_MARKET_DATA
            },

            Role.SYSTEM: set(Permission)  # All permissions for system processes
        }

        # User-role mappings (in production, this would be in database)
        self._user_roles: Dict[str, Set[Role]] = {}

        # Session tracking
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

        # Privilege escalation detection
        self._failed_auth_attempts: Dict[str, List[datetime]] = {}
        self._max_failed_attempts = 5
        self._lockout_duration = timedelta(minutes=30)

        # Database connection
        self.db: Optional[TimeSeriesDB] = None

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("AuthorizationManager initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()

            # Load user roles from database
            await self._load_user_roles()

            logger.info("AuthorizationManager initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize AuthorizationManager: {e}")
            raise

    async def _load_user_roles(self) -> None:
        """Load user-role mappings from database"""
        if not self.db:
            return

        try:
            # Load from database
            roles_df = await self.db.query_user_roles()

            for row in roles_df.iter_rows(named=True):
                user_id = row["user_id"]
                role_name = row["role"]

                try:
                    role = Role[role_name]

                    if user_id not in self._user_roles:
                        self._user_roles[user_id] = set()

                    self._user_roles[user_id].add(role)
                except KeyError:
                    logger.warning(f"Unknown role: {role_name}")

            logger.info(f"Loaded roles for {len(self._user_roles)} users")

        except Exception as e:
            logger.error(f"Failed to load user roles: {e}")

    async def assign_role(
        self,
        user_id: str,
        role: Role,
        assigned_by: str
    ) -> bool:
        """
        Assign role to user

        Args:
            user_id: User identifier
            role: Role to assign
            assigned_by: User assigning the role

        Returns:
            True if successful
        """
        try:
            # Check if assigner has permission
            if not await self.check_permission(assigned_by, Permission.MANAGE_USERS):
                logger.warning(f"User {assigned_by} attempted to assign role without permission")
                await self._log_security_event(
                    "UNAUTHORIZED_ROLE_ASSIGNMENT",
                    assigned_by,
                    {"target_user": user_id, "role": role.value}
                )
                return False

            # Add role
            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()

            self._user_roles[user_id].add(role)

            # Persist to database
            if self.db:
                await self.db.insert_user_role(user_id, role.value)

            # Audit log
            await self._log_security_event(
                "ROLE_ASSIGNED",
                assigned_by,
                {
                    "target_user": user_id,
                    "role": role.value,
                    "result": "SUCCESS"
                }
            )

            logger.info(f"Role {role.value} assigned to user {user_id} by {assigned_by}")
            return True

        except Exception as e:
            logger.error(f"Error assigning role: {e}")
            return False

    async def revoke_role(
        self,
        user_id: str,
        role: Role,
        revoked_by: str
    ) -> bool:
        """
        Revoke role from user

        Args:
            user_id: User identifier
            role: Role to revoke
            revoked_by: User revoking the role

        Returns:
            True if successful
        """
        try:
            # Check permission
            if not await self.check_permission(revoked_by, Permission.MANAGE_USERS):
                logger.warning(f"User {revoked_by} attempted to revoke role without permission")
                return False

            # Remove role
            if user_id in self._user_roles and role in self._user_roles[user_id]:
                self._user_roles[user_id].remove(role)

                # Persist to database
                if self.db:
                    await self.db.delete_user_role(user_id, role.value)

                # Audit log
                await self._log_security_event(
                    "ROLE_REVOKED",
                    revoked_by,
                    {
                        "target_user": user_id,
                        "role": role.value,
                        "result": "SUCCESS"
                    }
                )

                logger.info(f"Role {role.value} revoked from user {user_id} by {revoked_by}")
                return True
            else:
                logger.warning(f"User {user_id} does not have role {role.value}")
                return False

        except Exception as e:
            logger.error(f"Error revoking role: {e}")
            return False

    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles assigned to user"""
        return self._user_roles.get(user_id, set())

    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role"""
        return self._role_permissions.get(role, set())

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user (from all their roles)"""
        user_roles = self.get_user_roles(user_id)
        permissions: Set[Permission] = set()

        for role in user_roles:
            permissions.update(self.get_role_permissions(role))

        return permissions

    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """
        Check if user has specific permission

        Args:
            user_id: User identifier
            permission: Permission to check
            resource: Optional resource identifier for resource-specific checks

        Returns:
            True if user has permission
        """
        try:
            # Check if user is locked out
            if await self._is_user_locked_out(user_id):
                logger.warning(f"User {user_id} is locked out")
                return False

            # Get user permissions
            user_permissions = self.get_user_permissions(user_id)

            # Check permission
            has_permission = permission in user_permissions

            # Audit log (only for sensitive operations)
            if self._is_sensitive_permission(permission):
                await self._log_security_event(
                    "PERMISSION_CHECK",
                    user_id,
                    {
                        "permission": permission.value,
                        "resource": resource,
                        "result": "GRANTED" if has_permission else "DENIED"
                    }
                )

            # Track failed attempts
            if not has_permission:
                await self._track_failed_auth(user_id)

            return has_permission

        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return False

    async def check_multiple_permissions(
        self,
        user_id: str,
        permissions: List[Permission],
        require_all: bool = True
    ) -> bool:
        """
        Check if user has multiple permissions

        Args:
            user_id: User identifier
            permissions: List of permissions to check
            require_all: If True, user must have all permissions; if False, any permission

        Returns:
            True if permission check passes
        """
        user_permissions = self.get_user_permissions(user_id)

        if require_all:
            return all(perm in user_permissions for perm in permissions)
        else:
            return any(perm in user_permissions for perm in permissions)

    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str
    ) -> str:
        """
        Create authenticated session

        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session token
        """
        try:
            # Generate secure session token
            session_token = secrets.token_urlsafe(32)

            # Store session
            self._active_sessions[session_token] = {
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }

            # Audit log
            await self._log_security_event(
                "SESSION_CREATED",
                user_id,
                {
                    "ip_address": ip_address,
                    "session_token": self._hash_token(session_token)
                }
            )

            logger.info(f"Session created for user {user_id}")
            return session_token

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise

    async def validate_session(
        self,
        session_token: str,
        ip_address: str
    ) -> Optional[str]:
        """
        Validate session token

        Args:
            session_token: Session token
            ip_address: Client IP address

        Returns:
            User ID if valid, None otherwise
        """
        try:
            if session_token not in self._active_sessions:
                return None

            session = self._active_sessions[session_token]

            # Check session expiry (24 hours)
            session_age = datetime.utcnow() - session["created_at"]
            if session_age > timedelta(hours=24):
                await self.invalidate_session(session_token)
                return None

            # Check IP address (optional, can be disabled for mobile users)
            # if session["ip_address"] != ip_address:
            #     logger.warning(f"IP mismatch for session: {ip_address} != {session['ip_address']}")
            #     return None

            # Update last activity
            session["last_activity"] = datetime.utcnow()

            return session["user_id"]

        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None

    async def invalidate_session(self, session_token: str) -> None:
        """Invalidate session"""
        if session_token in self._active_sessions:
            session = self._active_sessions[session_token]

            await self._log_security_event(
                "SESSION_INVALIDATED",
                session["user_id"],
                {"session_token": self._hash_token(session_token)}
            )

            del self._active_sessions[session_token]
            logger.info(f"Session invalidated for user {session['user_id']}")

    async def _is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        if user_id not in self._failed_auth_attempts:
            return False

        recent_failures = [
            attempt for attempt in self._failed_auth_attempts[user_id]
            if datetime.utcnow() - attempt < self._lockout_duration
        ]

        if len(recent_failures) >= self._max_failed_attempts:
            logger.warning(f"User {user_id} is locked out due to {len(recent_failures)} failed attempts")
            return True

        return False

    async def _track_failed_auth(self, user_id: str) -> None:
        """Track failed authentication attempt"""
        if user_id not in self._failed_auth_attempts:
            self._failed_auth_attempts[user_id] = []

        self._failed_auth_attempts[user_id].append(datetime.utcnow())

        # Clean up old attempts
        self._failed_auth_attempts[user_id] = [
            attempt for attempt in self._failed_auth_attempts[user_id]
            if datetime.utcnow() - attempt < self._lockout_duration
        ]

        # Check for potential privilege escalation attack
        if len(self._failed_auth_attempts[user_id]) >= 3:
            await self._log_security_event(
                "POTENTIAL_PRIVILEGE_ESCALATION",
                user_id,
                {
                    "failed_attempts": len(self._failed_auth_attempts[user_id]),
                    "severity": "HIGH"
                }
            )

    def _is_sensitive_permission(self, permission: Permission) -> bool:
        """Check if permission is considered sensitive"""
        sensitive_permissions = {
            Permission.MODIFY_RISK_LIMITS,
            Permission.TRIGGER_CIRCUIT_BREAKER,
            Permission.OVERRIDE_RISK_CHECK,
            Permission.DEPLOY_STRATEGY,
            Permission.MODIFY_CONFIG,
            Permission.MANAGE_USERS,
            Permission.SYSTEM_ADMIN
        }

        return permission in sensitive_permissions

    def _hash_token(self, token: str) -> str:
        """Hash token for audit logging"""
        return hashlib.sha256(token.encode()).hexdigest()[:16]

    async def _log_security_event(
        self,
        operation: str,
        user_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Log security event to audit trail"""
        try:
            severity = details.get("severity", "INFO")
            if "UNAUTHORIZED" in operation or "ESCALATION" in operation:
                severity = "CRITICAL"

            audit_log = AuditLog(
                timestamp=datetime.utcnow(),
                operation=operation,
                user_id=user_id,
                component="AuthorizationManager",
                severity=severity,
                details=details,
                result=details.get("result", "SUCCESS")
            )

            if self.db:
                await self.db.insert_audit_log(audit_log)

        except Exception as e:
            logger.error(f"Failed to log security event: {e}")

    async def get_user_authorization_report(self, user_id: str) -> Dict:
        """
        Generate authorization report for user

        Args:
            user_id: User identifier

        Returns:
            Authorization report
        """
        try:
            roles = self.get_user_roles(user_id)
            permissions = self.get_user_permissions(user_id)

            # Get active sessions
            user_sessions = [
                {
                    "created_at": session["created_at"].isoformat(),
                    "last_activity": session["last_activity"].isoformat(),
                    "ip_address": session["ip_address"]
                }
                for session in self._active_sessions.values()
                if session["user_id"] == user_id
            ]

            # Check lockout status
            is_locked = await self._is_user_locked_out(user_id)

            report = {
                "user_id": user_id,
                "roles": [role.value for role in roles],
                "permissions": [perm.value for perm in permissions],
                "active_sessions": user_sessions,
                "is_locked_out": is_locked,
                "failed_attempts": len(self._failed_auth_attempts.get(user_id, []))
            }

            return report

        except Exception as e:
            logger.error(f"Error generating authorization report: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()

        logger.info("AuthorizationManager cleanup complete")
