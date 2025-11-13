"""
Authorization Module - Role-Based Access Control (RBAC)
CRITICAL: Production-ready RBAC system for trading operations
"""

import logging
from typing import Dict, Set, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass, field

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Trading system permissions"""
    # Order permissions
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    MODIFY_ORDER = "modify_order"
    VIEW_ORDERS = "view_orders"

    # Position permissions
    VIEW_POSITIONS = "view_positions"
    CLOSE_POSITION = "close_position"
    MODIFY_POSITION = "modify_position"

    # Account permissions
    VIEW_BALANCE = "view_balance"
    WITHDRAW_FUNDS = "withdraw_funds"
    DEPOSIT_FUNDS = "deposit_funds"

    # Trading permissions
    MANUAL_TRADING = "manual_trading"
    ALGO_TRADING = "algo_trading"
    HIGH_FREQUENCY_TRADING = "high_frequency_trading"

    # Risk management
    MODIFY_RISK_LIMITS = "modify_risk_limits"
    OVERRIDE_CIRCUIT_BREAKER = "override_circuit_breaker"
    EMERGENCY_STOP = "emergency_stop"

    # System permissions
    VIEW_LOGS = "view_logs"
    VIEW_METRICS = "view_metrics"
    MODIFY_CONFIG = "modify_config"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"

    # API permissions
    API_ACCESS = "api_access"
    WEBSOCKET_ACCESS = "websocket_access"

    # Market data
    VIEW_MARKET_DATA = "view_market_data"
    SUBSCRIBE_LEVEL2 = "subscribe_level2"


class Role(str, Enum):
    """Predefined system roles"""
    ADMIN = "admin"
    TRADER = "trader"
    ANALYST = "analyst"
    RISK_MANAGER = "risk_manager"
    VIEWER = "viewer"
    API_USER = "api_user"


@dataclass
class RoleDefinition:
    """Role definition with permissions and metadata"""
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserAuthorization:
    """User authorization state"""
    user_id: str
    roles: Set[str] = field(default_factory=set)
    additional_permissions: Set[Permission] = field(default_factory=set)
    revoked_permissions: Set[Permission] = field(default_factory=set)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AuthorizationManager:
    """Production RBAC authorization manager"""

    def __init__(self):
        """Initialize authorization manager with default roles"""
        self._config = get_config()
        self._roles: Dict[str, RoleDefinition] = {}
        self._user_authorizations: Dict[str, UserAuthorization] = {}
        self._permission_cache: Dict[str, Set[Permission]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        self._last_cache_clear = datetime.utcnow()
        self._lock = asyncio.Lock()

        # Initialize default roles
        self._initialize_default_roles()

        logger.info("Authorization manager initialized with RBAC")

    def _initialize_default_roles(self) -> None:
        """Initialize default role definitions"""
        # Admin role - full access
        self._roles[Role.ADMIN] = RoleDefinition(
            name=Role.ADMIN,
            permissions=set(Permission),  # All permissions
            description="Full system access"
        )

        # Trader role - trading operations
        self._roles[Role.TRADER] = RoleDefinition(
            name=Role.TRADER,
            permissions={
                Permission.PLACE_ORDER,
                Permission.CANCEL_ORDER,
                Permission.MODIFY_ORDER,
                Permission.VIEW_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.CLOSE_POSITION,
                Permission.VIEW_BALANCE,
                Permission.MANUAL_TRADING,
                Permission.ALGO_TRADING,
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_METRICS,
                Permission.API_ACCESS,
                Permission.WEBSOCKET_ACCESS,
            },
            description="Active trading operations"
        )

        # Analyst role - read-only analysis
        self._roles[Role.ANALYST] = RoleDefinition(
            name=Role.ANALYST,
            permissions={
                Permission.VIEW_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.VIEW_BALANCE,
                Permission.VIEW_LOGS,
                Permission.VIEW_METRICS,
                Permission.VIEW_MARKET_DATA,
                Permission.SUBSCRIBE_LEVEL2,
            },
            description="Read-only analysis access"
        )

        # Risk Manager role - risk oversight
        self._roles[Role.RISK_MANAGER] = RoleDefinition(
            name=Role.RISK_MANAGER,
            permissions={
                Permission.VIEW_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.VIEW_BALANCE,
                Permission.CLOSE_POSITION,
                Permission.MODIFY_RISK_LIMITS,
                Permission.OVERRIDE_CIRCUIT_BREAKER,
                Permission.EMERGENCY_STOP,
                Permission.VIEW_LOGS,
                Permission.VIEW_METRICS,
                Permission.VIEW_MARKET_DATA,
            },
            description="Risk management and oversight"
        )

        # Viewer role - minimal read access
        self._roles[Role.VIEWER] = RoleDefinition(
            name=Role.VIEWER,
            permissions={
                Permission.VIEW_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.VIEW_BALANCE,
                Permission.VIEW_MARKET_DATA,
            },
            description="View-only access"
        )

        # API User role - programmatic access
        self._roles[Role.API_USER] = RoleDefinition(
            name=Role.API_USER,
            permissions={
                Permission.PLACE_ORDER,
                Permission.CANCEL_ORDER,
                Permission.VIEW_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.VIEW_BALANCE,
                Permission.VIEW_MARKET_DATA,
                Permission.API_ACCESS,
                Permission.WEBSOCKET_ACCESS,
            },
            description="API programmatic access"
        )

    async def authorize(self, user_id: str, permission: Permission) -> bool:
        """
        Check if user is authorized for a specific permission

        Args:
            user_id: User identifier
            permission: Permission to check

        Returns:
            True if authorized, False otherwise
        """
        try:
            # Clear cache if TTL expired
            await self._check_cache_expiry()

            # Get user permissions (from cache if available)
            user_permissions = await self._get_user_permissions(user_id)

            # Check if permission is granted
            is_authorized = permission in user_permissions

            if is_authorized:
                logger.debug(f"User {user_id} authorized for {permission.value}")
            else:
                logger.warning(f"User {user_id} NOT authorized for {permission.value}")

            return is_authorized

        except Exception as e:
            logger.error(f"Authorization check failed for user {user_id}, permission {permission.value}: {e}")
            return False

    async def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Alias for authorize() - check if user has permission

        Args:
            user_id: User identifier
            permission: Permission to check

        Returns:
            True if authorized, False otherwise
        """
        return await self.authorize(user_id, permission)

    async def assign_role(self, user_id: str, role: str) -> bool:
        """
        Assign a role to a user

        Args:
            user_id: User identifier
            role: Role name to assign

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Validate role exists
                if role not in self._roles:
                    logger.error(f"Cannot assign unknown role {role} to user {user_id}")
                    return False

                # Get or create user authorization
                if user_id not in self._user_authorizations:
                    self._user_authorizations[user_id] = UserAuthorization(user_id=user_id)

                user_auth = self._user_authorizations[user_id]

                # Add role
                user_auth.roles.add(role)
                user_auth.last_updated = datetime.utcnow()

                # Clear permission cache for user
                if user_id in self._permission_cache:
                    del self._permission_cache[user_id]

                logger.info(f"Assigned role {role} to user {user_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to assign role {role} to user {user_id}: {e}")
                return False

    async def revoke_role(self, user_id: str, role: str) -> bool:
        """
        Revoke a role from a user

        Args:
            user_id: User identifier
            role: Role name to revoke

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                if user_id not in self._user_authorizations:
                    logger.warning(f"Cannot revoke role from unknown user {user_id}")
                    return False

                user_auth = self._user_authorizations[user_id]

                if role in user_auth.roles:
                    user_auth.roles.remove(role)
                    user_auth.last_updated = datetime.utcnow()

                    # Clear permission cache
                    if user_id in self._permission_cache:
                        del self._permission_cache[user_id]

                    logger.info(f"Revoked role {role} from user {user_id}")
                    return True
                else:
                    logger.warning(f"User {user_id} does not have role {role}")
                    return False

            except Exception as e:
                logger.error(f"Failed to revoke role {role} from user {user_id}: {e}")
                return False

    async def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Grant an additional permission to a user (beyond role permissions)

        Args:
            user_id: User identifier
            permission: Permission to grant

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Get or create user authorization
                if user_id not in self._user_authorizations:
                    self._user_authorizations[user_id] = UserAuthorization(user_id=user_id)

                user_auth = self._user_authorizations[user_id]
                user_auth.additional_permissions.add(permission)
                user_auth.last_updated = datetime.utcnow()

                # Clear cache
                if user_id in self._permission_cache:
                    del self._permission_cache[user_id]

                logger.info(f"Granted permission {permission.value} to user {user_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to grant permission {permission.value} to user {user_id}: {e}")
                return False

    async def revoke_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Revoke a specific permission from a user

        Args:
            user_id: User identifier
            permission: Permission to revoke

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                if user_id not in self._user_authorizations:
                    self._user_authorizations[user_id] = UserAuthorization(user_id=user_id)

                user_auth = self._user_authorizations[user_id]
                user_auth.revoked_permissions.add(permission)
                user_auth.last_updated = datetime.utcnow()

                # Clear cache
                if user_id in self._permission_cache:
                    del self._permission_cache[user_id]

                logger.info(f"Revoked permission {permission.value} from user {user_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to revoke permission {permission.value} from user {user_id}: {e}")
                return False

    async def get_user_roles(self, user_id: str) -> List[str]:
        """
        Get all roles assigned to a user

        Args:
            user_id: User identifier

        Returns:
            List of role names
        """
        try:
            if user_id not in self._user_authorizations:
                return []

            return list(self._user_authorizations[user_id].roles)

        except Exception as e:
            logger.error(f"Failed to get roles for user {user_id}: {e}")
            return []

    async def _get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all effective permissions for a user (cached)

        Args:
            user_id: User identifier

        Returns:
            Set of permissions
        """
        # Check cache
        if user_id in self._permission_cache:
            return self._permission_cache[user_id]

        # Calculate permissions
        permissions: Set[Permission] = set()

        if user_id in self._user_authorizations:
            user_auth = self._user_authorizations[user_id]

            # Add permissions from all assigned roles
            for role_name in user_auth.roles:
                if role_name in self._roles:
                    permissions.update(self._roles[role_name].permissions)

            # Add additional permissions
            permissions.update(user_auth.additional_permissions)

            # Remove revoked permissions
            permissions -= user_auth.revoked_permissions

        # Cache result
        self._permission_cache[user_id] = permissions

        return permissions

    async def _check_cache_expiry(self) -> None:
        """Clear permission cache if TTL expired"""
        now = datetime.utcnow()
        if (now - self._last_cache_clear).total_seconds() > self._cache_ttl_seconds:
            async with self._lock:
                self._permission_cache.clear()
                self._last_cache_clear = now
                logger.debug("Permission cache cleared due to TTL expiry")

    async def create_role(self, role_name: str, permissions: Set[Permission], description: str = "") -> bool:
        """
        Create a custom role

        Args:
            role_name: Name of the role
            permissions: Set of permissions for the role
            description: Role description

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                if role_name in self._roles:
                    logger.error(f"Role {role_name} already exists")
                    return False

                self._roles[role_name] = RoleDefinition(
                    name=role_name,
                    permissions=permissions,
                    description=description
                )

                logger.info(f"Created custom role {role_name} with {len(permissions)} permissions")
                return True

            except Exception as e:
                logger.error(f"Failed to create role {role_name}: {e}")
                return False

    async def get_role_permissions(self, role_name: str) -> Set[Permission]:
        """
        Get permissions for a role

        Args:
            role_name: Name of the role

        Returns:
            Set of permissions
        """
        try:
            if role_name in self._roles:
                return self._roles[role_name].permissions.copy()
            else:
                logger.warning(f"Role {role_name} not found")
                return set()

        except Exception as e:
            logger.error(f"Failed to get permissions for role {role_name}: {e}")
            return set()
