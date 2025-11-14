"""
Environment Configuration - Environment-specific settings management.

This module handles environment detection and configuration loading
for different deployment environments (dev, staging, production).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from structlog import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class EnvironmentConfig:
    """
    Environment-specific configuration settings.

    All values loaded from environment variables or config files.
    """

    environment: Environment
    debug: bool
    log_level: str
    database_url: str
    redis_url: str
    api_base_url: str
    enable_trading: bool
    enable_monitoring: bool
    enable_notifications: bool
    data_dir: Path
    logs_dir: Path
    max_workers: int
    timezone: str


class EnvironmentManager:
    """
    Manages environment detection and configuration loading.

    Provides environment-specific settings and validates configuration
    based on the deployment environment.

    Attributes:
        env: Current environment
        config: Environment-specific configuration

    Example:
        >>> env_mgr = EnvironmentManager()
        >>> config = env_mgr.get_config()
        >>> if env_mgr.is_production():
        ...     # Production-only logic
    """

    def __init__(self, env_var: str = 'QUANTUM_TRADER_ENV') -> None:
        """
        Initialize environment manager.

        Args:
            env_var: Environment variable name for environment detection

        Raises:
            ValueError: If environment invalid or config missing
        """
        self.env_var = env_var
        self.env = self._detect_environment()
        self.config = self._load_config()

        logger.info(
            "Environment initialized",
            environment=self.env.value,
            debug=self.config.debug
        )

    def _detect_environment(self) -> Environment:
        """
        Detect current deployment environment.

        Returns:
            Environment enum value

        Raises:
            ValueError: If environment value invalid
        """
        env_str = os.getenv(self.env_var, 'development').lower()

        try:
            return Environment(env_str)
        except ValueError:
            logger.warning(
                "Invalid environment, defaulting to development",
                provided=env_str
            )
            return Environment.DEVELOPMENT

    def _load_config(self) -> EnvironmentConfig:
        """
        Load environment-specific configuration.

        Returns:
            EnvironmentConfig with all settings

        Raises:
            ValueError: If required config missing
        """
        # Load from environment variables
        config = EnvironmentConfig(
            environment=self.env,
            debug=self._get_bool_env('DEBUG', self.env == Environment.DEVELOPMENT),
            log_level=os.getenv('LOG_LEVEL', 'INFO' if self.env == Environment.PRODUCTION else 'DEBUG'),
            database_url=self._get_required_env('DATABASE_URL'),
            redis_url=self._get_required_env('REDIS_URL'),
            api_base_url=self._get_required_env('API_BASE_URL'),
            enable_trading=self._get_bool_env('ENABLE_TRADING', self.env == Environment.PRODUCTION),
            enable_monitoring=self._get_bool_env('ENABLE_MONITORING', True),
            enable_notifications=self._get_bool_env('ENABLE_NOTIFICATIONS', self.env == Environment.PRODUCTION),
            data_dir=Path(os.getenv('DATA_DIR', '/data/quantum_trader')),
            logs_dir=Path(os.getenv('LOGS_DIR', '/var/log/quantum_trader')),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            timezone=os.getenv('TIMEZONE', 'UTC')
        )

        self._validate_config(config)
        return config

    def _get_required_env(self, key: str) -> str:
        """
        Get required environment variable.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ValueError: If environment variable not set
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable not set: {key}")
        return value

    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """
        Get boolean environment variable.

        Args:
            key: Environment variable name
            default: Default value if not set

        Returns:
            Boolean value
        """
        value = os.getenv(key)
        if value is None:
            return default

        return value.lower() in ('true', '1', 'yes', 'on')

    def _validate_config(self, config: EnvironmentConfig) -> None:
        """
        Validate environment configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration invalid
        """
        # Validate paths exist or can be created
        try:
            config.data_dir.mkdir(parents=True, exist_ok=True)
            config.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                "Failed to create directories",
                data_dir=str(config.data_dir),
                logs_dir=str(config.logs_dir),
                error=str(e)
            )

        # Validate URLs
        if not config.database_url:
            raise ValueError("DATABASE_URL cannot be empty")

        if not config.redis_url:
            raise ValueError("REDIS_URL cannot be empty")

        # Production-specific validations
        if config.environment == Environment.PRODUCTION:
            if config.debug:
                logger.warning("Debug mode enabled in production - this is not recommended")

            if not config.enable_monitoring:
                logger.warning("Monitoring disabled in production - this is not recommended")

    def get_config(self) -> EnvironmentConfig:
        """
        Get environment configuration.

        Returns:
            Current environment configuration
        """
        return self.config

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development."""
        return self.env == Environment.DEVELOPMENT

    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.env == Environment.STAGING

    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.env == Environment.TESTING

    def get_database_url(self, mask_password: bool = False) -> str:
        """
        Get database URL.

        Args:
            mask_password: Whether to mask password in URL

        Returns:
            Database URL string
        """
        url = self.config.database_url

        if mask_password and '@' in url:
            # Mask password: user:password@host -> user:***@host
            parts = url.split('@')
            if ':' in parts[0]:
                user_pass = parts[0].split(':')
                user_pass[1] = '***'
                parts[0] = ':'.join(user_pass)
                url = '@'.join(parts)

        return url

    def get_redis_url(self, mask_password: bool = False) -> str:
        """
        Get Redis URL.

        Args:
            mask_password: Whether to mask password

        Returns:
            Redis URL string
        """
        url = self.config.redis_url

        if mask_password and '@' in url:
            parts = url.split('@')
            if ':' in parts[0]:
                user_pass = parts[0].split(':')
                user_pass[1] = '***'
                parts[0] = ':'.join(user_pass)
                url = '@'.join(parts)

        return url

    def get_feature_flags(self) -> Dict[str, bool]:
        """
        Get environment-specific feature flags.

        Returns:
            Dictionary of feature flags
        """
        return {
            'trading_enabled': self.config.enable_trading,
            'monitoring_enabled': self.config.enable_monitoring,
            'notifications_enabled': self.config.enable_notifications,
            'debug_mode': self.config.debug
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Export environment config as dictionary.

        Returns:
            Dictionary representation of config
        """
        return {
            'environment': self.env.value,
            'debug': self.config.debug,
            'log_level': self.config.log_level,
            'database_url': self.get_database_url(mask_password=True),
            'redis_url': self.get_redis_url(mask_password=True),
            'api_base_url': self.config.api_base_url,
            'enable_trading': self.config.enable_trading,
            'enable_monitoring': self.config.enable_monitoring,
            'enable_notifications': self.config.enable_notifications,
            'data_dir': str(self.config.data_dir),
            'logs_dir': str(self.config.logs_dir),
            'max_workers': self.config.max_workers,
            'timezone': self.config.timezone
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvironmentManager':
        """
        Create EnvironmentManager from configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            EnvironmentManager instance
        """
        # Set environment variables from dict
        for key, value in config_dict.items():
            env_key = key.upper()
            if isinstance(value, bool):
                os.environ[env_key] = 'true' if value else 'false'
            else:
                os.environ[env_key] = str(value)

        return cls()


def get_environment() -> Environment:
    """
    Get current environment.

    Returns:
        Current Environment enum value
    """
    env_str = os.getenv('QUANTUM_TRADER_ENV', 'development').lower()
    try:
        return Environment(env_str)
    except ValueError:
        return Environment.DEVELOPMENT


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development environment."""
    return get_environment() == Environment.DEVELOPMENT


def load_env_file(env_file: Path) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file

    Raises:
        FileNotFoundError: If env file doesn't exist
    """
    if not env_file.exists():
        raise FileNotFoundError(f"Environment file not found: {env_file}")

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]

                os.environ[key] = value

    logger.info("Environment file loaded", path=str(env_file))
