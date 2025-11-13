"""
Configuration Loader Utility
CRITICAL: Centralized configuration loading from YAML files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files with environment variable substitution"""

    _instance: Optional['ConfigLoader'] = None
    _config_cache: Dict[str, Dict[str, Any]] = {}

    def __new__(cls) -> 'ConfigLoader':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.config_dir = Path(os.getenv('QUANTUM_CONFIG_DIR', '/home/user/quantum_trader_ai/config'))
        self.bot_config_dir = self.config_dir / 'bot'
        self.env_config_dir = self.config_dir / 'environments'

    def _substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute environment variables in config values"""
        if isinstance(value, str):
            # Check if value is an environment variable reference
            if value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logger.warning(f"Environment variable {env_var} not set, using empty string")
                    return ""
                return env_value
            return value
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        return value

    def load_config(self, config_name: str, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file

        Args:
            config_name: Name of config file (without .yaml extension)
            section: Optional specific section to return

        Returns:
            Configuration dictionary with env vars substituted
        """
        cache_key = f"{config_name}:{section}" if section else config_name

        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        config_path = self.bot_config_dir / f"{config_name}.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Substitute environment variables
            config = self._substitute_env_vars(config)

            # Extract section if specified
            if section:
                config = config.get(section, {})

            self._config_cache[cache_key] = config
            return config

        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config {config_path}: {e}")
            raise

    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation

        Args:
            config_name: Name of config file
            key_path: Dot-separated path to value (e.g., 'position_limits.max_position_size_usd')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config = self.load_config(config_name)

        keys = key_path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_decimal(self, config_name: str, key_path: str, default: str = "0") -> Decimal:
        """Get configuration value as Decimal (for financial calculations)"""
        value = self.get(config_name, key_path, default)
        try:
            return Decimal(str(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key_path}={value} to Decimal, using default {default}")
            return Decimal(str(default))

    def get_int(self, config_name: str, key_path: str, default: int = 0) -> int:
        """Get configuration value as integer"""
        value = self.get(config_name, key_path, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key_path}={value} to int, using default {default}")
            return default

    def get_float(self, config_name: str, key_path: str, default: float = 0.0) -> float:
        """Get configuration value as float (use sparingly, prefer Decimal for money)"""
        value = self.get(config_name, key_path, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key_path}={value} to float, using default {default}")
            return default

    def get_bool(self, config_name: str, key_path: str, default: bool = False) -> bool:
        """Get configuration value as boolean"""
        value = self.get(config_name, key_path, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)

    def get_list(self, config_name: str, key_path: str, default: Optional[list] = None) -> list:
        """Get configuration value as list"""
        value = self.get(config_name, key_path, default or [])
        return list(value) if value else []

    def reload(self, config_name: Optional[str] = None) -> None:
        """Reload configuration from disk (clear cache)"""
        if config_name:
            # Clear specific config from cache
            keys_to_remove = [k for k in self._config_cache.keys() if k.startswith(config_name)]
            for key in keys_to_remove:
                del self._config_cache[key]
        else:
            # Clear entire cache
            self._config_cache.clear()
        logger.info(f"Configuration cache cleared: {config_name or 'all'}")


# Global singleton instance
_config_loader = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get the global ConfigLoader instance"""
    return _config_loader
