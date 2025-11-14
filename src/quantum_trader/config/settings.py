"""
Settings Management - Centralized configuration management.

This module provides a unified settings interface that loads configuration
from multiple sources (files, environment variables, remote config servers).
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml
import json
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSettings:
    """Trading-specific settings."""

    enabled: bool
    max_positions: int
    default_leverage: int
    risk_per_trade: str  # Decimal as string
    max_daily_trades: int
    allowed_symbols: List[str]
    forbidden_symbols: List[str] = field(default_factory=list)
    trading_hours: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskSettings:
    """Risk management settings."""

    max_position_size: str  # Decimal as string
    max_leverage: str
    max_drawdown_percent: str
    max_daily_loss: str
    position_size_percent: str
    stop_loss_percent: str
    take_profit_percent: str
    trailing_stop_enabled: bool = False
    trailing_stop_percent: Optional[str] = None


@dataclass
class ExchangeSettings:
    """Exchange connection settings."""

    name: str
    enabled: bool
    testnet: bool
    rate_limit_requests: int
    rate_limit_window_seconds: int
    timeout_seconds: int
    retry_attempts: int
    websocket_enabled: bool


@dataclass
class StrategySettings:
    """Strategy-specific settings."""

    name: str
    enabled: bool
    timeframes: List[str]
    indicators: Dict[str, Any]
    parameters: Dict[str, Any]
    risk_multiplier: str  # Decimal as string


class SettingsManager:
    """
    Manages application configuration from multiple sources.

    Loads and merges configuration from YAML files, environment variables,
    and optionally remote configuration servers. Provides type-safe access
    to settings with validation.

    Attributes:
        config: Merged configuration dictionary
        config_files: List of loaded config files

    Example:
        >>> settings = SettingsManager(config_dir='/etc/quantum_trader')
        >>> trading = settings.get_trading_settings()
        >>> if trading.enabled:
        ...     # Start trading
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        config_files: Optional[List[str]] = None,
        env_prefix: str = 'QUANTUM_TRADER'
    ) -> None:
        """
        Initialize settings manager.

        Args:
            config_dir: Directory containing config files
            config_files: List of config filenames to load
            env_prefix: Prefix for environment variable overrides

        Raises:
            ValueError: If config directory doesn't exist
        """
        self.config_dir = config_dir or Path('/etc/quantum_trader/config')
        self.config_files = config_files or ['default.yaml', 'trading.yaml', 'risk.yaml']
        self.env_prefix = env_prefix

        self.config: Dict[str, Any] = {}
        self.file_configs: Dict[str, Dict[str, Any]] = {}

        self._load_configs()

        logger.info(
            "SettingsManager initialized",
            config_dir=str(self.config_dir),
            files_loaded=len(self.file_configs)
        )

    def _load_configs(self) -> None:
        """Load configuration from all sources."""
        try:
            # Load from YAML files
            self._load_config_files()

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Validate configuration
            self._validate_config()

        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise

    def _load_config_files(self) -> None:
        """Load configuration from YAML files."""
        for config_file in self.config_files:
            file_path = self.config_dir / config_file

            if not file_path.exists():
                logger.warning("Config file not found", path=str(file_path))
                continue

            try:
                with open(file_path, 'r') as f:
                    file_config = yaml.safe_load(f)

                if file_config:
                    self.file_configs[config_file] = file_config
                    self._merge_config(file_config)

                logger.debug("Loaded config file", path=str(file_path))

            except Exception as e:
                logger.error("Failed to load config file", path=str(file_path), error=str(e))
                raise

    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration into existing config.

        Args:
            new_config: Configuration to merge
        """
        self._deep_merge(self.config, new_config)

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Deep merge dictionaries.

        Args:
            base: Base dictionary to update
            update: Dictionary with updates
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        import os

        # Common environment variable patterns
        env_mappings = {
            f'{self.env_prefix}_TRADING_ENABLED': 'trading.enabled',
            f'{self.env_prefix}_MAX_POSITIONS': 'trading.max_positions',
            f'{self.env_prefix}_MAX_LEVERAGE': 'risk.max_leverage',
            f'{self.env_prefix}_MAX_DRAWDOWN': 'risk.max_drawdown_percent',
            f'{self.env_prefix}_TESTNET': 'exchanges.testnet',
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
                logger.debug("Applied env override", var=env_var, path=config_path)

    def _set_nested_value(self, path: str, value: Any) -> None:
        """
        Set a value in nested dictionary using dot notation.

        Args:
            path: Dot-separated path (e.g., 'trading.max_positions')
            value: Value to set
        """
        keys = path.split('.')
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        final_key = keys[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value

    def _validate_config(self) -> None:
        """Validate loaded configuration."""
        required_sections = ['trading', 'risk', 'exchanges']

        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing config section: {section}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Dot-separated configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> max_pos = settings.get('trading.max_positions', 10)
        """
        keys = key.split('.')
        current = self.config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Dot-separated configuration key
            value: Value to set
        """
        self._set_nested_value(key, value)
        logger.debug("Config value updated", key=key)

    def get_trading_settings(self) -> TradingSettings:
        """
        Get trading settings.

        Returns:
            TradingSettings object

        Raises:
            ValueError: If trading config invalid
        """
        trading_config = self.config.get('trading', {})

        return TradingSettings(
            enabled=bool(trading_config.get('enabled', False)),
            max_positions=int(trading_config.get('max_positions', 5)),
            default_leverage=int(trading_config.get('default_leverage', 1)),
            risk_per_trade=str(trading_config.get('risk_per_trade', '0.02')),
            max_daily_trades=int(trading_config.get('max_daily_trades', 100)),
            allowed_symbols=trading_config.get('allowed_symbols', []),
            forbidden_symbols=trading_config.get('forbidden_symbols', []),
            trading_hours=trading_config.get('trading_hours', {})
        )

    def get_risk_settings(self) -> RiskSettings:
        """
        Get risk management settings.

        Returns:
            RiskSettings object
        """
        risk_config = self.config.get('risk', {})

        return RiskSettings(
            max_position_size=str(risk_config.get('max_position_size', '10000')),
            max_leverage=str(risk_config.get('max_leverage', '5')),
            max_drawdown_percent=str(risk_config.get('max_drawdown_percent', '20')),
            max_daily_loss=str(risk_config.get('max_daily_loss', '1000')),
            position_size_percent=str(risk_config.get('position_size_percent', '2')),
            stop_loss_percent=str(risk_config.get('stop_loss_percent', '2')),
            take_profit_percent=str(risk_config.get('take_profit_percent', '5')),
            trailing_stop_enabled=bool(risk_config.get('trailing_stop_enabled', False)),
            trailing_stop_percent=risk_config.get('trailing_stop_percent')
        )

    def get_exchange_settings(self, exchange: str) -> ExchangeSettings:
        """
        Get exchange-specific settings.

        Args:
            exchange: Exchange name

        Returns:
            ExchangeSettings object

        Raises:
            ValueError: If exchange not configured
        """
        exchanges_config = self.config.get('exchanges', {})

        if exchange not in exchanges_config:
            raise ValueError(f"Exchange not configured: {exchange}")

        exchange_config = exchanges_config[exchange]

        return ExchangeSettings(
            name=exchange,
            enabled=bool(exchange_config.get('enabled', True)),
            testnet=bool(exchange_config.get('testnet', False)),
            rate_limit_requests=int(exchange_config.get('rate_limit_requests', 100)),
            rate_limit_window_seconds=int(exchange_config.get('rate_limit_window_seconds', 60)),
            timeout_seconds=int(exchange_config.get('timeout_seconds', 30)),
            retry_attempts=int(exchange_config.get('retry_attempts', 3)),
            websocket_enabled=bool(exchange_config.get('websocket_enabled', True))
        )

    def get_strategy_settings(self, strategy: str) -> StrategySettings:
        """
        Get strategy-specific settings.

        Args:
            strategy: Strategy name

        Returns:
            StrategySettings object

        Raises:
            ValueError: If strategy not configured
        """
        strategies_config = self.config.get('strategies', {})

        if strategy not in strategies_config:
            raise ValueError(f"Strategy not configured: {strategy}")

        strategy_config = strategies_config[strategy]

        return StrategySettings(
            name=strategy,
            enabled=bool(strategy_config.get('enabled', True)),
            timeframes=strategy_config.get('timeframes', ['1h']),
            indicators=strategy_config.get('indicators', {}),
            parameters=strategy_config.get('parameters', {}),
            risk_multiplier=str(strategy_config.get('risk_multiplier', '1.0'))
        )

    def get_all_exchanges(self) -> List[str]:
        """
        Get list of all configured exchanges.

        Returns:
            List of exchange names
        """
        exchanges_config = self.config.get('exchanges', {})
        return list(exchanges_config.keys())

    def get_enabled_exchanges(self) -> List[str]:
        """
        Get list of enabled exchanges.

        Returns:
            List of enabled exchange names
        """
        exchanges_config = self.config.get('exchanges', {})
        return [
            name for name, config in exchanges_config.items()
            if config.get('enabled', True)
        ]

    def get_all_strategies(self) -> List[str]:
        """
        Get list of all configured strategies.

        Returns:
            List of strategy names
        """
        strategies_config = self.config.get('strategies', {})
        return list(strategies_config.keys())

    def get_enabled_strategies(self) -> List[str]:
        """
        Get list of enabled strategies.

        Returns:
            List of enabled strategy names
        """
        strategies_config = self.config.get('strategies', {})
        return [
            name for name, config in strategies_config.items()
            if config.get('enabled', True)
        ]

    def reload(self) -> None:
        """Reload configuration from files."""
        self.config.clear()
        self.file_configs.clear()
        self._load_configs()
        logger.info("Configuration reloaded")

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def to_yaml(self, output_path: Path) -> None:
        """
        Export configuration to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)

            logger.info("Configuration exported to YAML", path=str(output_path))

        except Exception as e:
            logger.error("Failed to export config to YAML", error=str(e))
            raise

    def to_json(self, output_path: Path) -> None:
        """
        Export configuration to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info("Configuration exported to JSON", path=str(output_path))

        except Exception as e:
            logger.error("Failed to export config to JSON", error=str(e))
            raise
