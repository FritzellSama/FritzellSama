"""
Constants - System-wide constant definitions.

This module defines all constant values used across the trading system.
All values are loaded from configuration files to ensure zero hardcoding.
"""

from decimal import Decimal
from typing import Dict, Any, Final
from pathlib import Path
import os
from structlog import get_logger

logger = get_logger(__name__)


class TradingConstants:
    """
    Trading system constants loaded from configuration.

    All values are loaded from environment variables or config files.
    No hardcoded values are permitted in production code.

    Example:
        >>> constants = TradingConstants.from_config(config)
        >>> max_position = constants.MAX_POSITION_SIZE
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize constants from configuration.

        Args:
            config: Configuration dictionary containing all constant values

        Raises:
            ValueError: If required constants missing from config
        """
        self._config = config
        self._validate_config()

        # Precision and rounding
        self.DECIMAL_PRECISION: Final[int] = int(config['decimal_precision'])
        self.PRICE_PRECISION: Final[int] = int(config['price_precision'])
        self.QUANTITY_PRECISION: Final[int] = int(config['quantity_precision'])

        # Time constants
        self.TICK_INTERVAL_MS: Final[int] = int(config['tick_interval_ms'])
        self.ORDER_TIMEOUT_SECONDS: Final[int] = int(config['order_timeout_seconds'])
        self.HEARTBEAT_INTERVAL_SECONDS: Final[int] = int(config['heartbeat_interval_seconds'])
        self.RECONNECT_DELAY_SECONDS: Final[int] = int(config['reconnect_delay_seconds'])

        # Risk limits (all from config)
        self.MAX_POSITION_SIZE: Final[Decimal] = Decimal(str(config['max_position_size']))
        self.MAX_LEVERAGE: Final[Decimal] = Decimal(str(config['max_leverage']))
        self.MAX_DRAWDOWN_PERCENT: Final[Decimal] = Decimal(str(config['max_drawdown_percent']))
        self.MAX_DAILY_LOSS: Final[Decimal] = Decimal(str(config['max_daily_loss']))
        self.POSITION_SIZE_PERCENT: Final[Decimal] = Decimal(str(config['position_size_percent']))

        # Fee constants
        self.DEFAULT_MAKER_FEE: Final[Decimal] = Decimal(str(config['default_maker_fee']))
        self.DEFAULT_TAKER_FEE: Final[Decimal] = Decimal(str(config['default_taker_fee']))

        # Retry configuration
        self.MAX_RETRIES: Final[int] = int(config['max_retries'])
        self.RETRY_BACKOFF_BASE: Final[Decimal] = Decimal(str(config['retry_backoff_base']))
        self.RETRY_BACKOFF_MAX: Final[int] = int(config['retry_backoff_max'])

        # Data limits
        self.MAX_CANDLES_PER_REQUEST: Final[int] = int(config['max_candles_per_request'])
        self.MAX_ORDERBOOK_DEPTH: Final[int] = int(config['max_orderbook_depth'])
        self.MAX_TRADES_HISTORY: Final[int] = int(config['max_trades_history'])

        # Websocket configuration
        self.WS_PING_INTERVAL: Final[int] = int(config['ws_ping_interval'])
        self.WS_PING_TIMEOUT: Final[int] = int(config['ws_ping_timeout'])
        self.WS_MAX_MESSAGE_SIZE: Final[int] = int(config['ws_max_message_size'])

        # Database configuration
        self.DB_POOL_SIZE: Final[int] = int(config['db_pool_size'])
        self.DB_MAX_OVERFLOW: Final[int] = int(config['db_max_overflow'])
        self.DB_POOL_TIMEOUT: Final[int] = int(config['db_pool_timeout'])

        # Cache configuration
        self.CACHE_TTL_SECONDS: Final[int] = int(config['cache_ttl_seconds'])
        self.CACHE_MAX_SIZE: Final[int] = int(config['cache_max_size'])

        # Performance thresholds
        self.MIN_SHARPE_RATIO: Final[Decimal] = Decimal(str(config['min_sharpe_ratio']))
        self.MIN_WIN_RATE: Final[Decimal] = Decimal(str(config['min_win_rate']))
        self.MIN_PROFIT_FACTOR: Final[Decimal] = Decimal(str(config['min_profit_factor']))

        logger.info(
            "TradingConstants initialized",
            precision=self.DECIMAL_PRECISION,
            max_position=str(self.MAX_POSITION_SIZE)
        )

    def _validate_config(self) -> None:
        """Validate that all required constants are in config."""
        required_keys = [
            'decimal_precision',
            'price_precision',
            'quantity_precision',
            'tick_interval_ms',
            'order_timeout_seconds',
            'heartbeat_interval_seconds',
            'reconnect_delay_seconds',
            'max_position_size',
            'max_leverage',
            'max_drawdown_percent',
            'max_daily_loss',
            'position_size_percent',
            'default_maker_fee',
            'default_taker_fee',
            'max_retries',
            'retry_backoff_base',
            'retry_backoff_max',
            'max_candles_per_request',
            'max_orderbook_depth',
            'max_trades_history',
            'ws_ping_interval',
            'ws_ping_timeout',
            'ws_max_message_size',
            'db_pool_size',
            'db_max_overflow',
            'db_pool_timeout',
            'cache_ttl_seconds',
            'cache_max_size',
            'min_sharpe_ratio',
            'min_win_rate',
            'min_profit_factor'
        ]

        missing = [key for key in required_keys if key not in self._config]
        if missing:
            raise ValueError(f"Missing required constant keys in config: {missing}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TradingConstants':
        """
        Create TradingConstants from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            TradingConstants instance
        """
        return cls(config)

    @classmethod
    def from_env(cls, env_prefix: str = 'QUANTUM_TRADER') -> 'TradingConstants':
        """
        Create TradingConstants from environment variables.

        Args:
            env_prefix: Prefix for environment variables

        Returns:
            TradingConstants instance
        """
        config = {}

        # Map environment variables to config keys
        env_mappings = {
            f'{env_prefix}_DECIMAL_PRECISION': 'decimal_precision',
            f'{env_prefix}_PRICE_PRECISION': 'price_precision',
            f'{env_prefix}_QUANTITY_PRECISION': 'quantity_precision',
            f'{env_prefix}_TICK_INTERVAL_MS': 'tick_interval_ms',
            f'{env_prefix}_ORDER_TIMEOUT_SECONDS': 'order_timeout_seconds',
            f'{env_prefix}_HEARTBEAT_INTERVAL_SECONDS': 'heartbeat_interval_seconds',
            f'{env_prefix}_RECONNECT_DELAY_SECONDS': 'reconnect_delay_seconds',
            f'{env_prefix}_MAX_POSITION_SIZE': 'max_position_size',
            f'{env_prefix}_MAX_LEVERAGE': 'max_leverage',
            f'{env_prefix}_MAX_DRAWDOWN_PERCENT': 'max_drawdown_percent',
            f'{env_prefix}_MAX_DAILY_LOSS': 'max_daily_loss',
            f'{env_prefix}_POSITION_SIZE_PERCENT': 'position_size_percent',
            f'{env_prefix}_DEFAULT_MAKER_FEE': 'default_maker_fee',
            f'{env_prefix}_DEFAULT_TAKER_FEE': 'default_taker_fee',
            f'{env_prefix}_MAX_RETRIES': 'max_retries',
            f'{env_prefix}_RETRY_BACKOFF_BASE': 'retry_backoff_base',
            f'{env_prefix}_RETRY_BACKOFF_MAX': 'retry_backoff_max',
            f'{env_prefix}_MAX_CANDLES_PER_REQUEST': 'max_candles_per_request',
            f'{env_prefix}_MAX_ORDERBOOK_DEPTH': 'max_orderbook_depth',
            f'{env_prefix}_MAX_TRADES_HISTORY': 'max_trades_history',
            f'{env_prefix}_WS_PING_INTERVAL': 'ws_ping_interval',
            f'{env_prefix}_WS_PING_TIMEOUT': 'ws_ping_timeout',
            f'{env_prefix}_WS_MAX_MESSAGE_SIZE': 'ws_max_message_size',
            f'{env_prefix}_DB_POOL_SIZE': 'db_pool_size',
            f'{env_prefix}_DB_MAX_OVERFLOW': 'db_max_overflow',
            f'{env_prefix}_DB_POOL_TIMEOUT': 'db_pool_timeout',
            f'{env_prefix}_CACHE_TTL_SECONDS': 'cache_ttl_seconds',
            f'{env_prefix}_CACHE_MAX_SIZE': 'cache_max_size',
            f'{env_prefix}_MIN_SHARPE_RATIO': 'min_sharpe_ratio',
            f'{env_prefix}_MIN_WIN_RATE': 'min_win_rate',
            f'{env_prefix}_MIN_PROFIT_FACTOR': 'min_profit_factor'
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config[config_key] = value

        return cls(config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export constants as dictionary.

        Returns:
            Dictionary of all constants
        """
        return {
            'decimal_precision': self.DECIMAL_PRECISION,
            'price_precision': self.PRICE_PRECISION,
            'quantity_precision': self.QUANTITY_PRECISION,
            'tick_interval_ms': self.TICK_INTERVAL_MS,
            'order_timeout_seconds': self.ORDER_TIMEOUT_SECONDS,
            'heartbeat_interval_seconds': self.HEARTBEAT_INTERVAL_SECONDS,
            'reconnect_delay_seconds': self.RECONNECT_DELAY_SECONDS,
            'max_position_size': str(self.MAX_POSITION_SIZE),
            'max_leverage': str(self.MAX_LEVERAGE),
            'max_drawdown_percent': str(self.MAX_DRAWDOWN_PERCENT),
            'max_daily_loss': str(self.MAX_DAILY_LOSS),
            'position_size_percent': str(self.POSITION_SIZE_PERCENT),
            'default_maker_fee': str(self.DEFAULT_MAKER_FEE),
            'default_taker_fee': str(self.DEFAULT_TAKER_FEE),
            'max_retries': self.MAX_RETRIES,
            'retry_backoff_base': str(self.RETRY_BACKOFF_BASE),
            'retry_backoff_max': self.RETRY_BACKOFF_MAX,
            'max_candles_per_request': self.MAX_CANDLES_PER_REQUEST,
            'max_orderbook_depth': self.MAX_ORDERBOOK_DEPTH,
            'max_trades_history': self.MAX_TRADES_HISTORY,
            'ws_ping_interval': self.WS_PING_INTERVAL,
            'ws_ping_timeout': self.WS_PING_TIMEOUT,
            'ws_max_message_size': self.WS_MAX_MESSAGE_SIZE,
            'db_pool_size': self.DB_POOL_SIZE,
            'db_max_overflow': self.DB_MAX_OVERFLOW,
            'db_pool_timeout': self.DB_POOL_TIMEOUT,
            'cache_ttl_seconds': self.CACHE_TTL_SECONDS,
            'cache_max_size': self.CACHE_MAX_SIZE,
            'min_sharpe_ratio': str(self.MIN_SHARPE_RATIO),
            'min_win_rate': str(self.MIN_WIN_RATE),
            'min_profit_factor': str(self.MIN_PROFIT_FACTOR)
        }


# Exchange-specific constants
class ExchangeConstants:
    """Exchange-specific constant definitions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize exchange constants.

        Args:
            config: Exchange configuration dictionary
        """
        self._config = config

        # Rate limits (from config)
        self.RATE_LIMIT_REQUESTS: Final[int] = int(config['rate_limit_requests'])
        self.RATE_LIMIT_WINDOW_SECONDS: Final[int] = int(config['rate_limit_window_seconds'])

        # Order limits (from config)
        self.MIN_ORDER_SIZE: Final[Decimal] = Decimal(str(config['min_order_size']))
        self.MAX_ORDER_SIZE: Final[Decimal] = Decimal(str(config['max_order_size']))
        self.MIN_NOTIONAL: Final[Decimal] = Decimal(str(config['min_notional']))

        # API endpoints (from config)
        self.REST_API_URL: Final[str] = config['rest_api_url']
        self.WS_API_URL: Final[str] = config['ws_api_url']

        logger.info(
            "ExchangeConstants initialized",
            rate_limit=self.RATE_LIMIT_REQUESTS,
            rest_url=self.REST_API_URL
        )


# Symbol-specific constants
class SymbolConstants:
    """Symbol-specific trading constants."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize symbol constants.

        Args:
            config: Symbol configuration dictionary
        """
        self._config = config

        self.TICK_SIZE: Final[Decimal] = Decimal(str(config['tick_size']))
        self.LOT_SIZE: Final[Decimal] = Decimal(str(config['lot_size']))
        self.MIN_QUANTITY: Final[Decimal] = Decimal(str(config['min_quantity']))
        self.MAX_QUANTITY: Final[Decimal] = Decimal(str(config['max_quantity']))
        self.MIN_NOTIONAL: Final[Decimal] = Decimal(str(config['min_notional']))
