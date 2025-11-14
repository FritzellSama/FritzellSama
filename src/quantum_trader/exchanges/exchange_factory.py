"""Exchange factory for creating exchange connectors.

This module provides a factory for instantiating exchange connectors
with proper configuration and dependency injection.
"""

from typing import Dict, Any, Optional, Type
from decimal import Decimal
import os
from structlog import get_logger

logger = get_logger(__name__)


class ExchangeRegistry:
    """Registry for exchange connector classes."""

    _exchanges: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, exchange_class: Type) -> None:
        """Register an exchange connector.

        Args:
            name: Exchange name (lowercase)
            exchange_class: Exchange connector class
        """
        cls._exchanges[name.lower()] = exchange_class
        logger.info("exchange_registered", name=name, class_name=exchange_class.__name__)

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get exchange connector class by name.

        Args:
            name: Exchange name (lowercase)

        Returns:
            Exchange connector class or None
        """
        return cls._exchanges.get(name.lower())

    @classmethod
    def list_exchanges(cls) -> list:
        """Get list of registered exchange names.

        Returns:
            List of exchange names
        """
        return list(cls._exchanges.keys())


class ExchangeFactory:
    """Factory for creating exchange connector instances.

    This factory handles instantiation of exchange connectors with proper
    configuration, error handling, and dependency injection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize exchange factory.

        Args:
            config: Global configuration dictionary
        """
        self.config = config or {}
        self._instances: Dict[str, Any] = {}
        self._load_exchange_modules()

        logger.info("exchange_factory_initialized", config_keys=list(self.config.keys()))

    def _load_exchange_modules(self) -> None:
        """Dynamically load exchange modules to trigger registration."""
        exchange_names = self._get_configured_exchanges()

        for exchange_name in exchange_names:
            try:
                # Import exchange modules to trigger registration
                module_name = f"quantum_trader.exchanges.{exchange_name}"
                __import__(module_name)
                logger.info("exchange_module_loaded", exchange=exchange_name)
            except ImportError as e:
                logger.warning(
                    "exchange_module_not_found",
                    exchange=exchange_name,
                    error=str(e),
                )

    def _get_configured_exchanges(self) -> list:
        """Get list of exchanges from configuration or environment.

        Returns:
            List of exchange names
        """
        # Try config first
        if "exchanges" in self.config:
            return list(self.config["exchanges"].keys())

        # Try environment variable
        exchanges_env = os.getenv("QUANTUM_TRADER_EXCHANGES", "")
        if exchanges_env:
            return [e.strip().lower() for e in exchanges_env.split(",")]

        # Default exchanges
        return ["binance", "bybit", "okx", "kucoin", "bitget"]

    def create(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = False,
        passphrase: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create exchange connector instance.

        Args:
            exchange_name: Name of exchange (e.g., 'binance', 'bybit')
            api_key: API key (loaded from config/env if not provided)
            secret: API secret (loaded from config/env if not provided)
            testnet: Whether to use testnet
            passphrase: API passphrase (for exchanges that require it)
            **kwargs: Additional exchange-specific parameters

        Returns:
            Exchange connector instance

        Raises:
            ValueError: If exchange not found or credentials missing
        """
        exchange_name = exchange_name.lower()

        # Get exchange class
        exchange_class = ExchangeRegistry.get(exchange_name)
        if exchange_class is None:
            available = ExchangeRegistry.list_exchanges()
            raise ValueError(
                f"Exchange '{exchange_name}' not found. "
                f"Available exchanges: {available}"
            )

        # Load credentials from config/environment
        if api_key is None:
            api_key = self._get_credential(exchange_name, "api_key")
        if secret is None:
            secret = self._get_credential(exchange_name, "secret")
        if passphrase is None and exchange_name in ["kucoin"]:
            passphrase = self._get_credential(exchange_name, "passphrase")

        # Validate credentials
        if not api_key or not secret:
            raise ValueError(
                f"API credentials for {exchange_name} not found in config or environment"
            )

        # Build initialization parameters
        init_params: Dict[str, Any] = {
            "api_key": api_key,
            "secret": secret,
            "testnet": testnet,
            **kwargs,
        }

        # Add passphrase if required
        if passphrase:
            init_params["passphrase"] = passphrase

        # Create instance
        try:
            instance = exchange_class(**init_params)
            logger.info(
                "exchange_instance_created",
                exchange=exchange_name,
                testnet=testnet,
            )
            return instance

        except Exception as e:
            logger.error(
                "exchange_creation_failed",
                exchange=exchange_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _get_credential(self, exchange_name: str, credential_type: str) -> Optional[str]:
        """Get credential from config or environment.

        Args:
            exchange_name: Exchange name
            credential_type: Type of credential (api_key, secret, passphrase)

        Returns:
            Credential value or None
        """
        # Try config first
        if "exchanges" in self.config:
            exchange_config = self.config["exchanges"].get(exchange_name, {})
            if credential_type in exchange_config:
                return exchange_config[credential_type]

        # Try environment variable
        env_var_name = f"{exchange_name.upper()}_{credential_type.upper()}"
        return os.getenv(env_var_name)

    def get_or_create(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
        testnet: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Get existing instance or create new one (singleton pattern).

        Args:
            exchange_name: Name of exchange
            api_key: API key
            secret: API secret
            testnet: Whether to use testnet
            **kwargs: Additional parameters

        Returns:
            Exchange connector instance
        """
        cache_key = f"{exchange_name}_{testnet}"

        if cache_key in self._instances:
            logger.debug("exchange_instance_cached", exchange=exchange_name, testnet=testnet)
            return self._instances[cache_key]

        instance = self.create(
            exchange_name=exchange_name,
            api_key=api_key,
            secret=secret,
            testnet=testnet,
            **kwargs,
        )

        self._instances[cache_key] = instance
        return instance

    def create_all(self, testnet: bool = False) -> Dict[str, Any]:
        """Create instances for all configured exchanges.

        Args:
            testnet: Whether to use testnet for all exchanges

        Returns:
            Dictionary mapping exchange name to instance
        """
        instances = {}
        exchange_names = self._get_configured_exchanges()

        for exchange_name in exchange_names:
            try:
                instance = self.create(exchange_name=exchange_name, testnet=testnet)
                instances[exchange_name] = instance
                logger.info(
                    "exchange_created_in_batch",
                    exchange=exchange_name,
                    testnet=testnet,
                )
            except Exception as e:
                logger.error(
                    "exchange_creation_failed_in_batch",
                    exchange=exchange_name,
                    error=str(e),
                )

        return instances

    def clear_cache(self) -> None:
        """Clear cached exchange instances."""
        self._instances.clear()
        logger.info("exchange_cache_cleared")

    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get configuration for specific exchange.

        Args:
            exchange_name: Exchange name

        Returns:
            Exchange configuration dictionary
        """
        if "exchanges" in self.config:
            return self.config["exchanges"].get(exchange_name.lower(), {})

        return {}

    def validate_credentials(self, exchange_name: str) -> bool:
        """Validate that credentials exist for exchange.

        Args:
            exchange_name: Exchange name

        Returns:
            True if credentials found, False otherwise
        """
        api_key = self._get_credential(exchange_name, "api_key")
        secret = self._get_credential(exchange_name, "secret")

        has_credentials = bool(api_key and secret)

        logger.debug(
            "credentials_validated",
            exchange=exchange_name,
            has_credentials=has_credentials,
        )

        return has_credentials

    def list_available_exchanges(self) -> list:
        """List all available exchange names.

        Returns:
            List of exchange names
        """
        return ExchangeRegistry.list_exchanges()


# Global factory instance
_factory: Optional[ExchangeFactory] = None


def get_factory(config: Optional[Dict[str, Any]] = None) -> ExchangeFactory:
    """Get global exchange factory instance.

    Args:
        config: Configuration dictionary (only used on first call)

    Returns:
        ExchangeFactory instance
    """
    global _factory

    if _factory is None:
        _factory = ExchangeFactory(config=config)

    return _factory


def create_exchange(
    exchange_name: str,
    api_key: Optional[str] = None,
    secret: Optional[str] = None,
    testnet: bool = False,
    **kwargs: Any,
) -> Any:
    """Convenience function to create exchange instance.

    Args:
        exchange_name: Exchange name
        api_key: API key
        secret: API secret
        testnet: Whether to use testnet
        **kwargs: Additional parameters

    Returns:
        Exchange connector instance
    """
    factory = get_factory()
    return factory.create(
        exchange_name=exchange_name,
        api_key=api_key,
        secret=secret,
        testnet=testnet,
        **kwargs,
    )
