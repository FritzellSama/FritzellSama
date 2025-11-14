"""Strategy Factory for Dynamic Strategy Instantiation.

Creates and configures trading strategy instances dynamically based on
configuration and runtime parameters.

Performance Target: <100ms instantiation per strategy
Capital Allocation: N/A (factory module)
Risk: Configuration errors, invalid strategy types
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Type
from datetime import datetime, timezone
from enum import Enum
import importlib
import inspect

from structlog import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Available strategy types."""
    # High Frequency
    QUEUE_RACING = "queue_racing"
    SCALPING = "scalping"
    REBATE_CAPTURE = "rebate_capture"
    TICK_TRADING = "tick_trading"

    # Market Making
    MARKET_MAKING = "market_making"
    QUOTE_GENERATION = "quote_generation"

    # Mean Reversion
    RSI_REVERSION = "rsi_reversion"
    BOLLINGER_REVERSION = "bollinger_reversion"

    # Statistical Arbitrage
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARB = "statistical_arb"

    # Options
    OPTIONS_SPREADS = "options_spreads"

    # Momentum
    MOMENTUM = "momentum"
    TREND_FOLLOWING = "trend_following"


class StrategyFactory:
    """Factory for creating trading strategy instances.

    Dynamically creates and configures strategy instances based on
    strategy type and configuration parameters.

    Key Features:
    - Dynamic strategy loading
    - Configuration validation
    - Dependency injection
    - Strategy registry
    - Error handling and fallbacks

    Attributes:
        config: Global strategy configuration
        registry: Registered strategy classes
        instances: Created strategy instances
        risk_manager: Shared risk manager

    Example:
        >>> factory = StrategyFactory(config, risk_manager)
        >>> strategy = factory.create_strategy('queue_racing', strategy_config)
        >>> strategies = factory.create_multiple(['scalping', 'rsi_reversion'])
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize strategy factory.

        Args:
            config: Global configuration dictionary
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.risk_manager = risk_manager

        # Strategy registry (maps strategy type to module path and class)
        self.registry: Dict[str, Dict[str, str]] = self._initialize_registry()

        # Created instances cache
        self.instances: Dict[str, Any] = {}

        # Strategy metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Creation statistics
        self.strategies_created: int = 0
        self.creation_errors: int = 0

        logger.info(
            "strategy_factory_initialized",
            registered_strategies=len(self.registry),
            risk_manager_enabled=risk_manager is not None
        )

    def _initialize_registry(self) -> Dict[str, Dict[str, str]]:
        """Initialize strategy registry.

        Returns:
            Registry dictionary mapping strategy types to module paths
        """
        return {
            # High Frequency Strategies
            'queue_racing': {
                'module': 'quantum_trader.strategies.high_frequency.queue_racing',
                'class': 'QueueRacingStrategy'
            },
            'scalping': {
                'module': 'quantum_trader.strategies.high_frequency.scalping',
                'class': 'ScalpingStrategy'
            },
            'rebate_capture': {
                'module': 'quantum_trader.strategies.high_frequency.rebate_capture',
                'class': 'RestateCaptureStrategy'
            },
            'tick_trading': {
                'module': 'quantum_trader.strategies.high_frequency.tick_trading',
                'class': 'TickTradingStrategy'
            },

            # Market Making Strategies
            'quote_generation': {
                'module': 'quantum_trader.strategies.market_making.quote_generator',
                'class': 'QuoteGenerator'
            },

            # Mean Reversion Strategies
            'rsi_reversion': {
                'module': 'quantum_trader.strategies.mean_reversion.rsi_reversion',
                'class': 'RSIReversionStrategy'
            },

            # Statistical Arbitrage
            'statistical_arb': {
                'module': 'quantum_trader.strategies.arbitrage.statistical_arb',
                'class': 'StatisticalArbStrategy'
            },

            # Options Strategies
            'options_spreads': {
                'module': 'quantum_trader.strategies.options.spreads',
                'class': 'OptionsSpreadsStrategy'
            }
        }

    def create_strategy(
        self,
        strategy_type: str,
        strategy_config: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> Any:
        """Create a strategy instance.

        Args:
            strategy_type: Type of strategy to create
            strategy_config: Strategy-specific configuration
            cache: Whether to cache the created instance

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type unknown or creation fails
        """
        try:
            # Check if already cached
            if cache and strategy_type in self.instances:
                logger.debug("strategy_instance_cached", strategy_type=strategy_type)
                return self.instances[strategy_type]

            # Validate strategy type
            if strategy_type not in self.registry:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

            # Get registry info
            registry_info = self.registry[strategy_type]

            # Load strategy class
            strategy_class = self._load_strategy_class(
                registry_info['module'],
                registry_info['class']
            )

            # Merge configurations
            config = self._merge_config(strategy_type, strategy_config)

            # Validate configuration
            self._validate_strategy_config(strategy_type, config)

            # Create instance
            instance = self._instantiate_strategy(strategy_class, config)

            # Cache if requested
            if cache:
                self.instances[strategy_type] = instance

            # Store metadata
            self.metadata[strategy_type] = {
                'created_at': datetime.now(timezone.utc),
                'config': config,
                'class_name': registry_info['class'],
                'module': registry_info['module']
            }

            self.strategies_created += 1

            logger.info(
                "strategy_created",
                strategy_type=strategy_type,
                class_name=registry_info['class'],
                cached=cache
            )

            return instance

        except Exception as e:
            self.creation_errors += 1
            logger.error(
                "strategy_creation_failed",
                strategy_type=strategy_type,
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _load_strategy_class(self, module_path: str, class_name: str) -> Type:
        """Dynamically load strategy class.

        Args:
            module_path: Python module path
            class_name: Class name within module

        Returns:
            Strategy class

        Raises:
            ImportError: If module or class not found
        """
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)

            logger.debug(
                "strategy_class_loaded",
                module=module_path,
                class_name=class_name
            )

            return strategy_class

        except ImportError as e:
            logger.error(
                "strategy_module_import_failed",
                module=module_path,
                error=str(e)
            )
            raise

        except AttributeError as e:
            logger.error(
                "strategy_class_not_found",
                module=module_path,
                class_name=class_name,
                error=str(e)
            )
            raise

    def _merge_config(
        self,
        strategy_type: str,
        strategy_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge global and strategy-specific configurations.

        Args:
            strategy_type: Strategy type
            strategy_config: Strategy-specific config

        Returns:
            Merged configuration dictionary
        """
        # Start with global config
        merged = self.config.get('global', {}).copy()

        # Add strategy-type specific config from global
        type_config = self.config.get(strategy_type, {})
        merged.update(type_config)

        # Override with provided strategy config
        if strategy_config:
            merged.update(strategy_config)

        return merged

    def _validate_strategy_config(
        self,
        strategy_type: str,
        config: Dict[str, Any]
    ) -> None:
        """Validate strategy configuration.

        Args:
            strategy_type: Strategy type
            config: Configuration to validate

        Raises:
            ValueError: If configuration invalid
        """
        # Check for required global parameters
        if 'enabled' in config and not config['enabled']:
            raise ValueError(f"Strategy {strategy_type} is disabled in configuration")

        # Strategy-specific validation
        if strategy_type in ['queue_racing', 'scalping', 'rebate_capture']:
            # HFT strategies need specific params
            required = ['max_holding_time_ms', 'min_spread_bps']
            for param in required:
                if param not in config:
                    logger.warning(
                        "missing_hft_parameter",
                        strategy_type=strategy_type,
                        parameter=param
                    )

        elif strategy_type == 'rsi_reversion':
            # RSI strategy needs RSI parameters
            required = ['rsi_period', 'oversold_threshold', 'overbought_threshold']
            for param in required:
                if param not in config:
                    raise ValueError(f"Missing required parameter for {strategy_type}: {param}")

    def _instantiate_strategy(self, strategy_class: Type, config: Dict[str, Any]) -> Any:
        """Instantiate strategy with proper dependency injection.

        Args:
            strategy_class: Strategy class to instantiate
            config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            TypeError: If instantiation fails
        """
        try:
            # Check class signature
            sig = inspect.signature(strategy_class.__init__)
            params = list(sig.parameters.keys())

            # Prepare constructor arguments
            kwargs = {'config': config}

            # Add risk_manager if required
            if 'risk_manager' in params:
                kwargs['risk_manager'] = self.risk_manager

            # Create instance
            instance = strategy_class(**kwargs)

            return instance

        except TypeError as e:
            logger.error(
                "strategy_instantiation_failed",
                class_name=strategy_class.__name__,
                error=str(e)
            )
            raise

    def create_multiple(
        self,
        strategy_types: List[str],
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Create multiple strategy instances.

        Args:
            strategy_types: List of strategy types to create
            configs: Optional dict of strategy-specific configs

        Returns:
            Dictionary mapping strategy types to instances

        Raises:
            ValueError: If any strategy creation fails
        """
        instances = {}
        configs = configs or {}

        for strategy_type in strategy_types:
            try:
                strategy_config = configs.get(strategy_type)
                instance = self.create_strategy(strategy_type, strategy_config)
                instances[strategy_type] = instance

            except Exception as e:
                logger.error(
                    "multi_strategy_creation_failed",
                    strategy_type=strategy_type,
                    error=str(e)
                )
                # Continue with other strategies
                continue

        logger.info(
            "multiple_strategies_created",
            requested=len(strategy_types),
            created=len(instances),
            failed=len(strategy_types) - len(instances)
        )

        return instances

    def get_strategy(self, strategy_type: str) -> Optional[Any]:
        """Get cached strategy instance.

        Args:
            strategy_type: Strategy type

        Returns:
            Strategy instance or None if not cached
        """
        return self.instances.get(strategy_type)

    def list_available_strategies(self) -> List[str]:
        """List all available strategy types.

        Returns:
            List of strategy type identifiers
        """
        return list(self.registry.keys())

    def get_strategy_info(self, strategy_type: str) -> Optional[Dict[str, Any]]:
        """Get information about a strategy type.

        Args:
            strategy_type: Strategy type

        Returns:
            Dictionary with strategy information or None if unknown
        """
        if strategy_type not in self.registry:
            return None

        registry_info = self.registry[strategy_type]
        metadata = self.metadata.get(strategy_type, {})

        return {
            'strategy_type': strategy_type,
            'module': registry_info['module'],
            'class': registry_info['class'],
            'instantiated': strategy_type in self.instances,
            'created_at': metadata.get('created_at'),
            'config': metadata.get('config', {})
        }

    def clear_cache(self) -> None:
        """Clear all cached strategy instances."""
        count = len(self.instances)
        self.instances.clear()
        self.metadata.clear()

        logger.info("strategy_cache_cleared", instances_removed=count)

    def reload_strategy(self, strategy_type: str) -> Any:
        """Reload a strategy instance with fresh configuration.

        Args:
            strategy_type: Strategy type to reload

        Returns:
            Reloaded strategy instance

        Raises:
            ValueError: If strategy type unknown
        """
        # Remove from cache
        if strategy_type in self.instances:
            del self.instances[strategy_type]

        if strategy_type in self.metadata:
            old_config = self.metadata[strategy_type].get('config', {})
        else:
            old_config = None

        # Create new instance
        instance = self.create_strategy(strategy_type, old_config, cache=True)

        logger.info("strategy_reloaded", strategy_type=strategy_type)

        return instance

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get factory performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'strategies_created': self.strategies_created,
            'creation_errors': self.creation_errors,
            'cached_instances': len(self.instances),
            'registered_strategies': len(self.registry),
            'success_rate': (
                self.strategies_created / (self.strategies_created + self.creation_errors)
                if (self.strategies_created + self.creation_errors) > 0
                else 0
            )
        }
