"""
Indicator Factory - Centralized creation and management of technical indicators.

This module provides a factory pattern for creating and managing all technical indicators
used in the Quantum Trader AI system.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, Type, Protocol, List
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class IndicatorType(Enum):
    """Supported technical indicator types."""
    # Trend indicators
    MOVING_AVERAGE = "moving_average"
    ICHIMOKU = "ichimoku"

    # Momentum indicators
    MACD = "macd"
    RSI = "rsi"
    STOCHASTIC = "stochastic"

    # Volatility indicators
    BOLLINGER_BANDS = "bollinger_bands"
    ATR = "atr"
    KELTNER_CHANNELS = "keltner_channels"
    GARCH = "garch"

    # Volume indicators
    OBV = "obv"
    MFI = "mfi"
    VWAP = "vwap"

    # Market structure
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_PROFILE = "market_profile"

    # Custom indicators
    ORDER_FLOW = "order_flow"
    MICROSTRUCTURE = "microstructure"


class IndicatorProtocol(Protocol):
    """Protocol that all indicators must implement."""

    async def calculate(self, data: pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Calculate indicator values."""
        ...

    def _validate_config(self) -> None:
        """Validate indicator configuration."""
        ...

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data."""
        ...


class IndicatorFactory:
    """
    Factory for creating technical indicator instances.

    This factory manages indicator instantiation, configuration validation,
    and provides a centralized registry of all available indicators.

    Attributes:
        config: Global configuration dictionary
        indicator_registry: Registry mapping indicator types to classes
        indicator_cache: Cache of created indicator instances

    Example:
        >>> factory = IndicatorFactory(config)
        >>> macd = await factory.create_indicator(IndicatorType.MACD, macd_config)
        >>> result = await macd.calculate(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize indicator factory.

        Args:
            config: Global configuration dictionary containing:
                - indicator_config: Configuration for all indicators
                - enable_caching: Whether to cache indicator instances
                - max_cache_size: Maximum number of cached instances

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.enable_caching: bool = config.get("enable_caching", True)
        self.max_cache_size: int = int(config.get("max_cache_size", 100))

        self.indicator_registry: Dict[IndicatorType, Type] = {}
        self.indicator_cache: Dict[str, IndicatorProtocol] = {}

        self._register_indicators()

        logger.info(
            "Indicator factory initialized",
            registered_indicators=len(self.indicator_registry),
            caching_enabled=self.enable_caching
        )

    def _validate_config(self) -> None:
        """Validate factory configuration."""
        if "max_cache_size" in self.config:
            if int(self.config["max_cache_size"]) < 0:
                raise ValueError("max_cache_size must be non-negative")

        logger.debug("Indicator factory configuration validated")

    def _register_indicators(self) -> None:
        """Register all available indicator classes."""
        try:
            # Import indicator classes lazily to avoid circular dependencies
            from quantum_trader.indicators.momentum.macd import MACDIndicator
            from quantum_trader.indicators.trend.ichimoku import IchimokuIndicator
            from quantum_trader.indicators.volatility.garch import GARCHVolatilityIndicator
            from quantum_trader.indicators.volatility.keltner_channels import KeltnerChannelsIndicator
            from quantum_trader.indicators.volume.obv import OBVIndicator
            from quantum_trader.indicators.volume.mfi import MFIIndicator
            from quantum_trader.indicators.trend.moving_averages import MovingAverageIndicator
            from quantum_trader.indicators.market_structure.market_profile import MarketProfileIndicator
            from quantum_trader.indicators.custom.order_flow import OrderFlowIndicator
            from quantum_trader.indicators.custom.microstructure import MicrostructureIndicator

            # Register indicators
            self.indicator_registry[IndicatorType.MACD] = MACDIndicator
            self.indicator_registry[IndicatorType.ICHIMOKU] = IchimokuIndicator
            self.indicator_registry[IndicatorType.GARCH] = GARCHVolatilityIndicator
            self.indicator_registry[IndicatorType.KELTNER_CHANNELS] = KeltnerChannelsIndicator
            self.indicator_registry[IndicatorType.OBV] = OBVIndicator
            self.indicator_registry[IndicatorType.MFI] = MFIIndicator
            self.indicator_registry[IndicatorType.MOVING_AVERAGE] = MovingAverageIndicator
            self.indicator_registry[IndicatorType.MARKET_PROFILE] = MarketProfileIndicator
            self.indicator_registry[IndicatorType.ORDER_FLOW] = OrderFlowIndicator
            self.indicator_registry[IndicatorType.MICROSTRUCTURE] = MicrostructureIndicator

            logger.info(
                "Indicators registered",
                count=len(self.indicator_registry)
            )

        except ImportError as e:
            logger.warning(
                "Some indicators could not be imported",
                error=str(e)
            )

    async def create_indicator(
        self,
        indicator_type: IndicatorType,
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> IndicatorProtocol:
        """
        Create an indicator instance.

        Args:
            indicator_type: Type of indicator to create
            indicator_config: Configuration for the indicator

        Returns:
            Initialized indicator instance

        Raises:
            ValueError: If indicator type is not registered
            TypeError: If indicator configuration is invalid
        """
        try:
            # Check if indicator is registered
            if indicator_type not in self.indicator_registry:
                raise ValueError(f"Indicator type not registered: {indicator_type}")

            # Generate cache key
            cache_key = self._generate_cache_key(indicator_type, indicator_config)

            # Check cache if enabled
            if self.enable_caching and cache_key in self.indicator_cache:
                logger.debug(
                    "Returning cached indicator",
                    indicator_type=indicator_type.value
                )
                return self.indicator_cache[cache_key]

            # Get indicator class
            indicator_class = self.indicator_registry[indicator_type]

            # Merge with global config if available
            config = self._merge_config(indicator_type, indicator_config)

            # Create indicator instance
            indicator = indicator_class(config)

            # Cache if enabled
            if self.enable_caching:
                await self._cache_indicator(cache_key, indicator)

            logger.info(
                "Indicator created",
                indicator_type=indicator_type.value,
                cached=self.enable_caching
            )

            return indicator

        except Exception as e:
            logger.error(
                "Failed to create indicator",
                indicator_type=indicator_type.value,
                error=str(e)
            )
            raise

    async def create_multiple_indicators(
        self,
        indicator_configs: List[Dict[str, Any]]
    ) -> Dict[str, IndicatorProtocol]:
        """
        Create multiple indicators concurrently.

        Args:
            indicator_configs: List of configurations, each containing:
                - type: IndicatorType
                - config: Indicator-specific configuration
                - name: Optional custom name for the indicator

        Returns:
            Dictionary mapping indicator names to instances

        Raises:
            ValueError: If any indicator configuration is invalid
        """
        try:
            tasks = []
            names = []

            for idx, config in enumerate(indicator_configs):
                indicator_type = IndicatorType(config["type"])
                indicator_config = config.get("config", {})
                name = config.get("name", f"{indicator_type.value}_{idx}")

                tasks.append(self.create_indicator(indicator_type, indicator_config))
                names.append(name)

            # Create all indicators concurrently
            indicators = await asyncio.gather(*tasks, return_exceptions=True)

            # Build result dictionary
            result = {}
            for name, indicator in zip(names, indicators):
                if isinstance(indicator, Exception):
                    logger.error(
                        "Failed to create indicator",
                        name=name,
                        error=str(indicator)
                    )
                    continue
                result[name] = indicator

            logger.info(
                "Multiple indicators created",
                requested=len(indicator_configs),
                created=len(result)
            )

            return result

        except Exception as e:
            logger.error("Failed to create multiple indicators", error=str(e))
            raise

    def _merge_config(
        self,
        indicator_type: IndicatorType,
        indicator_config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge indicator-specific config with global config.

        Args:
            indicator_type: Type of indicator
            indicator_config: Indicator-specific configuration

        Returns:
            Merged configuration dictionary
        """
        # Start with global indicator config if available
        config = {}
        if "indicator_config" in self.config:
            global_config = self.config["indicator_config"].get(indicator_type.value, {})
            config.update(global_config)

        # Override with indicator-specific config
        if indicator_config:
            config.update(indicator_config)

        return config

    async def _cache_indicator(
        self,
        cache_key: str,
        indicator: IndicatorProtocol
    ) -> None:
        """
        Cache an indicator instance.

        Args:
            cache_key: Unique cache key
            indicator: Indicator instance to cache
        """
        try:
            # Check cache size limit
            if len(self.indicator_cache) >= self.max_cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.indicator_cache))
                del self.indicator_cache[oldest_key]
                logger.debug("Cache full, removed oldest indicator", key=oldest_key)

            self.indicator_cache[cache_key] = indicator

        except Exception as e:
            logger.warning("Failed to cache indicator", error=str(e))

    def _generate_cache_key(
        self,
        indicator_type: IndicatorType,
        config: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate a unique cache key for an indicator.

        Args:
            indicator_type: Type of indicator
            config: Indicator configuration

        Returns:
            Unique cache key string
        """
        import hashlib
        import json

        config_str = json.dumps(config or {}, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        return f"{indicator_type.value}:{config_hash}"

    async def clear_cache(self) -> None:
        """Clear all cached indicator instances."""
        cache_size = len(self.indicator_cache)
        self.indicator_cache.clear()
        logger.info("Indicator cache cleared", cleared_count=cache_size)

    async def get_registered_indicators(self) -> List[IndicatorType]:
        """
        Get list of all registered indicator types.

        Returns:
            List of registered indicator types
        """
        return list(self.indicator_registry.keys())

    async def is_indicator_registered(self, indicator_type: IndicatorType) -> bool:
        """
        Check if an indicator type is registered.

        Args:
            indicator_type: Indicator type to check

        Returns:
            True if registered, False otherwise
        """
        return indicator_type in self.indicator_registry

    async def register_custom_indicator(
        self,
        indicator_type: IndicatorType,
        indicator_class: Type[IndicatorProtocol]
    ) -> None:
        """
        Register a custom indicator class.

        Args:
            indicator_type: Type identifier for the indicator
            indicator_class: Indicator class implementing IndicatorProtocol

        Raises:
            ValueError: If indicator type already registered
            TypeError: If class doesn't implement required protocol
        """
        try:
            if indicator_type in self.indicator_registry:
                raise ValueError(
                    f"Indicator type already registered: {indicator_type}"
                )

            # Verify class implements required methods
            required_methods = ["calculate", "_validate_config", "_validate_data"]
            for method in required_methods:
                if not hasattr(indicator_class, method):
                    raise TypeError(
                        f"Indicator class must implement {method} method"
                    )

            self.indicator_registry[indicator_type] = indicator_class

            logger.info(
                "Custom indicator registered",
                indicator_type=indicator_type.value
            )

        except Exception as e:
            logger.error(
                "Failed to register custom indicator",
                indicator_type=indicator_type.value,
                error=str(e)
            )
            raise

    async def unregister_indicator(self, indicator_type: IndicatorType) -> None:
        """
        Unregister an indicator type.

        Args:
            indicator_type: Type of indicator to unregister
        """
        if indicator_type in self.indicator_registry:
            del self.indicator_registry[indicator_type]
            logger.info(
                "Indicator unregistered",
                indicator_type=indicator_type.value
            )

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "enabled": self.enable_caching,
            "size": len(self.indicator_cache),
            "max_size": self.max_cache_size,
            "utilization": len(self.indicator_cache) / self.max_cache_size if self.max_cache_size > 0 else 0
        }


async def create_indicator_factory(config: Dict[str, Any]) -> IndicatorFactory:
    """
    Factory function to create IndicatorFactory instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized IndicatorFactory
    """
    return IndicatorFactory(config)
