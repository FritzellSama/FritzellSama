"""
Indicator Manager - Resource and state management for technical indicators.

This module manages the lifecycle, state, and coordination of multiple technical
indicators in a thread-safe, event-driven manner.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Set, Callable
from enum import Enum
from datetime import datetime, timezone
import polars as pl
from structlog import get_logger
from dataclasses import dataclass, field
import threading

from quantum_trader.indicators.indicator_factory import (
    IndicatorFactory,
    IndicatorType,
    IndicatorProtocol
)

logger = get_logger(__name__)


class IndicatorState(Enum):
    """Indicator lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CALCULATING = "calculating"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class IndicatorEvent(Enum):
    """Indicator lifecycle events."""
    INITIALIZED = "initialized"
    CALCULATION_STARTED = "calculation_started"
    CALCULATION_COMPLETED = "calculation_completed"
    CALCULATION_FAILED = "calculation_failed"
    STATE_CHANGED = "state_changed"
    SHUTDOWN_STARTED = "shutdown_started"
    SHUTDOWN_COMPLETED = "shutdown_completed"


@dataclass
class IndicatorMetrics:
    """Performance metrics for an indicator."""
    name: str
    indicator_type: str
    total_calculations: int = 0
    successful_calculations: int = 0
    failed_calculations: int = 0
    total_calculation_time: Decimal = Decimal("0")
    average_calculation_time: Decimal = Decimal("0")
    last_calculation_time: Optional[datetime] = None
    last_error: Optional[str] = None
    state: IndicatorState = IndicatorState.UNINITIALIZED


@dataclass
class ManagedIndicator:
    """Wrapper for managed indicator instances."""
    name: str
    indicator_type: IndicatorType
    indicator: IndicatorProtocol
    state: IndicatorState = IndicatorState.READY
    metrics: IndicatorMetrics = field(default_factory=lambda: None)
    last_result: Optional[pl.DataFrame] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = IndicatorMetrics(
                name=self.name,
                indicator_type=self.indicator_type.value
            )


class IndicatorManager:
    """
    Manages lifecycle, state, and coordination of technical indicators.

    This manager provides:
    - Thread-safe indicator state management
    - Event-driven indicator lifecycle
    - Resource pooling and cleanup
    - Health monitoring
    - Performance metrics tracking

    Attributes:
        config: Configuration dictionary
        factory: IndicatorFactory instance
        indicators: Dictionary of managed indicators
        event_handlers: Event handler callbacks
        state: Manager state
        lock: Thread lock for state management

    Example:
        >>> config = {...}
        >>> manager = IndicatorManager(config)
        >>> await manager.initialize()
        >>> await manager.add_indicator("macd", IndicatorType.MACD, macd_config)
        >>> result = await manager.calculate_indicator("macd", price_data)
        >>> await manager.shutdown()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize indicator manager.

        Args:
            config: Configuration dictionary containing:
                - factory_config: Configuration for indicator factory
                - max_concurrent_calculations: Max concurrent indicator calculations
                - health_check_interval: Interval for health checks (seconds)
                - enable_metrics: Whether to collect performance metrics
                - calculation_timeout: Timeout for indicator calculations (seconds)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.max_concurrent_calculations: int = int(
            config.get("max_concurrent_calculations", 10)
        )
        self.health_check_interval: int = int(
            config.get("health_check_interval", 60)
        )
        self.enable_metrics: bool = config.get("enable_metrics", True)
        self.calculation_timeout: int = int(
            config.get("calculation_timeout", 30)
        )

        # Initialize factory
        factory_config = config.get("factory_config", {})
        self.factory: IndicatorFactory = IndicatorFactory(factory_config)

        # State management
        self.indicators: Dict[str, ManagedIndicator] = {}
        self.event_handlers: Dict[IndicatorEvent, List[Callable]] = {
            event: [] for event in IndicatorEvent
        }
        self.state: IndicatorState = IndicatorState.UNINITIALIZED
        self.lock: threading.Lock = threading.Lock()

        # Async resources
        self._calculation_semaphore: Optional[asyncio.Semaphore] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

        logger.info(
            "Indicator manager created",
            max_concurrent_calculations=self.max_concurrent_calculations,
            metrics_enabled=self.enable_metrics
        )

    def _validate_config(self) -> None:
        """Validate manager configuration."""
        if "max_concurrent_calculations" in self.config:
            if int(self.config["max_concurrent_calculations"]) < 1:
                raise ValueError("max_concurrent_calculations must be positive")

        if "health_check_interval" in self.config:
            if int(self.config["health_check_interval"]) < 1:
                raise ValueError("health_check_interval must be positive")

        if "calculation_timeout" in self.config:
            if int(self.config["calculation_timeout"]) < 1:
                raise ValueError("calculation_timeout must be positive")

        logger.debug("Indicator manager configuration validated")

    async def initialize(self) -> None:
        """
        Initialize the indicator manager.

        Sets up async resources and starts background tasks.
        """
        try:
            with self.lock:
                if self.state != IndicatorState.UNINITIALIZED:
                    logger.warning(
                        "Manager already initialized",
                        current_state=self.state.value
                    )
                    return

                self.state = IndicatorState.INITIALIZING

            # Initialize async resources
            self._calculation_semaphore = asyncio.Semaphore(
                self.max_concurrent_calculations
            )

            # Start health check task
            if self.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

            with self.lock:
                self.state = IndicatorState.READY

            await self._emit_event(IndicatorEvent.INITIALIZED, {})

            logger.info("Indicator manager initialized")

        except Exception as e:
            with self.lock:
                self.state = IndicatorState.ERROR
            logger.error("Failed to initialize indicator manager", error=str(e))
            raise

    async def add_indicator(
        self,
        name: str,
        indicator_type: IndicatorType,
        indicator_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new indicator to the manager.

        Args:
            name: Unique name for the indicator
            indicator_type: Type of indicator to create
            indicator_config: Configuration for the indicator

        Raises:
            ValueError: If indicator name already exists
            RuntimeError: If manager is not initialized
        """
        try:
            if self.state != IndicatorState.READY:
                raise RuntimeError(
                    f"Manager not ready: {self.state.value}"
                )

            with self.lock:
                if name in self.indicators:
                    raise ValueError(f"Indicator already exists: {name}")

            # Create indicator
            indicator = await self.factory.create_indicator(
                indicator_type,
                indicator_config
            )

            # Wrap in managed indicator
            managed = ManagedIndicator(
                name=name,
                indicator_type=indicator_type,
                indicator=indicator,
                state=IndicatorState.READY
            )

            with self.lock:
                self.indicators[name] = managed

            logger.info(
                "Indicator added",
                name=name,
                indicator_type=indicator_type.value
            )

        except Exception as e:
            logger.error(
                "Failed to add indicator",
                name=name,
                error=str(e)
            )
            raise

    async def remove_indicator(self, name: str) -> None:
        """
        Remove an indicator from the manager.

        Args:
            name: Name of indicator to remove

        Raises:
            ValueError: If indicator does not exist
        """
        try:
            with self.lock:
                if name not in self.indicators:
                    raise ValueError(f"Indicator not found: {name}")

                del self.indicators[name]

            logger.info("Indicator removed", name=name)

        except Exception as e:
            logger.error("Failed to remove indicator", name=name, error=str(e))
            raise

    async def calculate_indicator(
        self,
        name: str,
        data: pl.DataFrame,
        **kwargs
    ) -> pl.DataFrame:
        """
        Calculate an indicator with the given data.

        Args:
            name: Name of indicator to calculate
            data: Input data for calculation
            **kwargs: Additional arguments for indicator calculation

        Returns:
            DataFrame with calculated indicator values

        Raises:
            ValueError: If indicator does not exist
            asyncio.TimeoutError: If calculation times out
        """
        try:
            # Get managed indicator
            with self.lock:
                if name not in self.indicators:
                    raise ValueError(f"Indicator not found: {name}")
                managed = self.indicators[name]

            # Acquire calculation semaphore
            async with self._calculation_semaphore:
                # Update state
                with managed.lock:
                    managed.state = IndicatorState.CALCULATING

                await self._emit_event(
                    IndicatorEvent.CALCULATION_STARTED,
                    {"name": name}
                )

                start_time = datetime.now(timezone.utc)

                try:
                    # Calculate with timeout
                    result = await asyncio.wait_for(
                        managed.indicator.calculate(data, **kwargs),
                        timeout=self.calculation_timeout
                    )

                    # Update metrics
                    calculation_time = (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()

                    if self.enable_metrics:
                        await self._update_metrics(
                            managed,
                            success=True,
                            calculation_time=Decimal(str(calculation_time))
                        )

                    # Store result
                    with managed.lock:
                        managed.last_result = result
                        managed.state = IndicatorState.READY

                    await self._emit_event(
                        IndicatorEvent.CALCULATION_COMPLETED,
                        {"name": name, "calculation_time": calculation_time}
                    )

                    logger.debug(
                        "Indicator calculated",
                        name=name,
                        calculation_time=calculation_time
                    )

                    return result

                except asyncio.TimeoutError:
                    with managed.lock:
                        managed.state = IndicatorState.ERROR
                        managed.metrics.last_error = "Calculation timeout"

                    await self._emit_event(
                        IndicatorEvent.CALCULATION_FAILED,
                        {"name": name, "error": "timeout"}
                    )

                    logger.error("Indicator calculation timeout", name=name)
                    raise

                except Exception as e:
                    with managed.lock:
                        managed.state = IndicatorState.ERROR
                        managed.metrics.last_error = str(e)

                    if self.enable_metrics:
                        calculation_time = (
                            datetime.now(timezone.utc) - start_time
                        ).total_seconds()
                        await self._update_metrics(
                            managed,
                            success=False,
                            calculation_time=Decimal(str(calculation_time))
                        )

                    await self._emit_event(
                        IndicatorEvent.CALCULATION_FAILED,
                        {"name": name, "error": str(e)}
                    )

                    logger.error(
                        "Indicator calculation failed",
                        name=name,
                        error=str(e)
                    )
                    raise

        except Exception as e:
            logger.error("Failed to calculate indicator", name=name, error=str(e))
            raise

    async def calculate_multiple_indicators(
        self,
        calculations: List[Dict[str, Any]]
    ) -> Dict[str, pl.DataFrame]:
        """
        Calculate multiple indicators concurrently.

        Args:
            calculations: List of calculation requests, each containing:
                - name: Indicator name
                - data: Input data
                - kwargs: Optional additional arguments

        Returns:
            Dictionary mapping indicator names to results

        Raises:
            ValueError: If any indicator does not exist
        """
        try:
            tasks = []
            names = []

            for calc in calculations:
                name = calc["name"]
                data = calc["data"]
                kwargs = calc.get("kwargs", {})

                tasks.append(self.calculate_indicator(name, data, **kwargs))
                names.append(name)

            # Execute all calculations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build result dictionary
            result_dict = {}
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Calculation failed in batch",
                        name=name,
                        error=str(result)
                    )
                    continue
                result_dict[name] = result

            logger.info(
                "Multiple indicators calculated",
                requested=len(calculations),
                successful=len(result_dict)
            )

            return result_dict

        except Exception as e:
            logger.error("Failed to calculate multiple indicators", error=str(e))
            raise

    async def _update_metrics(
        self,
        managed: ManagedIndicator,
        success: bool,
        calculation_time: Decimal
    ) -> None:
        """Update indicator performance metrics."""
        with managed.lock:
            metrics = managed.metrics
            metrics.total_calculations += 1

            if success:
                metrics.successful_calculations += 1
            else:
                metrics.failed_calculations += 1

            metrics.total_calculation_time += calculation_time
            metrics.average_calculation_time = (
                metrics.total_calculation_time / metrics.total_calculations
            )
            metrics.last_calculation_time = datetime.now(timezone.utc)

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self.check_health()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Health check failed", error=str(e))

        except Exception as e:
            logger.error("Health check loop failed", error=str(e))

    async def check_health(self) -> Dict[str, Any]:
        """
        Perform health check on all managed indicators.

        Returns:
            Health status dictionary
        """
        try:
            health_status = {
                "manager_state": self.state.value,
                "total_indicators": len(self.indicators),
                "indicators": {}
            }

            with self.lock:
                for name, managed in self.indicators.items():
                    with managed.lock:
                        health_status["indicators"][name] = {
                            "state": managed.state.value,
                            "type": managed.indicator_type.value,
                            "last_calculation": (
                                managed.metrics.last_calculation_time.isoformat()
                                if managed.metrics.last_calculation_time
                                else None
                            ),
                            "total_calculations": managed.metrics.total_calculations,
                            "success_rate": (
                                managed.metrics.successful_calculations /
                                managed.metrics.total_calculations
                                if managed.metrics.total_calculations > 0
                                else 0
                            ),
                            "last_error": managed.metrics.last_error
                        }

            logger.debug("Health check completed", total_indicators=len(self.indicators))

            return health_status

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise

    async def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for indicators.

        Args:
            name: Optional indicator name. If None, returns all metrics.

        Returns:
            Dictionary with performance metrics
        """
        try:
            if name:
                with self.lock:
                    if name not in self.indicators:
                        raise ValueError(f"Indicator not found: {name}")
                    managed = self.indicators[name]

                with managed.lock:
                    return {
                        "name": managed.metrics.name,
                        "type": managed.metrics.indicator_type,
                        "total_calculations": managed.metrics.total_calculations,
                        "successful_calculations": managed.metrics.successful_calculations,
                        "failed_calculations": managed.metrics.failed_calculations,
                        "average_calculation_time": str(managed.metrics.average_calculation_time),
                        "last_calculation_time": (
                            managed.metrics.last_calculation_time.isoformat()
                            if managed.metrics.last_calculation_time
                            else None
                        ),
                        "last_error": managed.metrics.last_error
                    }
            else:
                metrics = {}
                with self.lock:
                    for name, managed in self.indicators.items():
                        with managed.lock:
                            metrics[name] = {
                                "type": managed.metrics.indicator_type,
                                "total_calculations": managed.metrics.total_calculations,
                                "successful_calculations": managed.metrics.successful_calculations,
                                "average_calculation_time": str(managed.metrics.average_calculation_time)
                            }
                return metrics

        except Exception as e:
            logger.error("Failed to get metrics", error=str(e))
            raise

    async def register_event_handler(
        self,
        event: IndicatorEvent,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register an event handler for indicator events.

        Args:
            event: Event type to handle
            handler: Callback function for the event
        """
        self.event_handlers[event].append(handler)
        logger.debug("Event handler registered", event=event.value)

    async def _emit_event(
        self,
        event: IndicatorEvent,
        data: Dict[str, Any]
    ) -> None:
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(
                    "Event handler failed",
                    event=event.value,
                    error=str(e)
                )

    async def shutdown(self) -> None:
        """Gracefully shutdown the indicator manager."""
        try:
            logger.info("Starting indicator manager shutdown")

            await self._emit_event(IndicatorEvent.SHUTDOWN_STARTED, {})

            with self.lock:
                self.state = IndicatorState.SHUTDOWN

            # Signal shutdown
            self._shutdown_event.set()

            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Clear all indicators
            with self.lock:
                self.indicators.clear()

            # Clear factory cache
            await self.factory.clear_cache()

            await self._emit_event(IndicatorEvent.SHUTDOWN_COMPLETED, {})

            logger.info("Indicator manager shutdown completed")

        except Exception as e:
            logger.error("Shutdown failed", error=str(e))
            raise


async def create_indicator_manager(config: Dict[str, Any]) -> IndicatorManager:
    """
    Factory function to create and initialize IndicatorManager.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized IndicatorManager instance
    """
    manager = IndicatorManager(config)
    await manager.initialize()
    return manager
