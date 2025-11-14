"""Market data manager for coordinating data collection and distribution.

This module provides centralized management of market data collection, processing,
and distribution across the trading system. Manages lifecycle of collectors,
processors, and feeds with health monitoring and graceful shutdown.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from structlog import get_logger

logger = get_logger(__name__)


class ManagerState(Enum):
    """Manager state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class MarketDataManager:
    """Centralized manager for market data operations.

    Coordinates all market data collection, processing, and distribution.
    Provides lifecycle management, health monitoring, and graceful shutdown.

    Attributes:
        config: Configuration dictionary
        exchanges: Dictionary of exchange connectors
        data_writer: Data writer instance
        state: Current manager state
        collectors: Active data collectors
        feeds: Active data feeds
        processors: Active data processors
        _health_check_task: Background health check task
        _shutdown_event: Event for coordinating shutdown

    Example:
        >>> config = {...}  # Full configuration
        >>> async with MarketDataManager(config, exchanges, writer) as manager:
        ...     await manager.start()
        ...     # Manager handles all data operations
        ...     await manager.wait_until_shutdown()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchanges: Dict[str, Any],
        data_writer: Any
    ) -> None:
        """Initialize market data manager.

        Args:
            config: Configuration dictionary
            exchanges: Dictionary of exchange connectors
            data_writer: Data writer instance

        Raises:
            ValueError: If required configuration is missing
        """
        self.config = config
        self._validate_config()

        self.exchanges = exchanges
        self.data_writer = data_writer

        # Manager state
        self.state = ManagerState.STOPPED
        self._state_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Component registries
        self.collectors: Dict[str, Any] = {}
        self.feeds: Dict[str, Any] = {}
        self.processors: Dict[str, Any] = {}

        # Health monitoring
        manager_config = config.get("market_data_manager", {})
        self.health_check_interval = manager_config.get("health_check_interval", 60)
        self.enable_health_checks = manager_config.get("enable_health_checks", True)
        self._health_check_task: Optional[asyncio.Task] = None

        # Component configuration
        self.enabled_collectors = manager_config.get("enabled_collectors", [])
        self.enabled_feeds = manager_config.get("enabled_feeds", [])
        self.enabled_processors = manager_config.get("enabled_processors", [])

        # Metrics
        self._metrics: Dict[str, Any] = {
            "start_time": None,
            "uptime_seconds": 0,
            "state_changes": 0,
            "health_checks": 0,
            "health_failures": 0,
            "restarts": 0
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required configuration is missing
        """
        if "market_data_manager" not in self.config:
            raise ValueError("Missing market_data_manager configuration")

        manager_config = self.config["market_data_manager"]

        required_keys = ["enabled_collectors", "enabled_feeds"]
        for key in required_keys:
            if key not in manager_config:
                raise ValueError(f"Missing required configuration: {key}")

    async def start(self) -> None:
        """Start the market data manager.

        Initializes and starts all enabled collectors, processors, and feeds.
        Begins health monitoring and enters running state.

        Raises:
            RuntimeError: If manager is not in STOPPED state
        """
        async with self._state_lock:
            if self.state != ManagerState.STOPPED:
                raise RuntimeError(f"Cannot start from state: {self.state}")

            self._change_state(ManagerState.STARTING)

        try:
            logger.info("market_data_manager_starting")

            # Initialize collectors
            await self._initialize_collectors()

            # Initialize processors
            await self._initialize_processors()

            # Initialize feeds
            await self._initialize_feeds()

            # Start health monitoring
            if self.enable_health_checks:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

            # Update state
            async with self._state_lock:
                self._change_state(ManagerState.RUNNING)
                self._metrics["start_time"] = datetime.now(timezone.utc)

            logger.info(
                "market_data_manager_started",
                collectors=len(self.collectors),
                processors=len(self.processors),
                feeds=len(self.feeds)
            )

        except Exception as e:
            logger.error("market_data_manager_start_failed", error=str(e))
            async with self._state_lock:
                self._change_state(ManagerState.ERROR)
            await self._cleanup_components()
            raise

    async def stop(self) -> None:
        """Stop the market data manager.

        Gracefully stops all collectors, processors, and feeds.
        Ensures all data is flushed before shutdown completes.
        """
        async with self._state_lock:
            if self.state == ManagerState.STOPPED:
                logger.warning("market_data_manager_already_stopped")
                return

            if self.state == ManagerState.STOPPING:
                logger.warning("market_data_manager_already_stopping")
                return

            self._change_state(ManagerState.STOPPING)

        try:
            logger.info("market_data_manager_stopping")

            # Stop health checks
            if self._health_check_task and not self._health_check_task.done():
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Stop all components
            await self._cleanup_components()

            # Signal shutdown complete
            self._shutdown_event.set()

            # Update state
            async with self._state_lock:
                self._change_state(ManagerState.STOPPED)

            # Calculate uptime
            if self._metrics["start_time"]:
                uptime = (
                    datetime.now(timezone.utc) - self._metrics["start_time"]
                ).total_seconds()
                self._metrics["uptime_seconds"] = uptime

            logger.info(
                "market_data_manager_stopped",
                uptime_seconds=self._metrics.get("uptime_seconds", 0)
            )

        except Exception as e:
            logger.error("market_data_manager_stop_failed", error=str(e))
            async with self._state_lock:
                self._change_state(ManagerState.ERROR)
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def wait_until_shutdown(self) -> None:
        """Wait until shutdown is signaled.

        Blocks until the manager receives a shutdown signal.
        """
        await self._shutdown_event.wait()

    def signal_shutdown(self) -> None:
        """Signal the manager to shutdown.

        Can be called from signal handlers or other components.
        """
        logger.info("shutdown_signal_received")
        asyncio.create_task(self.stop())

    async def _initialize_collectors(self) -> None:
        """Initialize and start all enabled collectors."""
        # Import collectors dynamically to avoid circular imports
        from quantum_trader.data.collectors.funding_collector import FundingCollector
        from quantum_trader.data.collectors.liquidation_collector import LiquidationCollector
        from quantum_trader.data.collectors.orderbook_collector import OrderbookCollector

        collector_classes = {
            "funding": FundingCollector,
            "liquidation": LiquidationCollector,
            "orderbook": OrderbookCollector
        }

        for collector_name in self.enabled_collectors:
            if collector_name in collector_classes:
                try:
                    collector_class = collector_classes[collector_name]
                    collector = collector_class(
                        self.config,
                        self.exchanges,
                        self.data_writer
                    )
                    await collector.start()
                    self.collectors[collector_name] = collector

                    logger.info(
                        "collector_initialized",
                        collector=collector_name
                    )

                except Exception as e:
                    logger.error(
                        "collector_initialization_failed",
                        collector=collector_name,
                        error=str(e)
                    )
                    raise

    async def _initialize_processors(self) -> None:
        """Initialize and start all enabled processors."""
        # Import processors dynamically
        from quantum_trader.data.processors.orderbook_processor import OrderbookProcessor
        from quantum_trader.data.processors.tick_processor import TickProcessor

        processor_classes = {
            "orderbook": OrderbookProcessor,
            "tick": TickProcessor
        }

        for processor_name in self.enabled_processors:
            if processor_name in processor_classes:
                try:
                    processor_class = processor_classes[processor_name]
                    processor = processor_class(self.config)
                    # Processors may not have start() method, just initialize
                    self.processors[processor_name] = processor

                    logger.info(
                        "processor_initialized",
                        processor=processor_name
                    )

                except Exception as e:
                    logger.error(
                        "processor_initialization_failed",
                        processor=processor_name,
                        error=str(e)
                    )
                    # Don't raise - processors are optional
                    logger.warning(
                        "continuing_without_processor",
                        processor=processor_name
                    )

    async def _initialize_feeds(self) -> None:
        """Initialize and start all enabled feeds."""
        # Import feeds dynamically
        from quantum_trader.data.feeds.orderbook_feed import OrderbookFeed
        from quantum_trader.data.feeds.price_feed import PriceFeed

        feed_classes = {
            "orderbook": OrderbookFeed,
            "price": PriceFeed
        }

        for feed_name in self.enabled_feeds:
            if feed_name in feed_classes:
                try:
                    feed_class = feed_classes[feed_name]
                    feed = feed_class(
                        self.config,
                        self.exchanges,
                        self.data_writer
                    )
                    await feed.start()
                    self.feeds[feed_name] = feed

                    logger.info(
                        "feed_initialized",
                        feed=feed_name
                    )

                except Exception as e:
                    logger.error(
                        "feed_initialization_failed",
                        feed=feed_name,
                        error=str(e)
                    )
                    raise

    async def _cleanup_components(self) -> None:
        """Stop and cleanup all components."""
        # Stop collectors
        for name, collector in self.collectors.items():
            try:
                await collector.stop()
                logger.info("collector_stopped", collector=name)
            except Exception as e:
                logger.error(
                    "collector_stop_failed",
                    collector=name,
                    error=str(e)
                )

        # Stop feeds
        for name, feed in self.feeds.items():
            try:
                await feed.stop()
                logger.info("feed_stopped", feed=name)
            except Exception as e:
                logger.error(
                    "feed_stop_failed",
                    feed=name,
                    error=str(e)
                )

        # Clear registries
        self.collectors.clear()
        self.feeds.clear()
        self.processors.clear()

    async def _health_check_loop(self) -> None:
        """Background task for health monitoring."""
        while self.state == ManagerState.RUNNING:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check health of all components
                health_status = await self._check_health()

                self._metrics["health_checks"] += 1

                if not health_status["healthy"]:
                    self._metrics["health_failures"] += 1
                    logger.warning(
                        "health_check_failed",
                        status=health_status
                    )

                    # Auto-restart unhealthy components if configured
                    if self.config.get("market_data_manager", {}).get("auto_restart", False):
                        await self._restart_unhealthy_components(
                            health_status["unhealthy_components"]
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def _check_health(self) -> Dict[str, Any]:
        """Check health of all components.

        Returns:
            Dictionary containing health status
        """
        unhealthy_components = []
        health_details = {}

        # Check collectors
        for name, collector in self.collectors.items():
            try:
                if hasattr(collector, "get_metrics"):
                    metrics = collector.get_metrics()
                    health_details[f"collector_{name}"] = metrics

                    # Check if collector is functioning
                    if not collector._running:
                        unhealthy_components.append(f"collector_{name}")

            except Exception as e:
                logger.error(
                    "collector_health_check_failed",
                    collector=name,
                    error=str(e)
                )
                unhealthy_components.append(f"collector_{name}")

        # Check feeds
        for name, feed in self.feeds.items():
            try:
                if hasattr(feed, "get_metrics"):
                    metrics = feed.get_metrics()
                    health_details[f"feed_{name}"] = metrics

                    if not feed._running:
                        unhealthy_components.append(f"feed_{name}")

            except Exception as e:
                logger.error(
                    "feed_health_check_failed",
                    feed=name,
                    error=str(e)
                )
                unhealthy_components.append(f"feed_{name}")

        return {
            "healthy": len(unhealthy_components) == 0,
            "unhealthy_components": unhealthy_components,
            "details": health_details,
            "timestamp": datetime.now(timezone.utc)
        }

    async def _restart_unhealthy_components(
        self,
        unhealthy: List[str]
    ) -> None:
        """Restart unhealthy components.

        Args:
            unhealthy: List of unhealthy component names
        """
        for component_name in unhealthy:
            try:
                component_type, name = component_name.split("_", 1)

                if component_type == "collector" and name in self.collectors:
                    logger.info("restarting_collector", collector=name)
                    collector = self.collectors[name]
                    await collector.stop()
                    await collector.start()
                    self._metrics["restarts"] += 1

                elif component_type == "feed" and name in self.feeds:
                    logger.info("restarting_feed", feed=name)
                    feed = self.feeds[name]
                    await feed.stop()
                    await feed.start()
                    self._metrics["restarts"] += 1

            except Exception as e:
                logger.error(
                    "component_restart_failed",
                    component=component_name,
                    error=str(e)
                )

    def _change_state(self, new_state: ManagerState) -> None:
        """Change manager state.

        Args:
            new_state: New state to transition to
        """
        old_state = self.state
        self.state = new_state
        self._metrics["state_changes"] += 1

        logger.info(
            "manager_state_changed",
            old_state=old_state.value,
            new_state=new_state.value
        )

    def get_state(self) -> ManagerState:
        """Get current manager state.

        Returns:
            Current state
        """
        return self.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()

        # Add component metrics
        metrics["collectors"] = {
            name: collector.get_metrics()
            for name, collector in self.collectors.items()
            if hasattr(collector, "get_metrics")
        }

        metrics["feeds"] = {
            name: feed.get_metrics()
            for name, feed in self.feeds.items()
            if hasattr(feed, "get_metrics")
        }

        return metrics

    def get_component(self, component_type: str, name: str) -> Optional[Any]:
        """Get a specific component by type and name.

        Args:
            component_type: Type of component (collector, feed, processor)
            name: Component name

        Returns:
            Component instance or None if not found
        """
        if component_type == "collector":
            return self.collectors.get(name)
        elif component_type == "feed":
            return self.feeds.get(name)
        elif component_type == "processor":
            return self.processors.get(name)
        return None
