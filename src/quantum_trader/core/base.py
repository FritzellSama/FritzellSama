"""
Base Classes - Foundational base classes for the trading system.

This module provides abstract base classes and common functionality
shared across all components of the trading system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from structlog import get_logger

logger = get_logger(__name__)


class BaseComponent(ABC):
    """
    Abstract base class for all system components.

    Provides common lifecycle methods, logging, and configuration management
    for all trading system components.

    Attributes:
        config: Component configuration
        name: Component name
        initialized: Whether component is initialized
        running: Whether component is running

    Example:
        >>> class MyComponent(BaseComponent):
        ...     async def _initialize(self):
        ...         # Setup code
        ...     async def _start(self):
        ...         # Start code
        ...     async def _stop(self):
        ...         # Cleanup code
    """

    def __init__(self, config: Dict[str, Any], name: Optional[str] = None) -> None:
        """
        Initialize base component.

        Args:
            config: Component configuration dictionary
            name: Component name (defaults to class name)

        Raises:
            ValueError: If config invalid
        """
        self.config = config
        self.name = name or self.__class__.__name__

        self.initialized = False
        self.running = False

        self._start_time: Optional[datetime] = None
        self._stop_time: Optional[datetime] = None

        logger.debug(f"{self.name} created")

    async def initialize(self) -> None:
        """
        Initialize component.

        Calls _initialize() which must be implemented by subclasses.

        Raises:
            RuntimeError: If already initialized
        """
        if self.initialized:
            raise RuntimeError(f"{self.name} already initialized")

        try:
            logger.info(f"{self.name} initializing")
            await self._initialize()
            self.initialized = True
            logger.info(f"{self.name} initialized successfully")

        except Exception as e:
            logger.error(f"{self.name} initialization failed", error=str(e))
            raise

    @abstractmethod
    async def _initialize(self) -> None:
        """
        Component-specific initialization logic.

        Must be implemented by subclasses.
        """
        pass

    async def start(self) -> None:
        """
        Start component.

        Calls _start() which must be implemented by subclasses.

        Raises:
            RuntimeError: If not initialized or already running
        """
        if not self.initialized:
            raise RuntimeError(f"{self.name} not initialized. Call initialize() first")

        if self.running:
            raise RuntimeError(f"{self.name} already running")

        try:
            logger.info(f"{self.name} starting")
            self._start_time = datetime.utcnow()
            await self._start()
            self.running = True
            logger.info(f"{self.name} started successfully")

        except Exception as e:
            logger.error(f"{self.name} start failed", error=str(e))
            raise

    @abstractmethod
    async def _start(self) -> None:
        """
        Component-specific start logic.

        Must be implemented by subclasses.
        """
        pass

    async def stop(self) -> None:
        """
        Stop component gracefully.

        Calls _stop() which must be implemented by subclasses.
        """
        if not self.running:
            logger.warning(f"{self.name} not running")
            return

        try:
            logger.info(f"{self.name} stopping")
            await self._stop()
            self.running = False
            self._stop_time = datetime.utcnow()
            logger.info(f"{self.name} stopped successfully")

        except Exception as e:
            logger.error(f"{self.name} stop failed", error=str(e))
            raise

    @abstractmethod
    async def _stop(self) -> None:
        """
        Component-specific stop logic.

        Must be implemented by subclasses.
        """
        pass

    async def restart(self) -> None:
        """Restart component (stop then start)."""
        logger.info(f"{self.name} restarting")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        await self.start()
        logger.info(f"{self.name} restarted successfully")

    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.

        Returns:
            Dictionary with status information
        """
        uptime = None
        if self._start_time and self.running:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            'name': self.name,
            'initialized': self.initialized,
            'running': self.running,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'stop_time': self._stop_time.isoformat() if self._stop_time else None,
            'uptime_seconds': uptime
        }

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def __repr__(self) -> str:
        """String representation."""
        status = 'running' if self.running else ('initialized' if self.initialized else 'created')
        return f"<{self.name} status={status}>"


class BaseService(BaseComponent):
    """
    Base class for long-running services.

    Extends BaseComponent with health check capabilities and
    periodic task management.

    Example:
        >>> class MonitoringService(BaseService):
        ...     async def _health_check(self):
        ...         return True
    """

    def __init__(self, config: Dict[str, Any], name: Optional[str] = None) -> None:
        """Initialize service."""
        super().__init__(config, name)

        self._health_check_interval = int(config.get('health_check_interval', 60))
        self._health_check_task: Optional[asyncio.Task] = None
        self._healthy = True

    async def _start(self) -> None:
        """Start service with health monitoring."""
        await self._start_service()

        # Start health check task
        if self._health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._run_health_checks())

    @abstractmethod
    async def _start_service(self) -> None:
        """
        Service-specific start logic.

        Must be implemented by subclasses.
        """
        pass

    async def _stop(self) -> None:
        """Stop service and cancel health checks."""
        # Cancel health check task
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await self._stop_service()

    @abstractmethod
    async def _stop_service(self) -> None:
        """
        Service-specific stop logic.

        Must be implemented by subclasses.
        """
        pass

    async def _run_health_checks(self) -> None:
        """Run periodic health checks."""
        while self.running:
            try:
                self._healthy = await self._health_check()

                if not self._healthy:
                    logger.warning(f"{self.name} health check failed")
                else:
                    logger.debug(f"{self.name} health check passed")

                await asyncio.sleep(self._health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.name} health check error", error=str(e))
                self._healthy = False
                await asyncio.sleep(self._health_check_interval)

    async def _health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if healthy, False otherwise

        Can be overridden by subclasses for custom health checks.
        """
        return self.running

    def is_healthy(self) -> bool:
        """
        Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy and self.running

    def get_status(self) -> Dict[str, Any]:
        """Get service status including health."""
        status = super().get_status()
        status['healthy'] = self._healthy
        status['health_check_interval'] = self._health_check_interval
        return status


class Singleton(type):
    """
    Metaclass for singleton pattern.

    Ensures only one instance of a class exists.

    Example:
        >>> class ConfigManager(metaclass=Singleton):
        ...     pass
        >>> mgr1 = ConfigManager()
        >>> mgr2 = ConfigManager()
        >>> assert mgr1 is mgr2
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def clear(mcs) -> None:
        """Clear all singleton instances."""
        mcs._instances.clear()
