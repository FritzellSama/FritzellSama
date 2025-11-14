"""Health Checker for Quantum Trader AI.

Production-ready health checking system that monitors all critical
components and provides detailed health status reports.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading

from prometheus_client import Counter, Gauge, Histogram
from structlog import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Health status for a single component.

    Attributes:
        name: Component name
        status: Health status
        message: Status message
        last_check: Last health check timestamp
        details: Additional health details
    """

    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize component health.

        Args:
            name: Component name
            status: Health status
            message: Status message
            details: Additional details
        """
        self.name = name
        self.status = status
        self.message = message
        self.last_check = datetime.now(timezone.utc)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "details": self.details
        }


class HealthChecker:
    """Monitor system health across all components.

    Performs periodic health checks on:
    - Database connections
    - Exchange connections
    - Message queues
    - Cache services
    - External APIs
    - Internal services

    Attributes:
        config: Configuration dictionary
        health_checks: Registered health check functions
        component_status: Current health status per component

    Example:
        >>> config = {
        ...     "check_interval": 30,
        ...     "timeout_seconds": 5,
        ...     "unhealthy_threshold": 3
        ... }
        >>> checker = HealthChecker(config)
        >>> await checker.start()
        >>> health = await checker.get_overall_health()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize health checker.

        Args:
            config: Configuration dictionary containing:
                - check_interval: Seconds between health checks
                - timeout_seconds: Timeout for each check
                - unhealthy_threshold: Failed checks before unhealthy
                - enable_detailed_checks: Enable detailed component checks
                - notification_on_degraded: Send alerts on degradation

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.check_interval = self.config.get("check_interval", 30)
        self.timeout_seconds = self.config.get("timeout_seconds", 5)
        self.unhealthy_threshold = self.config.get("unhealthy_threshold", 3)
        self.enable_detailed = self.config.get("enable_detailed_checks", True)
        self.notification_on_degraded = self.config.get("notification_on_degraded", True)

        # Thread safety
        self._lock = threading.RLock()

        # Health check registry
        self.health_checks: Dict[str, Callable] = {}
        self.component_status: Dict[str, ComponentHealth] = {}
        self._failure_counts: Dict[str, int] = {}

        # Background tasks
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        # Metrics
        self._health_check_duration = Histogram(
            "quantum_trader_health_check_duration_seconds",
            "Health check duration",
            ["component"]
        )
        self._component_health_status = Gauge(
            "quantum_trader_component_health",
            "Component health status (1=healthy, 0.5=degraded, 0=unhealthy)",
            ["component"]
        )
        self._health_check_failures = Counter(
            "quantum_trader_health_check_failures_total",
            "Health check failures",
            ["component"]
        )

        logger.info(
            "health_checker_initialized",
            check_interval=self.check_interval,
            timeout=self.timeout_seconds,
            unhealthy_threshold=self.unhealthy_threshold
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        check_interval = self.config.get("check_interval", 30)
        if check_interval < 1:
            raise ValueError(f"check_interval must be >= 1, got {check_interval}")

        timeout = self.config.get("timeout_seconds", 5)
        if timeout < 1:
            raise ValueError(f"timeout_seconds must be >= 1, got {timeout}")

        threshold = self.config.get("unhealthy_threshold", 3)
        if threshold < 1:
            raise ValueError(f"unhealthy_threshold must be >= 1, got {threshold}")

    async def start(self) -> None:
        """Start health checking.

        Raises:
            RuntimeError: If already running
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Health checker already running")

            self._running = True

        logger.info("starting_health_checker")

        # Register default health checks
        await self._register_default_checks()

        # Start background task
        self._check_task = asyncio.create_task(self._health_check_loop())

        logger.info("health_checker_started", checks_registered=len(self.health_checks))

    async def stop(self) -> None:
        """Stop health checking."""
        logger.info("stopping_health_checker")

        with self._lock:
            self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("health_checker_stopped")

    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], asyncio.Future[ComponentHealth]]
    ) -> None:
        """Register a health check function.

        Args:
            name: Component name
            check_function: Async function that returns ComponentHealth

        Raises:
            ValueError: If name is invalid or already registered
        """
        if not name:
            raise ValueError("Component name cannot be empty")

        with self._lock:
            if name in self.health_checks:
                raise ValueError(f"Health check already registered: {name}")

            self.health_checks[name] = check_function
            self._failure_counts[name] = 0

        logger.info("health_check_registered", component=name)

    async def _register_default_checks(self) -> None:
        """Register default health checks."""
        try:
            # System health check
            self.register_health_check("system", self._check_system_health)

            # Memory health check
            self.register_health_check("memory", self._check_memory_health)

            # Event loop health check
            self.register_health_check("event_loop", self._check_event_loop_health)

        except Exception as e:
            logger.error("default_checks_registration_failed", error=str(e))

    async def _check_system_health(self) -> ComponentHealth:
        """Check overall system health.

        Returns:
            Component health status
        """
        try:
            # Basic system liveness check
            current_time = datetime.now(timezone.utc)

            return ComponentHealth(
                name="system",
                status=HealthStatus.HEALTHY,
                message="System operational",
                details={
                    "timestamp": current_time.isoformat(),
                    "uptime_seconds": (
                        current_time - datetime.now(timezone.utc)
                    ).total_seconds()
                }
            )

        except Exception as e:
            logger.error("system_health_check_failed", error=str(e))
            return ComponentHealth(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}"
            )

    async def _check_memory_health(self) -> ComponentHealth:
        """Check memory usage health.

        Returns:
            Component health status
        """
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical memory usage"
            elif memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High memory usage"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"

            return ComponentHealth(
                name="memory",
                status=status,
                message=message,
                details={
                    "percent": memory_percent,
                    "available_mb": memory.available / 1024 / 1024,
                    "total_mb": memory.total / 1024 / 1024
                }
            )

        except ImportError:
            # psutil not available, skip check
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory monitoring not available"
            )
        except Exception as e:
            logger.error("memory_health_check_failed", error=str(e))
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}"
            )

    async def _check_event_loop_health(self) -> ComponentHealth:
        """Check event loop health.

        Returns:
            Component health status
        """
        try:
            loop = asyncio.get_event_loop()

            # Check if loop is running
            if not loop.is_running():
                return ComponentHealth(
                    name="event_loop",
                    status=HealthStatus.UNHEALTHY,
                    message="Event loop not running"
                )

            return ComponentHealth(
                name="event_loop",
                status=HealthStatus.HEALTHY,
                message="Event loop operational"
            )

        except Exception as e:
            logger.error("event_loop_health_check_failed", error=str(e))
            return ComponentHealth(
                name="event_loop",
                status=HealthStatus.UNHEALTHY,
                message=f"Event loop check failed: {str(e)}"
            )

    async def check_component(self, name: str) -> ComponentHealth:
        """Run health check for specific component.

        Args:
            name: Component name

        Returns:
            Component health status

        Raises:
            ValueError: If component not registered
        """
        try:
            with self._lock:
                if name not in self.health_checks:
                    raise ValueError(f"Component not registered: {name}")

                check_function = self.health_checks[name]

            start_time = datetime.now(timezone.utc)

            # Run check with timeout
            try:
                health = await asyncio.wait_for(
                    check_function(),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                health = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timeout after {self.timeout_seconds}s"
                )

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._health_check_duration.labels(component=name).observe(duration)

            # Update failure count
            with self._lock:
                if health.status == HealthStatus.UNHEALTHY:
                    self._failure_counts[name] += 1
                    self._health_check_failures.labels(component=name).inc()

                    if self._failure_counts[name] >= self.unhealthy_threshold:
                        logger.warning(
                            "component_unhealthy",
                            component=name,
                            failures=self._failure_counts[name],
                            message=health.message
                        )
                else:
                    self._failure_counts[name] = 0

                # Update status
                self.component_status[name] = health

                # Update metrics
                status_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0.0,
                    HealthStatus.UNKNOWN: -1.0
                }[health.status]

                self._component_health_status.labels(component=name).set(status_value)

            logger.debug(
                "component_health_checked",
                component=name,
                status=health.status.value,
                duration=duration
            )

            return health

        except Exception as e:
            logger.error("component_health_check_failed", component=name, error=str(e))
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}"
            )

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """Run health checks for all components.

        Returns:
            Dictionary of component health statuses
        """
        try:
            with self._lock:
                components = list(self.health_checks.keys())

            results = {}

            # Run all checks concurrently
            tasks = [self.check_component(name) for name in components]
            health_statuses = await asyncio.gather(*tasks, return_exceptions=True)

            for name, health in zip(components, health_statuses):
                if isinstance(health, Exception):
                    results[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(health)}"
                    )
                else:
                    results[name] = health

            return results

        except Exception as e:
            logger.error("check_all_failed", error=str(e))
            return {}

    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status.

        Returns:
            Overall health report
        """
        try:
            with self._lock:
                components = dict(self.component_status)

            if not components:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "message": "No health checks registered",
                    "components": {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Determine overall status
            statuses = [c.status for c in components.values()]

            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                overall_status = HealthStatus.UNHEALTHY
                message = "One or more components unhealthy"
            elif any(s == HealthStatus.DEGRADED for s in statuses):
                overall_status = HealthStatus.DEGRADED
                message = "One or more components degraded"
            elif any(s == HealthStatus.UNKNOWN for s in statuses):
                overall_status = HealthStatus.DEGRADED
                message = "Some components status unknown"
            else:
                overall_status = HealthStatus.HEALTHY
                message = "All components healthy"

            healthy_count = sum(1 for s in statuses if s == HealthStatus.HEALTHY)
            degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)
            unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)

            return {
                "status": overall_status.value,
                "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total": len(components),
                    "healthy": healthy_count,
                    "degraded": degraded_count,
                    "unhealthy": unhealthy_count
                },
                "components": {
                    name: health.to_dict()
                    for name, health in components.items()
                }
            }

        except Exception as e:
            logger.error("get_overall_health_failed", error=str(e))
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Health check failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def _health_check_loop(self) -> None:
        """Background task to run periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)

                logger.debug("running_periodic_health_checks")
                await self.check_all()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_loop_error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """Perform self health check.

        Returns:
            Health status dictionary
        """
        return {
            "healthy": self._running,
            "registered_checks": len(self.health_checks),
            "last_check": max(
                (h.last_check for h in self.component_status.values()),
                default=None
            )
        }
