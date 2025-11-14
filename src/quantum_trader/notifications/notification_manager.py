"""
Notification Manager - Resource and state management for notification system.

Handles event-driven notifications, state management, and graceful shutdown.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import os
from collections import defaultdict
import threading
from structlog import get_logger

logger = get_logger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationChannel(Enum):
    """Available notification channels."""
    TELEGRAM = "TELEGRAM"
    EMAIL = "EMAIL"
    SLACK = "SLACK"
    WEBHOOK = "WEBHOOK"
    SMS = "SMS"


@dataclass
class Notification:
    """Notification message structure."""
    level: NotificationLevel
    message: str
    timestamp: datetime
    channel: NotificationChannel
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class NotificationStats:
    """Statistics for notification delivery."""
    total_sent: int = 0
    total_failed: int = 0
    by_channel: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class NotificationManager:
    """
    Thread-safe notification manager with event-driven architecture.

    Manages notification lifecycle, routing, and delivery across multiple channels.
    Provides graceful shutdown and health monitoring.

    Attributes:
        config: Configuration dictionary from environment
        channels: Registered notification channels
        queue: Async queue for notifications
        stats: Delivery statistics
        running: Manager state flag
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize notification manager.

        Args:
            config: Optional configuration override. If None, loads from environment.
        """
        self.config = config or self._load_config()
        self._validate_config()

        # Thread-safe state
        self._lock = threading.RLock()
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Notification channels (to be registered)
        self._channels: Dict[NotificationChannel, Any] = {}

        # Async queue for notification processing
        self._queue: Optional[asyncio.Queue] = None
        self._max_queue_size = int(self.config.get("max_queue_size", os.getenv("NOTIFICATION_MAX_QUEUE_SIZE", "1000")))

        # Statistics
        self._stats = NotificationStats()

        # Event listeners
        self._listeners: Dict[str, List[callable]] = defaultdict(list)

        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._num_workers = int(self.config.get("num_workers", os.getenv("NOTIFICATION_NUM_WORKERS", "3")))

        logger.info(
            "notification_manager_initialized",
            max_queue_size=self._max_queue_size,
            num_workers=self._num_workers
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            "max_queue_size": os.getenv("NOTIFICATION_MAX_QUEUE_SIZE", "1000"),
            "num_workers": os.getenv("NOTIFICATION_NUM_WORKERS", "3"),
            "health_check_interval": os.getenv("NOTIFICATION_HEALTH_CHECK_INTERVAL", "60"),
            "retry_delay_base": os.getenv("NOTIFICATION_RETRY_DELAY_BASE", "2"),
            "retry_delay_max": os.getenv("NOTIFICATION_RETRY_DELAY_MAX", "300"),
            "enable_rate_limiting": os.getenv("NOTIFICATION_ENABLE_RATE_LIMITING", "true").lower() == "true",
        }

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ["max_queue_size", "num_workers"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if int(self.config["max_queue_size"]) < 1:
            raise ValueError("max_queue_size must be >= 1")

        if int(self.config["num_workers"]) < 1:
            raise ValueError("num_workers must be >= 1")

    async def start(self) -> None:
        """
        Start the notification manager.

        Initializes queue, starts worker tasks, and begins processing.
        """
        with self._lock:
            if self._running:
                logger.warning("notification_manager_already_running")
                return

            self._running = True
            self._shutdown_event.clear()

        try:
            # Initialize queue
            self._queue = asyncio.Queue(maxsize=self._max_queue_size)

            # Start worker tasks
            for i in range(self._num_workers):
                task = asyncio.create_task(self._worker(i))
                self._workers.append(task)

            # Start health check task
            health_task = asyncio.create_task(self._health_check_loop())
            self._workers.append(health_task)

            logger.info(
                "notification_manager_started",
                num_workers=self._num_workers,
                max_queue_size=self._max_queue_size
            )

        except Exception as e:
            logger.error("failed_to_start_notification_manager", error=str(e))
            with self._lock:
                self._running = False
            raise

    async def stop(self, timeout: Optional[float] = None) -> None:
        """
        Gracefully shutdown the notification manager.

        Args:
            timeout: Maximum time to wait for shutdown in seconds.
                    Defaults to environment variable or 30 seconds.
        """
        if timeout is None:
            timeout = float(self.config.get("shutdown_timeout", os.getenv("NOTIFICATION_SHUTDOWN_TIMEOUT", "30")))

        with self._lock:
            if not self._running:
                logger.warning("notification_manager_not_running")
                return

            self._running = False
            self._shutdown_event.set()

        logger.info("notification_manager_shutting_down", timeout=timeout)

        try:
            # Process remaining items in queue
            if self._queue and not self._queue.empty():
                logger.info("processing_remaining_notifications", queue_size=self._queue.qsize())

            # Wait for workers to complete
            if self._workers:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=timeout
                )

            # Close all channels
            await self._close_channels()

            logger.info(
                "notification_manager_stopped",
                total_sent=self._stats.total_sent,
                total_failed=self._stats.total_failed
            )

        except asyncio.TimeoutError:
            logger.error("notification_manager_shutdown_timeout", timeout=timeout)
            # Cancel remaining tasks
            for task in self._workers:
                if not task.done():
                    task.cancel()
        except Exception as e:
            logger.error("notification_manager_shutdown_error", error=str(e))
        finally:
            self._workers.clear()
            self._queue = None

    def register_channel(self, channel_type: NotificationChannel, channel_instance: Any) -> None:
        """
        Register a notification channel.

        Args:
            channel_type: Type of channel to register
            channel_instance: Channel implementation instance
        """
        with self._lock:
            self._channels[channel_type] = channel_instance
            logger.info("notification_channel_registered", channel=channel_type.value)

    def unregister_channel(self, channel_type: NotificationChannel) -> None:
        """
        Unregister a notification channel.

        Args:
            channel_type: Type of channel to unregister
        """
        with self._lock:
            if channel_type in self._channels:
                del self._channels[channel_type]
                logger.info("notification_channel_unregistered", channel=channel_type.value)

    async def send_notification(self, notification: Notification) -> bool:
        """
        Queue a notification for delivery.

        Args:
            notification: Notification to send

        Returns:
            True if queued successfully, False otherwise
        """
        if not self._running or not self._queue:
            logger.error("notification_manager_not_running", level=notification.level.value)
            return False

        try:
            # Set timestamp if not provided
            if not notification.timestamp:
                notification.timestamp = datetime.utcnow()

            # Add to queue (non-blocking)
            self._queue.put_nowait(notification)

            logger.debug(
                "notification_queued",
                level=notification.level.value,
                channel=notification.channel.value,
                queue_size=self._queue.qsize()
            )

            return True

        except asyncio.QueueFull:
            logger.error(
                "notification_queue_full",
                level=notification.level.value,
                max_size=self._max_queue_size
            )
            return False
        except Exception as e:
            logger.error("failed_to_queue_notification", error=str(e))
            return False

    def add_listener(self, event: str, callback: callable) -> None:
        """
        Add event listener for notification events.

        Args:
            event: Event name (e.g., 'notification_sent', 'notification_failed')
            callback: Callback function to invoke
        """
        with self._lock:
            self._listeners[event].append(callback)
            logger.debug("event_listener_added", event=event)

    async def _worker(self, worker_id: int) -> None:
        """
        Worker task for processing notifications.

        Args:
            worker_id: Unique worker identifier
        """
        logger.info("notification_worker_started", worker_id=worker_id)

        while self._running or (self._queue and not self._queue.empty()):
            try:
                # Get notification with timeout
                try:
                    notification = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process notification
                success = await self._process_notification(notification)

                # Update stats
                with self._lock:
                    if success:
                        self._stats.total_sent += 1
                        self._stats.by_channel[notification.channel.value] += 1
                        self._stats.by_level[notification.level.value] += 1
                    else:
                        self._stats.total_failed += 1

                # Mark as done
                self._queue.task_done()

            except Exception as e:
                logger.error("notification_worker_error", worker_id=worker_id, error=str(e))
                await asyncio.sleep(1)

        logger.info("notification_worker_stopped", worker_id=worker_id)

    async def _process_notification(self, notification: Notification) -> bool:
        """
        Process and deliver a notification.

        Args:
            notification: Notification to process

        Returns:
            True if delivered successfully, False otherwise
        """
        channel = self._channels.get(notification.channel)

        if not channel:
            logger.error("notification_channel_not_registered", channel=notification.channel.value)
            return False

        retry_delay_base = float(self.config.get("retry_delay_base", os.getenv("NOTIFICATION_RETRY_DELAY_BASE", "2")))
        retry_delay_max = float(self.config.get("retry_delay_max", os.getenv("NOTIFICATION_RETRY_DELAY_MAX", "300")))

        for attempt in range(notification.max_retries + 1):
            try:
                # Deliver notification
                await channel.send(notification)

                logger.info(
                    "notification_sent",
                    channel=notification.channel.value,
                    level=notification.level.value,
                    attempt=attempt + 1
                )

                # Trigger event listeners
                await self._trigger_event("notification_sent", notification)

                return True

            except Exception as e:
                logger.error(
                    "notification_delivery_failed",
                    channel=notification.channel.value,
                    attempt=attempt + 1,
                    error=str(e)
                )

                if attempt < notification.max_retries:
                    # Exponential backoff
                    delay = min(retry_delay_base ** (attempt + 1), retry_delay_max)
                    await asyncio.sleep(delay)
                else:
                    # Max retries reached
                    await self._trigger_event("notification_failed", notification)
                    return False

        return False

    async def _trigger_event(self, event: str, notification: Notification) -> None:
        """
        Trigger event listeners.

        Args:
            event: Event name
            notification: Notification that triggered the event
        """
        listeners = self._listeners.get(event, [])

        for callback in listeners:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification)
                else:
                    callback(notification)
            except Exception as e:
                logger.error("event_listener_error", event=event, error=str(e))

    async def _close_channels(self) -> None:
        """Close all registered channels gracefully."""
        for channel_type, channel in self._channels.items():
            try:
                if hasattr(channel, 'close'):
                    if asyncio.iscoroutinefunction(channel.close):
                        await channel.close()
                    else:
                        channel.close()
                logger.info("notification_channel_closed", channel=channel_type.value)
            except Exception as e:
                logger.error("failed_to_close_channel", channel=channel_type.value, error=str(e))

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        interval = float(self.config.get("health_check_interval", os.getenv("NOTIFICATION_HEALTH_CHECK_INTERVAL", "60")))

        while self._running:
            try:
                await asyncio.sleep(interval)

                health = await self.health_check()

                if not health["healthy"]:
                    logger.warning("notification_manager_unhealthy", **health)

            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on notification manager.

        Returns:
            Health check results including status and metrics
        """
        with self._lock:
            queue_size = self._queue.qsize() if self._queue else 0
            queue_full_pct = (queue_size / self._max_queue_size * 100) if self._max_queue_size > 0 else 0

            healthy = (
                self._running and
                queue_full_pct < 90 and  # Queue not >90% full
                len(self._workers) > 0
            )

            return {
                "healthy": healthy,
                "running": self._running,
                "queue_size": queue_size,
                "queue_capacity": self._max_queue_size,
                "queue_full_percentage": round(queue_full_pct, 2),
                "active_workers": len([w for w in self._workers if not w.done()]),
                "total_workers": len(self._workers),
                "registered_channels": list(self._channels.keys()),
                "stats": {
                    "total_sent": self._stats.total_sent,
                    "total_failed": self._stats.total_failed,
                    "by_channel": dict(self._stats.by_channel),
                    "by_level": dict(self._stats.by_level),
                }
            }

    def get_stats(self) -> NotificationStats:
        """
        Get notification statistics.

        Returns:
            Current notification statistics
        """
        with self._lock:
            return NotificationStats(
                total_sent=self._stats.total_sent,
                total_failed=self._stats.total_failed,
                by_channel=dict(self._stats.by_channel),
                by_level=dict(self._stats.by_level)
            )

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        with self._lock:
            return self._running
