"""
Event-Driven Architecture using Redis for Event Bus
Production-ready event publishing and subscription system
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable, Coroutine
import logging
import asyncio
import json
import os
from datetime import datetime
from enum import Enum
import redis.asyncio as aioredis
from dataclasses import dataclass, asdict

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of event types"""
    ORDER_CREATED = "order.created"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    TRADE_EXECUTED = "trade.executed"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    RISK_BREACH = "risk.breach"
    SYSTEM_ERROR = "system.error"
    MARKET_DATA = "market.data"
    BALANCE_UPDATE = "balance.update"
    STATE_TRANSITION = "state.transition"


@dataclass
class Event:
    """Event data structure"""
    event_type: str
    event_id: str
    timestamp: str
    data: Dict[str, Any]
    source: str
    priority: int = 0

    def to_json(self) -> str:
        """Serialize event to JSON"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize event from JSON"""
        data = json.loads(json_str)
        return cls(**data)


class EventBus:
    """Event bus implementation using Redis pub/sub"""

    def __init__(self) -> None:
        """Initialize event bus with configuration"""
        self.config = get_config()
        self._load_config()
        self._redis: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = None
        self._worker_tasks: List[asyncio.Task] = []
        self._running = False
        logger.info("EventBus initialized")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            redis_url = 'redis://localhost:6379/0'
            logger.warning(f"REDIS_URL not set, using default: {redis_url}")

        self.redis_url = redis_url
        self.backend = self.config.get_string('engine', 'event_bus.backend')
        self.max_queue_size = self.config.get_int('engine', 'event_bus.max_queue_size')
        self.worker_threads = self.config.get_int('engine', 'event_bus.worker_threads')
        self.event_retention_seconds = self.config.get_int('engine', 'event_bus.event_retention_seconds')
        self.dead_letter_queue_enabled = self.config.get_bool('engine', 'event_bus.dead_letter_queue_enabled')

        logger.info(f"EventBus configured: backend={self.backend}, workers={self.worker_threads}")

    async def connect(self) -> None:
        """Connect to Redis"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=50
                )

                # Test connection
                await self._redis.ping()

                # Initialize pubsub
                self._pubsub = self._redis.pubsub()

                # Initialize event queue
                self._event_queue = asyncio.Queue(maxsize=self.max_queue_size)

                logger.info("Successfully connected to Redis event bus")
                return

            except Exception as e:
                logger.error(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ConnectionError(f"Failed to connect to Redis after {max_retries} attempts: {e}")

    async def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        source: str,
        priority: int = 0
    ) -> str:
        """
        Publish an event to the bus

        Args:
            event_type: Type of event (use EventType enum values)
            data: Event data payload
            source: Source system/component
            priority: Event priority (0=normal, 1=high, 2=critical)

        Returns:
            str: Event ID

        Raises:
            RuntimeError: If publishing fails after retries
        """
        if not self._redis:
            await self.connect()

        import uuid

        event_id = str(uuid.uuid4())
        event = Event(
            event_type=event_type,
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            source=source,
            priority=priority
        )

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Publish to Redis channel
                channel = f"events:{event_type}"
                message = event.to_json()

                await self._redis.publish(channel, message)

                # Store in Redis with TTL for replay capability
                event_key = f"event:{event_id}"
                await self._redis.setex(
                    event_key,
                    self.event_retention_seconds,
                    message
                )

                # Add to sorted set for ordered retrieval (score = timestamp)
                timestamp_score = datetime.utcnow().timestamp()
                await self._redis.zadd(
                    f"events:timeline:{event_type}",
                    {event_id: timestamp_score}
                )

                logger.debug(f"Published event {event_id} of type {event_type}")
                return event_id

            except Exception as e:
                logger.error(f"Publish event attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    if self.dead_letter_queue_enabled:
                        await self._send_to_dead_letter_queue(event)
                    raise RuntimeError(f"Failed to publish event after {max_retries} attempts: {e}")

        return event_id

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> bool:
        """
        Subscribe to events of a specific type

        Args:
            event_type: Type of event to subscribe to
            callback: Async callback function to handle events

        Returns:
            bool: True if subscription successful
        """
        if not self._redis:
            await self.connect()

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Add callback to subscribers
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []

                self._subscribers[event_type].append(callback)

                # Subscribe to Redis channel
                channel = f"events:{event_type}"
                await self._pubsub.subscribe(channel)

                logger.info(f"Subscribed to event type: {event_type}")
                return True

            except Exception as e:
                logger.error(f"Subscribe attempt {attempt + 1} failed for {event_type}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to subscribe to {event_type} after {max_retries} attempts: {e}")

        return False

    async def unsubscribe(self, event_type: str) -> bool:
        """
        Unsubscribe from events of a specific type

        Args:
            event_type: Type of event to unsubscribe from

        Returns:
            bool: True if unsubscription successful
        """
        if not self._pubsub:
            logger.warning("Not connected to event bus")
            return False

        try:
            # Remove from subscribers
            if event_type in self._subscribers:
                del self._subscribers[event_type]

            # Unsubscribe from Redis channel
            channel = f"events:{event_type}"
            await self._pubsub.unsubscribe(channel)

            logger.info(f"Unsubscribed from event type: {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {event_type}: {e}")
            return False

    async def process_events(self) -> None:
        """
        Process events from subscriptions

        This is the main event processing loop that should run continuously
        """
        if not self._pubsub:
            await self.connect()

        self._running = True

        # Start worker tasks
        for i in range(self.worker_threads):
            task = asyncio.create_task(self._event_worker(worker_id=i))
            self._worker_tasks.append(task)

        logger.info(f"Started {self.worker_threads} event worker threads")

        try:
            async for message in self._pubsub.listen():
                if not self._running:
                    break

                if message['type'] == 'message':
                    try:
                        # Parse event
                        event = Event.from_json(message['data'])

                        # Add to processing queue
                        await self._event_queue.put(event)

                    except Exception as e:
                        logger.error(f"Error parsing event message: {e}")
                        continue

        except asyncio.CancelledError:
            logger.info("Event processing cancelled")
        except Exception as e:
            logger.error(f"Error in event processing loop: {e}")
        finally:
            self._running = False
            await self._shutdown_workers()

    async def _event_worker(self, worker_id: int) -> None:
        """
        Worker task to process events from the queue

        Args:
            worker_id: Unique worker identifier
        """
        logger.info(f"Event worker {worker_id} started")

        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                # Find and execute callbacks
                callbacks = self._subscribers.get(event.event_type, [])

                for callback in callbacks:
                    try:
                        await callback(event)
                    except Exception as e:
                        logger.error(f"Error in event callback for {event.event_type}: {e}")
                        if self.dead_letter_queue_enabled:
                            await self._send_to_dead_letter_queue(event)

                # Mark task as done
                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Event worker {worker_id} stopped")

    async def _send_to_dead_letter_queue(self, event: Event) -> None:
        """Send failed event to dead letter queue"""
        try:
            if self._redis:
                dlq_key = f"dlq:events:{event.event_type}"
                await self._redis.lpush(dlq_key, event.to_json())
                logger.warning(f"Event {event.event_id} sent to dead letter queue")
        except Exception as e:
            logger.error(f"Failed to send event to DLQ: {e}")

    async def _shutdown_workers(self) -> None:
        """Shutdown all worker tasks"""
        logger.info("Shutting down event workers")

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def get_event_history(
        self,
        event_type: str,
        limit: int = 100
    ) -> List[Event]:
        """
        Retrieve historical events of a specific type

        Args:
            event_type: Type of events to retrieve
            limit: Maximum number of events to return

        Returns:
            List of Event objects
        """
        if not self._redis:
            await self.connect()

        try:
            # Get event IDs from sorted set (most recent first)
            event_ids = await self._redis.zrevrange(
                f"events:timeline:{event_type}",
                0,
                limit - 1
            )

            events = []
            for event_id in event_ids:
                event_key = f"event:{event_id}"
                event_json = await self._redis.get(event_key)

                if event_json:
                    event = Event.from_json(event_json)
                    events.append(event)

            logger.info(f"Retrieved {len(events)} historical events of type {event_type}")
            return events

        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []

    async def close(self) -> None:
        """Close event bus connections"""
        self._running = False

        await self._shutdown_workers()

        if self._pubsub:
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        logger.info("EventBus closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
