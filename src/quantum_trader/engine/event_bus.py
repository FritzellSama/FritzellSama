"""
Quantum Trader AI - High-Performance Event Bus
Production-grade pub/sub event system for distributed trading

CRITICAL: Low-latency event routing for time-sensitive trading
CRITICAL: Guaranteed delivery with dead letter queue
CRITICAL: Event replay for disaster recovery
"""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set
from uuid import uuid4

import polars as pl
import yaml


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 0  # Emergency shutdown, circuit breaker
    HIGH = 1      # Order fills, risk breaches
    NORMAL = 2    # Regular trading events
    LOW = 3       # Analytics, monitoring


class EventType(Enum):
    """System event types"""
    # Order events
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL = "ORDER_PARTIAL"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_FAILED = "ORDER_FAILED"

    # Position events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_UPDATED = "POSITION_UPDATED"

    # Risk events
    RISK_BREACH = "RISK_BREACH"
    STOP_LOSS_TRIGGERED = "STOP_LOSS_TRIGGERED"
    TAKE_PROFIT_TRIGGERED = "TAKE_PROFIT_TRIGGERED"
    MARGIN_CALL = "MARGIN_CALL"

    # System events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CIRCUIT_BREAKER_TRIGGERED = "CIRCUIT_BREAKER_TRIGGERED"
    FAILOVER_INITIATED = "FAILOVER_INITIATED"

    # Market events
    MARKET_DATA_UPDATE = "MARKET_DATA_UPDATE"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"

    # Error events
    EXCHANGE_ERROR = "EXCHANGE_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


@dataclass
class Event:
    """
    Trading system event

    CRITICAL: All monetary values use Decimal
    """
    event_id: str
    event_type: EventType
    priority: EventPriority
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self._serialize_data(self.data),
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'correlation_id': self.correlation_id
        }

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data with Decimal handling"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, Decimal):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            else:
                serialized[key] = value
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create Event from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            priority=EventPriority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data['data'],
            metadata=data.get('metadata', {}),
            retry_count=data.get('retry_count', 0),
            correlation_id=data.get('correlation_id')
        )


@dataclass
class EventMetrics:
    """Performance metrics for event bus"""
    total_published: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    total_dead_lettered: int = 0
    avg_latency_ms: Decimal = Decimal('0')
    max_latency_ms: Decimal = Decimal('0')
    events_per_second: Decimal = Decimal('0')
    queue_depth: int = 0
    dlq_depth: int = 0


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    High-performance event bus with pub/sub pattern

    Features:
    - Priority-based event queuing
    - Event filtering and routing
    - Dead letter queue for failed events
    - Event replay from history
    - Performance monitoring
    - Async event handling
    - Guaranteed delivery
    """

    def __init__(
        self,
        config_path: str = '/home/user/FritzellSama/config/environments/production.yaml',
        bot_config_path: str = '/home/user/FritzellSama/config/bot/bot.yaml'
    ) -> None:
        """Initialize event bus"""
        self.config = self._load_config(config_path)
        self.bot_config = self._load_config(bot_config_path)

        # Configuration
        self.max_queue_size = int(os.getenv('EVENT_BUS_MAX_QUEUE_SIZE', '10000'))
        self.max_dlq_size = int(os.getenv('EVENT_BUS_MAX_DLQ_SIZE', '5000'))
        self.max_retries = int(os.getenv('EVENT_BUS_MAX_RETRIES', '3'))
        self.retry_delay_ms = int(os.getenv('EVENT_BUS_RETRY_DELAY_MS', '1000'))
        self.event_ttl_seconds = int(os.getenv('EVENT_BUS_TTL_SECONDS', '3600'))
        self.enable_persistence = os.getenv('EVENT_BUS_ENABLE_PERSISTENCE', 'true').lower() == 'true'
        self.enable_replay = os.getenv('EVENT_BUS_ENABLE_REPLAY', 'true').lower() == 'true'

        # Priority queues (one per priority level)
        self._queues: Dict[EventPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=self.max_queue_size)
            for priority in EventPriority
        }

        # Dead letter queue for failed events
        self._dlq: Deque[Event] = deque(maxlen=self.max_dlq_size)

        # Event history for replay
        self._event_history: Deque[Event] = deque(maxlen=100000)

        # Subscribers organized by event type and filters
        self._subscribers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._filtered_subscribers: List[tuple[Callable[[Event], bool], EventHandler]] = []

        # Wildcard subscribers (receive all events)
        self._wildcard_subscribers: List[EventHandler] = []

        # Metrics
        self._metrics = EventMetrics()
        self._latency_samples: Deque[Decimal] = deque(maxlen=1000)

        # Processing tasks
        self._processor_tasks: List[asyncio.Task] = []
        self._running = False

        # Metrics tracking
        self._last_metrics_time = time.time()
        self._events_since_last_metric = 0

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config if config else {}
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    async def start(self) -> None:
        """Start event bus processors"""
        if self._running:
            return

        self._running = True

        # Start one processor per priority level
        for priority in EventPriority:
            task = asyncio.create_task(self._process_events(priority))
            self._processor_tasks.append(task)

        # Start metrics collector
        metrics_task = asyncio.create_task(self._metrics_collector())
        self._processor_tasks.append(metrics_task)

    async def stop(self) -> None:
        """Stop event bus gracefully"""
        self._running = False

        # Cancel all processor tasks
        for task in self._processor_tasks:
            task.cancel()

        if self._processor_tasks:
            await asyncio.gather(*self._processor_tasks, return_exceptions=True)

        self._processor_tasks.clear()

    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        source: str = "system",
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Publish event to the bus

        Args:
            event_type: Type of event
            data: Event payload
            priority: Event priority
            source: Source component
            correlation_id: Optional correlation ID for tracking
            metadata: Optional metadata

        Returns:
            Event ID

        CRITICAL: Handles back-pressure with queue size limits
        """
        event = Event(
            event_id=str(uuid4()),
            event_type=event_type,
            priority=priority,
            timestamp=datetime.utcnow(),
            source=source,
            data=data,
            metadata=metadata or {},
            correlation_id=correlation_id
        )

        publish_start = time.time()

        try:
            # Add to priority queue
            queue = self._queues[priority]

            # Non-blocking put with timeout
            try:
                await asyncio.wait_for(queue.put(event), timeout=5.0)
            except asyncio.TimeoutError:
                raise RuntimeError(f"Event queue full for priority {priority.name}")

            # Track metrics
            self._metrics.total_published += 1
            self._events_since_last_metric += 1

            # Store in history for replay
            if self.enable_replay:
                self._event_history.append(event)

            # Calculate latency
            latency_ms = Decimal(str((time.time() - publish_start) * 1000))
            self._latency_samples.append(latency_ms)

            return event.event_id

        except Exception as e:
            self._metrics.total_failed += 1
            raise RuntimeError(f"Failed to publish event: {e}")

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> None:
        """
        Subscribe to specific event type

        Args:
            event_type: Event type to subscribe to
            handler: Async handler function
        """
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events (wildcard subscription)

        Args:
            handler: Async handler function
        """
        self._wildcard_subscribers.append(handler)

    def subscribe_filtered(
        self,
        filter_func: Callable[[Event], bool],
        handler: EventHandler
    ) -> None:
        """
        Subscribe with custom filter

        Args:
            filter_func: Function to filter events
            handler: Async handler function
        """
        self._filtered_subscribers.append((filter_func, handler))

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> None:
        """
        Unsubscribe from event type

        Args:
            event_type: Event type to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass

    async def _process_events(self, priority: EventPriority) -> None:
        """
        Process events from priority queue

        Args:
            priority: Priority level to process
        """
        queue = self._queues[priority]

        while self._running:
            try:
                # Get next event from queue
                event = await asyncio.wait_for(queue.get(), timeout=1.0)

                # Process event
                await self._handle_event(event)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing events for priority {priority.name}: {e}")

    async def _handle_event(self, event: Event) -> None:
        """
        Handle single event by dispatching to subscribers

        Args:
            event: Event to handle
        """
        handlers: List[EventHandler] = []

        # Get type-specific subscribers
        if event.event_type in self._subscribers:
            handlers.extend(self._subscribers[event.event_type])

        # Add filtered subscribers
        for filter_func, handler in self._filtered_subscribers:
            try:
                if filter_func(event):
                    handlers.append(handler)
            except Exception as e:
                print(f"Error in filter function: {e}")

        # Add wildcard subscribers
        handlers.extend(self._wildcard_subscribers)

        if not handlers:
            # No subscribers, just count as delivered
            self._metrics.total_delivered += 1
            return

        # Dispatch to all handlers
        results = await asyncio.gather(
            *[self._invoke_handler(handler, event) for handler in handlers],
            return_exceptions=True
        )

        # Check for failures
        failures = [r for r in results if isinstance(r, Exception)]

        if failures:
            # Some handlers failed
            if event.retry_count < self.max_retries:
                # Retry
                event.retry_count += 1
                await asyncio.sleep(self.retry_delay_ms / 1000.0)
                await self._queues[event.priority].put(event)
            else:
                # Max retries exceeded, move to DLQ
                self._dlq.append(event)
                self._metrics.total_dead_lettered += 1
        else:
            # All handlers succeeded
            self._metrics.total_delivered += 1

    async def _invoke_handler(self, handler: EventHandler, event: Event) -> None:
        """
        Invoke event handler with error handling

        Args:
            handler: Handler function
            event: Event to process
        """
        try:
            await handler(event)
        except Exception as e:
            print(f"Handler error for event {event.event_id}: {e}")
            raise

    async def _metrics_collector(self) -> None:
        """Background task to collect metrics"""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                # Calculate events per second
                now = time.time()
                elapsed = now - self._last_metrics_time
                if elapsed > 0:
                    self._metrics.events_per_second = Decimal(
                        str(self._events_since_last_metric / elapsed)
                    )

                self._last_metrics_time = now
                self._events_since_last_metric = 0

                # Calculate average latency
                if self._latency_samples:
                    self._metrics.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
                    self._metrics.max_latency_ms = max(self._latency_samples)

                # Update queue depths
                self._metrics.queue_depth = sum(q.qsize() for q in self._queues.values())
                self._metrics.dlq_depth = len(self._dlq)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error collecting metrics: {e}")

    def get_metrics(self) -> EventMetrics:
        """Get current event bus metrics"""
        return self._metrics

    def get_metrics_df(self) -> pl.DataFrame:
        """
        Get metrics as polars DataFrame

        Returns:
            DataFrame with current metrics
        """
        data = {
            'metric': [
                'total_published',
                'total_delivered',
                'total_failed',
                'total_dead_lettered',
                'avg_latency_ms',
                'max_latency_ms',
                'events_per_second',
                'queue_depth',
                'dlq_depth'
            ],
            'value': [
                self._metrics.total_published,
                self._metrics.total_delivered,
                self._metrics.total_failed,
                self._metrics.total_dead_lettered,
                float(self._metrics.avg_latency_ms),
                float(self._metrics.max_latency_ms),
                float(self._metrics.events_per_second),
                self._metrics.queue_depth,
                self._metrics.dlq_depth
            ],
            'timestamp': [datetime.utcnow().isoformat()] * 9
        }

        return pl.DataFrame(data)

    async def replay_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
        correlation_id: Optional[str] = None
    ) -> int:
        """
        Replay events from history

        Args:
            start_time: Start of replay window
            end_time: End of replay window
            event_types: Optional filter by event types
            correlation_id: Optional filter by correlation ID

        Returns:
            Number of events replayed

        CRITICAL: Used for disaster recovery and testing
        """
        if not self.enable_replay:
            raise RuntimeError("Event replay is disabled")

        replayed = 0

        for event in self._event_history:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if correlation_id and event.correlation_id != correlation_id:
                continue

            # Replay event
            try:
                await self._queues[event.priority].put(event)
                replayed += 1
            except Exception as e:
                print(f"Failed to replay event {event.event_id}: {e}")

        return replayed

    def get_dead_letter_queue(self) -> List[Event]:
        """Get all events in dead letter queue"""
        return list(self._dlq)

    def get_dead_letter_queue_df(self) -> pl.DataFrame:
        """
        Get dead letter queue as polars DataFrame

        Returns:
            DataFrame with failed events
        """
        if not self._dlq:
            return pl.DataFrame(
                schema={
                    'event_id': pl.Utf8,
                    'event_type': pl.Utf8,
                    'priority': pl.Int32,
                    'timestamp': pl.Utf8,
                    'source': pl.Utf8,
                    'retry_count': pl.Int32,
                    'correlation_id': pl.Utf8
                }
            )

        data = {
            'event_id': [e.event_id for e in self._dlq],
            'event_type': [e.event_type.value for e in self._dlq],
            'priority': [e.priority.value for e in self._dlq],
            'timestamp': [e.timestamp.isoformat() for e in self._dlq],
            'source': [e.source for e in self._dlq],
            'retry_count': [e.retry_count for e in self._dlq],
            'correlation_id': [e.correlation_id or '' for e in self._dlq]
        }

        return pl.DataFrame(data)

    async def retry_dead_letter(self, event_id: str) -> bool:
        """
        Retry specific event from dead letter queue

        Args:
            event_id: Event ID to retry

        Returns:
            True if event was found and retried
        """
        for i, event in enumerate(self._dlq):
            if event.event_id == event_id:
                # Remove from DLQ
                event_to_retry = self._dlq[i]
                del self._dlq[i]

                # Reset retry count
                event_to_retry.retry_count = 0

                # Re-queue
                await self._queues[event_to_retry.priority].put(event_to_retry)
                return True

        return False

    async def clear_dead_letter_queue(self) -> int:
        """
        Clear dead letter queue

        Returns:
            Number of events cleared
        """
        count = len(self._dlq)
        self._dlq.clear()
        return count

    def get_event_history(
        self,
        limit: int = 100,
        event_type: Optional[EventType] = None
    ) -> pl.DataFrame:
        """
        Get recent event history as DataFrame

        Args:
            limit: Maximum events to return
            event_type: Optional filter by type

        Returns:
            polars DataFrame with event history
        """
        if not self.enable_replay:
            raise RuntimeError("Event replay/history is disabled")

        # Filter events
        filtered_events = self._event_history
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        # Limit results
        events = list(filtered_events)[-limit:]

        if not events:
            return pl.DataFrame(
                schema={
                    'event_id': pl.Utf8,
                    'event_type': pl.Utf8,
                    'priority': pl.Int32,
                    'timestamp': pl.Utf8,
                    'source': pl.Utf8,
                    'correlation_id': pl.Utf8
                }
            )

        data = {
            'event_id': [e.event_id for e in events],
            'event_type': [e.event_type.value for e in events],
            'priority': [e.priority.value for e in events],
            'timestamp': [e.timestamp.isoformat() for e in events],
            'source': [e.source for e in events],
            'correlation_id': [e.correlation_id or '' for e in events]
        }

        return pl.DataFrame(data)
