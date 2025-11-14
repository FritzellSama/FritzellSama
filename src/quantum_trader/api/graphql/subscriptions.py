"""
GraphQL Subscriptions - Real-time data streaming via GraphQL subscriptions.

This module provides GraphQL subscription endpoints for real-time updates
of market data, orders, positions, and system events.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime, timezone
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class SubscriptionManager:
    """Manages GraphQL subscription lifecycle and event distribution.

    Attributes:
        config: Subscription configuration from environment
        subscribers: Active subscription registry
        event_queue: Async queue for event distribution
    """

    def __init__(self) -> None:
        """Initialize subscription manager.

        Raises:
            ConfigurationError: If required configuration missing
        """
        self.config: Dict[str, Any] = self._load_config()
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['event_queue_size']
        )
        self._running: bool = False
        self._event_task: Optional[asyncio.Task] = None

        logger.info("SubscriptionManager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load subscription configuration from environment.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'event_queue_size': int(os.getenv('GRAPHQL_EVENT_QUEUE_SIZE', '10000')),
                'max_subscribers_per_topic': int(os.getenv('GRAPHQL_MAX_SUBSCRIBERS', '1000')),
                'heartbeat_interval': int(os.getenv('GRAPHQL_HEARTBEAT_INTERVAL', '30')),
                'subscriber_timeout': int(os.getenv('GRAPHQL_SUBSCRIBER_TIMEOUT', '300')),
            }

            logger.debug("GraphQL subscription config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load subscription config", error=str(e))
            raise

    async def start(self) -> None:
        """Start subscription manager event loop."""
        if self._running:
            logger.warning("SubscriptionManager already running")
            return

        self._running = True
        self._event_task = asyncio.create_task(self._event_distributor())

        logger.info("SubscriptionManager started")

    async def stop(self) -> None:
        """Stop subscription manager and cleanup."""
        if not self._running:
            return

        self._running = False

        if self._event_task:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass

        # Clear all subscribers
        for topic in list(self.subscribers.keys()):
            await self._clear_topic(topic)

        logger.info("SubscriptionManager stopped")

    async def subscribe_market_data(
        self,
        symbol: str,
        exchange: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to real-time market data updates.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name

        Yields:
            Market data update events

        Example:
            >>> async for update in manager.subscribe_market_data('BTC/USDT', 'BINANCE'):
            ...     print(update['price'])
        """
        topic = f"market_data:{exchange}:{symbol}"
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        try:
            # Register subscriber
            await self._register_subscriber(topic, queue)

            logger.info(
                "Market data subscription started",
                symbol=symbol,
                exchange=exchange,
                topic=topic
            )

            # Yield events from queue
            while self._running:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config['heartbeat_interval']
                    )

                    yield event

                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

        except asyncio.CancelledError:
            logger.info("Market data subscription cancelled", topic=topic)
        except Exception as e:
            logger.error("Market data subscription error", error=str(e), topic=topic)
        finally:
            await self._unregister_subscriber(topic, queue)

    async def subscribe_orders(
        self,
        strategy_id: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to order status updates.

        Args:
            strategy_id: Filter by strategy (None for all)
            exchange: Filter by exchange (None for all)

        Yields:
            Order update events

        Example:
            >>> async for order in manager.subscribe_orders(strategy_id='mom_v1'):
            ...     print(order['status'])
        """
        topic_parts = ['orders']
        if strategy_id:
            topic_parts.append(f"strategy:{strategy_id}")
        if exchange:
            topic_parts.append(f"exchange:{exchange}")

        topic = ':'.join(topic_parts)
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        try:
            await self._register_subscriber(topic, queue)

            logger.info(
                "Order subscription started",
                strategy_id=strategy_id,
                exchange=exchange,
                topic=topic
            )

            while self._running:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config['heartbeat_interval']
                    )

                    yield event

                except asyncio.TimeoutError:
                    yield {
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

        except asyncio.CancelledError:
            logger.info("Order subscription cancelled", topic=topic)
        except Exception as e:
            logger.error("Order subscription error", error=str(e), topic=topic)
        finally:
            await self._unregister_subscriber(topic, queue)

    async def subscribe_positions(
        self,
        strategy_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to position updates.

        Args:
            strategy_id: Filter by strategy (None for all)

        Yields:
            Position update events
        """
        topic = f"positions:{strategy_id}" if strategy_id else "positions:all"
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        try:
            await self._register_subscriber(topic, queue)

            logger.info("Position subscription started", strategy_id=strategy_id, topic=topic)

            while self._running:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config['heartbeat_interval']
                    )

                    yield event

                except asyncio.TimeoutError:
                    yield {
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

        except asyncio.CancelledError:
            logger.info("Position subscription cancelled", topic=topic)
        except Exception as e:
            logger.error("Position subscription error", error=str(e), topic=topic)
        finally:
            await self._unregister_subscriber(topic, queue)

    async def subscribe_performance(
        self,
        strategy_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to strategy performance updates.

        Args:
            strategy_id: Strategy identifier

        Yields:
            Performance metric updates

        Example:
            >>> async for perf in manager.subscribe_performance('mom_v1'):
            ...     print(perf['total_pnl'])
        """
        topic = f"performance:{strategy_id}"
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        try:
            await self._register_subscriber(topic, queue)

            logger.info("Performance subscription started", strategy_id=strategy_id, topic=topic)

            while self._running:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config['heartbeat_interval']
                    )

                    yield event

                except asyncio.TimeoutError:
                    yield {
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

        except asyncio.CancelledError:
            logger.info("Performance subscription cancelled", topic=topic)
        except Exception as e:
            logger.error("Performance subscription error", error=str(e), topic=topic)
        finally:
            await self._unregister_subscriber(topic, queue)

    async def subscribe_system_events(self) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to system-level events (errors, warnings, etc).

        Yields:
            System event updates
        """
        topic = "system:events"
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        try:
            await self._register_subscriber(topic, queue)

            logger.info("System events subscription started", topic=topic)

            while self._running:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.config['heartbeat_interval']
                    )

                    yield event

                except asyncio.TimeoutError:
                    yield {
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

        except asyncio.CancelledError:
            logger.info("System events subscription cancelled", topic=topic)
        except Exception as e:
            logger.error("System events subscription error", error=str(e), topic=topic)
        finally:
            await self._unregister_subscriber(topic, queue)

    async def publish_event(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish an event to all subscribers of a topic.

        Args:
            topic: Event topic
            event: Event data

        Raises:
            asyncio.QueueFull: If event queue is full
        """
        try:
            await self.event_queue.put((topic, event))

            logger.debug(
                "Event published",
                topic=topic,
                event_type=event.get('type', 'unknown')
            )

        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event", topic=topic)
        except Exception as e:
            logger.error("Failed to publish event", error=str(e), topic=topic)

    async def _event_distributor(self) -> None:
        """Background task to distribute events to subscribers."""
        logger.info("Event distributor started")

        while self._running:
            try:
                # Get event from queue with timeout
                topic, event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )

                # Distribute to all subscribers of this topic
                await self._distribute_to_subscribers(topic, event)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Event distribution error", error=str(e))
                await asyncio.sleep(0.1)

        logger.info("Event distributor stopped")

    async def _distribute_to_subscribers(
        self,
        topic: str,
        event: Dict[str, Any]
    ) -> None:
        """Distribute event to all subscribers of a topic.

        Args:
            topic: Event topic
            event: Event data
        """
        if topic not in self.subscribers:
            return

        failed_queues: List[asyncio.Queue] = []

        for queue in self.subscribers[topic]:
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(queue.put(event), timeout=0.1)

            except asyncio.TimeoutError:
                logger.warning("Subscriber queue full", topic=topic)
                failed_queues.append(queue)
            except Exception as e:
                logger.error("Failed to deliver event", error=str(e), topic=topic)
                failed_queues.append(queue)

        # Remove failed subscribers
        for queue in failed_queues:
            if queue in self.subscribers[topic]:
                self.subscribers[topic].remove(queue)

    async def _register_subscriber(self, topic: str, queue: asyncio.Queue) -> None:
        """Register a new subscriber for a topic.

        Args:
            topic: Subscription topic
            queue: Subscriber queue

        Raises:
            ValueError: If max subscribers exceeded
        """
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        if len(self.subscribers[topic]) >= self.config['max_subscribers_per_topic']:
            raise ValueError(f"Max subscribers for topic {topic} exceeded")

        self.subscribers[topic].append(queue)

        logger.debug(
            "Subscriber registered",
            topic=topic,
            total_subscribers=len(self.subscribers[topic])
        )

    async def _unregister_subscriber(self, topic: str, queue: asyncio.Queue) -> None:
        """Unregister a subscriber from a topic.

        Args:
            topic: Subscription topic
            queue: Subscriber queue
        """
        if topic in self.subscribers and queue in self.subscribers[topic]:
            self.subscribers[topic].remove(queue)

            # Remove topic if no subscribers left
            if not self.subscribers[topic]:
                del self.subscribers[topic]

            logger.debug(
                "Subscriber unregistered",
                topic=topic,
                remaining_subscribers=len(self.subscribers.get(topic, []))
            )

    async def _clear_topic(self, topic: str) -> None:
        """Clear all subscribers from a topic.

        Args:
            topic: Topic to clear
        """
        if topic in self.subscribers:
            del self.subscribers[topic]
            logger.debug("Topic cleared", topic=topic)


# Global subscription manager instance
_subscription_manager: Optional[SubscriptionManager] = None


async def get_subscription_manager() -> SubscriptionManager:
    """Get or create the global subscription manager.

    Returns:
        SubscriptionManager instance
    """
    global _subscription_manager

    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
        await _subscription_manager.start()

    return _subscription_manager
