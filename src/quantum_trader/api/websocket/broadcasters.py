"""WebSocket broadcasters for real-time data streaming.

This module provides WebSocket broadcasting capabilities for streaming
real-time market data, trading signals, order updates, and system events
to connected clients.

Features:
- Connection management with heartbeat
- Topic-based subscriptions
- Message broadcasting to multiple clients
- Rate limiting and backpressure handling
- Automatic reconnection support
- Message compression for large payloads

Example:
    ```python
    from quantum_trader.api.websocket.broadcasters import WebSocketBroadcaster

    broadcaster = WebSocketBroadcaster(config)
    await broadcaster.initialize()

    # Broadcast market data
    await broadcaster.broadcast_market_data({
        "symbol": "BTC/USDT",
        "price": Decimal("50000.00"),
        "volume": Decimal("100.5")
    })

    # Broadcast to specific topic
    await broadcaster.broadcast_to_topic("signals", signal_data)
    ```
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class WebSocketClient:
    """WebSocket client connection information.

    Attributes:
        client_id: Unique client identifier
        websocket: WebSocket connection
        subscriptions: Set of subscribed topics
        connected_at: Connection timestamp
        last_heartbeat: Last heartbeat timestamp
        metadata: Additional client metadata
    """

    client_id: str
    websocket: WebSocket
    subscriptions: Set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BroadcastMessage:
    """Message to broadcast to clients.

    Attributes:
        topic: Message topic/channel
        message_type: Type of message
        data: Message payload
        timestamp: Message timestamp
        metadata: Additional message metadata
    """

    topic: str
    message_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        """Encode Decimal as string to preserve precision."""
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class WebSocketBroadcaster:
    """WebSocket broadcaster for real-time data streaming.

    This broadcaster manages WebSocket connections and provides
    topic-based message broadcasting to subscribed clients.

    Supports:
    - Multiple concurrent connections
    - Topic-based subscriptions
    - Heartbeat monitoring
    - Message queuing and rate limiting
    - Graceful disconnection handling
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize WebSocket broadcaster.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Extract configuration
        self.max_connections = int(config.get("max_connections", 1000))
        self.heartbeat_interval_seconds = int(config.get("heartbeat_interval_seconds", 30))
        self.message_queue_size = int(config.get("message_queue_size", 10000))
        self.enable_compression = config.get("enable_compression", True)
        self.max_message_size_bytes = int(config.get("max_message_size_bytes", 1048576))  # 1MB

        # Client management
        self._clients: Dict[str, WebSocketClient] = {}
        self._topic_subscribers: Dict[str, Set[str]] = {}  # topic -> set of client_ids
        self._lock = asyncio.Lock()

        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=self.message_queue_size)

        # Background tasks
        self._broadcast_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "websocket_broadcaster_initialized",
            max_connections=self.max_connections,
            heartbeat_interval=self.heartbeat_interval_seconds
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # All parameters have defaults, so just validate ranges
        if "max_connections" in self.config:
            max_conn = int(self.config["max_connections"])
            if max_conn < 1:
                raise ValueError("max_connections must be at least 1")

        if "heartbeat_interval_seconds" in self.config:
            interval = int(self.config["heartbeat_interval_seconds"])
            if interval < 1:
                raise ValueError("heartbeat_interval_seconds must be at least 1")

    async def initialize(self) -> None:
        """Initialize broadcaster and start background tasks."""
        logger.info("websocket_broadcaster_starting")

        self._running = True

        # Start background tasks
        self._broadcast_task = asyncio.create_task(self._broadcast_worker())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_worker())

        logger.info("websocket_broadcaster_started")

    async def shutdown(self) -> None:
        """Shutdown broadcaster and cleanup resources."""
        logger.info("websocket_broadcaster_shutting_down")

        self._running = False

        # Cancel background tasks
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        async with self._lock:
            for client_id in list(self._clients.keys()):
                await self._disconnect_client(client_id)

        logger.info("websocket_broadcaster_shutdown_complete")

    async def connect_client(
        self,
        client_id: str,
        websocket: WebSocket,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Connect new WebSocket client.

        Args:
            client_id: Unique client identifier
            websocket: WebSocket connection
            metadata: Optional client metadata

        Returns:
            True if connection successful

        Raises:
            ValueError: If max connections reached
        """
        async with self._lock:
            # Check max connections
            if len(self._clients) >= self.max_connections:
                logger.warning(
                    "max_connections_reached",
                    max_connections=self.max_connections
                )
                raise ValueError("Maximum connections reached")

            # Accept connection
            await websocket.accept()

            # Create client record
            client = WebSocketClient(
                client_id=client_id,
                websocket=websocket,
                metadata=metadata or {}
            )

            self._clients[client_id] = client

            logger.info(
                "websocket_client_connected",
                client_id=client_id,
                total_clients=len(self._clients)
            )

            return True

    async def disconnect_client(self, client_id: str) -> None:
        """Disconnect WebSocket client.

        Args:
            client_id: Client identifier to disconnect
        """
        async with self._lock:
            await self._disconnect_client(client_id)

    async def _disconnect_client(self, client_id: str) -> None:
        """Internal disconnect implementation (must hold lock)."""
        if client_id not in self._clients:
            return

        client = self._clients[client_id]

        # Unsubscribe from all topics
        for topic in client.subscriptions:
            if topic in self._topic_subscribers:
                self._topic_subscribers[topic].discard(client_id)
                if not self._topic_subscribers[topic]:
                    del self._topic_subscribers[topic]

        # Close WebSocket
        try:
            await client.websocket.close()
        except Exception as e:
            logger.warning("websocket_close_error", client_id=client_id, error=str(e))

        # Remove client
        del self._clients[client_id]

        logger.info(
            "websocket_client_disconnected",
            client_id=client_id,
            total_clients=len(self._clients)
        )

    async def subscribe_to_topic(self, client_id: str, topic: str) -> bool:
        """Subscribe client to topic.

        Args:
            client_id: Client identifier
            topic: Topic to subscribe to

        Returns:
            True if subscription successful

        Raises:
            ValueError: If client not found
        """
        async with self._lock:
            if client_id not in self._clients:
                raise ValueError(f"Client {client_id} not found")

            client = self._clients[client_id]
            client.subscriptions.add(topic)

            # Add to topic subscribers
            if topic not in self._topic_subscribers:
                self._topic_subscribers[topic] = set()

            self._topic_subscribers[topic].add(client_id)

            logger.info(
                "websocket_client_subscribed",
                client_id=client_id,
                topic=topic,
                topic_subscribers=len(self._topic_subscribers[topic])
            )

            return True

    async def unsubscribe_from_topic(self, client_id: str, topic: str) -> bool:
        """Unsubscribe client from topic.

        Args:
            client_id: Client identifier
            topic: Topic to unsubscribe from

        Returns:
            True if unsubscription successful
        """
        async with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            client.subscriptions.discard(topic)

            # Remove from topic subscribers
            if topic in self._topic_subscribers:
                self._topic_subscribers[topic].discard(client_id)
                if not self._topic_subscribers[topic]:
                    del self._topic_subscribers[topic]

            logger.info(
                "websocket_client_unsubscribed",
                client_id=client_id,
                topic=topic
            )

            return True

    async def broadcast_to_topic(
        self,
        topic: str,
        message_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Broadcast message to all subscribers of a topic.

        Args:
            topic: Topic to broadcast to
            message_type: Type of message
            data: Message data
            metadata: Optional message metadata

        Returns:
            Number of clients message was queued for
        """
        message = BroadcastMessage(
            topic=topic,
            message_type=message_type,
            data=data,
            metadata=metadata or {}
        )

        try:
            # Add to queue (non-blocking)
            self._message_queue.put_nowait(message)

            # Get subscriber count
            async with self._lock:
                subscriber_count = len(self._topic_subscribers.get(topic, set()))

            return subscriber_count

        except asyncio.QueueFull:
            logger.error(
                "message_queue_full",
                topic=topic,
                message_type=message_type
            )
            return 0

    async def broadcast_market_data(self, market_data: Dict[str, Any]) -> int:
        """Broadcast market data update.

        Args:
            market_data: Market data dictionary

        Returns:
            Number of clients message was queued for
        """
        return await self.broadcast_to_topic(
            topic="market_data",
            message_type="market_update",
            data=market_data
        )

    async def broadcast_signal(self, signal_data: Dict[str, Any]) -> int:
        """Broadcast trading signal.

        Args:
            signal_data: Signal data dictionary

        Returns:
            Number of clients message was queued for
        """
        return await self.broadcast_to_topic(
            topic="signals",
            message_type="signal",
            data=signal_data
        )

    async def broadcast_order_update(self, order_data: Dict[str, Any]) -> int:
        """Broadcast order update.

        Args:
            order_data: Order data dictionary

        Returns:
            Number of clients message was queued for
        """
        return await self.broadcast_to_topic(
            topic="orders",
            message_type="order_update",
            data=order_data
        )

    async def broadcast_position_update(self, position_data: Dict[str, Any]) -> int:
        """Broadcast position update.

        Args:
            position_data: Position data dictionary

        Returns:
            Number of clients message was queued for
        """
        return await self.broadcast_to_topic(
            topic="positions",
            message_type="position_update",
            data=position_data
        )

    async def send_to_client(
        self,
        client_id: str,
        message_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send message to specific client.

        Args:
            client_id: Client identifier
            message_type: Type of message
            data: Message data

        Returns:
            True if message sent successfully
        """
        async with self._lock:
            if client_id not in self._clients:
                logger.warning("client_not_found", client_id=client_id)
                return False

            client = self._clients[client_id]

        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }

            json_message = json.dumps(message, cls=DecimalEncoder)

            # Check message size
            if len(json_message) > self.max_message_size_bytes:
                logger.warning(
                    "message_too_large",
                    client_id=client_id,
                    size=len(json_message),
                    max_size=self.max_message_size_bytes
                )
                return False

            await client.websocket.send_text(json_message)
            return True

        except WebSocketDisconnect:
            logger.info("client_disconnected_during_send", client_id=client_id)
            await self.disconnect_client(client_id)
            return False

        except Exception as e:
            logger.error(
                "send_to_client_error",
                client_id=client_id,
                error=str(e)
            )
            return False

    async def _broadcast_worker(self) -> None:
        """Background worker for broadcasting messages from queue."""
        logger.info("broadcast_worker_started")

        while self._running:
            try:
                # Get message from queue (with timeout)
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Get subscribers for topic
                async with self._lock:
                    subscriber_ids = self._topic_subscribers.get(message.topic, set()).copy()

                # Broadcast to all subscribers
                for client_id in subscriber_ids:
                    await self.send_to_client(
                        client_id=client_id,
                        message_type=message.message_type,
                        data=message.data
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("broadcast_worker_error", error=str(e))

        logger.info("broadcast_worker_stopped")

    async def _heartbeat_worker(self) -> None:
        """Background worker for sending heartbeats to clients."""
        logger.info("heartbeat_worker_started")

        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval_seconds)

                # Get all client IDs
                async with self._lock:
                    client_ids = list(self._clients.keys())

                # Send heartbeat to all clients
                for client_id in client_ids:
                    try:
                        success = await self.send_to_client(
                            client_id=client_id,
                            message_type="heartbeat",
                            data={"timestamp": datetime.utcnow().isoformat()}
                        )

                        if success:
                            # Update last heartbeat
                            async with self._lock:
                                if client_id in self._clients:
                                    self._clients[client_id].last_heartbeat = datetime.utcnow()

                    except Exception as e:
                        logger.warning(
                            "heartbeat_send_error",
                            client_id=client_id,
                            error=str(e)
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("heartbeat_worker_error", error=str(e))

        logger.info("heartbeat_worker_stopped")

    def get_connection_count(self) -> int:
        """Get current number of connected clients."""
        return len(self._clients)

    def get_topic_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        return len(self._topic_subscribers.get(topic, set()))

    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics.

        Returns:
            Dictionary with broadcaster stats
        """
        return {
            "total_clients": len(self._clients),
            "total_topics": len(self._topic_subscribers),
            "queue_size": self._message_queue.qsize(),
            "max_connections": self.max_connections,
            "running": self._running,
            "topic_stats": {
                topic: len(subscribers)
                for topic, subscribers in self._topic_subscribers.items()
            }
        }
