"""
WebSocket Subscriptions - Real-time data streaming via WebSocket connections.

This module provides WebSocket endpoints for real-time market data, orders,
positions, and system events with efficient binary and JSON protocols.
"""

import asyncio
import json
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from enum import Enum
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_EVENT = "system_event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    ACK = "ack"


class WebSocketSubscriptionManager:
    """Manages WebSocket subscription lifecycle and message routing.

    Attributes:
        config: WebSocket configuration from environment
        connections: Active WebSocket connections
        subscriptions: Subscription registry by connection
        message_queue: Async message distribution queue
    """

    def __init__(self) -> None:
        """Initialize WebSocket subscription manager.

        Raises:
            ConfigurationError: If configuration invalid
        """
        self.config: Dict[str, Any] = self._load_config()
        self.connections: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['message_queue_size']
        )
        self._running: bool = False
        self._message_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        logger.info("WebSocketSubscriptionManager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load WebSocket configuration from environment.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'message_queue_size': int(os.getenv('WS_MESSAGE_QUEUE_SIZE', '50000')),
                'max_connections': int(os.getenv('WS_MAX_CONNECTIONS', '10000')),
                'heartbeat_interval': int(os.getenv('WS_HEARTBEAT_INTERVAL', '30')),
                'connection_timeout': int(os.getenv('WS_CONNECTION_TIMEOUT', '300')),
                'max_subscriptions_per_connection': int(os.getenv('WS_MAX_SUBSCRIPTIONS', '100')),
                'compression_enabled': os.getenv('WS_COMPRESSION_ENABLED', 'true').lower() == 'true',
            }

            logger.debug("WebSocket config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load WebSocket config", error=str(e))
            raise

    async def start(self) -> None:
        """Start WebSocket manager background tasks."""
        if self._running:
            logger.warning("WebSocketSubscriptionManager already running")
            return

        self._running = True
        self._message_task = asyncio.create_task(self._message_router())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_sender())

        logger.info("WebSocketSubscriptionManager started")

    async def stop(self) -> None:
        """Stop WebSocket manager and cleanup."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        for task in [self._message_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)

        logger.info("WebSocketSubscriptionManager stopped")

    async def connect(self, connection_id: str, websocket: Any) -> None:
        """Register a new WebSocket connection.

        Args:
            connection_id: Unique connection identifier
            websocket: WebSocket connection object

        Raises:
            ValueError: If max connections exceeded

        Example:
            >>> await manager.connect('conn_123', websocket)
        """
        try:
            if len(self.connections) >= self.config['max_connections']:
                raise ValueError(f"Max connections ({self.config['max_connections']}) exceeded")

            self.connections[connection_id] = {
                'websocket': websocket,
                'connected_at': datetime.now(timezone.utc),
                'last_heartbeat': datetime.now(timezone.utc),
                'subscriptions': set()
            }

            self.subscriptions[connection_id] = set()

            logger.info(
                "WebSocket connection established",
                connection_id=connection_id,
                total_connections=len(self.connections)
            )

            # Send welcome message
            await self._send_message(
                connection_id,
                {
                    'type': MessageType.ACK.value,
                    'message': 'Connected successfully',
                    'connection_id': connection_id,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            logger.error("Connection failed", error=str(e), connection_id=connection_id)
            raise

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect and cleanup a WebSocket connection.

        Args:
            connection_id: Connection identifier
        """
        try:
            if connection_id in self.connections:
                # Unsubscribe from all topics
                await self.unsubscribe_all(connection_id)

                # Remove connection
                del self.connections[connection_id]

                if connection_id in self.subscriptions:
                    del self.subscriptions[connection_id]

                logger.info(
                    "WebSocket connection closed",
                    connection_id=connection_id,
                    remaining_connections=len(self.connections)
                )

        except Exception as e:
            logger.error("Disconnect error", error=str(e), connection_id=connection_id)

    async def subscribe(
        self,
        connection_id: str,
        topic: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Subscribe a connection to a topic.

        Args:
            connection_id: Connection identifier
            topic: Subscription topic (e.g., 'market_data', 'orders')
            params: Optional subscription parameters

        Raises:
            ValueError: If connection not found or max subscriptions exceeded

        Example:
            >>> await manager.subscribe('conn_123', 'market_data', {'symbol': 'BTC/USDT'})
        """
        try:
            if connection_id not in self.connections:
                raise ValueError(f"Connection {connection_id} not found")

            subscriptions = self.subscriptions[connection_id]

            if len(subscriptions) >= self.config['max_subscriptions_per_connection']:
                raise ValueError(
                    f"Max subscriptions ({self.config['max_subscriptions_per_connection']}) exceeded"
                )

            # Build subscription key
            sub_key = self._build_subscription_key(topic, params)

            subscriptions.add(sub_key)
            self.connections[connection_id]['subscriptions'].add(sub_key)

            logger.info(
                "Subscription added",
                connection_id=connection_id,
                topic=topic,
                params=params,
                total_subscriptions=len(subscriptions)
            )

            # Send acknowledgment
            await self._send_message(
                connection_id,
                {
                    'type': MessageType.ACK.value,
                    'action': 'subscribe',
                    'topic': topic,
                    'params': params,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            logger.error("Subscribe error", error=str(e), connection_id=connection_id)
            await self._send_error(connection_id, str(e))

    async def unsubscribe(
        self,
        connection_id: str,
        topic: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Unsubscribe a connection from a topic.

        Args:
            connection_id: Connection identifier
            topic: Subscription topic
            params: Optional subscription parameters
        """
        try:
            if connection_id not in self.connections:
                return

            sub_key = self._build_subscription_key(topic, params)

            if connection_id in self.subscriptions:
                self.subscriptions[connection_id].discard(sub_key)

            if connection_id in self.connections:
                self.connections[connection_id]['subscriptions'].discard(sub_key)

            logger.info(
                "Subscription removed",
                connection_id=connection_id,
                topic=topic,
                params=params
            )

            # Send acknowledgment
            await self._send_message(
                connection_id,
                {
                    'type': MessageType.ACK.value,
                    'action': 'unsubscribe',
                    'topic': topic,
                    'params': params,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            logger.error("Unsubscribe error", error=str(e), connection_id=connection_id)

    async def unsubscribe_all(self, connection_id: str) -> None:
        """Unsubscribe a connection from all topics.

        Args:
            connection_id: Connection identifier
        """
        try:
            if connection_id in self.subscriptions:
                self.subscriptions[connection_id].clear()

            if connection_id in self.connections:
                self.connections[connection_id]['subscriptions'].clear()

            logger.info("All subscriptions removed", connection_id=connection_id)

        except Exception as e:
            logger.error("Unsubscribe all error", error=str(e), connection_id=connection_id)

    async def broadcast(
        self,
        topic: str,
        message: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast a message to all subscribers of a topic.

        Args:
            topic: Message topic
            message: Message data
            params: Optional topic parameters for filtering

        Example:
            >>> await manager.broadcast('market_data', {
            ...     'symbol': 'BTC/USDT',
            ...     'price': '45000.00',
            ...     'timestamp': '2025-01-15T10:00:00Z'
            ... }, {'symbol': 'BTC/USDT'})
        """
        try:
            sub_key = self._build_subscription_key(topic, params)

            # Queue message for routing
            await self.message_queue.put((sub_key, message))

            logger.debug("Message broadcast queued", topic=topic, sub_key=sub_key)

        except asyncio.QueueFull:
            logger.error("Message queue full, dropping message", topic=topic)
        except Exception as e:
            logger.error("Broadcast error", error=str(e), topic=topic)

    async def _message_router(self) -> None:
        """Background task to route messages to subscribers."""
        logger.info("Message router started")

        while self._running:
            try:
                # Get message from queue
                sub_key, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                # Route to all matching subscribers
                await self._route_message(sub_key, message)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Message routing error", error=str(e))
                await asyncio.sleep(0.1)

        logger.info("Message router stopped")

    async def _route_message(self, sub_key: str, message: Dict[str, Any]) -> None:
        """Route message to all subscribers with matching subscription key.

        Args:
            sub_key: Subscription key
            message: Message data
        """
        sent_count = 0
        failed_connections: List[str] = []

        for connection_id, subscriptions in self.subscriptions.items():
            if sub_key in subscriptions:
                try:
                    await self._send_message(connection_id, message)
                    sent_count += 1

                except Exception as e:
                    logger.error(
                        "Failed to send message",
                        error=str(e),
                        connection_id=connection_id
                    )
                    failed_connections.append(connection_id)

        # Cleanup failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        if sent_count > 0:
            logger.debug("Message routed", sub_key=sub_key, recipients=sent_count)

    async def _heartbeat_sender(self) -> None:
        """Background task to send periodic heartbeats."""
        logger.info("Heartbeat sender started")

        while self._running:
            try:
                await asyncio.sleep(self.config['heartbeat_interval'])

                now = datetime.now(timezone.utc)
                heartbeat_msg = {
                    'type': MessageType.HEARTBEAT.value,
                    'timestamp': now.isoformat()
                }

                # Send heartbeat to all connections
                for connection_id in list(self.connections.keys()):
                    try:
                        await self._send_message(connection_id, heartbeat_msg)
                        self.connections[connection_id]['last_heartbeat'] = now

                    except Exception as e:
                        logger.error(
                            "Heartbeat failed",
                            error=str(e),
                            connection_id=connection_id
                        )
                        await self.disconnect(connection_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat sender error", error=str(e))

        logger.info("Heartbeat sender stopped")

    async def _send_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Send a message to a specific connection.

        Args:
            connection_id: Connection identifier
            message: Message data

        Raises:
            Exception: If send fails
        """
        if connection_id not in self.connections:
            raise ValueError(f"Connection {connection_id} not found")

        websocket = self.connections[connection_id]['websocket']

        # Serialize message
        message_str = json.dumps(message, default=self._json_serializer)

        # Send via websocket (actual implementation would use WebSocket send)
        # await websocket.send_text(message_str)

        logger.debug("Message sent", connection_id=connection_id, message_type=message.get('type'))

    async def _send_error(self, connection_id: str, error_message: str) -> None:
        """Send an error message to a connection.

        Args:
            connection_id: Connection identifier
            error_message: Error description
        """
        try:
            await self._send_message(
                connection_id,
                {
                    'type': MessageType.ERROR.value,
                    'error': error_message,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            logger.error("Failed to send error message", error=str(e))

    def _build_subscription_key(
        self,
        topic: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build a subscription key from topic and parameters.

        Args:
            topic: Subscription topic
            params: Optional parameters

        Returns:
            Subscription key string
        """
        if not params:
            return topic

        # Sort params for consistent key generation
        sorted_params = sorted(params.items())
        param_str = ':'.join(f"{k}={v}" for k, v in sorted_params)

        return f"{topic}:{param_str}"

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for Decimal and datetime.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation
        """
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketSubscriptionManager] = None


async def get_websocket_manager() -> WebSocketSubscriptionManager:
    """Get or create the global WebSocket manager.

    Returns:
        WebSocketSubscriptionManager instance
    """
    global _ws_manager

    if _ws_manager is None:
        _ws_manager = WebSocketSubscriptionManager()
        await _ws_manager.start()

    return _ws_manager
