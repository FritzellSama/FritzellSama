"""
WebSocket Broadcasters

Production-ready WebSocket connection management and broadcasting.
Handles real-time market data, trade updates, and system notifications.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from collections import defaultdict
import json
import os

from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class ConnectionManager:
    """
    Manage WebSocket connections and message broadcasting.

    Handles connection lifecycle, subscription management,
    and efficient message broadcasting to multiple clients.

    Attributes:
        active_connections: Set of active WebSocket connections
        subscriptions: Mapping of topics to subscribed connections
        connection_metadata: Metadata for each connection

    Example:
        >>> manager = ConnectionManager()
        >>> await manager.connect(websocket, client_id)
        >>> await manager.broadcast("market_data", {"price": "50000"})
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        # Active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Topic subscriptions: topic -> set of websockets
        self.subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)

        # Connection metadata: websocket -> metadata dict
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Message queue for each connection
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}

        # Connection limits
        self.max_connections: int = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
        self.max_subscriptions_per_connection: int = int(
            os.getenv("WS_MAX_SUBSCRIPTIONS_PER_CONNECTION", "50")
        )

        logger.info(
            "connection_manager_initialized",
            max_connections=self.max_connections
        )

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Accept and register new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
            metadata: Optional connection metadata

        Returns:
            True if connection accepted, False if rejected

        Raises:
            Exception: If connection fails
        """
        try:
            # Check connection limit
            if len(self.active_connections) >= self.max_connections:
                logger.warning("max_connections_reached", current=len(self.active_connections))
                await websocket.close(code=1008, reason="Maximum connections reached")
                return False

            # Accept connection
            await websocket.accept()

            # Register connection
            self.active_connections.add(websocket)

            # Store metadata
            self.connection_metadata[websocket] = {
                "client_id": client_id or f"client_{id(websocket)}",
                "connected_at": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "message_count": 0
            }

            # Create message queue
            self.message_queues[websocket] = asyncio.Queue()

            logger.info(
                "websocket_connected",
                client_id=self.connection_metadata[websocket]["client_id"],
                total_connections=len(self.active_connections)
            )

            return True

        except Exception as e:
            logger.error("websocket_connection_failed", error=str(e))
            raise

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect and cleanup WebSocket connection.

        Args:
            websocket: WebSocket connection to disconnect
        """
        try:
            # Remove from active connections
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

            # Remove from all subscriptions
            for topic_subscribers in self.subscriptions.values():
                topic_subscribers.discard(websocket)

            # Clean up metadata
            client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]

            # Clean up message queue
            if websocket in self.message_queues:
                del self.message_queues[websocket]

            logger.info(
                "websocket_disconnected",
                client_id=client_id,
                total_connections=len(self.active_connections)
            )

        except Exception as e:
            logger.error("websocket_disconnect_failed", error=str(e))

    async def subscribe(
        self,
        websocket: WebSocket,
        topic: str
    ) -> bool:
        """
        Subscribe connection to topic.

        Args:
            websocket: WebSocket connection
            topic: Topic to subscribe to

        Returns:
            True if subscription successful
        """
        try:
            # Check subscription limit
            current_subs = sum(1 for subs in self.subscriptions.values() if websocket in subs)
            if current_subs >= self.max_subscriptions_per_connection:
                logger.warning(
                    "max_subscriptions_reached",
                    client_id=self.connection_metadata.get(websocket, {}).get("client_id")
                )
                return False

            # Add subscription
            self.subscriptions[topic].add(websocket)

            client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            logger.info("websocket_subscribed", client_id=client_id, topic=topic)

            return True

        except Exception as e:
            logger.error("subscription_failed", topic=topic, error=str(e))
            return False

    async def unsubscribe(
        self,
        websocket: WebSocket,
        topic: str
    ) -> None:
        """
        Unsubscribe connection from topic.

        Args:
            websocket: WebSocket connection
            topic: Topic to unsubscribe from
        """
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(websocket)

                client_id = self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
                logger.info("websocket_unsubscribed", client_id=client_id, topic=topic)

        except Exception as e:
            logger.error("unsubscribe_failed", topic=topic, error=str(e))

    async def send_personal_message(
        self,
        websocket: WebSocket,
        message: Dict[str, Any]
    ) -> None:
        """
        Send message to specific connection.

        Args:
            websocket: WebSocket connection
            message: Message to send
        """
        try:
            message_json = json.dumps(message, default=str)
            await websocket.send_text(message_json)

            # Update message count
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1

        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            logger.error("send_personal_message_failed", error=str(e))
            await self.disconnect(websocket)

    async def broadcast(
        self,
        topic: str,
        message: Dict[str, Any]
    ) -> int:
        """
        Broadcast message to all subscribers of topic.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        try:
            if topic not in self.subscriptions:
                return 0

            subscribers = self.subscriptions[topic].copy()
            message_json = json.dumps(message, default=str)

            # Send to all subscribers concurrently
            send_tasks = []
            for websocket in subscribers:
                send_tasks.append(self._send_safe(websocket, message_json))

            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Count successful sends
            success_count = sum(1 for r in results if r is True)

            logger.debug(
                "broadcast_completed",
                topic=topic,
                sent=success_count,
                total_subscribers=len(subscribers)
            )

            return success_count

        except Exception as e:
            logger.error("broadcast_failed", topic=topic, error=str(e))
            return 0

    async def _send_safe(
        self,
        websocket: WebSocket,
        message_json: str
    ) -> bool:
        """
        Safely send message, handling disconnections.

        Args:
            websocket: WebSocket connection
            message_json: JSON-encoded message

        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_text(message_json)

            # Update message count
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["message_count"] += 1

            return True

        except WebSocketDisconnect:
            await self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error("send_safe_failed", error=str(e))
            await self.disconnect(websocket)
            return False

    async def broadcast_all(
        self,
        message: Dict[str, Any]
    ) -> int:
        """
        Broadcast message to all active connections.

        Args:
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        try:
            connections = self.active_connections.copy()
            message_json = json.dumps(message, default=str)

            # Send to all connections concurrently
            send_tasks = []
            for websocket in connections:
                send_tasks.append(self._send_safe(websocket, message_json))

            results = await asyncio.gather(*send_tasks, return_exceptions=True)

            # Count successful sends
            success_count = sum(1 for r in results if r is True)

            logger.debug(
                "broadcast_all_completed",
                sent=success_count,
                total_connections=len(connections)
            )

            return success_count

        except Exception as e:
            logger.error("broadcast_all_failed", error=str(e))
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection manager statistics.

        Returns:
            Dictionary with statistics
        """
        # Calculate total subscriptions
        total_subscriptions = sum(len(subs) for subs in self.subscriptions.values())

        # Calculate total messages sent
        total_messages = sum(
            meta.get("message_count", 0)
            for meta in self.connection_metadata.values()
        )

        return {
            "active_connections": len(self.active_connections),
            "total_topics": len(self.subscriptions),
            "total_subscriptions": total_subscriptions,
            "total_messages_sent": total_messages,
            "topics": {
                topic: len(subs)
                for topic, subs in self.subscriptions.items()
            }
        }


class MarketDataBroadcaster:
    """
    Specialized broadcaster for market data.

    Handles high-frequency market data broadcasting with
    rate limiting and data aggregation.
    """

    def __init__(self, connection_manager: ConnectionManager) -> None:
        """
        Initialize market data broadcaster.

        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager

        # Rate limiting
        self.max_updates_per_second: int = int(os.getenv("WS_MAX_UPDATES_PER_SECOND", "10"))

        # Last update timestamps
        self.last_updates: Dict[str, datetime] = {}

        logger.info("market_data_broadcaster_initialized")

    async def broadcast_ticker(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime
    ) -> int:
        """
        Broadcast ticker update.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Volume
            timestamp: Update timestamp

        Returns:
            Number of recipients
        """
        # Rate limit check
        topic = f"ticker:{symbol}"
        if not self._should_send(topic):
            return 0

        message = {
            "type": "ticker",
            "symbol": symbol,
            "price": str(price),
            "volume": str(volume),
            "timestamp": timestamp.isoformat()
        }

        count = await self.connection_manager.broadcast(topic, message)

        self.last_updates[topic] = datetime.now(timezone.utc)

        return count

    async def broadcast_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
        timestamp: datetime
    ) -> int:
        """
        Broadcast orderbook update.

        Args:
            symbol: Trading symbol
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            timestamp: Update timestamp

        Returns:
            Number of recipients
        """
        topic = f"orderbook:{symbol}"

        message = {
            "type": "orderbook",
            "symbol": symbol,
            "bids": [[str(p), str(s)] for p, s in bids],
            "asks": [[str(p), str(s)] for p, s in asks],
            "timestamp": timestamp.isoformat()
        }

        count = await self.connection_manager.broadcast(topic, message)

        return count

    async def broadcast_trade(
        self,
        symbol: str,
        trade_id: str,
        price: Decimal,
        quantity: Decimal,
        side: str,
        timestamp: datetime
    ) -> int:
        """
        Broadcast trade update.

        Args:
            symbol: Trading symbol
            trade_id: Trade ID
            price: Trade price
            quantity: Trade quantity
            side: Trade side (BUY/SELL)
            timestamp: Trade timestamp

        Returns:
            Number of recipients
        """
        topic = f"trades:{symbol}"

        message = {
            "type": "trade",
            "symbol": symbol,
            "trade_id": trade_id,
            "price": str(price),
            "quantity": str(quantity),
            "side": side,
            "timestamp": timestamp.isoformat()
        }

        count = await self.connection_manager.broadcast(topic, message)

        return count

    def _should_send(self, topic: str) -> bool:
        """Check if update should be sent based on rate limit."""
        if topic not in self.last_updates:
            return True

        time_since_last = (datetime.now(timezone.utc) - self.last_updates[topic]).total_seconds()
        min_interval = 1.0 / self.max_updates_per_second

        return time_since_last >= min_interval


# Global connection manager instance
connection_manager = ConnectionManager()
market_data_broadcaster = MarketDataBroadcaster(connection_manager)
