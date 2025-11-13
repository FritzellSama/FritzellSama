"""
WebSocket broadcasters for Quantum Trader AI.

Handles broadcasting of real-time data to connected WebSocket clients.
"""

import asyncio
import json
from decimal import Decimal
from typing import Optional, Dict, List, Any, Set
from datetime import datetime
from enum import Enum
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BroadcastChannel(str, Enum):
    """Broadcast channel enumeration."""
    MARKET_DATA = "market_data"
    ORDER_UPDATES = "order_updates"
    POSITION_UPDATES = "position_updates"
    TRADE_UPDATES = "trade_updates"
    ALERTS = "alerts"
    SYSTEM = "system"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        """Encode Decimal as string.

        Args:
            obj: Object to encode

        Returns:
            Encoded object
        """
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class WebSocketBroadcaster:
    """Production-ready WebSocket broadcaster.

    Manages broadcasting of real-time data to connected WebSocket clients
    with channel subscription, filtering, and rate limiting.

    Attributes:
        config: Configuration dictionary
        connections: Active WebSocket connections
        subscriptions: Channel subscriptions by connection
        rate_limiter: Rate limiting configuration
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize WebSocket broadcaster.

        Args:
            config: Configuration from config files

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config

        # Connection tracking
        self.connections: Dict[str, Any] = {}  # connection_id -> websocket
        self.subscriptions: Dict[str, Set[BroadcastChannel]] = {}  # connection_id -> channels
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids

        # Rate limiting
        broadcast_config = config.get("websocket", {}).get("broadcast", {})
        self.max_messages_per_second = broadcast_config.get("max_messages_per_second", 100)
        self.max_message_size_bytes = broadcast_config.get("max_message_size_bytes", 65536)

        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None

        self._validate_config()
        logger.info("WebSocket broadcaster initialized")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_messages_per_second <= 0:
            raise ValueError("max_messages_per_second must be positive")
        if self.max_message_size_bytes <= 0:
            raise ValueError("max_message_size_bytes must be positive")

    async def start(self) -> None:
        """Start broadcaster background task."""
        if self._broadcast_task is None:
            self._broadcast_task = asyncio.create_task(self._broadcast_worker())
            logger.info("Broadcaster started")

    async def stop(self) -> None:
        """Stop broadcaster background task."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
            logger.info("Broadcaster stopped")

    def register_connection(
        self,
        connection_id: str,
        websocket: Any,
        user_id: Optional[str] = None
    ) -> None:
        """Register WebSocket connection.

        Args:
            connection_id: Unique connection identifier
            websocket: WebSocket connection object
            user_id: User ID (optional)
        """
        self.connections[connection_id] = websocket
        self.subscriptions[connection_id] = set()

        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)

        logger.info(
            "Connection registered",
            connection_id=connection_id,
            user_id=user_id
        )

    def unregister_connection(self, connection_id: str) -> None:
        """Unregister WebSocket connection.

        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.connections:
            del self.connections[connection_id]

        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]

        # Remove from user connections
        for user_id, conn_ids in self.user_connections.items():
            if connection_id in conn_ids:
                conn_ids.remove(connection_id)
                break

        logger.info("Connection unregistered", connection_id=connection_id)

    def subscribe(
        self,
        connection_id: str,
        channels: List[BroadcastChannel]
    ) -> None:
        """Subscribe connection to channels.

        Args:
            connection_id: Connection identifier
            channels: List of channels to subscribe to
        """
        if connection_id not in self.subscriptions:
            logger.warning("Subscription for unknown connection", connection_id=connection_id)
            return

        for channel in channels:
            self.subscriptions[connection_id].add(channel)

        logger.info(
            "Channel subscription",
            connection_id=connection_id,
            channels=[c.value for c in channels]
        )

    def unsubscribe(
        self,
        connection_id: str,
        channels: List[BroadcastChannel]
    ) -> None:
        """Unsubscribe connection from channels.

        Args:
            connection_id: Connection identifier
            channels: List of channels to unsubscribe from
        """
        if connection_id not in self.subscriptions:
            return

        for channel in channels:
            self.subscriptions[connection_id].discard(channel)

        logger.info(
            "Channel unsubscription",
            connection_id=connection_id,
            channels=[c.value for c in channels]
        )

    async def broadcast_to_channel(
        self,
        channel: BroadcastChannel,
        data: Dict[str, Any],
        user_filter: Optional[Set[str]] = None
    ) -> None:
        """Broadcast message to channel.

        Args:
            channel: Broadcast channel
            data: Data to broadcast
            user_filter: Optional set of user IDs to broadcast to
        """
        try:
            # Serialize message
            message = json.dumps(
                {
                    "channel": channel.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": data
                },
                cls=DecimalEncoder
            )

            # Check message size
            if len(message.encode()) > self.max_message_size_bytes:
                logger.warning(
                    "Message exceeds size limit",
                    channel=channel.value,
                    size=len(message.encode())
                )
                return

            # Queue message
            await self._message_queue.put({
                "channel": channel,
                "message": message,
                "user_filter": user_filter
            })

        except Exception as e:
            logger.error("Broadcast queueing failed", error=str(e), channel=channel.value)

    async def broadcast_to_user(
        self,
        user_id: str,
        channel: BroadcastChannel,
        data: Dict[str, Any]
    ) -> None:
        """Broadcast message to specific user.

        Args:
            user_id: User identifier
            channel: Broadcast channel
            data: Data to broadcast
        """
        await self.broadcast_to_channel(
            channel=channel,
            data=data,
            user_filter={user_id}
        )

    async def _broadcast_worker(self) -> None:
        """Background worker for broadcasting messages."""
        logger.info("Broadcast worker started")

        # Rate limiting
        interval = Decimal("1.0") / Decimal(str(self.max_messages_per_second))
        last_send = datetime.utcnow()

        try:
            while True:
                try:
                    # Get message from queue
                    msg_data = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=1.0
                    )

                    # Rate limiting
                    now = datetime.utcnow()
                    elapsed = (now - last_send).total_seconds()
                    if elapsed < float(interval):
                        await asyncio.sleep(float(interval) - elapsed)

                    # Broadcast message
                    await self._send_to_subscribers(
                        channel=msg_data["channel"],
                        message=msg_data["message"],
                        user_filter=msg_data.get("user_filter")
                    )

                    last_send = datetime.utcnow()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Broadcast worker error", error=str(e))
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Broadcast worker cancelled")
            raise

    async def _send_to_subscribers(
        self,
        channel: BroadcastChannel,
        message: str,
        user_filter: Optional[Set[str]] = None
    ) -> None:
        """Send message to channel subscribers.

        Args:
            channel: Broadcast channel
            message: Serialized message
            user_filter: Optional user filter
        """
        failed_connections = []

        for connection_id, subscribed_channels in self.subscriptions.items():
            # Check if subscribed to channel
            if channel not in subscribed_channels:
                continue

            # Check user filter
            if user_filter:
                user_id = None
                for uid, conn_ids in self.user_connections.items():
                    if connection_id in conn_ids:
                        user_id = uid
                        break

                if not user_id or user_id not in user_filter:
                    continue

            # Send message
            try:
                websocket = self.connections.get(connection_id)
                if websocket:
                    await websocket.send_text(message)
            except Exception as e:
                logger.warning(
                    "Failed to send message",
                    connection_id=connection_id,
                    error=str(e)
                )
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            self.unregister_connection(connection_id)

    async def broadcast_market_data(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> None:
        """Broadcast market data update.

        Args:
            symbol: Trading symbol
            data: Market data
        """
        await self.broadcast_to_channel(
            channel=BroadcastChannel.MARKET_DATA,
            data={
                "symbol": symbol,
                **data
            }
        )

    async def broadcast_order_update(
        self,
        user_id: str,
        order_data: Dict[str, Any]
    ) -> None:
        """Broadcast order update to user.

        Args:
            user_id: User identifier
            order_data: Order information
        """
        await self.broadcast_to_user(
            user_id=user_id,
            channel=BroadcastChannel.ORDER_UPDATES,
            data=order_data
        )

    async def broadcast_position_update(
        self,
        user_id: str,
        position_data: Dict[str, Any]
    ) -> None:
        """Broadcast position update to user.

        Args:
            user_id: User identifier
            position_data: Position information
        """
        await self.broadcast_to_user(
            user_id=user_id,
            channel=BroadcastChannel.POSITION_UPDATES,
            data=position_data
        )

    async def broadcast_trade_update(
        self,
        user_id: str,
        trade_data: Dict[str, Any]
    ) -> None:
        """Broadcast trade update to user.

        Args:
            user_id: User identifier
            trade_data: Trade information
        """
        await self.broadcast_to_user(
            user_id=user_id,
            channel=BroadcastChannel.TRADE_UPDATES,
            data=trade_data
        )

    async def broadcast_alert(
        self,
        user_id: str,
        alert_data: Dict[str, Any]
    ) -> None:
        """Broadcast alert to user.

        Args:
            user_id: User identifier
            alert_data: Alert information
        """
        await self.broadcast_to_user(
            user_id=user_id,
            channel=BroadcastChannel.ALERTS,
            data=alert_data
        )

    async def broadcast_system_message(
        self,
        message: str,
        severity: str = "info"
    ) -> None:
        """Broadcast system message to all connected clients.

        Args:
            message: System message
            severity: Message severity (info, warning, error)
        """
        await self.broadcast_to_channel(
            channel=BroadcastChannel.SYSTEM,
            data={
                "message": message,
                "severity": severity
            }
        )

    def get_connection_count(self) -> int:
        """Get number of active connections.

        Returns:
            Number of active connections
        """
        return len(self.connections)

    def get_subscriptions_count(self, channel: BroadcastChannel) -> int:
        """Get number of subscriptions for channel.

        Args:
            channel: Broadcast channel

        Returns:
            Number of subscriptions
        """
        return sum(
            1 for subs in self.subscriptions.values()
            if channel in subs
        )
