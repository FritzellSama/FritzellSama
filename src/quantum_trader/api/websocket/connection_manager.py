"""
WebSocket Connection Manager for real-time trading updates.

Manages WebSocket connections with thread-safe state management,
event-driven architecture, and graceful resource lifecycle management.
"""

import asyncio
from decimal import Decimal
from typing import Dict, Set, List, Optional, Any
from datetime import datetime
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_DATA = "market_data"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    SIGNAL_UPDATE = "signal_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class Connection:
    """Represents a WebSocket connection with metadata."""

    websocket: WebSocket
    client_id: str
    subscriptions: Set[str] = field(default_factory=set)
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime = field(default_factory=lambda: datetime.utcnow())
    last_ping: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """
    Thread-safe WebSocket connection manager.

    Manages multiple WebSocket connections with subscription management,
    state tracking, health checks, and graceful shutdown capabilities.

    Attributes:
        config: Configuration dictionary
        active_connections: Map of client_id to Connection
        subscriptions: Map of channel to set of client_ids
        _lock: Asyncio lock for thread-safe operations
        _running: Flag for manager state
        _health_check_task: Background health check task
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize connection manager.

        Args:
            config: Configuration with ping_interval, ping_timeout, max_connections

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.active_connections: Dict[str, Connection] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info(
            "connection_manager_initialized",
            ping_interval=self.config.get("ping_interval"),
            max_connections=self.config.get("max_connections")
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ["ping_interval", "ping_timeout", "max_connections"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config["ping_interval"] <= 0:
            raise ValueError("ping_interval must be positive")

        if self.config["ping_timeout"] <= 0:
            raise ValueError("ping_timeout must be positive")

        if self.config["max_connections"] <= 0:
            raise ValueError("max_connections must be positive")

    async def start(self) -> None:
        """Start the connection manager and health check task."""
        async with self._lock:
            if self._running:
                logger.warning("connection_manager_already_running")
                return

            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info("connection_manager_started")

    async def stop(self) -> None:
        """Stop the connection manager gracefully."""
        async with self._lock:
            if not self._running:
                logger.warning("connection_manager_not_running")
                return

            self._running = False

            # Cancel health check task
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # Disconnect all clients
            client_ids = list(self.active_connections.keys())
            for client_id in client_ids:
                try:
                    await self._disconnect_internal(client_id, "server_shutdown")
                except Exception as e:
                    logger.error(
                        "error_disconnecting_client",
                        client_id=client_id,
                        error=str(e)
                    )

            logger.info("connection_manager_stopped")

    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Unique client identifier
            metadata: Optional client metadata

        Raises:
            ValueError: If max connections reached
            RuntimeError: If connection manager not running
        """
        async with self._lock:
            if not self._running:
                raise RuntimeError("Connection manager not running")

            if len(self.active_connections) >= self.config["max_connections"]:
                raise ValueError(
                    f"Max connections reached: {self.config['max_connections']}"
                )

            if client_id in self.active_connections:
                # Disconnect existing connection
                await self._disconnect_internal(client_id, "duplicate_connection")

            await websocket.accept()

            connection = Connection(
                websocket=websocket,
                client_id=client_id,
                state=ConnectionState.CONNECTED,
                metadata=metadata or {}
            )

            self.active_connections[client_id] = connection

            logger.info(
                "client_connected",
                client_id=client_id,
                total_connections=len(self.active_connections),
                metadata=metadata
            )

    async def disconnect(self, client_id: str, reason: str = "client_disconnect") -> None:
        """
        Disconnect a client.

        Args:
            client_id: Client to disconnect
            reason: Disconnection reason
        """
        async with self._lock:
            await self._disconnect_internal(client_id, reason)

    async def _disconnect_internal(self, client_id: str, reason: str) -> None:
        """Internal disconnect method without lock (assumes lock is held)."""
        if client_id not in self.active_connections:
            return

        connection = self.active_connections[client_id]
        connection.state = ConnectionState.DISCONNECTING

        # Remove from all subscriptions
        for channel in list(connection.subscriptions):
            if channel in self.subscriptions:
                self.subscriptions[channel].discard(client_id)
                if not self.subscriptions[channel]:
                    del self.subscriptions[channel]

        # Close WebSocket
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.warning(
                "error_closing_websocket",
                client_id=client_id,
                error=str(e)
            )

        connection.state = ConnectionState.DISCONNECTED
        del self.active_connections[client_id]

        logger.info(
            "client_disconnected",
            client_id=client_id,
            reason=reason,
            total_connections=len(self.active_connections)
        )

    async def subscribe(self, client_id: str, channels: List[str]) -> None:
        """
        Subscribe client to channels.

        Args:
            client_id: Client identifier
            channels: List of channel names to subscribe to

        Raises:
            ValueError: If client not connected
        """
        async with self._lock:
            if client_id not in self.active_connections:
                raise ValueError(f"Client not connected: {client_id}")

            connection = self.active_connections[client_id]

            for channel in channels:
                connection.subscriptions.add(channel)
                self.subscriptions[channel].add(client_id)

            logger.info(
                "client_subscribed",
                client_id=client_id,
                channels=channels,
                total_subscriptions=len(connection.subscriptions)
            )

    async def unsubscribe(self, client_id: str, channels: List[str]) -> None:
        """
        Unsubscribe client from channels.

        Args:
            client_id: Client identifier
            channels: List of channel names to unsubscribe from

        Raises:
            ValueError: If client not connected
        """
        async with self._lock:
            if client_id not in self.active_connections:
                raise ValueError(f"Client not connected: {client_id}")

            connection = self.active_connections[client_id]

            for channel in channels:
                connection.subscriptions.discard(channel)
                if channel in self.subscriptions:
                    self.subscriptions[channel].discard(client_id)
                    if not self.subscriptions[channel]:
                        del self.subscriptions[channel]

            logger.info(
                "client_unsubscribed",
                client_id=client_id,
                channels=channels,
                total_subscriptions=len(connection.subscriptions)
            )

    async def broadcast(
        self,
        channel: str,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None
    ) -> int:
        """
        Broadcast message to all subscribers of a channel.

        Args:
            channel: Channel name
            message: Message data to broadcast
            exclude: Optional set of client_ids to exclude

        Returns:
            Number of clients message was sent to
        """
        exclude = exclude or set()
        sent_count = 0

        async with self._lock:
            if channel not in self.subscriptions:
                return 0

            subscribers = self.subscriptions[channel] - exclude

            # Prepare message
            message_data = {
                "type": MessageType.MARKET_DATA.value,
                "channel": channel,
                "data": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            message_str = json.dumps(message_data, default=str)

            # Send to all subscribers
            for client_id in list(subscribers):
                try:
                    connection = self.active_connections.get(client_id)
                    if connection and connection.state == ConnectionState.CONNECTED:
                        await connection.websocket.send_text(message_str)
                        sent_count += 1
                except Exception as e:
                    logger.error(
                        "broadcast_error",
                        client_id=client_id,
                        channel=channel,
                        error=str(e)
                    )
                    # Disconnect failed client
                    try:
                        await self._disconnect_internal(client_id, "send_error")
                    except Exception:
                        pass

        return sent_count

    async def send_personal(
        self,
        client_id: str,
        message: Dict[str, Any],
        message_type: MessageType = MessageType.ORDER_UPDATE
    ) -> bool:
        """
        Send message to specific client.

        Args:
            client_id: Target client
            message: Message data
            message_type: Type of message

        Returns:
            True if sent successfully, False otherwise
        """
        async with self._lock:
            connection = self.active_connections.get(client_id)
            if not connection or connection.state != ConnectionState.CONNECTED:
                return False

            try:
                message_data = {
                    "type": message_type.value,
                    "data": message,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await connection.websocket.send_text(
                    json.dumps(message_data, default=str)
                )
                return True
            except Exception as e:
                logger.error(
                    "send_personal_error",
                    client_id=client_id,
                    error=str(e)
                )
                try:
                    await self._disconnect_internal(client_id, "send_error")
                except Exception:
                    pass
                return False

    async def _health_check_loop(self) -> None:
        """Background task to ping connections and check health."""
        while self._running:
            try:
                await asyncio.sleep(self.config["ping_interval"])
                await self._ping_all_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def _ping_all_connections(self) -> None:
        """Send ping to all connections and disconnect stale ones."""
        now = datetime.utcnow()
        timeout_seconds = self.config["ping_timeout"]

        async with self._lock:
            for client_id in list(self.active_connections.keys()):
                connection = self.active_connections[client_id]

                # Check if last ping timed out
                if connection.last_ping:
                    elapsed = (now - connection.last_ping).total_seconds()
                    if elapsed > timeout_seconds:
                        logger.warning(
                            "connection_timeout",
                            client_id=client_id,
                            elapsed=elapsed
                        )
                        await self._disconnect_internal(client_id, "ping_timeout")
                        continue

                # Send ping
                try:
                    ping_message = {
                        "type": MessageType.PING.value,
                        "timestamp": now.isoformat()
                    }
                    await connection.websocket.send_text(
                        json.dumps(ping_message)
                    )
                    connection.last_ping = now
                except Exception as e:
                    logger.error(
                        "ping_error",
                        client_id=client_id,
                        error=str(e)
                    )
                    await self._disconnect_internal(client_id, "ping_error")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_subscription_count(self, channel: str) -> int:
        """Get number of subscribers for a channel."""
        return len(self.subscriptions.get(channel, set()))

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of connection manager.

        Returns:
            Dictionary with health metrics
        """
        return {
            "running": self._running,
            "active_connections": len(self.active_connections),
            "max_connections": self.config["max_connections"],
            "total_channels": len(self.subscriptions),
            "connection_details": [
                {
                    "client_id": conn.client_id,
                    "state": conn.state.value,
                    "subscriptions": len(conn.subscriptions),
                    "connected_at": conn.connected_at.isoformat(),
                    "last_ping": conn.last_ping.isoformat() if conn.last_ping else None
                }
                for conn in self.active_connections.values()
            ]
        }
