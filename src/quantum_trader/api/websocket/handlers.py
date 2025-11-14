"""
WebSocket message handlers for real-time trading data.

Handles incoming WebSocket messages, processes subscriptions,
and manages real-time data streams with batch optimization
and memory efficiency.
"""

import asyncio
import json
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    ACK = "ack"


class ChannelType(Enum):
    """Available subscription channels."""
    MARKET_DATA = "market_data"
    ORDER_UPDATES = "order_updates"
    POSITION_UPDATES = "position_updates"
    SIGNAL_UPDATES = "signal_updates"
    TRADE_UPDATES = "trade_updates"
    RISK_ALERTS = "risk_alerts"
    SYSTEM_STATUS = "system_status"


@dataclass
class WebSocketMessage:
    """Parsed WebSocket message."""

    type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    message_id: Optional[str] = None


@dataclass
class StreamBuffer:
    """
    Buffer for batching stream updates.

    Attributes:
        messages: Buffered messages
        max_size: Maximum buffer size
        max_age_ms: Maximum buffer age in milliseconds
        created_at: Buffer creation time
    """

    messages: List[Dict[str, Any]] = field(default_factory=list)
    max_size: int = 100
    max_age_ms: int = 1000
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self.messages) >= self.max_size:
            return True

        age_ms = (datetime.utcnow() - self.created_at).total_seconds() * 1000
        if age_ms >= self.max_age_ms:
            return True

        return False

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to buffer."""
        self.messages.append(message)

    def flush(self) -> List[Dict[str, Any]]:
        """Flush buffer and return messages."""
        messages = self.messages.copy()
        self.messages.clear()
        self.created_at = datetime.utcnow()
        return messages


class WebSocketHandler:
    """
    Handles WebSocket message processing and stream management.

    Provides message parsing, validation, subscription management,
    and batch optimization for high-throughput data streams.

    Attributes:
        config: Handler configuration
        connection_manager: WebSocket connection manager
        _stream_buffers: Per-client stream buffers
        _running: Handler running state
        _flush_task: Background buffer flush task
    """

    def __init__(
        self,
        config: Dict[str, Any],
        connection_manager: Any
    ) -> None:
        """
        Initialize WebSocket handler.

        Args:
            config: Configuration with buffer settings, rate limits
            connection_manager: ConnectionManager instance

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.connection_manager = connection_manager
        self._validate_config()

        self._stream_buffers: Dict[str, StreamBuffer] = {}
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None

        # Valid channels
        self._valid_channels = {ct.value for ct in ChannelType}

        logger.info(
            "websocket_handler_initialized",
            valid_channels=len(self._valid_channels),
            buffer_size=self.config.get("buffer_max_size")
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = [
            "buffer_max_size",
            "buffer_max_age_ms",
            "flush_interval_ms",
            "max_subscriptions_per_client"
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config["buffer_max_size"] <= 0:
            raise ValueError("buffer_max_size must be positive")

        if self.config["buffer_max_age_ms"] <= 0:
            raise ValueError("buffer_max_age_ms must be positive")

    async def start(self) -> None:
        """Start the message handler."""
        if self._running:
            logger.warning("websocket_handler_already_running")
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._buffer_flush_loop())

        logger.info("websocket_handler_started")

    async def stop(self) -> None:
        """Stop the message handler gracefully."""
        if not self._running:
            logger.warning("websocket_handler_not_running")
            return

        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush all remaining buffers
        for client_id in list(self._stream_buffers.keys()):
            await self._flush_buffer(client_id)

        self._stream_buffers.clear()

        logger.info("websocket_handler_stopped")

    async def handle_message(
        self,
        websocket: WebSocket,
        client_id: str,
        raw_message: str
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            websocket: WebSocket connection
            client_id: Client identifier
            raw_message: Raw JSON message string

        Example:
            >>> await handler.handle_message(
            ...     websocket,
            ...     "client123",
            ...     '{"type": "subscribe", "channels": ["market_data:BTC/USDT"]}'
            ... )
        """
        try:
            # Parse message
            message = self._parse_message(raw_message)

            logger.debug(
                "message_received",
                client_id=client_id,
                type=message.type,
                message_id=message.message_id
            )

            # Route to appropriate handler
            if message.type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(client_id, message)

            elif message.type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(client_id, message)

            elif message.type == MessageType.HEARTBEAT.value:
                await self._handle_heartbeat(client_id, message)

            else:
                await self._send_error(
                    websocket,
                    f"Unknown message type: {message.type}",
                    message.message_id
                )

            # Send acknowledgment
            await self._send_ack(websocket, message)

        except json.JSONDecodeError as e:
            logger.error("invalid_json", client_id=client_id, error=str(e))
            await self._send_error(websocket, "Invalid JSON format")

        except Exception as e:
            logger.error(
                "message_handling_error",
                client_id=client_id,
                error=str(e),
                error_type=type(e).__name__
            )
            await self._send_error(websocket, f"Error processing message: {str(e)}")

    def _parse_message(self, raw_message: str) -> WebSocketMessage:
        """
        Parse raw WebSocket message.

        Args:
            raw_message: Raw JSON string

        Returns:
            Parsed WebSocketMessage

        Raises:
            json.JSONDecodeError: If JSON is invalid
            ValueError: If message structure is invalid
        """
        data = json.loads(raw_message)

        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object")

        if "type" not in data:
            raise ValueError("Message must have 'type' field")

        return WebSocketMessage(
            type=data["type"],
            payload=data.get("payload", {}),
            message_id=data.get("message_id")
        )

    async def _handle_subscribe(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> None:
        """
        Handle subscription request.

        Args:
            client_id: Client identifier
            message: Parsed message

        Raises:
            ValueError: If subscription is invalid
        """
        channels = message.payload.get("channels", [])

        if not isinstance(channels, list):
            raise ValueError("'channels' must be a list")

        if not channels:
            raise ValueError("'channels' cannot be empty")

        # Validate channels
        valid_channels = []
        for channel in channels:
            if not isinstance(channel, str):
                logger.warning("invalid_channel_type", channel=channel)
                continue

            # Parse channel format: "type:symbol" or just "type"
            parts = channel.split(":", 1)
            channel_type = parts[0]

            if channel_type not in self._valid_channels:
                logger.warning(
                    "invalid_channel_type",
                    channel_type=channel_type,
                    client_id=client_id
                )
                continue

            valid_channels.append(channel)

        if not valid_channels:
            raise ValueError("No valid channels specified")

        # Check subscription limit
        max_subs = self.config["max_subscriptions_per_client"]
        # Get current subscription count from connection manager
        # This is simplified - actual implementation would check against limit

        # Subscribe to channels
        await self.connection_manager.subscribe(client_id, valid_channels)

        logger.info(
            "client_subscribed",
            client_id=client_id,
            channels=valid_channels,
            count=len(valid_channels)
        )

    async def _handle_unsubscribe(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> None:
        """
        Handle unsubscription request.

        Args:
            client_id: Client identifier
            message: Parsed message
        """
        channels = message.payload.get("channels", [])

        if not isinstance(channels, list):
            raise ValueError("'channels' must be a list")

        if not channels:
            raise ValueError("'channels' cannot be empty")

        # Unsubscribe from channels
        await self.connection_manager.unsubscribe(client_id, channels)

        logger.info(
            "client_unsubscribed",
            client_id=client_id,
            channels=channels,
            count=len(channels)
        )

    async def _handle_heartbeat(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> None:
        """
        Handle heartbeat message.

        Args:
            client_id: Client identifier
            message: Parsed message
        """
        logger.debug("heartbeat_received", client_id=client_id)

    async def _send_ack(
        self,
        websocket: WebSocket,
        message: WebSocketMessage
    ) -> None:
        """Send acknowledgment for message."""
        if not message.message_id:
            return

        ack = {
            "type": MessageType.ACK.value,
            "message_id": message.message_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            await websocket.send_text(json.dumps(ack))
        except Exception as e:
            logger.error("ack_send_error", error=str(e))

    async def _send_error(
        self,
        websocket: WebSocket,
        error: str,
        message_id: Optional[str] = None
    ) -> None:
        """Send error message to client."""
        error_msg = {
            "type": MessageType.ERROR.value,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }

        if message_id:
            error_msg["message_id"] = message_id

        try:
            await websocket.send_text(json.dumps(error_msg))
        except Exception as e:
            logger.error("error_send_failed", error=str(e))

    async def publish_market_data(
        self,
        channel: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Publish market data to subscribers.

        Args:
            channel: Channel name (e.g., "market_data:BTC/USDT")
            data: Market data to publish

        Returns:
            Number of clients message was sent to

        Example:
            >>> count = await handler.publish_market_data(
            ...     "market_data:BTC/USDT",
            ...     {"price": "50000.00", "volume": "123.45"}
            ... )
        """
        return await self.connection_manager.broadcast(channel, data)

    async def publish_order_update(
        self,
        client_id: str,
        order_data: Dict[str, Any]
    ) -> bool:
        """
        Publish order update to specific client.

        Args:
            client_id: Target client
            order_data: Order update data

        Returns:
            True if sent successfully

        Example:
            >>> await handler.publish_order_update(
            ...     "client123",
            ...     {"order_id": "123", "status": "filled"}
            ... )
        """
        from quantum_trader.api.websocket.connection_manager import MessageType as CMMessageType

        return await self.connection_manager.send_personal(
            client_id,
            order_data,
            CMMessageType.ORDER_UPDATE
        )

    async def publish_position_update(
        self,
        client_id: str,
        position_data: Dict[str, Any]
    ) -> bool:
        """
        Publish position update to specific client.

        Args:
            client_id: Target client
            position_data: Position update data

        Returns:
            True if sent successfully
        """
        from quantum_trader.api.websocket.connection_manager import MessageType as CMMessageType

        return await self.connection_manager.send_personal(
            client_id,
            position_data,
            CMMessageType.POSITION_UPDATE
        )

    async def _buffer_flush_loop(self) -> None:
        """Background task to flush stream buffers."""
        flush_interval = self.config["flush_interval_ms"] / 1000.0

        while self._running:
            try:
                await asyncio.sleep(flush_interval)
                await self._flush_all_buffers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("buffer_flush_loop_error", error=str(e))

    async def _flush_all_buffers(self) -> None:
        """Flush all stream buffers that should be flushed."""
        for client_id in list(self._stream_buffers.keys()):
            buffer = self._stream_buffers[client_id]
            if buffer.should_flush():
                await self._flush_buffer(client_id)

    async def _flush_buffer(self, client_id: str) -> None:
        """
        Flush buffer for specific client.

        Args:
            client_id: Client identifier
        """
        if client_id not in self._stream_buffers:
            return

        buffer = self._stream_buffers[client_id]
        messages = buffer.flush()

        if not messages:
            return

        # Send batched messages
        try:
            for message in messages:
                await self.connection_manager.broadcast(
                    message["channel"],
                    message["data"]
                )

            logger.debug(
                "buffer_flushed",
                client_id=client_id,
                message_count=len(messages)
            )

        except Exception as e:
            logger.error(
                "buffer_flush_error",
                client_id=client_id,
                error=str(e)
            )

    def get_handler_stats(self) -> Dict[str, Any]:
        """
        Get handler statistics.

        Returns:
            Dictionary with handler metrics
        """
        total_buffered = sum(
            len(buffer.messages)
            for buffer in self._stream_buffers.values()
        )

        return {
            "running": self._running,
            "active_buffers": len(self._stream_buffers),
            "total_buffered_messages": total_buffered,
            "valid_channels": list(self._valid_channels),
            "config": {
                "buffer_max_size": self.config["buffer_max_size"],
                "buffer_max_age_ms": self.config["buffer_max_age_ms"],
                "flush_interval_ms": self.config["flush_interval_ms"]
            }
        }
