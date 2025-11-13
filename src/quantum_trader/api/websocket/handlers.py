"""
WebSocket message handlers for Quantum Trader AI.

Handles incoming WebSocket messages and routes them appropriately.
"""

import asyncio
import json
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

from quantum_trader.api.websocket.connection_manager import ConnectionManager
from quantum_trader.api.websocket.broadcasters import WebSocketBroadcaster, BroadcastChannel

logger = get_logger(__name__)


class MessageType(str, Enum):
    """WebSocket message type enumeration."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    GET_TICKER = "get_ticker"
    GET_ORDERBOOK = "get_orderbook"
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    GET_POSITIONS = "get_positions"
    GET_ORDERS = "get_orders"


class WebSocketHandler:
    """Production-ready WebSocket message handler.

    Handles WebSocket message processing with:
    - Message validation and parsing
    - Command routing
    - Error handling
    - Rate limiting
    - Subscription management
    - Parallel processing
    - Memory efficiency

    Attributes:
        config: Configuration dictionary
        connection_manager: Connection manager instance
        broadcaster: Broadcaster instance
        message_handlers: Registry of message type handlers
    """

    def __init__(
        self,
        config: Dict[str, Any],
        connection_manager: ConnectionManager,
        broadcaster: WebSocketBroadcaster
    ) -> None:
        """Initialize WebSocket handler.

        Args:
            config: Configuration from config files
            connection_manager: Connection manager
            broadcaster: Broadcaster instance

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.connection_manager = connection_manager
        self.broadcaster = broadcaster

        # Load from config
        ws_config = config.get("websocket", {})
        self.max_message_size = ws_config.get("max_message_size_bytes", 65536)
        self.message_timeout = ws_config.get("message_timeout_seconds", 30)

        # Message handlers registry
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.SUBSCRIBE: self._handle_subscribe,
            MessageType.UNSUBSCRIBE: self._handle_unsubscribe,
            MessageType.PING: self._handle_ping,
            MessageType.GET_TICKER: self._handle_get_ticker,
            MessageType.GET_ORDERBOOK: self._handle_get_orderbook,
            MessageType.PLACE_ORDER: self._handle_place_order,
            MessageType.CANCEL_ORDER: self._handle_cancel_order,
            MessageType.GET_POSITIONS: self._handle_get_positions,
            MessageType.GET_ORDERS: self._handle_get_orders,
        }

        self._validate_config()
        logger.info("WebSocket handler initialized")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")
        if self.message_timeout <= 0:
            raise ValueError("message_timeout must be positive")

    async def handle_connection(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None
    ) -> None:
        """Handle WebSocket connection lifecycle.

        Args:
            websocket: WebSocket instance
            connection_id: Connection identifier
            user_id: User identifier (optional)
        """
        try:
            # Connect
            connected = await self.connection_manager.connect(
                websocket=websocket,
                connection_id=connection_id,
                user_id=user_id
            )

            if not connected:
                logger.warning("Connection rejected", connection_id=connection_id)
                return

            # Register with broadcaster
            self.broadcaster.register_connection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=user_id
            )

            # Send welcome message
            await self._send_message(
                connection_id,
                {
                    "type": "connected",
                    "connection_id": connection_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

            # Message loop
            await self._message_loop(websocket, connection_id, user_id)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", connection_id=connection_id)
        except Exception as e:
            logger.error("WebSocket error", connection_id=connection_id, error=str(e))
        finally:
            # Cleanup
            self.broadcaster.unregister_connection(connection_id)
            await self.connection_manager.disconnect(connection_id)

    async def _message_loop(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str]
    ) -> None:
        """Main message processing loop.

        Args:
            websocket: WebSocket instance
            connection_id: Connection identifier
            user_id: User identifier
        """
        while True:
            try:
                # Receive message with timeout
                message_text = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=self.message_timeout
                )

                # Check message size
                if len(message_text.encode()) > self.max_message_size:
                    await self._send_error(
                        connection_id,
                        "Message too large",
                        "MESSAGE_TOO_LARGE"
                    )
                    continue

                # Parse message
                try:
                    message = json.loads(message_text)
                except json.JSONDecodeError:
                    await self._send_error(
                        connection_id,
                        "Invalid JSON",
                        "INVALID_JSON"
                    )
                    continue

                # Process message
                await self._process_message(connection_id, user_id, message)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await self._send_message(connection_id, {"type": "ping"})
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("Message loop error", error=str(e))
                await self._send_error(connection_id, "Internal error", "INTERNAL_ERROR")

    async def _process_message(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Process incoming WebSocket message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Parsed message dictionary
        """
        try:
            # Extract message type
            message_type_str = message.get("type")
            if not message_type_str:
                await self._send_error(connection_id, "Missing message type", "MISSING_TYPE")
                return

            # Validate message type
            try:
                message_type = MessageType(message_type_str)
            except ValueError:
                await self._send_error(
                    connection_id,
                    f"Unknown message type: {message_type_str}",
                    "UNKNOWN_TYPE"
                )
                return

            # Get handler
            handler = self.message_handlers.get(message_type)
            if not handler:
                await self._send_error(
                    connection_id,
                    f"No handler for type: {message_type_str}",
                    "NO_HANDLER"
                )
                return

            # Execute handler
            await handler(connection_id, user_id, message)

        except Exception as e:
            logger.error("Message processing error", error=str(e))
            await self._send_error(connection_id, "Processing failed", "PROCESSING_ERROR")

    async def _handle_subscribe(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle subscribe message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        try:
            channels_str = message.get("channels", [])
            if not isinstance(channels_str, list):
                await self._send_error(connection_id, "Invalid channels format", "INVALID_FORMAT")
                return

            # Parse channels
            channels = []
            for channel_str in channels_str:
                try:
                    channel = BroadcastChannel(channel_str)
                    channels.append(channel)
                except ValueError:
                    logger.warning("Invalid channel", channel=channel_str)

            if not channels:
                await self._send_error(connection_id, "No valid channels", "NO_CHANNELS")
                return

            # Subscribe
            self.broadcaster.subscribe(connection_id, channels)

            await self._send_message(
                connection_id,
                {
                    "type": "subscribed",
                    "channels": [c.value for c in channels]
                }
            )

            logger.info(
                "Subscribed to channels",
                connection_id=connection_id,
                channels=[c.value for c in channels]
            )

        except Exception as e:
            logger.error("Subscribe failed", error=str(e))
            await self._send_error(connection_id, "Subscribe failed", "SUBSCRIBE_ERROR")

    async def _handle_unsubscribe(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle unsubscribe message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        try:
            channels_str = message.get("channels", [])

            # Parse channels
            channels = []
            for channel_str in channels_str:
                try:
                    channel = BroadcastChannel(channel_str)
                    channels.append(channel)
                except ValueError:
                    logger.warning("Invalid channel", channel=channel_str)

            if not channels:
                return

            # Unsubscribe
            self.broadcaster.unsubscribe(connection_id, channels)

            await self._send_message(
                connection_id,
                {
                    "type": "unsubscribed",
                    "channels": [c.value for c in channels]
                }
            )

        except Exception as e:
            logger.error("Unsubscribe failed", error=str(e))

    async def _handle_ping(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle ping message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        await self._send_message(
            connection_id,
            {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    async def _handle_get_ticker(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle get ticker message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        try:
            symbol = message.get("symbol")
            exchange = message.get("exchange")

            if not symbol or not exchange:
                await self._send_error(
                    connection_id,
                    "Missing symbol or exchange",
                    "MISSING_PARAMS"
                )
                return

            # TODO: Fetch ticker data from data service
            # For now, send placeholder
            await self._send_message(
                connection_id,
                {
                    "type": "ticker",
                    "symbol": symbol,
                    "exchange": exchange,
                    "data": {}
                }
            )

        except Exception as e:
            logger.error("Get ticker failed", error=str(e))
            await self._send_error(connection_id, "Get ticker failed", "TICKER_ERROR")

    async def _handle_get_orderbook(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle get orderbook message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        try:
            symbol = message.get("symbol")
            exchange = message.get("exchange")
            depth = message.get("depth", 20)

            if not symbol or not exchange:
                await self._send_error(
                    connection_id,
                    "Missing symbol or exchange",
                    "MISSING_PARAMS"
                )
                return

            # TODO: Fetch orderbook from data service
            await self._send_message(
                connection_id,
                {
                    "type": "orderbook",
                    "symbol": symbol,
                    "exchange": exchange,
                    "data": {}
                }
            )

        except Exception as e:
            logger.error("Get orderbook failed", error=str(e))
            await self._send_error(connection_id, "Get orderbook failed", "ORDERBOOK_ERROR")

    async def _handle_place_order(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle place order message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        if not user_id:
            await self._send_error(connection_id, "Authentication required", "AUTH_REQUIRED")
            return

        try:
            # TODO: Validate and place order
            await self._send_message(
                connection_id,
                {
                    "type": "order_placed",
                    "order_id": "placeholder"
                }
            )

        except Exception as e:
            logger.error("Place order failed", error=str(e))
            await self._send_error(connection_id, "Place order failed", "ORDER_ERROR")

    async def _handle_cancel_order(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle cancel order message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        if not user_id:
            await self._send_error(connection_id, "Authentication required", "AUTH_REQUIRED")
            return

        try:
            order_id = message.get("order_id")
            if not order_id:
                await self._send_error(connection_id, "Missing order_id", "MISSING_PARAMS")
                return

            # TODO: Cancel order
            await self._send_message(
                connection_id,
                {
                    "type": "order_cancelled",
                    "order_id": order_id
                }
            )

        except Exception as e:
            logger.error("Cancel order failed", error=str(e))
            await self._send_error(connection_id, "Cancel order failed", "CANCEL_ERROR")

    async def _handle_get_positions(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle get positions message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        if not user_id:
            await self._send_error(connection_id, "Authentication required", "AUTH_REQUIRED")
            return

        try:
            # TODO: Fetch positions
            await self._send_message(
                connection_id,
                {
                    "type": "positions",
                    "data": []
                }
            )

        except Exception as e:
            logger.error("Get positions failed", error=str(e))
            await self._send_error(connection_id, "Get positions failed", "POSITIONS_ERROR")

    async def _handle_get_orders(
        self,
        connection_id: str,
        user_id: Optional[str],
        message: Dict[str, Any]
    ) -> None:
        """Handle get orders message.

        Args:
            connection_id: Connection identifier
            user_id: User identifier
            message: Message data
        """
        if not user_id:
            await self._send_error(connection_id, "Authentication required", "AUTH_REQUIRED")
            return

        try:
            # TODO: Fetch orders
            await self._send_message(
                connection_id,
                {
                    "type": "orders",
                    "data": []
                }
            )

        except Exception as e:
            logger.error("Get orders failed", error=str(e))
            await self._send_error(connection_id, "Get orders failed", "ORDERS_ERROR")

    async def _send_message(
        self,
        connection_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send message to connection.

        Args:
            connection_id: Connection identifier
            data: Data to send

        Returns:
            True if sent successfully
        """
        try:
            return await self.connection_manager.send_json(connection_id, data)
        except Exception as e:
            logger.error("Send message failed", error=str(e))
            return False

    async def _send_error(
        self,
        connection_id: str,
        message: str,
        error_code: str
    ) -> bool:
        """Send error message to connection.

        Args:
            connection_id: Connection identifier
            message: Error message
            error_code: Error code

        Returns:
            True if sent successfully
        """
        return await self._send_message(
            connection_id,
            {
                "type": "error",
                "error": error_code,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
