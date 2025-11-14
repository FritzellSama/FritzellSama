"""
Bitget WebSocket Client - Real-time market data and account updates.

Implements Bitget WebSocket API V2 for real-time order book, trades, positions,
and order updates with automatic reconnection and authentication.
"""

import asyncio
import json
import time
import hmac
import hashlib
import base64
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import websockets
from websockets.client import WebSocketClientProtocol
import structlog

logger = structlog.get_logger(__name__)


class BitgetChannelType(Enum):
    """Bitget WebSocket channel types."""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"


@dataclass
class BitgetSubscription:
    """Bitget WebSocket subscription."""
    channel: str
    inst_type: str  # mc (USDT futures), sp (spot), etc.
    inst_id: str  # Trading pair
    channel_type: BitgetChannelType
    callback: Callable
    active: bool = True
    last_update: Optional[datetime] = None
    message_count: int = 0


class BitgetWebSocket:
    """
    Production-grade Bitget WebSocket client.

    Handles both public and private WebSocket streams with authentication,
    automatic reconnection, and subscription management for Bitget V2 API.

    Attributes:
        config: Configuration dictionary
        api_key: API key for private channels
        api_secret: API secret for authentication
        passphrase: API passphrase
        subscriptions: Active channel subscriptions

    Example:
        >>> config = {
        ...     'public_ws_url': 'wss://ws.bitget.com/v2/ws/public',
        ...     'private_ws_url': 'wss://ws.bitget.com/v2/ws/private',
        ...     'reconnect_interval': 5
        ... }
        >>> ws_client = BitgetWebSocket(config, api_key='key', api_secret='secret', passphrase='pass')
        >>> await ws_client.connect()
        >>>
        >>> async def handle_trade(data):
        ...     print(f"Trade: {data}")
        >>>
        >>> await ws_client.subscribe_trade('BTCUSDT', 'mc', handle_trade)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None
    ) -> None:
        """
        Initialize Bitget WebSocket client.

        Args:
            config: Configuration dictionary
            api_key: API key for private channels
            api_secret: API secret for authentication
            passphrase: API passphrase

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        self.public_ws_url = config['public_ws_url']
        self.private_ws_url = config.get('private_ws_url')
        self.testnet = config.get('testnet', False)

        # WebSocket connections
        self.public_ws: Optional[WebSocketClientProtocol] = None
        self.private_ws: Optional[WebSocketClientProtocol] = None
        self.running = False

        self._connect_lock = asyncio.Lock()
        self._reconnect_interval = Decimal(
            str(config.get('reconnect_interval', 5))
        )

        # Subscription management
        self.subscriptions: Dict[str, BitgetSubscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Background tasks
        self._public_receive_task: Optional[asyncio.Task] = None
        self._private_receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        # Keep-alive
        self._ping_interval = Decimal(str(config.get('ping_interval', 20)))

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_messages': 0,
            'public_messages': 0,
            'private_messages': 0,
            'messages_by_channel': {},
            'reconnect_count': 0,
            'connection_errors': 0,
            'auth_errors': 0,
            'last_connected': None
        }

        logger.info(
            "bitget_websocket_initialized",
            public_url=self.public_ws_url,
            private_enabled=bool(self.private_ws_url and api_key),
            testnet=self.testnet
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if 'public_ws_url' not in self.config:
            raise ValueError("Missing 'public_ws_url' in configuration")

        if not self.config['public_ws_url'].startswith('wss://'):
            raise ValueError("WebSocket URL must use wss:// protocol")

    async def connect(self) -> None:
        """
        Establish WebSocket connections (public and private).

        Raises:
            RuntimeError: If connection fails
        """
        async with self._connect_lock:
            if self.running:
                logger.warning("websocket_already_connected")
                return

            try:
                logger.info("connecting_to_bitget_websocket")

                # Connect to public WebSocket
                self.public_ws = await websockets.connect(
                    self.public_ws_url,
                    ping_interval=None,
                    close_timeout=float(self.config.get('close_timeout', 10)),
                    max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                )

                # Connect to private WebSocket if configured
                if self.private_ws_url and self.api_key and self.api_secret and self.passphrase:
                    self.private_ws = await websockets.connect(
                        self.private_ws_url,
                        ping_interval=None,
                        close_timeout=float(self.config.get('close_timeout', 10)),
                        max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                    )

                    # Authenticate private WebSocket
                    await self._authenticate_private_ws()

                self.running = True
                self._metrics['last_connected'] = datetime.now(timezone.utc)

                # Start background tasks
                self._public_receive_task = asyncio.create_task(
                    self._receive_loop(self.public_ws, BitgetChannelType.PUBLIC)
                )

                if self.private_ws:
                    self._private_receive_task = asyncio.create_task(
                        self._receive_loop(self.private_ws, BitgetChannelType.PRIVATE)
                    )

                self._ping_task = asyncio.create_task(self._ping_loop())

                logger.info(
                    "bitget_websocket_connected",
                    public=True,
                    private=bool(self.private_ws)
                )

                # Resubscribe to channels
                await self._resubscribe_all()

            except Exception as e:
                self._metrics['connection_errors'] += 1
                logger.error(
                    "websocket_connection_failed",
                    error=str(e)
                )
                raise RuntimeError(f"Failed to connect to Bitget WebSocket: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket connections and cleanup."""
        logger.info("disconnecting_bitget_websocket")

        self.running = False

        # Cancel background tasks
        for task in [
            self._public_receive_task,
            self._private_receive_task,
            self._ping_task
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket connections
        for ws in [self.public_ws, self.private_ws]:
            if ws:
                try:
                    await ws.close()
                except Exception as e:
                    logger.warning(
                        "websocket_close_error",
                        error=str(e)
                    )

        self.public_ws = None
        self.private_ws = None

        logger.info("bitget_websocket_disconnected")

    async def _authenticate_private_ws(self) -> None:
        """Authenticate private WebSocket connection."""
        if not self.private_ws or not self.api_key or not self.api_secret or not self.passphrase:
            return

        try:
            # Generate authentication signature
            timestamp = str(int(time.time()))
            message = timestamp + 'GET' + '/user/verify'

            signature = base64.b64encode(
                hmac.new(
                    self.api_secret.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            # Send login message
            login_message = {
                "op": "login",
                "args": [
                    {
                        "apiKey": self.api_key,
                        "passphrase": self.passphrase,
                        "timestamp": timestamp,
                        "sign": signature
                    }
                ]
            }

            await self.private_ws.send(json.dumps(login_message))

            # Wait for auth response
            response = await asyncio.wait_for(
                self.private_ws.recv(),
                timeout=float(self.config.get('auth_timeout', 10))
            )

            data = json.loads(response)

            if data.get('event') == 'login' and data.get('code') == '0':
                logger.info("private_websocket_authenticated")
            else:
                self._metrics['auth_errors'] += 1
                raise RuntimeError(f"Authentication failed: {data}")

        except Exception as e:
            self._metrics['auth_errors'] += 1
            logger.error(
                "authentication_failed",
                error=str(e)
            )
            raise

    async def subscribe_trade(
        self,
        inst_id: str,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to public trade stream.

        Args:
            inst_id: Instrument ID (e.g., 'BTCUSDT')
            inst_type: Instrument type (mc, sp, etc.)
            callback: Async callback function for trade data
        """
        channel = "trade"
        await self._subscribe(
            channel,
            inst_type,
            inst_id,
            BitgetChannelType.PUBLIC,
            callback
        )

    async def subscribe_orderbook(
        self,
        inst_id: str,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: str = "books15"
    ) -> None:
        """
        Subscribe to order book stream.

        Args:
            inst_id: Instrument ID
            inst_type: Instrument type
            callback: Async callback function
            depth: Order book depth (books1, books5, books15)
        """
        channel = depth
        await self._subscribe(
            channel,
            inst_type,
            inst_id,
            BitgetChannelType.PUBLIC,
            callback
        )

    async def subscribe_ticker(
        self,
        inst_id: str,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to ticker stream.

        Args:
            inst_id: Instrument ID
            inst_type: Instrument type
            callback: Async callback function
        """
        channel = "ticker"
        await self._subscribe(
            channel,
            inst_type,
            inst_id,
            BitgetChannelType.PUBLIC,
            callback
        )

    async def subscribe_candle(
        self,
        inst_id: str,
        inst_type: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to candlestick stream.

        Args:
            inst_id: Instrument ID
            inst_type: Instrument type
            interval: Candle interval (1m, 5m, 1H, etc.)
            callback: Async callback function
        """
        channel = f"candle{interval}"
        await self._subscribe(
            channel,
            inst_type,
            inst_id,
            BitgetChannelType.PUBLIC,
            callback
        )

    async def subscribe_account(
        self,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to account updates (private).

        Args:
            inst_type: Instrument type
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        channel = "account"
        await self._subscribe(
            channel,
            inst_type,
            "default",
            BitgetChannelType.PRIVATE,
            callback
        )

    async def subscribe_positions(
        self,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to position updates (private).

        Args:
            inst_type: Instrument type
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        channel = "positions"
        await self._subscribe(
            channel,
            inst_type,
            "default",
            BitgetChannelType.PRIVATE,
            callback
        )

    async def subscribe_orders(
        self,
        inst_type: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to order updates (private).

        Args:
            inst_type: Instrument type
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        channel = "orders"
        await self._subscribe(
            channel,
            inst_type,
            "default",
            BitgetChannelType.PRIVATE,
            callback
        )

    async def _subscribe(
        self,
        channel: str,
        inst_type: str,
        inst_id: str,
        channel_type: BitgetChannelType,
        callback: Callable
    ) -> None:
        """
        Internal subscription handler.

        Args:
            channel: Channel name
            inst_type: Instrument type
            inst_id: Instrument ID
            channel_type: PUBLIC or PRIVATE
            callback: Message callback
        """
        async with self._subscription_lock:
            # Create subscription key
            sub_key = f"{channel}:{inst_type}:{inst_id}"

            subscription = BitgetSubscription(
                channel=channel,
                inst_type=inst_type,
                inst_id=inst_id,
                channel_type=channel_type,
                callback=callback
            )

            self.subscriptions[sub_key] = subscription

            # Send subscription if connected
            if self.running:
                ws = self.public_ws if channel_type == BitgetChannelType.PUBLIC else self.private_ws
                if ws:
                    await self._send_subscribe(ws, channel, inst_type, inst_id)

            logger.info(
                "subscribed_to_channel",
                channel=channel,
                inst_type=inst_type,
                inst_id=inst_id,
                channel_type=channel_type.value
            )

    async def _send_subscribe(
        self,
        ws: WebSocketClientProtocol,
        channel: str,
        inst_type: str,
        inst_id: str
    ) -> None:
        """Send subscription message."""
        message = {
            "op": "subscribe",
            "args": [
                {
                    "instType": inst_type,
                    "channel": channel,
                    "instId": inst_id
                }
            ]
        }

        try:
            await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "subscribe_send_failed",
                channel=channel,
                error=str(e)
            )

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active channels after reconnect."""
        async with self._subscription_lock:
            for sub_key, subscription in self.subscriptions.items():
                ws = (self.public_ws if subscription.channel_type == BitgetChannelType.PUBLIC
                      else self.private_ws)
                if ws:
                    await self._send_subscribe(
                        ws,
                        subscription.channel,
                        subscription.inst_type,
                        subscription.inst_id
                    )

        logger.info(
            "resubscribed_all_channels",
            count=len(self.subscriptions)
        )

    async def _ping_loop(self) -> None:
        """Background task for sending ping messages."""
        while self.running:
            try:
                await asyncio.sleep(float(self._ping_interval))

                # Ping all active connections
                for ws in [self.public_ws, self.private_ws]:
                    if ws and not ws.closed:
                        await ws.send("ping")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "ping_loop_error",
                    error=str(e)
                )

    async def _receive_loop(
        self,
        ws: WebSocketClientProtocol,
        channel_type: BitgetChannelType
    ) -> None:
        """
        Background task for receiving messages.

        Args:
            ws: WebSocket connection
            channel_type: Channel type
        """
        while self.running:
            try:
                if ws.closed:
                    await asyncio.sleep(1)
                    continue

                message = await ws.recv()

                # Handle pong
                if message == "pong":
                    continue

                # Parse and handle message
                try:
                    data = json.loads(message)
                    await self._handle_message(data, channel_type)

                except json.JSONDecodeError as e:
                    logger.warning(
                        "json_decode_error",
                        error=str(e),
                        message=message[:200]
                    )

            except websockets.ConnectionClosed:
                logger.warning(
                    "websocket_connection_closed",
                    channel=channel_type.value
                )
                if self.running:
                    await self._reconnect()
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(
                    "receive_loop_error",
                    channel=channel_type.value,
                    error=str(e)
                )
                await asyncio.sleep(1)

    async def _handle_message(
        self,
        data: Dict[str, Any],
        channel_type: BitgetChannelType
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed message data
            channel_type: Channel type
        """
        self._metrics['total_messages'] += 1

        if channel_type == BitgetChannelType.PUBLIC:
            self._metrics['public_messages'] += 1
        else:
            self._metrics['private_messages'] += 1

        # Handle event messages
        event = data.get('event')
        if event in ['subscribe', 'unsubscribe', 'login', 'error']:
            if event == 'error':
                logger.error(
                    "websocket_error",
                    data=data
                )
            return

        # Handle data messages
        if 'arg' in data and 'data' in data:
            arg = data['arg']
            channel = arg.get('channel')
            inst_type = arg.get('instType', '')
            inst_id = arg.get('instId', 'default')

            # Find matching subscription
            sub_key = f"{channel}:{inst_type}:{inst_id}"
            subscription = self.subscriptions.get(sub_key)

            if subscription and subscription.active:
                subscription.message_count += 1
                subscription.last_update = datetime.now(timezone.utc)

                # Track by channel
                self._metrics['messages_by_channel'][channel] = \
                    self._metrics['messages_by_channel'].get(channel, 0) + 1

                # Call callback with data
                try:
                    message_data = data['data']
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(message_data)
                    else:
                        subscription.callback(message_data)

                except Exception as e:
                    logger.error(
                        "callback_error",
                        channel=channel,
                        error=str(e)
                    )

    async def _reconnect(self) -> None:
        """Reconnect to WebSocket with exponential backoff."""
        self._metrics['reconnect_count'] += 1

        logger.info(
            "attempting_reconnect",
            attempt=self._metrics['reconnect_count']
        )

        await self.disconnect()

        attempt = 0
        while self.running:
            try:
                delay = min(
                    self._reconnect_interval * (Decimal("2") ** Decimal(str(attempt))),
                    Decimal("300")
                )
                await asyncio.sleep(float(delay))

                await self.connect()
                logger.info("reconnect_successful")
                return

            except Exception as e:
                attempt += 1
                logger.warning(
                    "reconnect_attempt_failed",
                    attempt=attempt,
                    error=str(e)
                )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get WebSocket metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            **self._metrics,
            'public_connected': bool(self.public_ws and not self.public_ws.closed),
            'private_connected': bool(self.private_ws and not self.private_ws.closed),
            'active_subscriptions': len(self.subscriptions)
        }

    async def __aenter__(self) -> 'BitgetWebSocket':
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.disconnect()
