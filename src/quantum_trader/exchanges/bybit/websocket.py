"""
Bybit WebSocket Client - Real-time market data and account updates.

Implements Bybit WebSocket API V5 for real-time order book, trades, positions,
and order updates with automatic reconnection and authentication.
"""

import asyncio
import json
import time
import hmac
import hashlib
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import websockets
from websockets.client import WebSocketClientProtocol
import structlog

logger = structlog.get_logger(__name__)


class BybitChannelType(Enum):
    """Bybit WebSocket channel types."""
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"


class TopicType(Enum):
    """Bybit WebSocket topic types."""
    ORDERBOOK = "orderbook"
    TRADE = "publicTrade"
    TICKER = "tickers"
    KLINE = "kline"
    LIQUIDATION = "liquidation"
    # Private topics
    POSITION = "position"
    EXECUTION = "execution"
    ORDER = "order"
    WALLET = "wallet"


@dataclass
class TopicSubscription:
    """WebSocket topic subscription."""
    topic: str
    topic_type: TopicType
    channel_type: BybitChannelType
    symbol: Optional[str]
    callback: Callable
    active: bool = True
    last_update: Optional[datetime] = None
    message_count: int = 0


class BybitWebSocket:
    """
    Production-grade Bybit WebSocket client.

    Handles both public and private WebSocket streams with authentication,
    automatic reconnection, and subscription management.

    Attributes:
        config: Configuration dictionary
        api_key: API key for private channels
        api_secret: API secret for authentication
        subscriptions: Active topic subscriptions

    Example:
        >>> config = {
        ...     'public_ws_url': 'wss://stream.bybit.com/v5/public/linear',
        ...     'private_ws_url': 'wss://stream.bybit.com/v5/private',
        ...     'reconnect_interval': 5
        ... }
        >>> ws_client = BybitWebSocket(config, api_key='key', api_secret='secret')
        >>> await ws_client.connect()
        >>>
        >>> async def handle_trade(data):
        ...     print(f"Trade: {data}")
        >>>
        >>> await ws_client.subscribe_trade('BTCUSDT', handle_trade)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ) -> None:
        """
        Initialize Bybit WebSocket client.

        Args:
            config: Configuration dictionary
            api_key: API key for private channels
            api_secret: API secret for authentication

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.api_key = api_key
        self.api_secret = api_secret

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
        self.subscriptions: Dict[str, TopicSubscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Background tasks
        self._public_receive_task: Optional[asyncio.Task] = None
        self._private_receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        # Keep-alive
        self._ping_interval = Decimal(str(config.get('ping_interval', 20)))
        self._last_public_pong: Optional[datetime] = None
        self._last_private_pong: Optional[datetime] = None

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_messages': 0,
            'public_messages': 0,
            'private_messages': 0,
            'messages_by_topic': {},
            'reconnect_count': 0,
            'connection_errors': 0,
            'auth_errors': 0,
            'last_connected': None
        }

        logger.info(
            "bybit_websocket_initialized",
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
                logger.info("connecting_to_bybit_websocket")

                # Connect to public WebSocket
                self.public_ws = await websockets.connect(
                    self.public_ws_url,
                    ping_interval=None,
                    close_timeout=float(self.config.get('close_timeout', 10)),
                    max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                )

                self._last_public_pong = datetime.now(timezone.utc)

                # Connect to private WebSocket if configured
                if self.private_ws_url and self.api_key and self.api_secret:
                    self.private_ws = await websockets.connect(
                        self.private_ws_url,
                        ping_interval=None,
                        close_timeout=float(self.config.get('close_timeout', 10)),
                        max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                    )

                    self._last_private_pong = datetime.now(timezone.utc)

                    # Authenticate private WebSocket
                    await self._authenticate_private_ws()

                self.running = True
                self._metrics['last_connected'] = datetime.now(timezone.utc)

                # Start background tasks
                self._public_receive_task = asyncio.create_task(
                    self._receive_loop(self.public_ws, BybitChannelType.PUBLIC)
                )

                if self.private_ws:
                    self._private_receive_task = asyncio.create_task(
                        self._receive_loop(self.private_ws, BybitChannelType.PRIVATE)
                    )

                self._ping_task = asyncio.create_task(self._ping_loop())

                logger.info(
                    "bybit_websocket_connected",
                    public=True,
                    private=bool(self.private_ws)
                )

                # Resubscribe to topics
                await self._resubscribe_all()

            except Exception as e:
                self._metrics['connection_errors'] += 1
                logger.error(
                    "websocket_connection_failed",
                    error=str(e)
                )
                raise RuntimeError(f"Failed to connect to Bybit WebSocket: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket connections and cleanup."""
        logger.info("disconnecting_bybit_websocket")

        self.running = False

        # Cancel background tasks
        for task in [self._public_receive_task, self._private_receive_task, self._ping_task]:
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

        logger.info("bybit_websocket_disconnected")

    async def _authenticate_private_ws(self) -> None:
        """Authenticate private WebSocket connection."""
        if not self.private_ws or not self.api_key or not self.api_secret:
            return

        try:
            # Generate authentication signature
            expires = int((time.time() + 10) * 1000)
            signature_payload = f"GET/realtime{expires}"

            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Send auth message
            auth_message = {
                "op": "auth",
                "args": [self.api_key, expires, signature]
            }

            await self.private_ws.send(json.dumps(auth_message))

            # Wait for auth response
            response = await asyncio.wait_for(
                self.private_ws.recv(),
                timeout=float(self.config.get('auth_timeout', 10))
            )

            data = json.loads(response)

            if data.get('success') or data.get('op') == 'auth' and not data.get('success', True) is False:
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
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to public trade stream.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            callback: Async callback function for trade data
        """
        topic = f"publicTrade.{symbol}"
        await self._subscribe(
            topic,
            TopicType.TRADE,
            BybitChannelType.PUBLIC,
            symbol,
            callback
        )

    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        depth: int = 50
    ) -> None:
        """
        Subscribe to order book stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
            depth: Order book depth (1, 50, 200, 500)
        """
        topic = f"orderbook.{depth}.{symbol}"
        await self._subscribe(
            topic,
            TopicType.ORDERBOOK,
            BybitChannelType.PUBLIC,
            symbol,
            callback
        )

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to ticker stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
        """
        topic = f"tickers.{symbol}"
        await self._subscribe(
            topic,
            TopicType.TICKER,
            BybitChannelType.PUBLIC,
            symbol,
            callback
        )

    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to kline stream.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1, 3, 5, 15, 30, 60, etc.)
            callback: Async callback function
        """
        topic = f"kline.{interval}.{symbol}"
        await self._subscribe(
            topic,
            TopicType.KLINE,
            BybitChannelType.PUBLIC,
            symbol,
            callback
        )

    async def subscribe_position(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to position updates (private).

        Args:
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        topic = "position"
        await self._subscribe(
            topic,
            TopicType.POSITION,
            BybitChannelType.PRIVATE,
            None,
            callback
        )

    async def subscribe_order(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to order updates (private).

        Args:
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        topic = "order"
        await self._subscribe(
            topic,
            TopicType.ORDER,
            BybitChannelType.PRIVATE,
            None,
            callback
        )

    async def subscribe_execution(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to execution (trade) updates (private).

        Args:
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        topic = "execution"
        await self._subscribe(
            topic,
            TopicType.EXECUTION,
            BybitChannelType.PRIVATE,
            None,
            callback
        )

    async def subscribe_wallet(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to wallet updates (private).

        Args:
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        topic = "wallet"
        await self._subscribe(
            topic,
            TopicType.WALLET,
            BybitChannelType.PRIVATE,
            None,
            callback
        )

    async def _subscribe(
        self,
        topic: str,
        topic_type: TopicType,
        channel_type: BybitChannelType,
        symbol: Optional[str],
        callback: Callable
    ) -> None:
        """
        Internal subscription handler.

        Args:
            topic: Topic name
            topic_type: Type of topic
            channel_type: PUBLIC or PRIVATE
            symbol: Trading pair (if applicable)
            callback: Message callback
        """
        async with self._subscription_lock:
            subscription = TopicSubscription(
                topic=topic,
                topic_type=topic_type,
                channel_type=channel_type,
                symbol=symbol,
                callback=callback
            )

            self.subscriptions[topic] = subscription

            # Send subscription if connected
            if self.running:
                ws = self.public_ws if channel_type == BybitChannelType.PUBLIC else self.private_ws
                if ws:
                    await self._send_subscribe(ws, topic)

            logger.info(
                "subscribed_to_topic",
                topic=topic,
                type=topic_type.value,
                channel=channel_type.value
            )

    async def unsubscribe(self, topic: str) -> None:
        """
        Unsubscribe from topic.

        Args:
            topic: Topic name to unsubscribe
        """
        async with self._subscription_lock:
            subscription = self.subscriptions.get(topic)
            if subscription:
                if self.running:
                    ws = (self.public_ws if subscription.channel_type == BybitChannelType.PUBLIC
                          else self.private_ws)
                    if ws:
                        await self._send_unsubscribe(ws, topic)

                del self.subscriptions[topic]

                logger.info(
                    "unsubscribed_from_topic",
                    topic=topic
                )

    async def _send_subscribe(
        self,
        ws: WebSocketClientProtocol,
        topic: str
    ) -> None:
        """Send subscription message."""
        message = {
            "op": "subscribe",
            "args": [topic]
        }

        try:
            await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "subscribe_send_failed",
                topic=topic,
                error=str(e)
            )

    async def _send_unsubscribe(
        self,
        ws: WebSocketClientProtocol,
        topic: str
    ) -> None:
        """Send unsubscription message."""
        message = {
            "op": "unsubscribe",
            "args": [topic]
        }

        try:
            await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "unsubscribe_send_failed",
                topic=topic,
                error=str(e)
            )

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active topics after reconnect."""
        async with self._subscription_lock:
            for topic, subscription in self.subscriptions.items():
                ws = (self.public_ws if subscription.channel_type == BybitChannelType.PUBLIC
                      else self.private_ws)
                if ws:
                    await self._send_subscribe(ws, topic)

        logger.info(
            "resubscribed_all_topics",
            count=len(self.subscriptions)
        )

    async def _ping_loop(self) -> None:
        """Background task for sending ping messages."""
        while self.running:
            try:
                await asyncio.sleep(float(self._ping_interval))

                # Ping public WebSocket
                if self.public_ws and not self.public_ws.closed:
                    await self.public_ws.send(json.dumps({"op": "ping"}))

                # Ping private WebSocket
                if self.private_ws and not self.private_ws.closed:
                    await self.private_ws.send(json.dumps({"op": "ping"}))

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
        channel_type: BybitChannelType
    ) -> None:
        """
        Background task for receiving messages.

        Args:
            ws: WebSocket connection
            channel_type: Channel type (PUBLIC or PRIVATE)
        """
        while self.running:
            try:
                if ws.closed:
                    await asyncio.sleep(1)
                    continue

                message = await ws.recv()

                # Update pong time
                if channel_type == BybitChannelType.PUBLIC:
                    self._last_public_pong = datetime.now(timezone.utc)
                else:
                    self._last_private_pong = datetime.now(timezone.utc)

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
        channel_type: BybitChannelType
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed message data
            channel_type: Channel type
        """
        self._metrics['total_messages'] += 1

        if channel_type == BybitChannelType.PUBLIC:
            self._metrics['public_messages'] += 1
        else:
            self._metrics['private_messages'] += 1

        # Handle pong
        if data.get('op') == 'pong':
            return

        # Handle subscription response
        if data.get('op') in ['subscribe', 'unsubscribe']:
            if not data.get('success'):
                logger.warning(
                    "subscription_failed",
                    data=data
                )
            return

        # Handle auth response
        if data.get('op') == 'auth':
            return

        # Handle topic data
        if 'topic' in data:
            topic = data['topic']
            topic_data = data.get('data', [])

            subscription = self.subscriptions.get(topic)
            if subscription and subscription.active:
                subscription.message_count += 1
                subscription.last_update = datetime.now(timezone.utc)

                # Track by topic type
                topic_type = subscription.topic_type.value
                self._metrics['messages_by_topic'][topic_type] = \
                    self._metrics['messages_by_topic'].get(topic_type, 0) + 1

                # Call callback
                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(topic_data)
                    else:
                        subscription.callback(topic_data)

                except Exception as e:
                    logger.error(
                        "callback_error",
                        topic=topic,
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

    async def __aenter__(self) -> 'BybitWebSocket':
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.disconnect()
