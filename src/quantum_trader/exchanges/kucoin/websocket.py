"""
KuCoin WebSocket Client - Real-time market data streaming.

Implements KuCoin WebSocket API for real-time order book, trades, and account updates
with token-based connection and automatic reconnection.
"""

import asyncio
import json
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import websockets
from websockets.client import WebSocketClientProtocol
import structlog

logger = structlog.get_logger(__name__)


class KuCoinTopicType(Enum):
    """KuCoin WebSocket topic types."""
    TICKER = "ticker"
    LEVEL2 = "level2"
    MATCH = "match"
    CANDLES = "candles"
    # Private topics
    ACCOUNT = "account"
    POSITION = "position"
    ORDER = "orderChange"


@dataclass
class KuCoinSubscription:
    """KuCoin WebSocket subscription."""
    topic: str
    topic_type: KuCoinTopicType
    symbol: Optional[str]
    is_private: bool
    callback: Callable
    active: bool = True
    last_update: Optional[datetime] = None
    message_count: int = 0


class KuCoinWebSocket:
    """
    Production-grade KuCoin WebSocket client.

    Handles both public and private WebSocket streams with token-based authentication
    and automatic reconnection. Uses KuCoin's bullet-connect protocol.

    Attributes:
        config: Configuration dictionary
        rest_client: REST client for obtaining WebSocket tokens
        subscriptions: Active topic subscriptions

    Example:
        >>> config = {
        ...     'reconnect_interval': 5,
        ...     'ping_interval': 20
        ... }
        >>> ws_client = KuCoinWebSocket(config, rest_client)
        >>> await ws_client.connect()
        >>>
        >>> async def handle_ticker(data):
        ...     print(f"Ticker: {data}")
        >>>
        >>> await ws_client.subscribe_ticker('BTC-USDT', handle_ticker)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        rest_client: Any
    ) -> None:
        """
        Initialize KuCoin WebSocket client.

        Args:
            config: Configuration dictionary
            rest_client: REST client for token requests

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.rest_client = rest_client

        # Connection details (obtained from API)
        self.ws_url: Optional[str] = None
        self.token: Optional[str] = None
        self.ping_interval_ms: Optional[int] = None
        self.ping_timeout_ms: Optional[int] = None

        # WebSocket connections
        self.public_ws: Optional[WebSocketClientProtocol] = None
        self.private_ws: Optional[WebSocketClientProtocol] = None
        self.running = False

        self._connect_lock = asyncio.Lock()
        self._reconnect_interval = Decimal(
            str(config.get('reconnect_interval', 5))
        )

        # Subscription management
        self.subscriptions: Dict[str, KuCoinSubscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Background tasks
        self._public_receive_task: Optional[asyncio.Task] = None
        self._private_receive_task: Optional[asyncio.Task] = None
        self._public_ping_task: Optional[asyncio.Task] = None
        self._private_ping_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_messages': 0,
            'public_messages': 0,
            'private_messages': 0,
            'messages_by_topic': {},
            'reconnect_count': 0,
            'connection_errors': 0,
            'last_connected': None
        }

        logger.info("kucoin_websocket_initialized")

    async def connect(self) -> None:
        """
        Establish WebSocket connections.

        Raises:
            RuntimeError: If connection fails
        """
        async with self._connect_lock:
            if self.running:
                logger.warning("websocket_already_connected")
                return

            try:
                logger.info("connecting_to_kucoin_websocket")

                # Get public bullet token
                public_token_data = await self._get_public_token()
                self.ws_url = public_token_data['instanceServers'][0]['endpoint']
                self.token = public_token_data['token']
                self.ping_interval_ms = public_token_data['instanceServers'][0].get('pingInterval', 18000)
                self.ping_timeout_ms = public_token_data['instanceServers'][0].get('pingTimeout', 10000)

                # Connect public WebSocket
                public_url = f"{self.ws_url}?token={self.token}"
                self.public_ws = await websockets.connect(
                    public_url,
                    ping_interval=None,
                    close_timeout=float(self.config.get('close_timeout', 10)),
                    max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                )

                # Get private bullet token if authenticated
                if hasattr(self.rest_client, 'api_key') and self.rest_client.api_key:
                    try:
                        private_token_data = await self._get_private_token()
                        private_url = f"{private_token_data['instanceServers'][0]['endpoint']}?token={private_token_data['token']}"

                        self.private_ws = await websockets.connect(
                            private_url,
                            ping_interval=None,
                            close_timeout=float(self.config.get('close_timeout', 10)),
                            max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                        )

                        logger.info("private_websocket_connected")

                    except Exception as e:
                        logger.warning(
                            "private_websocket_connection_failed",
                            error=str(e)
                        )

                self.running = True
                self._metrics['last_connected'] = datetime.now(timezone.utc)

                # Start background tasks
                self._public_receive_task = asyncio.create_task(
                    self._receive_loop(self.public_ws, False)
                )
                self._public_ping_task = asyncio.create_task(
                    self._ping_loop(self.public_ws)
                )

                if self.private_ws:
                    self._private_receive_task = asyncio.create_task(
                        self._receive_loop(self.private_ws, True)
                    )
                    self._private_ping_task = asyncio.create_task(
                        self._ping_loop(self.private_ws)
                    )

                logger.info(
                    "kucoin_websocket_connected",
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
                raise RuntimeError(f"Failed to connect to KuCoin WebSocket: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket connections and cleanup."""
        logger.info("disconnecting_kucoin_websocket")

        self.running = False

        # Cancel background tasks
        for task in [
            self._public_receive_task,
            self._private_receive_task,
            self._public_ping_task,
            self._private_ping_task
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

        logger.info("kucoin_websocket_disconnected")

    async def _get_public_token(self) -> Dict[str, Any]:
        """
        Get public WebSocket bullet token.

        Returns:
            Token data from API
        """
        response = await self.rest_client.request(
            'POST',
            '/api/v1/bullet-public',
            auth_required=False
        )

        if response.get('code') != '200000':
            raise RuntimeError(f"Failed to get public token: {response}")

        return response['data']

    async def _get_private_token(self) -> Dict[str, Any]:
        """
        Get private WebSocket bullet token.

        Returns:
            Token data from API
        """
        response = await self.rest_client.request(
            'POST',
            '/api/v1/bullet-private',
            auth_required=True
        )

        if response.get('code') != '200000':
            raise RuntimeError(f"Failed to get private token: {response}")

        return response['data']

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to ticker stream.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            callback: Async callback function
        """
        topic = f"/market/ticker:{symbol}"
        await self._subscribe(
            topic,
            KuCoinTopicType.TICKER,
            symbol,
            False,
            callback
        )

    async def subscribe_level2(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to level2 order book stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
        """
        topic = f"/market/level2:{symbol}"
        await self._subscribe(
            topic,
            KuCoinTopicType.LEVEL2,
            symbol,
            False,
            callback
        )

    async def subscribe_match(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to trade match stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
        """
        topic = f"/market/match:{symbol}"
        await self._subscribe(
            topic,
            KuCoinTopicType.MATCH,
            symbol,
            False,
            callback
        )

    async def subscribe_candles(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to candlestick stream.

        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1min, 5min, 1hour, etc.)
            callback: Async callback function
        """
        topic = f"/market/candles:{symbol}_{interval}"
        await self._subscribe(
            topic,
            KuCoinTopicType.CANDLES,
            symbol,
            False,
            callback
        )

    async def subscribe_account(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to account balance updates (private).

        Args:
            callback: Async callback function
        """
        if not self.private_ws:
            raise RuntimeError("Private WebSocket not configured")

        topic = "/account/balance"
        await self._subscribe(
            topic,
            KuCoinTopicType.ACCOUNT,
            None,
            True,
            callback
        )

    async def subscribe_orders(
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

        topic = "/spotMarket/tradeOrders"
        await self._subscribe(
            topic,
            KuCoinTopicType.ORDER,
            None,
            True,
            callback
        )

    async def _subscribe(
        self,
        topic: str,
        topic_type: KuCoinTopicType,
        symbol: Optional[str],
        is_private: bool,
        callback: Callable
    ) -> None:
        """
        Internal subscription handler.

        Args:
            topic: Topic path
            topic_type: Type of topic
            symbol: Trading pair (if applicable)
            is_private: Whether private channel
            callback: Message callback
        """
        async with self._subscription_lock:
            subscription = KuCoinSubscription(
                topic=topic,
                topic_type=topic_type,
                symbol=symbol,
                is_private=is_private,
                callback=callback
            )

            self.subscriptions[topic] = subscription

            # Send subscription if connected
            if self.running:
                ws = self.private_ws if is_private else self.public_ws
                if ws:
                    await self._send_subscribe(ws, topic, is_private)

            logger.info(
                "subscribed_to_topic",
                topic=topic,
                type=topic_type.value,
                private=is_private
            )

    async def _send_subscribe(
        self,
        ws: WebSocketClientProtocol,
        topic: str,
        is_private: bool
    ) -> None:
        """Send subscription message."""
        subscribe_id = str(int(time.time() * 1000))

        message = {
            "id": subscribe_id,
            "type": "subscribe",
            "topic": topic,
            "privateChannel": is_private,
            "response": True
        }

        try:
            await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "subscribe_send_failed",
                topic=topic,
                error=str(e)
            )

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active topics after reconnect."""
        async with self._subscription_lock:
            for topic, subscription in self.subscriptions.items():
                ws = self.private_ws if subscription.is_private else self.public_ws
                if ws:
                    await self._send_subscribe(ws, topic, subscription.is_private)

        logger.info(
            "resubscribed_all_topics",
            count=len(self.subscriptions)
        )

    async def _ping_loop(self, ws: WebSocketClientProtocol) -> None:
        """
        Background task for sending ping messages.

        Args:
            ws: WebSocket connection
        """
        if not self.ping_interval_ms:
            return

        ping_interval_seconds = Decimal(str(self.ping_interval_ms)) / Decimal("1000")

        while self.running:
            try:
                await asyncio.sleep(float(ping_interval_seconds))

                if ws and not ws.closed:
                    ping_message = {
                        "id": str(int(time.time() * 1000)),
                        "type": "ping"
                    }
                    await ws.send(json.dumps(ping_message))

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
        is_private: bool
    ) -> None:
        """
        Background task for receiving messages.

        Args:
            ws: WebSocket connection
            is_private: Whether private channel
        """
        while self.running:
            try:
                if ws.closed:
                    await asyncio.sleep(1)
                    continue

                message = await ws.recv()

                # Parse and handle message
                try:
                    data = json.loads(message)
                    await self._handle_message(data, is_private)

                except json.JSONDecodeError as e:
                    logger.warning(
                        "json_decode_error",
                        error=str(e),
                        message=message[:200]
                    )

            except websockets.ConnectionClosed:
                logger.warning(
                    "websocket_connection_closed",
                    private=is_private
                )
                if self.running:
                    await self._reconnect()
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(
                    "receive_loop_error",
                    private=is_private,
                    error=str(e)
                )
                await asyncio.sleep(1)

    async def _handle_message(
        self,
        data: Dict[str, Any],
        is_private: bool
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed message data
            is_private: Whether from private channel
        """
        self._metrics['total_messages'] += 1

        if is_private:
            self._metrics['private_messages'] += 1
        else:
            self._metrics['public_messages'] += 1

        # Handle pong
        if data.get('type') == 'pong':
            return

        # Handle welcome message
        if data.get('type') == 'welcome':
            logger.info("websocket_welcome_received")
            return

        # Handle ack
        if data.get('type') == 'ack':
            return

        # Handle error
        if data.get('type') == 'error':
            logger.error(
                "websocket_error_message",
                data=data
            )
            return

        # Handle message data
        if data.get('type') == 'message':
            topic = data.get('topic')
            subject = data.get('subject')

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
                    message_data = data.get('data', {})
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(message_data)
                    else:
                        subscription.callback(message_data)

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

    async def __aenter__(self) -> 'KuCoinWebSocket':
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.disconnect()
