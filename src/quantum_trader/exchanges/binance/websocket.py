"""
Binance WebSocket Client - Real-time market data streaming.

Implements Binance WebSocket API for real-time order book, trades, and ticker updates
with automatic reconnection and subscription management.
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


class StreamType(Enum):
    """Binance WebSocket stream types."""
    TRADE = "trade"
    TICKER = "ticker"
    DEPTH = "depth"
    KLINE = "kline"
    AGG_TRADE = "aggTrade"
    BOOK_TICKER = "bookTicker"
    MINI_TICKER = "miniTicker"


@dataclass
class StreamSubscription:
    """WebSocket stream subscription."""
    stream_name: str
    stream_type: StreamType
    symbol: str
    callback: Callable
    active: bool = True
    last_update: Optional[datetime] = None
    message_count: int = 0


class BinanceWebSocket:
    """
    Production-grade Binance WebSocket client.

    Handles real-time market data streaming with automatic reconnection,
    subscription management, and message routing.

    Attributes:
        config: Configuration dictionary
        subscriptions: Active stream subscriptions
        ws: WebSocket connection
        running: Connection state

    Example:
        >>> config = {
        ...     'ws_url': 'wss://stream.binance.com:9443/ws',
        ...     'reconnect_interval': 5,
        ...     'ping_interval': 20
        ... }
        >>> ws_client = BinanceWebSocket(config)
        >>> await ws_client.connect()
        >>>
        >>> async def handle_trade(data):
        ...     print(f"Trade: {data['p']} @ {data['q']}")
        >>>
        >>> await ws_client.subscribe_trade('BTCUSDT', handle_trade)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Binance WebSocket client.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.ws_url = config['ws_url']
        self.testnet = config.get('testnet', False)

        # Connection management
        self.ws: Optional[WebSocketClientProtocol] = None
        self.running = False
        self._connect_lock = asyncio.Lock()
        self._reconnect_interval = Decimal(
            str(config.get('reconnect_interval', 5))
        )
        self._max_reconnect_attempts = int(
            config.get('max_reconnect_attempts', 0)  # 0 = infinite
        )

        # Subscription management
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Keep-alive
        self._ping_interval = Decimal(str(config.get('ping_interval', 20)))
        self._pong_timeout = Decimal(str(config.get('pong_timeout', 10)))
        self._ping_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._last_pong_time: Optional[datetime] = None

        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=int(config.get('message_queue_size', 1000))
        )

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_messages': 0,
            'messages_by_type': {},
            'reconnect_count': 0,
            'connection_errors': 0,
            'last_connected': None,
            'uptime_seconds': Decimal("0")
        }

        logger.info(
            "binance_websocket_initialized",
            url=self.ws_url,
            testnet=self.testnet
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if 'ws_url' not in self.config:
            raise ValueError("Missing 'ws_url' in configuration")

        if not self.config['ws_url'].startswith('wss://'):
            raise ValueError("WebSocket URL must use wss:// protocol")

    async def connect(self) -> None:
        """
        Establish WebSocket connection.

        Raises:
            RuntimeError: If connection fails
        """
        async with self._connect_lock:
            if self.running and self.ws:
                logger.warning("websocket_already_connected")
                return

            try:
                logger.info("connecting_to_binance_websocket")

                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=None,  # We handle pings manually
                    ping_timeout=float(self._pong_timeout),
                    close_timeout=float(self.config.get('close_timeout', 10)),
                    max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024))
                )

                self.running = True
                self._last_pong_time = datetime.now(timezone.utc)
                self._metrics['last_connected'] = datetime.now(timezone.utc)

                # Start background tasks
                self._ping_task = asyncio.create_task(self._ping_loop())
                self._receive_task = asyncio.create_task(self._receive_loop())

                logger.info("binance_websocket_connected")

                # Resubscribe to streams
                await self._resubscribe_all()

            except Exception as e:
                self._metrics['connection_errors'] += 1
                logger.error(
                    "websocket_connection_failed",
                    error=str(e)
                )
                raise RuntimeError(f"Failed to connect to Binance WebSocket: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup."""
        logger.info("disconnecting_binance_websocket")

        self.running = False

        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(
                    "websocket_close_error",
                    error=str(e)
                )

        self.ws = None

        # Update uptime
        if self._metrics['last_connected']:
            uptime = (
                datetime.now(timezone.utc) - self._metrics['last_connected']
            ).total_seconds()
            self._metrics['uptime_seconds'] += Decimal(str(uptime))

        logger.info("binance_websocket_disconnected")

    async def subscribe_trade(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to trade stream.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            callback: Async callback function for trade data

        Example:
            >>> async def on_trade(data):
            ...     price = Decimal(data['p'])
            ...     quantity = Decimal(data['q'])
            ...     print(f"Trade: {price} x {quantity}")
            >>> await ws.subscribe_trade('BTCUSDT', on_trade)
        """
        stream_name = f"{symbol.lower()}@trade"
        await self._subscribe(stream_name, StreamType.TRADE, symbol, callback)

    async def subscribe_depth(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        levels: str = "20",
        update_speed: str = "100ms"
    ) -> None:
        """
        Subscribe to order book depth stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
            levels: Depth levels (5, 10, 20)
            update_speed: Update speed (100ms, 1000ms)
        """
        stream_name = f"{symbol.lower()}@depth{levels}@{update_speed}"
        await self._subscribe(stream_name, StreamType.DEPTH, symbol, callback)

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
        stream_name = f"{symbol.lower()}@ticker"
        await self._subscribe(stream_name, StreamType.TICKER, symbol, callback)

    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to kline/candlestick stream.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 1h, etc.)
            callback: Async callback function
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe(stream_name, StreamType.KLINE, symbol, callback)

    async def subscribe_book_ticker(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to best bid/ask stream.

        Args:
            symbol: Trading pair symbol
            callback: Async callback function
        """
        stream_name = f"{symbol.lower()}@bookTicker"
        await self._subscribe(stream_name, StreamType.BOOK_TICKER, symbol, callback)

    async def _subscribe(
        self,
        stream_name: str,
        stream_type: StreamType,
        symbol: str,
        callback: Callable
    ) -> None:
        """
        Internal subscription handler.

        Args:
            stream_name: Stream name
            stream_type: Type of stream
            symbol: Trading pair
            callback: Message callback
        """
        async with self._subscription_lock:
            subscription = StreamSubscription(
                stream_name=stream_name,
                stream_type=stream_type,
                symbol=symbol,
                callback=callback
            )

            self.subscriptions[stream_name] = subscription

            # Send subscription if connected
            if self.running and self.ws:
                await self._send_subscribe(stream_name)

            logger.info(
                "subscribed_to_stream",
                stream=stream_name,
                type=stream_type.value
            )

    async def unsubscribe(self, stream_name: str) -> None:
        """
        Unsubscribe from stream.

        Args:
            stream_name: Stream name to unsubscribe
        """
        async with self._subscription_lock:
            if stream_name in self.subscriptions:
                if self.running and self.ws:
                    await self._send_unsubscribe(stream_name)

                del self.subscriptions[stream_name]

                logger.info(
                    "unsubscribed_from_stream",
                    stream=stream_name
                )

    async def _send_subscribe(self, stream_name: str) -> None:
        """Send subscription message to WebSocket."""
        if not self.ws:
            return

        message = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time() * 1000)
        }

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "subscribe_send_failed",
                stream=stream_name,
                error=str(e)
            )

    async def _send_unsubscribe(self, stream_name: str) -> None:
        """Send unsubscription message to WebSocket."""
        if not self.ws:
            return

        message = {
            "method": "UNSUBSCRIBE",
            "params": [stream_name],
            "id": int(time.time() * 1000)
        }

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "unsubscribe_send_failed",
                stream=stream_name,
                error=str(e)
            )

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active streams after reconnect."""
        async with self._subscription_lock:
            for stream_name in list(self.subscriptions.keys()):
                await self._send_subscribe(stream_name)

        logger.info(
            "resubscribed_all_streams",
            count=len(self.subscriptions)
        )

    async def _ping_loop(self) -> None:
        """Background task for sending ping messages."""
        while self.running:
            try:
                await asyncio.sleep(float(self._ping_interval))

                if self.ws and not self.ws.closed:
                    await self.ws.ping()

                    # Check for pong timeout
                    if self._last_pong_time:
                        elapsed = (
                            datetime.now(timezone.utc) - self._last_pong_time
                        ).total_seconds()

                        if Decimal(str(elapsed)) > self._pong_timeout * Decimal("2"):
                            logger.warning("pong_timeout_reconnecting")
                            await self._reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "ping_loop_error",
                    error=str(e)
                )

    async def _receive_loop(self) -> None:
        """Background task for receiving messages."""
        while self.running:
            try:
                if not self.ws or self.ws.closed:
                    await asyncio.sleep(1)
                    continue

                message = await self.ws.recv()
                self._last_pong_time = datetime.now(timezone.utc)

                # Parse and route message
                try:
                    data = json.loads(message)
                    await self._handle_message(data)

                except json.JSONDecodeError as e:
                    logger.warning(
                        "json_decode_error",
                        error=str(e),
                        message=message[:200]
                    )

            except websockets.ConnectionClosed:
                logger.warning("websocket_connection_closed")
                if self.running:
                    await self._reconnect()
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(
                    "receive_loop_error",
                    error=str(e)
                )
                if self.running:
                    await self._reconnect()
                    await asyncio.sleep(1)

    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            data: Parsed message data
        """
        self._metrics['total_messages'] += 1

        # Handle subscription response
        if 'result' in data:
            return

        # Handle stream data
        if 'stream' in data:
            stream_name = data['stream']
            stream_data = data['data']

            subscription = self.subscriptions.get(stream_name)
            if subscription and subscription.active:
                subscription.message_count += 1
                subscription.last_update = datetime.now(timezone.utc)

                # Track message type
                stream_type = subscription.stream_type.value
                self._metrics['messages_by_type'][stream_type] = \
                    self._metrics['messages_by_type'].get(stream_type, 0) + 1

                # Call callback
                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(stream_data)
                    else:
                        subscription.callback(stream_data)

                except Exception as e:
                    logger.error(
                        "callback_error",
                        stream=stream_name,
                        error=str(e)
                    )

        # Handle single stream format
        elif 'e' in data:
            event_type = data['e']
            symbol = data.get('s', '').lower()

            # Find matching subscription
            for stream_name, subscription in self.subscriptions.items():
                if symbol in stream_name and subscription.active:
                    subscription.message_count += 1
                    subscription.last_update = datetime.now(timezone.utc)

                    try:
                        if asyncio.iscoroutinefunction(subscription.callback):
                            await subscription.callback(data)
                        else:
                            subscription.callback(data)

                    except Exception as e:
                        logger.error(
                            "callback_error",
                            stream=stream_name,
                            error=str(e)
                        )
                    break

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
            if self._max_reconnect_attempts > 0 and attempt >= self._max_reconnect_attempts:
                logger.error("max_reconnect_attempts_reached")
                self.running = False
                break

            try:
                delay = min(
                    self._reconnect_interval * (Decimal("2") ** Decimal(str(attempt))),
                    Decimal("300")  # Max 5 minutes
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
            'connected': self.running and self.ws and not self.ws.closed,
            'active_subscriptions': len(self.subscriptions),
            'uptime_seconds': float(self._metrics['uptime_seconds'])
        }

    async def __aenter__(self) -> 'BinanceWebSocket':
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.disconnect()
