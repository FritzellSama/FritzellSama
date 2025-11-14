"""
WebSocket Connector - Universal WebSocket connector for cryptocurrency exchanges.

Provides robust WebSocket client with automatic reconnection, subscription management,
heartbeat monitoring, and message routing for exchange WebSocket APIs.
"""

import asyncio
import json
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
import websockets
from websockets.client import WebSocketClientProtocol
import structlog

logger = structlog.get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass
class Subscription:
    """WebSocket subscription."""
    channel: str
    params: Dict[str, Any]
    callback: Callable
    active: bool = True
    last_message: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0


@dataclass
class ConnectionMetrics:
    """WebSocket connection metrics."""
    total_messages: int = 0
    total_bytes: int = 0
    messages_per_channel: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    connection_count: int = 0
    reconnection_count: int = 0
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    uptime_seconds: Decimal = Decimal("0")
    error_count: int = 0


class WebSocketConnector:
    """
    Universal WebSocket connector for cryptocurrency exchanges.

    Provides robust WebSocket connection management with automatic reconnection,
    subscription management, heartbeat monitoring, and message routing.

    Attributes:
        config: Configuration dictionary
        url: WebSocket URL
        state: Connection state
        subscriptions: Active subscriptions
        metrics: Connection metrics

    Example:
        >>> config = {
        ...     'url': 'wss://stream.example.com/ws',
        ...     'reconnect_interval': 5,
        ...     'ping_interval': 20,
        ...     'max_reconnect_attempts': 10
        ... }
        >>> connector = WebSocketConnector(config)
        >>> await connector.connect()
        >>>
        >>> async def handle_message(data):
        ...     print(f"Received: {data}")
        >>>
        >>> await connector.subscribe('ticker', {'symbol': 'BTC/USDT'}, handle_message)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize WebSocket connector.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.url = config['url']
        self.state = ConnectionState.DISCONNECTED

        # Connection management
        self.ws: Optional[WebSocketClientProtocol] = None
        self._connect_lock = asyncio.Lock()

        # Reconnection configuration
        self._reconnect_interval = Decimal(
            str(config.get('reconnect_interval', 5))
        )
        self._max_reconnect_attempts = int(
            config.get('max_reconnect_attempts', 0)  # 0 = infinite
        )
        self._exponential_backoff = config.get('exponential_backoff', True)
        self._max_backoff = Decimal(str(config.get('max_backoff', 300)))

        # Heartbeat configuration
        self._ping_interval = Decimal(str(config.get('ping_interval', 20)))
        self._pong_timeout = Decimal(str(config.get('pong_timeout', 10)))
        self._last_pong: Optional[datetime] = None
        self._custom_ping_message = config.get('ping_message')
        self._custom_pong_check = config.get('pong_check_field')

        # Subscription management
        self.subscriptions: Dict[str, Subscription] = {}
        self._subscription_lock = asyncio.Lock()

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=int(config.get('message_queue_size', 1000))
        )

        # Metrics
        self.metrics = ConnectionMetrics()

        # Callbacks
        self._on_connect_callback: Optional[Callable] = None
        self._on_disconnect_callback: Optional[Callable] = None
        self._on_error_callback: Optional[Callable] = None

        logger.info(
            "websocket_connector_initialized",
            url=self.url,
            reconnect_interval=float(self._reconnect_interval)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if 'url' not in self.config:
            raise ValueError("Missing 'url' in configuration")

        url = self.config['url']
        if not url.startswith('ws://') and not url.startswith('wss://'):
            raise ValueError(f"Invalid WebSocket URL: {url}")

    def set_callbacks(
        self,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> None:
        """
        Set connection lifecycle callbacks.

        Args:
            on_connect: Called when connected
            on_disconnect: Called when disconnected
            on_error: Called on errors
        """
        self._on_connect_callback = on_connect
        self._on_disconnect_callback = on_disconnect
        self._on_error_callback = on_error

    async def connect(self) -> None:
        """
        Establish WebSocket connection.

        Raises:
            RuntimeError: If connection fails
        """
        async with self._connect_lock:
            if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
                logger.warning("websocket_already_connected_or_connecting")
                return

            try:
                self.state = ConnectionState.CONNECTING
                logger.info("connecting_to_websocket", url=self.url)

                # Build connection parameters
                timeout = aiohttp.ClientTimeout(
                    total=float(self.config.get('connect_timeout', 30))
                ) if 'connect_timeout' in self.config else None

                extra_headers = self.config.get('headers', {})

                self.ws = await websockets.connect(
                    self.url,
                    ping_interval=None,  # We handle pings manually
                    close_timeout=float(self.config.get('close_timeout', 10)),
                    max_size=int(self.config.get('max_message_size', 10 * 1024 * 1024)),
                    extra_headers=extra_headers
                )

                self.state = ConnectionState.CONNECTED
                self._last_pong = datetime.now(timezone.utc)
                self.metrics.connection_count += 1
                self.metrics.last_connected = datetime.now(timezone.utc)

                # Start background tasks
                self._receive_task = asyncio.create_task(self._receive_loop())
                self._ping_task = asyncio.create_task(self._ping_loop())
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                logger.info("websocket_connected", url=self.url)

                # Resubscribe to channels
                await self._resubscribe_all()

                # Call connect callback
                if self._on_connect_callback:
                    try:
                        if asyncio.iscoroutinefunction(self._on_connect_callback):
                            await self._on_connect_callback()
                        else:
                            self._on_connect_callback()
                    except Exception as e:
                        logger.error(
                            "connect_callback_error",
                            error=str(e)
                        )

            except Exception as e:
                self.state = ConnectionState.FAILED
                self.metrics.error_count += 1
                logger.error(
                    "websocket_connection_failed",
                    url=self.url,
                    error=str(e)
                )
                raise RuntimeError(f"Failed to connect to WebSocket: {e}")

    async def disconnect(self) -> None:
        """Disconnect WebSocket and cleanup."""
        logger.info("disconnecting_websocket")

        previous_state = self.state
        self.state = ConnectionState.DISCONNECTED

        # Cancel background tasks
        for task in [self._receive_task, self._ping_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
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

        # Update metrics
        if self.metrics.last_connected:
            uptime = (
                datetime.now(timezone.utc) - self.metrics.last_connected
            ).total_seconds()
            self.metrics.uptime_seconds += Decimal(str(uptime))

        self.metrics.last_disconnected = datetime.now(timezone.utc)

        logger.info("websocket_disconnected")

        # Call disconnect callback
        if self._on_disconnect_callback and previous_state == ConnectionState.CONNECTED:
            try:
                if asyncio.iscoroutinefunction(self._on_disconnect_callback):
                    await self._on_disconnect_callback()
                else:
                    self._on_disconnect_callback()
            except Exception as e:
                logger.error(
                    "disconnect_callback_error",
                    error=str(e)
                )

    async def subscribe(
        self,
        channel: str,
        params: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to WebSocket channel.

        Args:
            channel: Channel name
            params: Subscription parameters
            callback: Callback function for messages

        Example:
            >>> await connector.subscribe(
            ...     'orderbook',
            ...     {'symbol': 'BTC/USDT', 'depth': 20},
            ...     handle_orderbook
            ... )
        """
        async with self._subscription_lock:
            sub_key = self._make_subscription_key(channel, params)

            subscription = Subscription(
                channel=channel,
                params=params,
                callback=callback
            )

            self.subscriptions[sub_key] = subscription

            # Send subscription if connected
            if self.state == ConnectionState.CONNECTED and self.ws:
                await self._send_subscription(channel, params)

            logger.info(
                "subscribed_to_channel",
                channel=channel,
                params=params
            )

    async def unsubscribe(
        self,
        channel: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Unsubscribe from WebSocket channel.

        Args:
            channel: Channel name
            params: Subscription parameters
        """
        async with self._subscription_lock:
            sub_key = self._make_subscription_key(channel, params)

            if sub_key in self.subscriptions:
                if self.state == ConnectionState.CONNECTED and self.ws:
                    await self._send_unsubscription(channel, params)

                del self.subscriptions[sub_key]

                logger.info(
                    "unsubscribed_from_channel",
                    channel=channel,
                    params=params
                )

    def _make_subscription_key(
        self,
        channel: str,
        params: Dict[str, Any]
    ) -> str:
        """Create unique subscription key."""
        params_str = json.dumps(params, sort_keys=True)
        return f"{channel}:{params_str}"

    async def _send_subscription(
        self,
        channel: str,
        params: Dict[str, Any]
    ) -> None:
        """
        Send subscription message.

        Args:
            channel: Channel name
            params: Subscription parameters
        """
        if not self.ws:
            return

        # Build subscription message (exchange-specific)
        message = self.config.get('subscription_format', {}).copy()
        message['channel'] = channel
        message.update(params)

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "subscription_send_failed",
                channel=channel,
                error=str(e)
            )

    async def _send_unsubscription(
        self,
        channel: str,
        params: Dict[str, Any]
    ) -> None:
        """Send unsubscription message."""
        if not self.ws:
            return

        message = self.config.get('unsubscription_format', {}).copy()
        message['channel'] = channel
        message.update(params)

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(
                "unsubscription_send_failed",
                channel=channel,
                error=str(e)
            )

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all active channels after reconnect."""
        async with self._subscription_lock:
            for sub_key, subscription in list(self.subscriptions.items()):
                if subscription.active:
                    await self._send_subscription(
                        subscription.channel,
                        subscription.params
                    )

        logger.info(
            "resubscribed_all_channels",
            count=len(self.subscriptions)
        )

    async def _ping_loop(self) -> None:
        """Background task for sending ping messages."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(float(self._ping_interval))

                if self.ws and not self.ws.closed:
                    if self._custom_ping_message:
                        # Send custom ping message
                        await self.ws.send(
                            json.dumps(self._custom_ping_message)
                            if isinstance(self._custom_ping_message, dict)
                            else self._custom_ping_message
                        )
                    else:
                        # Send standard ping
                        await self.ws.ping()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "ping_loop_error",
                    error=str(e)
                )

    async def _health_check_loop(self) -> None:
        """Background task for health monitoring."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(float(self._pong_timeout))

                if self._last_pong:
                    elapsed = (
                        datetime.now(timezone.utc) - self._last_pong
                    ).total_seconds()

                    timeout_threshold = float(
                        self._ping_interval + self._pong_timeout
                    )

                    if elapsed > timeout_threshold:
                        logger.warning(
                            "pong_timeout_detected",
                            elapsed=elapsed,
                            threshold=timeout_threshold
                        )
                        await self._reconnect()
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_check_error",
                    error=str(e)
                )

    async def _receive_loop(self) -> None:
        """Background task for receiving messages."""
        while self.state == ConnectionState.CONNECTED:
            try:
                if not self.ws or self.ws.closed:
                    await asyncio.sleep(1)
                    continue

                message = await self.ws.recv()
                self._last_pong = datetime.now(timezone.utc)

                # Update metrics
                self.metrics.total_messages += 1
                self.metrics.total_bytes += len(message) if isinstance(message, (str, bytes)) else 0

                # Parse and route message
                try:
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')

                    data = json.loads(message)
                    await self._route_message(data)

                except json.JSONDecodeError as e:
                    logger.warning(
                        "json_decode_error",
                        error=str(e),
                        message=message[:200]
                    )

            except websockets.ConnectionClosed:
                logger.warning("websocket_connection_closed")
                if self.state == ConnectionState.CONNECTED:
                    await self._reconnect()
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                self.metrics.error_count += 1
                logger.error(
                    "receive_loop_error",
                    error=str(e)
                )

                if self._on_error_callback:
                    try:
                        if asyncio.iscoroutinefunction(self._on_error_callback):
                            await self._on_error_callback(e)
                        else:
                            self._on_error_callback(e)
                    except Exception as callback_error:
                        logger.error(
                            "error_callback_failed",
                            error=str(callback_error)
                        )

    async def _route_message(self, data: Dict[str, Any]) -> None:
        """
        Route message to appropriate subscription callback.

        Args:
            data: Parsed message data
        """
        # Extract channel from message (exchange-specific)
        channel_field = self.config.get('channel_field', 'channel')
        channel = data.get(channel_field)

        if not channel:
            # Message might be system message (pong, ack, etc.)
            return

        # Update metrics
        self.metrics.messages_per_channel[channel] += 1

        # Find matching subscriptions
        for sub_key, subscription in self.subscriptions.items():
            if subscription.channel == channel and subscription.active:
                subscription.message_count += 1
                subscription.last_message = datetime.now(timezone.utc)

                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(data)
                    else:
                        subscription.callback(data)

                except Exception as e:
                    subscription.error_count += 1
                    logger.error(
                        "subscription_callback_error",
                        channel=channel,
                        error=str(e)
                    )

    async def _reconnect(self) -> None:
        """Reconnect to WebSocket with exponential backoff."""
        self.state = ConnectionState.RECONNECTING
        self.metrics.reconnection_count += 1

        logger.info(
            "attempting_reconnect",
            attempt=self.metrics.reconnection_count
        )

        await self.disconnect()

        attempt = 0
        while self.state == ConnectionState.RECONNECTING:
            if self._max_reconnect_attempts > 0 and attempt >= self._max_reconnect_attempts:
                logger.error("max_reconnect_attempts_reached")
                self.state = ConnectionState.FAILED
                break

            try:
                # Calculate backoff delay
                if self._exponential_backoff:
                    delay = min(
                        self._reconnect_interval * (Decimal("2") ** Decimal(str(attempt))),
                        self._max_backoff
                    )
                else:
                    delay = self._reconnect_interval

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
        Get connector metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            'state': self.state.value,
            'total_messages': self.metrics.total_messages,
            'total_bytes': self.metrics.total_bytes,
            'messages_per_channel': dict(self.metrics.messages_per_channel),
            'connection_count': self.metrics.connection_count,
            'reconnection_count': self.metrics.reconnection_count,
            'error_count': self.metrics.error_count,
            'uptime_seconds': float(self.metrics.uptime_seconds),
            'active_subscriptions': len(self.subscriptions),
            'last_connected': (
                self.metrics.last_connected.isoformat()
                if self.metrics.last_connected else None
            ),
            'last_disconnected': (
                self.metrics.last_disconnected.isoformat()
                if self.metrics.last_disconnected else None
            )
        }

    async def __aenter__(self) -> 'WebSocketConnector':
        """Context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.disconnect()
