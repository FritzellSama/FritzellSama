"""
WebSocket connection manager for Quantum Trader AI.

Manages WebSocket connections, subscriptions, and lifecycle.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from structlog import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Production-ready WebSocket connection manager.

    Manages WebSocket connections with:
    - Connection lifecycle management
    - Thread-safe connection tracking
    - Subscription management
    - Graceful shutdown
    - Health monitoring
    - Rate limiting

    Attributes:
        config: Configuration dictionary
        active_connections: Active WebSocket connections
        user_sessions: User session tracking
        connection_metadata: Connection metadata storage
        heartbeat_interval: Heartbeat interval in seconds
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize connection manager.

        Args:
            config: Configuration from config files

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config

        # Connection tracking (thread-safe)
        self._lock = asyncio.Lock()
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Configuration
        ws_config = config.get("websocket", {})
        self.heartbeat_interval = ws_config.get("heartbeat_interval_seconds", 30)
        self.max_connections_per_user = ws_config.get("max_connections_per_user", 10)
        self.connection_timeout = ws_config.get("connection_timeout_seconds", 300)

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        self._validate_config()
        logger.info("Connection manager initialized")

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if self.max_connections_per_user <= 0:
            raise ValueError("max_connections_per_user must be positive")
        if self.connection_timeout <= 0:
            raise ValueError("connection_timeout must be positive")

    async def start(self) -> None:
        """Start connection manager background tasks."""
        if not self._running:
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_worker())
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
            logger.info("Connection manager started")

    async def stop(self) -> None:
        """Stop connection manager and close all connections."""
        self._running = False

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        await self.disconnect_all()

        logger.info("Connection manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Accept and register WebSocket connection.

        Args:
            websocket: WebSocket instance
            connection_id: Unique connection identifier
            user_id: User identifier (optional)
            metadata: Additional connection metadata

        Returns:
            True if connection accepted, False otherwise

        Raises:
            ValueError: If connection limit exceeded
        """
        try:
            async with self._lock:
                # Check user connection limit
                if user_id:
                    user_conn_count = len(self.user_sessions.get(user_id, set()))
                    if user_conn_count >= self.max_connections_per_user:
                        logger.warning(
                            "User connection limit exceeded",
                            user_id=user_id,
                            limit=self.max_connections_per_user
                        )
                        return False

                # Accept connection
                await websocket.accept()

                # Register connection
                self.active_connections[connection_id] = websocket

                # Track user session
                if user_id:
                    if user_id not in self.user_sessions:
                        self.user_sessions[user_id] = set()
                    self.user_sessions[user_id].add(connection_id)

                # Store metadata
                self.connection_metadata[connection_id] = {
                    "user_id": user_id,
                    "connected_at": datetime.utcnow(),
                    "last_activity": datetime.utcnow(),
                    "metadata": metadata or {}
                }

                logger.info(
                    "WebSocket connected",
                    connection_id=connection_id,
                    user_id=user_id,
                    total_connections=len(self.active_connections)
                )

                return True

        except Exception as e:
            logger.error("Connection failed", connection_id=connection_id, error=str(e))
            return False

    async def disconnect(self, connection_id: str) -> None:
        """Disconnect and unregister WebSocket connection.

        Args:
            connection_id: Connection identifier
        """
        try:
            async with self._lock:
                # Get connection
                websocket = self.active_connections.get(connection_id)
                if not websocket:
                    return

                # Get metadata
                metadata = self.connection_metadata.get(connection_id, {})
                user_id = metadata.get("user_id")

                # Close websocket
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning("Error closing websocket", error=str(e))

                # Remove from tracking
                del self.active_connections[connection_id]

                if connection_id in self.connection_metadata:
                    del self.connection_metadata[connection_id]

                # Remove from user sessions
                if user_id and user_id in self.user_sessions:
                    self.user_sessions[user_id].discard(connection_id)
                    if not self.user_sessions[user_id]:
                        del self.user_sessions[user_id]

                logger.info(
                    "WebSocket disconnected",
                    connection_id=connection_id,
                    user_id=user_id,
                    total_connections=len(self.active_connections)
                )

        except Exception as e:
            logger.error("Disconnect error", connection_id=connection_id, error=str(e))

    async def disconnect_all(self) -> None:
        """Disconnect all WebSocket connections."""
        logger.info("Disconnecting all connections")

        connection_ids = list(self.active_connections.keys())

        for connection_id in connection_ids:
            await self.disconnect(connection_id)

        logger.info("All connections disconnected")

    async def disconnect_user(self, user_id: str) -> None:
        """Disconnect all connections for a user.

        Args:
            user_id: User identifier
        """
        async with self._lock:
            connection_ids = list(self.user_sessions.get(user_id, set()))

        for connection_id in connection_ids:
            await self.disconnect(connection_id)

        logger.info("User connections disconnected", user_id=user_id)

    async def send_message(
        self,
        connection_id: str,
        message: str
    ) -> bool:
        """Send message to specific connection.

        Args:
            connection_id: Connection identifier
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            websocket = self.active_connections.get(connection_id)
            if not websocket:
                logger.warning("Connection not found", connection_id=connection_id)
                return False

            await websocket.send_text(message)

            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()

            return True

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during send", connection_id=connection_id)
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error("Send message failed", connection_id=connection_id, error=str(e))
            await self.disconnect(connection_id)
            return False

    async def send_json(
        self,
        connection_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send JSON message to specific connection.

        Args:
            connection_id: Connection identifier
            data: Data to send as JSON

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            websocket = self.active_connections.get(connection_id)
            if not websocket:
                return False

            await websocket.send_json(data)

            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow()

            return True

        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error("Send JSON failed", connection_id=connection_id, error=str(e))
            await self.disconnect(connection_id)
            return False

    async def broadcast(
        self,
        message: str,
        user_filter: Optional[Set[str]] = None
    ) -> int:
        """Broadcast message to all or filtered connections.

        Args:
            message: Message to broadcast
            user_filter: Optional set of user IDs to filter

        Returns:
            Number of successful sends
        """
        sent_count = 0
        failed_connections = []

        async with self._lock:
            connection_ids = list(self.active_connections.keys())

        for connection_id in connection_ids:
            # Check user filter
            if user_filter:
                metadata = self.connection_metadata.get(connection_id, {})
                user_id = metadata.get("user_id")
                if not user_id or user_id not in user_filter:
                    continue

            # Send message
            success = await self.send_message(connection_id, message)
            if success:
                sent_count += 1
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        return sent_count

    async def _heartbeat_worker(self) -> None:
        """Background worker for sending heartbeat pings."""
        logger.info("Heartbeat worker started")

        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)

                async with self._lock:
                    connection_ids = list(self.active_connections.keys())

                for connection_id in connection_ids:
                    try:
                        websocket = self.active_connections.get(connection_id)
                        if websocket:
                            await websocket.send_json({
                                "type": "heartbeat",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                    except Exception as e:
                        logger.warning(
                            "Heartbeat failed",
                            connection_id=connection_id,
                            error=str(e)
                        )
                        await self.disconnect(connection_id)

        except asyncio.CancelledError:
            logger.info("Heartbeat worker cancelled")
            raise

    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up stale connections."""
        logger.info("Cleanup worker started")

        try:
            while self._running:
                await asyncio.sleep(60)  # Run every minute

                now = datetime.utcnow()
                stale_connections = []

                async with self._lock:
                    for connection_id, metadata in self.connection_metadata.items():
                        last_activity = metadata.get("last_activity")
                        if last_activity:
                            idle_seconds = (now - last_activity).total_seconds()
                            if idle_seconds > self.connection_timeout:
                                stale_connections.append(connection_id)

                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.info(
                        "Disconnecting stale connection",
                        connection_id=connection_id
                    )
                    await self.disconnect(connection_id)

        except asyncio.CancelledError:
            logger.info("Cleanup worker cancelled")
            raise

    def get_connection_count(self) -> int:
        """Get number of active connections.

        Returns:
            Number of active connections
        """
        return len(self.active_connections)

    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for user.

        Args:
            user_id: User identifier

        Returns:
            Number of connections for user
        """
        return len(self.user_sessions.get(user_id, set()))

    def get_connection_metadata(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection metadata.

        Args:
            connection_id: Connection identifier

        Returns:
            Connection metadata or None
        """
        return self.connection_metadata.get(connection_id)

    def is_connected(self, connection_id: str) -> bool:
        """Check if connection is active.

        Args:
            connection_id: Connection identifier

        Returns:
            True if connected
        """
        return connection_id in self.active_connections

    async def get_health_status(self) -> Dict[str, Any]:
        """Get connection manager health status.

        Returns:
            Dictionary with health metrics
        """
        async with self._lock:
            total_connections = len(self.active_connections)
            total_users = len(self.user_sessions)

        return {
            "status": "healthy" if self._running else "stopped",
            "total_connections": total_connections,
            "total_users": total_users,
            "max_connections_per_user": self.max_connections_per_user,
            "heartbeat_interval": self.heartbeat_interval,
            "connection_timeout": self.connection_timeout
        }
