"""WebSocket API for real-time communication.

This module provides WebSocket endpoint implementations for real-time market
data streaming, trading updates, and portfolio notifications. WebSocket
connections enable low-latency bidirectional communication between clients
and the trading system.

Features:
- Real-time market data streaming (OHLCV, order book, trades)
- Live trading order status updates
- Portfolio position changes and P&L updates
- User notifications and alerts
- Automatic reconnection and message queuing
- Connection authentication and authorization

Channels are organized by topic:
- market/<symbol> - Market data for specific symbols
- trading/orders - Personal order updates
- trading/fills - Trade execution updates
- portfolio/positions - Position changes
- portfolio/pnl - P&L updates
- notifications - System alerts and messages
"""

from typing import Callable, Optional, Dict, Any

__all__ = [
    "WebSocketManager",
    "WebSocketConnection",
    "MessageHandler",
    "ConnectionPool",
    "create_websocket_handler",
]

# Placeholder exports - import actual classes when available
# from .manager import WebSocketManager
# from .connection import WebSocketConnection
# from .handler import MessageHandler, create_websocket_handler
# from .pool import ConnectionPool

__version__ = "1.0.0"
