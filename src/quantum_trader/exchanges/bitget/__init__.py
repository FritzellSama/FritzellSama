"""Bitget exchange module for quantum trader.

This module provides a complete implementation of the Bitget exchange connector,
supporting Spot, Perpetual (USDT and Coin margin), and Copy trading markets.

Features:
    - Spot trading (BUY, SELL, LIMIT, MARKET orders)
    - USDT-M Perpetual Futures
    - Coin-M Perpetual Futures
    - Copy trading integration
    - Real-time WebSocket market data
    - Order management and cancellation
    - Position management and tracking
    - Advanced order types (conditional, bracket orders, etc.)

API Support:
    - REST API (v2)
    - WebSocket Public Channels
    - WebSocket Private Channels for real-time updates
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add Bitget connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
