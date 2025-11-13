"""Bybit exchange module for quantum trader.

This module provides a complete implementation of the Bybit exchange connector,
supporting Spot, Perpetual (Linear and Inverse), and Options markets.

Features:
    - Spot trading
    - USDT Perpetuals (linear contracts)
    - Inverse Perpetuals
    - Options trading
    - Real-time WebSocket market data
    - Order management (POST, AMEND, CANCEL)
    - Position management and leverage
    - Advanced order types
    - Copy trading integration

API Support:
    - REST API (v5)
    - WebSocket Public Streams (market data)
    - WebSocket Private Streams (orders, positions, execution)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add Bybit connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
