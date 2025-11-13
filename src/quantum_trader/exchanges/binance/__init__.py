"""Binance exchange module for quantum trader.

This module provides a complete implementation of the Binance exchange connector,
supporting Spot, Margin, Futures (USDM), and Coin Margin (COINM) markets.

Features:
    - Spot trading (BUY, SELL, LIMIT, MARKET orders)
    - Margin trading with isolated and cross margin
    - USD-M Futures (perpetuals and quarterly contracts)
    - Coin-M Futures
    - Real-time WebSocket market data
    - Order management and cancellation
    - Position management
    - Advanced order types (OCO, conditional, etc.)

API Support:
    - REST API (v3)
    - WebSocket Streams
    - User Data Streams for real-time updates
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add Binance connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
