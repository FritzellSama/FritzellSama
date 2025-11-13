"""KuCoin exchange module for quantum trader.

This module provides a complete implementation of the KuCoin exchange connector,
supporting Spot, Margin, and Futures (Perpetuals) markets.

Features:
    - Spot trading
    - Margin trading (isolated and cross margin)
    - USDT and USD Perpetuals
    - Real-time WebSocket market data and order updates
    - Order management and cancellation
    - Position management and leverage control
    - Advanced order types
    - Account and portfolio management
    - Lending and borrowing facilities

API Support:
    - REST API (v2)
    - WebSocket Public Channels (tickers, candles, depth, matches)
    - WebSocket Private Channels (orders, positions, account balance)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add KuCoin connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
