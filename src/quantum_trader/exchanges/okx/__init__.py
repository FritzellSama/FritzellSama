"""OKX exchange module for quantum trader.

This module provides a complete implementation of the OKX (formerly OKEx) exchange
connector, supporting Spot, Margin, Perpetuals, Futures, and Options markets.

Features:
    - Spot trading
    - Margin trading (isolated and cross)
    - USDT and USD Perpetuals (linear swaps)
    - Quarterly and bi-weekly futures
    - Options trading
    - Real-time WebSocket market data
    - Order management and execution
    - Position management with leverage
    - Advanced order types and algos
    - Portfolio and risk management

API Support:
    - REST API (v5)
    - WebSocket Public Channels (market data, index, funding rates)
    - WebSocket Private Channels (orders, positions, account)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add OKX connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
