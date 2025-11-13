"""Exchanges module for quantum trader.

This module provides unified interfaces and implementations for connecting to and
trading on multiple cryptocurrency exchanges. It abstracts away exchange-specific
differences and provides a consistent API for order execution, market data, and
account management.

Supported Exchanges:
    - Binance: Spot, Futures, Margin
    - Bybit: Perpetual, Inverse, Spot
    - OKX: Spot, Perpetual, Options
    - KuCoin: Spot, Margin, Futures

Features:
    - Unified API across exchanges
    - Order execution and management
    - Real-time market data feeds
    - Account and portfolio management
    - Risk management and position tracking
    - WebSocket and REST API support
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add exchange classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
