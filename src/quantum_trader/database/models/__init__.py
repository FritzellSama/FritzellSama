"""Database models module for quantum trader.

This module contains all ORM models for the quantum trader system, including
models for market data, trading signals, positions, trades, and system configuration.

Models:
    - MarketData: OHLCV and derived market data
    - TradingSignal: Trading signals and indicators
    - Position: Open and closed trading positions
    - Trade: Individual trade records
    - Order: Order management and history
    - Account: Account and portfolio information
    - Configuration: System configuration and parameters
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add model classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
