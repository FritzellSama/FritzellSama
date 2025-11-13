"""Database repositories module for quantum trader.

This module implements the repository pattern for database access, providing
a clean abstraction layer over the ORM for data persistence and retrieval.

Repositories:
    - MarketDataRepository: OHLCV and market data persistence
    - SignalRepository: Trading signal management
    - TradeRepository: Trade history and management
    - PositionRepository: Position tracking and analysis
    - OrderRepository: Order management and execution tracking
    - AccountRepository: Account state and portfolio management
    - ConfigurationRepository: System configuration storage
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add repository classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
