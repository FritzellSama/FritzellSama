"""Exchange connectors module for quantum trader.

This module provides abstract base classes and interfaces for exchange connectors,
defining the standard contract that all exchange implementations must follow.

Interfaces:
    - BaseExchangeConnector: Core exchange connection interface
    - MarketDataConnector: Real-time market data interface
    - TradeExecutionConnector: Order execution interface
    - AccountConnector: Account and portfolio interface
    - WebSocketConnector: Real-time data streaming interface

Features:
    - Unified connection management
    - Authentication and security
    - Error handling and recovery
    - Rate limiting and throttling
    - Retry mechanisms
    - Event publishing
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = [
    # Add connector classes here as they are implemented
]

__version__ = "1.0.0"
__author__ = "QuantumTrader Team"
