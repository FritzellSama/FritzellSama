"""
Quantum Trader AI - Production Trading System
Manages billions in capital across multiple exchanges
"""

__version__ = "1.0.0"
__author__ = "Quantum Trading Team"

# Export all models
from .models import (
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    SignalAction,
    Exchange,
    TimeFrame,
    # Data Models
    Order,
    Position,
    Signal,
    ExecutionResult,
    Trade,
    MarketData,
    OrderBook,
    Balance,
)

# Export utility functions
from .utils import ConfigLoader, get_config

__all__ = [
    # Version
    '__version__',
    # Enums
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'SignalAction',
    'Exchange',
    'TimeFrame',
    # Models
    'Order',
    'Position',
    'Signal',
    'ExecutionResult',
    'Trade',
    'MarketData',
    'OrderBook',
    'Balance',
    # Utils
    'ConfigLoader',
    'get_config',
]
