"""
Order flow trading strategies module.

This module provides order flow analysis and trading strategies that monitor
and react to market microstructure signals. Strategies analyze order book
dynamics, trade execution patterns, and institutional order flow.

Features:
    - Order book imbalance detection
    - Trade flow classification
    - Institutional order detection
    - Market depth analysis
    - Real-time execution signals
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .order_book_analysis import OrderBookAnalysis
    from .trade_flow import TradeFlowAnalysis
    from .institutional_detection import InstitutionalOrderDetection

__version__: str = "1.0.0"

__all__: list[str] = [
    "OrderBookAnalysis",
    "TradeFlowAnalysis",
    "InstitutionalOrderDetection",
]
