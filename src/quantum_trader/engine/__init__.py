"""Trading engine module for order execution and portfolio management.

This module provides the core trading engine functionality including:
- Order execution and management
- Portfolio state tracking and updates
- Position management and lifecycle
- Trade execution strategies
- Real-time market interaction
- Order matching with price-time priority
- Post-trade reconciliation and break detection
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_trader.engine.matching_engine import MatchingEngine, Match, OrderBookLevel
    from quantum_trader.engine.order_manager import OrderManager, OrderLifecycleEvent
    from quantum_trader.engine.reconciliation import (
        ReconciliationEngine,
        Discrepancy,
        DiscrepancyType,
        DiscrepancySeverity,
        ResolutionStatus,
    )

__all__: list[str] = [
    "MatchingEngine",
    "Match",
    "OrderBookLevel",
    "OrderManager",
    "OrderLifecycleEvent",
    "ReconciliationEngine",
    "Discrepancy",
    "DiscrepancyType",
    "DiscrepancySeverity",
    "ResolutionStatus",
]

__version__ = "1.0.0"
