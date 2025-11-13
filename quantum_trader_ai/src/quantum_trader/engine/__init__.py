"""
Quantum Trader Trading Engine Module
CRITICAL: Order management, execution, matching, and settlement
"""

# Event Bus
from quantum_trader.engine.event_bus import (
    EventBus,
    Event,
    EventType
)

# Execution Engine
from quantum_trader.engine.execution_engine import (
    ExecutionEngine,
    ExecutionOrder,
    ExecutionResult,
    ExecutionStatus,
    OrderSide,
    OrderType
)

# Matching Engine
from quantum_trader.engine.matching_engine import (
    MatchingEngine,
    OrderBook,
    OrderBookEntry,
    MatchResult
)

# Order Manager
from quantum_trader.engine.order_manager import (
    OrderManager,
    Order,
    OrderValidationResult,
    OrderStatus,
    TimeInForce
)

# Reconciliation
from quantum_trader.engine.reconciliation import (
    Reconciliation,
    Trade,
    Balance,
    Discrepancy,
    DiscrepancyType,
    ReconciliationStatus
)

# Settlement
from quantum_trader.engine.settlement import (
    Settlement,
    SettlementInstruction,
    SettlementResult,
    NettingResult,
    SettlementStatus,
    SettlementType
)

# State Machine
from quantum_trader.engine.state_machine import (
    StateMachine,
    StateTransition,
    State,
    TransitionReason
)

# Trading Engine
from quantum_trader.engine.trading_engine import (
    TradingEngine,
    TradingSignal,
    Position,
    SignalType
)

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "EventType",

    # Execution Engine
    "ExecutionEngine",
    "ExecutionOrder",
    "ExecutionResult",
    "ExecutionStatus",
    "OrderSide",
    "OrderType",

    # Matching Engine
    "MatchingEngine",
    "OrderBook",
    "OrderBookEntry",
    "MatchResult",

    # Order Manager
    "OrderManager",
    "Order",
    "OrderValidationResult",
    "OrderStatus",
    "TimeInForce",

    # Reconciliation
    "Reconciliation",
    "Trade",
    "Balance",
    "Discrepancy",
    "DiscrepancyType",
    "ReconciliationStatus",

    # Settlement
    "Settlement",
    "SettlementInstruction",
    "SettlementResult",
    "NettingResult",
    "SettlementStatus",
    "SettlementType",

    # State Machine
    "StateMachine",
    "StateTransition",
    "State",
    "TransitionReason",

    # Trading Engine
    "TradingEngine",
    "TradingSignal",
    "Position",
    "SignalType",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader Development Team"
