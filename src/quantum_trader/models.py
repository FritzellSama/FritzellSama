"""Core data models for Quantum Trader AI.

This module defines the core data structures used throughout the trading system,
including signals, orders, positions, and market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict


class SignalAction(Enum):
    """Trading signal action types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Signal:
    """Trading signal from strategy.

    Attributes:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        action: Signal action type (BUY, SELL, HOLD, CLOSE)
        strength: Signal strength from 0.0 to 1.0
        confidence: Confidence level from 0.0 to 1.0
        timestamp: UTC timestamp when signal was generated
        strategy: Name of strategy that generated the signal
        timeframe: Timeframe for the signal (e.g., '1m', '5m', '1h')
        indicators: Dictionary of technical indicators and their values
        metadata: Additional metadata about the signal
    """

    symbol: str
    action: SignalAction
    strength: Decimal
    confidence: Decimal
    timestamp: datetime
    strategy: str
    timeframe: str
    indicators: Dict[str, Decimal] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal data after initialization."""
        if not Decimal("0.0") <= self.strength <= Decimal("1.0"):
            raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {self.strength}")

        if not Decimal("0.0") <= self.confidence <= Decimal("1.0"):
            raise ValueError(f"Signal confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not isinstance(self.action, SignalAction):
            raise ValueError(f"Signal action must be SignalAction enum, got {type(self.action)}")


__all__ = [
    "Signal",
    "SignalAction",
]
