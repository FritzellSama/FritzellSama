"""Core data models for Quantum Trader AI.

This module contains the fundamental data structures used throughout the trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Signal:
    """Trading signal from strategy.

    Attributes:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        action: Signal action (BUY, SELL, HOLD, CLOSE)
        strength: Signal strength from 0.0 to 1.0
        confidence: Signal confidence from 0.0 to 1.0
        timestamp: UTC timestamp when signal was generated
        strategy: Name of strategy that generated the signal
        timeframe: Timeframe for the signal (e.g., '1m', '5m', '1h')
        indicators: Dictionary of indicator values
        metadata: Additional metadata

    Example:
        >>> from decimal import Decimal
        >>> from datetime import datetime, timezone
        >>> signal = Signal(
        ...     symbol="BTCUSDT",
        ...     action=SignalAction.BUY,
        ...     strength=Decimal("0.85"),
        ...     confidence=Decimal("0.92"),
        ...     timestamp=datetime.now(timezone.utc),
        ...     strategy="momentum_strategy",
        ...     timeframe="5m",
        ...     indicators={"rsi": Decimal("35.5"), "macd": Decimal("12.3")},
        ...     metadata={"market_regime": "trending"}
        ... )
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
        if not Decimal("0") <= self.strength <= Decimal("1"):
            raise ValueError(f"Strength must be between 0 and 1, got {self.strength}")
        if not Decimal("0") <= self.confidence <= Decimal("1"):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
