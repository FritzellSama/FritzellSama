"""Core data models for Quantum Trader AI.

This module defines the fundamental data structures used throughout the trading system.
All models use Decimal for monetary values and UTC timestamps for time-based data.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Tuple


@dataclass
class OrderBook:
    """Order book snapshot.

    Represents a point-in-time view of the order book for a trading symbol,
    containing bid and ask price levels with their sizes.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        bids: List of (price, size) tuples sorted by price descending
        asks: List of (price, size) tuples sorted by price ascending
        timestamp: UTC timestamp when the snapshot was captured

    Example:
        >>> from decimal import Decimal
        >>> from datetime import datetime, timezone
        >>> orderbook = OrderBook(
        ...     symbol="BTC/USDT",
        ...     bids=[(Decimal("50000.00"), Decimal("1.5"))],
        ...     asks=[(Decimal("50001.00"), Decimal("2.0"))],
        ...     timestamp=datetime.now(timezone.utc)
        ... )
    """
    symbol: str
    bids: List[Tuple[Decimal, Decimal]]
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: datetime


@dataclass
class Ticker:
    """Real-time ticker data.

    Represents current market data for a trading symbol including best bid/ask,
    last traded price, and 24-hour volume.

    Attributes:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        bid: Best bid price
        ask: Best ask price
        last: Last traded price
        volume: 24-hour trading volume in base currency
        timestamp: UTC timestamp when the ticker data was captured

    Example:
        >>> from decimal import Decimal
        >>> from datetime import datetime, timezone
        >>> ticker = Ticker(
        ...     symbol="BTC/USDT",
        ...     bid=Decimal("50000.00"),
        ...     ask=Decimal("50001.00"),
        ...     last=Decimal("50000.50"),
        ...     volume=Decimal("1234.56"),
        ...     timestamp=datetime.now(timezone.utc)
        ... )
    """
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime
