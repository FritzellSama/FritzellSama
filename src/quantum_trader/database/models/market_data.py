"""Market data models for time-series storage in TimescaleDB.

Defines models for ticks, OHLCV, order books, and other market data
with Decimal precision and UTC timestamps.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class MarketDataType(Enum):
    """Types of market data."""
    TICK = "TICK"
    OHLCV = "OHLCV"
    ORDER_BOOK = "ORDER_BOOK"
    TRADE = "TRADE"
    FUNDING_RATE = "FUNDING_RATE"
    LIQUIDATION = "LIQUIDATION"


class TimeFrame(Enum):
    """Standard timeframes for OHLCV data."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

    def to_seconds(self) -> int:
        """Convert timeframe to seconds.

        Returns:
            Timeframe duration in seconds
        """
        mapping = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800,
            '1M': 2592000  # Approximate
        }
        return mapping.get(self.value, 0)


@dataclass
class MarketTick:
    """Individual market tick (price update).

    Attributes:
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange: Exchange name
        bid: Best bid price
        ask: Best ask price
        last: Last trade price
        volume: 24h volume
        timestamp: UTC timestamp
        metadata: Additional tick data
        tick_id: Unique identifier (set by database)
    """
    symbol: str
    exchange: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tick_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate market tick after initialization."""
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            if self.timestamp.tzinfo is None:
                raise ValueError("timestamp must have UTC timezone")

        # Validate all prices are Decimal
        for field_name in ['bid', 'ask', 'last', 'volume']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise ValueError(f"{field_name} must be Decimal, got {type(value)}")

        # Validate price sanity
        if self.bid <= Decimal('0') or self.ask <= Decimal('0') or self.last <= Decimal('0'):
            raise ValueError("Prices must be positive")

        if self.bid > self.ask:
            raise ValueError(f"Bid ({self.bid}) cannot exceed ask ({self.ask})")

        if self.volume < Decimal('0'):
            raise ValueError("Volume cannot be negative")

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread.

        Returns:
            Spread as Decimal
        """
        return self.ask - self.bid

    @property
    def spread_percentage(self) -> Decimal:
        """Calculate spread as percentage of mid price.

        Returns:
            Spread percentage as Decimal
        """
        mid = (self.bid + self.ask) / Decimal('2')
        if mid == Decimal('0'):
            return Decimal('0')

        return (self.spread / mid * Decimal('100')).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price.

        Returns:
            Mid price as Decimal
        """
        return ((self.bid + self.ask) / Decimal('2')).quantize(
            Decimal('0.00000001'), rounding=ROUND_HALF_UP
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'tick_id': self.tick_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'bid': str(self.bid),
            'ask': str(self.ask),
            'last': str(self.last),
            'volume': str(self.volume),
            'timestamp': self.timestamp.isoformat(),
            'spread': str(self.spread),
            'spread_percentage': str(self.spread_percentage),
            'mid_price': str(self.mid_price),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketTick':
        """Create MarketTick from dictionary.

        Args:
            data: Dictionary with tick data

        Returns:
            MarketTick instance
        """
        return cls(
            symbol=data['symbol'],
            exchange=data['exchange'],
            bid=Decimal(data['bid']),
            ask=Decimal(data['ask']),
            last=Decimal(data['last']),
            volume=Decimal(data['volume']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {}),
            tick_id=data.get('tick_id')
        )


@dataclass
class OHLCV:
    """OHLCV candlestick data.

    Attributes:
        symbol: Trading pair
        exchange: Exchange name
        timeframe: Candle timeframe
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        timestamp: Candle start time (UTC)
        num_trades: Number of trades in candle
        metadata: Additional candle data
        ohlcv_id: Unique identifier (set by database)
    """
    symbol: str
    exchange: str
    timeframe: TimeFrame
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    timestamp: datetime
    num_trades: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ohlcv_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate OHLCV after initialization."""
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            if self.timestamp.tzinfo is None:
                raise ValueError("timestamp must have UTC timezone")

        # Validate all prices are Decimal
        for field_name in ['open', 'high', 'low', 'close', 'volume']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise ValueError(f"{field_name} must be Decimal, got {type(value)}")

        # Validate OHLC relationships
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")

        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= open and close")

        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= open and close")

        if self.volume < Decimal('0'):
            raise ValueError("Volume cannot be negative")

        if self.num_trades < 0:
            raise ValueError("num_trades cannot be negative")

    @property
    def price_change(self) -> Decimal:
        """Calculate price change (close - open).

        Returns:
            Price change as Decimal
        """
        return self.close - self.open

    @property
    def price_change_percentage(self) -> Decimal:
        """Calculate price change percentage.

        Returns:
            Price change percentage as Decimal
        """
        if self.open == Decimal('0'):
            return Decimal('0')

        return ((self.close - self.open) / self.open * Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )

    @property
    def range(self) -> Decimal:
        """Calculate candle range (high - low).

        Returns:
            Range as Decimal
        """
        return self.high - self.low

    @property
    def range_percentage(self) -> Decimal:
        """Calculate range as percentage of open.

        Returns:
            Range percentage as Decimal
        """
        if self.open == Decimal('0'):
            return Decimal('0')

        return (self.range / self.open * Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open).

        Returns:
            True if bullish, False otherwise
        """
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open).

        Returns:
            True if bearish, False otherwise
        """
        return self.close < self.open

    @property
    def typical_price(self) -> Decimal:
        """Calculate typical price (HLC/3).

        Returns:
            Typical price as Decimal
        """
        return ((self.high + self.low + self.close) / Decimal('3')).quantize(
            Decimal('0.00000001'), rounding=ROUND_HALF_UP
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'ohlcv_id': self.ohlcv_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe.value,
            'open': str(self.open),
            'high': str(self.high),
            'low': str(self.low),
            'close': str(self.close),
            'volume': str(self.volume),
            'timestamp': self.timestamp.isoformat(),
            'num_trades': self.num_trades,
            'price_change': str(self.price_change),
            'price_change_percentage': str(self.price_change_percentage),
            'range': str(self.range),
            'is_bullish': self.is_bullish,
            'typical_price': str(self.typical_price),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCV':
        """Create OHLCV from dictionary.

        Args:
            data: Dictionary with OHLCV data

        Returns:
            OHLCV instance
        """
        return cls(
            symbol=data['symbol'],
            exchange=data['exchange'],
            timeframe=TimeFrame(data['timeframe']),
            open=Decimal(data['open']),
            high=Decimal(data['high']),
            low=Decimal(data['low']),
            close=Decimal(data['close']),
            volume=Decimal(data['volume']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            num_trades=data.get('num_trades', 0),
            metadata=data.get('metadata', {}),
            ohlcv_id=data.get('ohlcv_id')
        )


@dataclass
class OrderBookSnapshot:
    """Order book snapshot at a point in time.

    Attributes:
        symbol: Trading pair
        exchange: Exchange name
        bids: List of (price, size) tuples for bids
        asks: List of (price, size) tuples for asks
        timestamp: UTC timestamp of snapshot
        sequence_number: Optional sequence number for ordering
        metadata: Additional snapshot data
        snapshot_id: Unique identifier (set by database)
    """
    symbol: str
    exchange: str
    bids: List[Tuple[Decimal, Decimal]]
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: datetime
    sequence_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshot_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate order book snapshot."""
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            if self.timestamp.tzinfo is None:
                raise ValueError("timestamp must have UTC timezone")

        if not self.bids:
            raise ValueError("bids cannot be empty")

        if not self.asks:
            raise ValueError("asks cannot be empty")

        # Validate bid/ask levels are Decimal tuples
        for bid in self.bids:
            if not isinstance(bid[0], Decimal) or not isinstance(bid[1], Decimal):
                raise ValueError("Bid levels must be (Decimal, Decimal) tuples")

        for ask in self.asks:
            if not isinstance(ask[0], Decimal) or not isinstance(ask[1], Decimal):
                raise ValueError("Ask levels must be (Decimal, Decimal) tuples")

    @property
    def best_bid(self) -> Tuple[Decimal, Decimal]:
        """Get best bid (highest price).

        Returns:
            (price, size) tuple
        """
        return max(self.bids, key=lambda x: x[0])

    @property
    def best_ask(self) -> Tuple[Decimal, Decimal]:
        """Get best ask (lowest price).

        Returns:
            (price, size) tuple
        """
        return min(self.asks, key=lambda x: x[0])

    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread.

        Returns:
            Spread as Decimal
        """
        return self.best_ask[0] - self.best_bid[0]

    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price.

        Returns:
            Mid price as Decimal
        """
        return ((self.best_bid[0] + self.best_ask[0]) / Decimal('2')).quantize(
            Decimal('0.00000001'), rounding=ROUND_HALF_UP
        )

    @property
    def total_bid_volume(self) -> Decimal:
        """Calculate total bid volume.

        Returns:
            Total bid volume as Decimal
        """
        return sum(size for _, size in self.bids)

    @property
    def total_ask_volume(self) -> Decimal:
        """Calculate total ask volume.

        Returns:
            Total ask volume as Decimal
        """
        return sum(size for _, size in self.asks)

    @property
    def imbalance_ratio(self) -> Decimal:
        """Calculate order book imbalance ratio.

        Returns:
            Imbalance ratio (bid_volume / (bid_volume + ask_volume))
        """
        total = self.total_bid_volume + self.total_ask_volume
        if total == Decimal('0'):
            return Decimal('0.5')

        return (self.total_bid_volume / total).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )

    def get_depth_at_price(self, price: Decimal, side: str = 'bid') -> Decimal:
        """Get cumulative volume up to a price level.

        Args:
            price: Price level
            side: 'bid' or 'ask'

        Returns:
            Cumulative volume as Decimal
        """
        if side == 'bid':
            levels = [(p, s) for p, s in self.bids if p >= price]
        else:
            levels = [(p, s) for p, s in self.asks if p <= price]

        return sum(size for _, size in levels)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'snapshot_id': self.snapshot_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'bids': [[str(p), str(s)] for p, s in self.bids],
            'asks': [[str(p), str(s)] for p, s in self.asks],
            'timestamp': self.timestamp.isoformat(),
            'sequence_number': self.sequence_number,
            'best_bid': [str(self.best_bid[0]), str(self.best_bid[1])],
            'best_ask': [str(self.best_ask[0]), str(self.best_ask[1])],
            'spread': str(self.spread),
            'mid_price': str(self.mid_price),
            'total_bid_volume': str(self.total_bid_volume),
            'total_ask_volume': str(self.total_ask_volume),
            'imbalance_ratio': str(self.imbalance_ratio),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBookSnapshot':
        """Create OrderBookSnapshot from dictionary.

        Args:
            data: Dictionary with snapshot data

        Returns:
            OrderBookSnapshot instance
        """
        return cls(
            symbol=data['symbol'],
            exchange=data['exchange'],
            bids=[(Decimal(p), Decimal(s)) for p, s in data['bids']],
            asks=[(Decimal(p), Decimal(s)) for p, s in data['asks']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            sequence_number=data.get('sequence_number'),
            metadata=data.get('metadata', {}),
            snapshot_id=data.get('snapshot_id')
        )


def aggregate_ticks_to_ohlcv(
    ticks: List[MarketTick],
    timeframe: TimeFrame,
    symbol: str,
    exchange: str
) -> List[OHLCV]:
    """Aggregate tick data into OHLCV candles.

    Args:
        ticks: List of market ticks
        timeframe: Target timeframe for candles
        symbol: Trading pair
        exchange: Exchange name

    Returns:
        List of OHLCV candles

    Raises:
        ValueError: If ticks empty or invalid
    """
    if not ticks:
        raise ValueError("Ticks list cannot be empty")

    # Sort ticks by timestamp
    sorted_ticks = sorted(ticks, key=lambda t: t.timestamp)

    # Group ticks into timeframe buckets
    interval_seconds = timeframe.to_seconds()
    candles: Dict[datetime, List[MarketTick]] = {}

    for tick in sorted_ticks:
        # Round timestamp down to timeframe boundary
        timestamp_unix = int(tick.timestamp.timestamp())
        bucket_unix = (timestamp_unix // interval_seconds) * interval_seconds
        bucket_time = datetime.fromtimestamp(bucket_unix, tz=timezone.utc)

        if bucket_time not in candles:
            candles[bucket_time] = []
        candles[bucket_time].append(tick)

    # Create OHLCV candles
    ohlcv_list = []

    for timestamp, bucket_ticks in sorted(candles.items()):
        prices = [t.last for t in bucket_ticks]
        volumes = [t.volume for t in bucket_ticks]

        ohlcv = OHLCV(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            timestamp=timestamp,
            num_trades=len(bucket_ticks)
        )
        ohlcv_list.append(ohlcv)

    logger.debug(
        "Ticks aggregated to OHLCV",
        symbol=symbol,
        ticks=len(ticks),
        candles=len(ohlcv_list),
        timeframe=timeframe.value
    )

    return ohlcv_list
