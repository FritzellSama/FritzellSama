"""Tick Trading High-Frequency Strategy.

Ultra-high-frequency trading on individual tick data, capturing micro-price
movements with sub-millisecond execution targeting.

Performance Target: <1ms latency, 5000+ trades/day
Capital Allocation: Configurable via config
Risk: Minimal per-trade, extreme execution sensitivity
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class TickDirection(Enum):
    """Tick direction classification."""
    UPTICK = "uptick"
    DOWNTICK = "downtick"
    ZERO_UPTICK = "zero_uptick"
    ZERO_DOWNTICK = "zero_downtick"


@dataclass
class Tick:
    """Individual market tick."""
    symbol: str
    price: Decimal
    size: Decimal
    timestamp: datetime
    direction: TickDirection
    is_bid: bool  # True if at bid, False if at ask
    bid_price: Decimal
    ask_price: Decimal
    sequence: int


@dataclass
class TickPattern:
    """Detected tick pattern."""
    pattern_type: str
    symbol: str
    ticks: List[Tick]
    expected_direction: str  # 'up' or 'down'
    confidence: Decimal
    timestamp: datetime


class TickTradingStrategy:
    """Tick-level high-frequency trading strategy.

    Analyzes individual ticks to identify micro-patterns and execute
    ultra-fast trades capturing tiny price movements.

    Key Features:
    - Tick-by-tick analysis
    - Micro-pattern detection
    - Sub-millisecond targeting
    - Order flow toxicity detection
    - Aggressive latency optimization

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        tick_buffer: Recent ticks buffer
        active_positions: Current positions
        pattern_detectors: Pattern detection functions

    Example:
        >>> config = load_config('strategies.yaml')['high_frequency']['tick_trading']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = TickTradingStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(tick_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize tick trading strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Strategy parameters
        self.tick_buffer_size = config.get('tick_buffer_size', 100)
        self.min_tick_pattern_length = config.get('min_pattern_length', 3)
        self.max_holding_ticks = config.get('max_holding_ticks', 10)
        self.profit_target_ticks = Decimal(str(config.get('profit_target_ticks', 2)))
        self.stop_loss_ticks = Decimal(str(config.get('stop_loss_ticks', 3)))

        # Pattern detection parameters
        self.imbalance_threshold = Decimal(str(config.get('order_flow_imbalance_threshold', 0.6)))
        self.aggressive_threshold = Decimal(str(config.get('aggressive_trade_threshold', 0.7)))

        # Position sizing
        self.base_position_size = Decimal(str(config.get('base_position_size', 100)))
        self.max_position_size = Decimal(str(config.get('max_position_size', 500)))

        # State tracking
        self.tick_buffers: Dict[str, deque] = {}
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.tick_patterns: Dict[str, List[TickPattern]] = {}

        # Performance tracking
        self.total_ticks_processed: int = 0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.ticks_per_trade: deque = deque(maxlen=1000)

        logger.info(
            "tick_trading_strategy_initialized",
            tick_buffer_size=self.tick_buffer_size,
            profit_target_ticks=float(self.profit_target_ticks),
            stop_loss_ticks=float(self.stop_loss_ticks)
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = [
            'tick_buffer_size',
            'profit_target_ticks',
            'stop_loss_ticks'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['profit_target_ticks'] <= 0:
            raise ValueError("profit_target_ticks must be positive")

    async def generate_signals(self, tick_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate tick trading signals.

        Args:
            tick_data: Polars DataFrame with tick data:
                - symbol: str
                - timestamp: datetime
                - price: Decimal
                - size: Decimal
                - bid_price: Decimal
                - ask_price: Decimal
                - is_bid: bool (trade at bid=True, at ask=False)
                - sequence: int

        Returns:
            List of signal dictionaries

        Raises:
            ValueError: If tick_data invalid
        """
        if tick_data.is_empty():
            logger.warning("empty_tick_data_received")
            return []

        try:
            signals = []

            # Process each symbol
            for symbol in tick_data['symbol'].unique():
                symbol_ticks = tick_data.filter(pl.col('symbol') == symbol).sort('sequence')

                # Update tick buffer
                self._update_tick_buffer(symbol, symbol_ticks)

                # Check existing positions first
                if symbol in self.active_positions:
                    exit_signal = self._check_position_exit(symbol, symbol_ticks)
                    if exit_signal:
                        signals.append(exit_signal)
                        continue  # Don't enter new position if exiting

                # Detect patterns
                patterns = self._detect_tick_patterns(symbol)

                # Generate entry signals
                for pattern in patterns:
                    entry_signal = self._create_tick_signal(pattern, symbol_ticks)
                    if entry_signal:
                        signals.append(entry_signal)
                        break  # One signal per symbol

            self.total_ticks_processed += tick_data.height

            logger.debug(
                "tick_signals_generated",
                count=len(signals),
                ticks_processed=tick_data.height,
                total_ticks=self.total_ticks_processed
            )

            return signals

        except Exception as e:
            logger.error(
                "tick_signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _update_tick_buffer(self, symbol: str, tick_data: pl.DataFrame) -> None:
        """Update tick buffer for symbol.

        Args:
            symbol: Trading symbol
            tick_data: New tick data
        """
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = deque(maxlen=self.tick_buffer_size)

        # Convert rows to Tick objects
        for row in tick_data.iter_rows(named=True):
            price = Decimal(str(row['price']))
            prev_price = self.tick_buffers[symbol][-1].price if self.tick_buffers[symbol] else price

            # Classify tick direction
            if price > prev_price:
                direction = TickDirection.UPTICK
            elif price < prev_price:
                direction = TickDirection.DOWNTICK
            elif row.get('is_bid', True):
                direction = TickDirection.ZERO_DOWNTICK
            else:
                direction = TickDirection.ZERO_UPTICK

            tick = Tick(
                symbol=symbol,
                price=price,
                size=Decimal(str(row['size'])),
                timestamp=row['timestamp'],
                direction=direction,
                is_bid=row.get('is_bid', True),
                bid_price=Decimal(str(row.get('bid_price', price))),
                ask_price=Decimal(str(row.get('ask_price', price))),
                sequence=row.get('sequence', 0)
            )

            self.tick_buffers[symbol].append(tick)

    def _detect_tick_patterns(self, symbol: str) -> List[TickPattern]:
        """Detect tradeable tick patterns.

        Args:
            symbol: Trading symbol

        Returns:
            List of detected patterns
        """
        if symbol not in self.tick_buffers:
            return []

        buffer = list(self.tick_buffers[symbol])
        if len(buffer) < self.min_tick_pattern_length:
            return []

        patterns = []

        # Pattern 1: Consecutive upticks (bullish)
        consecutive_up = self._detect_consecutive_direction(buffer, TickDirection.UPTICK)
        if consecutive_up:
            patterns.append(consecutive_up)

        # Pattern 2: Consecutive downticks (bearish)
        consecutive_down = self._detect_consecutive_direction(buffer, TickDirection.DOWNTICK)
        if consecutive_down:
            patterns.append(consecutive_down)

        # Pattern 3: Order flow imbalance
        imbalance_pattern = self._detect_order_flow_imbalance(buffer)
        if imbalance_pattern:
            patterns.append(imbalance_pattern)

        # Pattern 4: Aggressive buying/selling
        aggressive_pattern = self._detect_aggressive_trading(buffer)
        if aggressive_pattern:
            patterns.append(aggressive_pattern)

        return patterns

    def _detect_consecutive_direction(
        self,
        buffer: List[Tick],
        direction: TickDirection
    ) -> Optional[TickPattern]:
        """Detect consecutive ticks in same direction.

        Args:
            buffer: Tick buffer
            direction: Direction to look for

        Returns:
            TickPattern if detected, None otherwise
        """
        # Look at recent ticks
        recent_ticks = buffer[-self.min_tick_pattern_length:]

        consecutive_count = sum(1 for t in recent_ticks if t.direction == direction)

        if consecutive_count >= self.min_tick_pattern_length:
            confidence = Decimal(str(consecutive_count)) / Decimal(str(len(recent_ticks)))

            expected_direction = 'up' if direction in [TickDirection.UPTICK, TickDirection.ZERO_UPTICK] else 'down'

            return TickPattern(
                pattern_type=f'consecutive_{direction.value}',
                symbol=buffer[0].symbol,
                ticks=recent_ticks,
                expected_direction=expected_direction,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

        return None

    def _detect_order_flow_imbalance(self, buffer: List[Tick]) -> Optional[TickPattern]:
        """Detect order flow imbalance.

        Args:
            buffer: Tick buffer

        Returns:
            TickPattern if imbalance detected
        """
        recent_ticks = buffer[-10:]  # Last 10 ticks

        buy_volume = sum(t.size for t in recent_ticks if not t.is_bid)  # Trades at ask = buys
        sell_volume = sum(t.size for t in recent_ticks if t.is_bid)  # Trades at bid = sells

        total_volume = buy_volume + sell_volume

        if total_volume == Decimal('0'):
            return None

        imbalance = (buy_volume - sell_volume) / total_volume

        if abs(imbalance) > self.imbalance_threshold:
            confidence = min(abs(imbalance), Decimal('1'))

            return TickPattern(
                pattern_type='order_flow_imbalance',
                symbol=buffer[0].symbol,
                ticks=recent_ticks,
                expected_direction='up' if imbalance > 0 else 'down',
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

        return None

    def _detect_aggressive_trading(self, buffer: List[Tick]) -> Optional[TickPattern]:
        """Detect aggressive buying or selling.

        Args:
            buffer: Tick buffer

        Returns:
            TickPattern if detected
        """
        recent_ticks = buffer[-5:]

        # Count aggressive trades (large size, crossing spread)
        aggressive_buys = sum(
            1 for t in recent_ticks
            if not t.is_bid and t.size > Decimal('100')
        )

        aggressive_sells = sum(
            1 for t in recent_ticks
            if t.is_bid and t.size > Decimal('100')
        )

        total_trades = len(recent_ticks)

        if aggressive_buys > total_trades * float(self.aggressive_threshold):
            return TickPattern(
                pattern_type='aggressive_buying',
                symbol=buffer[0].symbol,
                ticks=recent_ticks,
                expected_direction='up',
                confidence=Decimal(str(aggressive_buys / total_trades)),
                timestamp=datetime.now(timezone.utc)
            )

        if aggressive_sells > total_trades * float(self.aggressive_threshold):
            return TickPattern(
                pattern_type='aggressive_selling',
                symbol=buffer[0].symbol,
                ticks=recent_ticks,
                expected_direction='down',
                confidence=Decimal(str(aggressive_sells / total_trades)),
                timestamp=datetime.now(timezone.utc)
            )

        return None

    def _create_tick_signal(
        self,
        pattern: TickPattern,
        tick_data: pl.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Create trading signal from tick pattern.

        Args:
            pattern: Detected tick pattern
            tick_data: Recent tick data

        Returns:
            Signal dictionary or None
        """
        latest_tick = pattern.ticks[-1]

        # Determine entry price and side
        if pattern.expected_direction == 'up':
            action = 'BUY'
            entry_price = latest_tick.ask_price
            tick_size = self._get_tick_size(entry_price)
            take_profit = entry_price + (self.profit_target_ticks * tick_size)
            stop_loss = entry_price - (self.stop_loss_ticks * tick_size)
        else:
            action = 'SELL'
            entry_price = latest_tick.bid_price
            tick_size = self._get_tick_size(entry_price)
            take_profit = entry_price - (self.profit_target_ticks * tick_size)
            stop_loss = entry_price + (self.stop_loss_ticks * tick_size)

        # Calculate position size
        quantity = self._calculate_position_size(pattern.confidence)

        signal = {
            'symbol': pattern.symbol,
            'action': action,
            'price': entry_price,
            'quantity': quantity,
            'strategy': 'tick_trading',
            'confidence': pattern.confidence,
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'pattern_type': pattern.pattern_type,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'tick_size': tick_size,
                'expected_ticks': self.profit_target_ticks,
                'pattern_confidence': pattern.confidence
            }
        }

        # Track position
        self.active_positions[pattern.symbol] = {
            'entry_price': entry_price,
            'entry_tick_sequence': latest_tick.sequence,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'action': action,
            'quantity': quantity,
            'entry_time': datetime.now(timezone.utc)
        }

        self.total_trades += 1

        return signal

    def _check_position_exit(
        self,
        symbol: str,
        tick_data: pl.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check if position should be exited.

        Args:
            symbol: Trading symbol
            tick_data: Recent tick data

        Returns:
            Exit signal or None
        """
        position = self.active_positions.get(symbol)
        if not position:
            return None

        latest_tick = list(self.tick_buffers[symbol])[-1]
        current_price = latest_tick.price

        # Check take profit
        if position['action'] == 'BUY' and current_price >= position['take_profit']:
            return self._create_exit_signal(symbol, 'SELL', current_price, 'take_profit')
        elif position['action'] == 'SELL' and current_price <= position['take_profit']:
            return self._create_exit_signal(symbol, 'BUY', current_price, 'take_profit')

        # Check stop loss
        if position['action'] == 'BUY' and current_price <= position['stop_loss']:
            return self._create_exit_signal(symbol, 'SELL', current_price, 'stop_loss')
        elif position['action'] == 'SELL' and current_price >= position['stop_loss']:
            return self._create_exit_signal(symbol, 'BUY', current_price, 'stop_loss')

        # Check max holding ticks
        ticks_held = latest_tick.sequence - position['entry_tick_sequence']
        if ticks_held >= self.max_holding_ticks:
            exit_action = 'SELL' if position['action'] == 'BUY' else 'BUY'
            return self._create_exit_signal(symbol, exit_action, current_price, 'max_ticks')

        return None

    def _create_exit_signal(
        self,
        symbol: str,
        action: str,
        price: Decimal,
        reason: str
    ) -> Dict[str, Any]:
        """Create exit signal.

        Args:
            symbol: Trading symbol
            action: Exit action
            price: Exit price
            reason: Exit reason

        Returns:
            Exit signal dictionary
        """
        position = self.active_positions[symbol]

        # Calculate P&L
        if position['action'] == 'BUY':
            pnl = (price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - price) * position['quantity']

        # Track win/loss
        if pnl > 0:
            self.winning_trades += 1

        # Calculate ticks held
        latest_tick = list(self.tick_buffers[symbol])[-1]
        ticks_held = latest_tick.sequence - position['entry_tick_sequence']
        self.ticks_per_trade.append(ticks_held)

        signal = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': position['quantity'],
            'strategy': 'tick_trading',
            'confidence': Decimal('0.9'),
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'exit_reason': reason,
                'pnl': pnl,
                'ticks_held': ticks_held,
                'entry_price': position['entry_price'],
                'exit_price': price
            }
        }

        # Remove position
        del self.active_positions[symbol]

        return signal

    def _get_tick_size(self, price: Decimal) -> Decimal:
        """Get tick size for price level.

        Args:
            price: Price level

        Returns:
            Tick size
        """
        if price >= Decimal('1000'):
            return Decimal('1.0')
        elif price >= Decimal('100'):
            return Decimal('0.1')
        elif price >= Decimal('10'):
            return Decimal('0.01')
        else:
            return Decimal('0.001')

    def _calculate_position_size(self, confidence: Decimal) -> Decimal:
        """Calculate position size based on confidence.

        Args:
            confidence: Signal confidence

        Returns:
            Position quantity
        """
        # Scale base size by confidence
        size = self.base_position_size * confidence

        # Cap at max
        size = min(size, self.max_position_size)

        return size

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate performance indicators.

        Args:
            data: Historical data

        Returns:
            Dictionary of indicators
        """
        win_rate = (
            Decimal(str(self.winning_trades)) / Decimal(str(self.total_trades))
            if self.total_trades > 0 else Decimal('0')
        )

        avg_ticks_per_trade = (
            Decimal(str(sum(self.ticks_per_trade))) / Decimal(str(len(self.ticks_per_trade)))
            if self.ticks_per_trade else Decimal('0')
        )

        return {
            'total_ticks_processed': Decimal(str(self.total_ticks_processed)),
            'total_trades': Decimal(str(self.total_trades)),
            'winning_trades': Decimal(str(self.winning_trades)),
            'win_rate': win_rate,
            'avg_ticks_per_trade': avg_ticks_per_trade,
            'active_positions': Decimal(str(len(self.active_positions)))
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'total_ticks_processed': self.total_ticks_processed,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'active_positions': len(self.active_positions),
            'avg_ticks_per_trade': sum(self.ticks_per_trade) / len(self.ticks_per_trade) if self.ticks_per_trade else 0
        }
