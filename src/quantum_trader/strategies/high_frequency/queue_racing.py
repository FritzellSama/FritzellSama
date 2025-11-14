"""Queue Racing High-Frequency Trading Strategy.

This strategy exploits order book dynamics by racing to be first in the queue
at key price levels, capturing spread and rebates through maker orders.

Performance Target: <5ms latency, 1000+ trades/day
Capital Allocation: Configurable via config
Risk: Ultra-low per-trade, high volume
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookLevel:
    """Order book price level snapshot."""
    price: Decimal
    quantity: Decimal
    order_count: int
    timestamp: datetime


@dataclass
class QueuePosition:
    """Our position in the order book queue."""
    symbol: str
    side: str  # 'BID' or 'ASK'
    price: Decimal
    our_quantity: Decimal
    ahead_quantity: Decimal  # Volume ahead of us in queue
    total_quantity: Decimal
    timestamp: datetime

    @property
    def queue_percentage(self) -> Decimal:
        """Calculate our position as percentage (0-100)."""
        if self.total_quantity == Decimal('0'):
            return Decimal('100')
        return (self.ahead_quantity / self.total_quantity * Decimal('100'))


class QueueRacingStrategy:
    """Queue racing high-frequency trading strategy.

    This strategy monitors order book dynamics and races to establish
    priority in the queue at optimal price levels to capture spread
    and exchange rebates.

    Key Features:
    - Sub-millisecond order placement
    - Real-time queue position tracking
    - Dynamic price level selection
    - Rebate optimization
    - Smart order cancellation

    Attributes:
        config: Strategy configuration from YAML
        risk_manager: Risk management instance
        active_orders: Currently active maker orders
        queue_positions: Tracked queue positions
        price_history: Recent price level history

    Example:
        >>> config = load_config('strategies.yaml')['high_frequency']['queue_racing']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = QueueRacingStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(market_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize queue racing strategy.

        Args:
            config: Strategy configuration dictionary
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Strategy state
        self.active_orders: Dict[str, Dict[str, Any]] = {}
        self.queue_positions: Dict[str, QueuePosition] = {}
        self.price_history: Dict[str, deque] = {}

        # Performance tracking
        self.fills_count: int = 0
        self.cancels_count: int = 0
        self.rebates_earned: Decimal = Decimal('0')

        # Configuration parameters
        self.max_queue_position: Decimal = Decimal(str(config.get('max_queue_position_pct', 20)))
        self.min_spread_bps: Decimal = Decimal(str(config.get('min_spread_bps', 2)))
        self.quote_ttl_ms: int = config.get('quote_ttl_ms', 500)
        self.max_levels: int = config.get('max_price_levels', 3)
        self.min_level_size: Decimal = Decimal(str(config.get('min_level_size', 10000)))
        self.rebate_threshold: Decimal = Decimal(str(config.get('rebate_threshold_bps', 0.5)))

        logger.info(
            "queue_racing_strategy_initialized",
            max_queue_pos=float(self.max_queue_position),
            min_spread_bps=float(self.min_spread_bps),
            quote_ttl_ms=self.quote_ttl_ms
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If required config parameters missing or invalid
        """
        required_fields = [
            'max_queue_position_pct',
            'min_spread_bps',
            'quote_ttl_ms',
            'max_price_levels'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['max_queue_position_pct'] <= 0 or self.config['max_queue_position_pct'] > 100:
            raise ValueError("max_queue_position_pct must be between 0 and 100")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals from order book data.

        Args:
            market_data: Polars DataFrame with columns:
                - symbol: str
                - bid_price_{1-10}: Decimal
                - bid_size_{1-10}: Decimal
                - ask_price_{1-10}: Decimal
                - ask_size_{1-10}: Decimal
                - timestamp: datetime

        Returns:
            List of signal dictionaries with:
                - symbol: str
                - action: 'BUY' or 'SELL'
                - price: Decimal
                - quantity: Decimal
                - strategy: 'queue_racing'
                - confidence: Decimal (0-1)
                - metadata: Dict

        Raises:
            ValueError: If market_data invalid or empty
        """
        if market_data.is_empty():
            logger.warning("empty_market_data_received")
            return []

        try:
            signals = []

            # Process each symbol
            for row in market_data.iter_rows(named=True):
                symbol = row['symbol']

                # Analyze order book levels
                bid_levels = self._extract_bid_levels(row)
                ask_levels = self._extract_ask_levels(row)

                if not bid_levels or not ask_levels:
                    continue

                # Calculate current spread
                best_bid = bid_levels[0].price
                best_ask = ask_levels[0].price
                spread_bps = self._calculate_spread_bps(best_bid, best_ask)

                # Check if spread meets minimum threshold
                if spread_bps < self.min_spread_bps:
                    logger.debug(
                        "spread_too_tight",
                        symbol=symbol,
                        spread_bps=float(spread_bps),
                        min_required=float(self.min_spread_bps)
                    )
                    continue

                # Find optimal levels to join
                bid_signal = self._evaluate_bid_level(symbol, bid_levels, spread_bps)
                if bid_signal:
                    signals.append(bid_signal)

                ask_signal = self._evaluate_ask_level(symbol, ask_levels, spread_bps)
                if ask_signal:
                    signals.append(ask_signal)

            logger.info(
                "signals_generated",
                signal_count=len(signals),
                symbols_analyzed=market_data.height
            )

            return signals

        except Exception as e:
            logger.error(
                "signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _extract_bid_levels(self, row: Dict[str, Any]) -> List[OrderBookLevel]:
        """Extract bid levels from market data row.

        Args:
            row: Market data row dictionary

        Returns:
            List of OrderBookLevel objects for bids
        """
        levels = []

        for i in range(1, self.max_levels + 1):
            price_key = f'bid_price_{i}'
            size_key = f'bid_size_{i}'

            if price_key not in row or size_key not in row:
                break

            price = Decimal(str(row[price_key]))
            size = Decimal(str(row[size_key]))

            if price > 0 and size > 0:
                levels.append(OrderBookLevel(
                    price=price,
                    quantity=size,
                    order_count=int(row.get(f'bid_orders_{i}', 1)),
                    timestamp=row.get('timestamp', datetime.now(timezone.utc))
                ))

        return levels

    def _extract_ask_levels(self, row: Dict[str, Any]) -> List[OrderBookLevel]:
        """Extract ask levels from market data row.

        Args:
            row: Market data row dictionary

        Returns:
            List of OrderBookLevel objects for asks
        """
        levels = []

        for i in range(1, self.max_levels + 1):
            price_key = f'ask_price_{i}'
            size_key = f'ask_size_{i}'

            if price_key not in row or size_key not in row:
                break

            price = Decimal(str(row[price_key]))
            size = Decimal(str(row[size_key]))

            if price > 0 and size > 0:
                levels.append(OrderBookLevel(
                    price=price,
                    quantity=size,
                    order_count=int(row.get(f'ask_orders_{i}', 1)),
                    timestamp=row.get('timestamp', datetime.now(timezone.utc))
                ))

        return levels

    def _calculate_spread_bps(self, bid: Decimal, ask: Decimal) -> Decimal:
        """Calculate spread in basis points.

        Args:
            bid: Best bid price
            ask: Best ask price

        Returns:
            Spread in basis points
        """
        if bid == Decimal('0'):
            return Decimal('0')

        mid = (bid + ask) / Decimal('2')
        spread = ask - bid
        return (spread / mid) * Decimal('10000')

    def _evaluate_bid_level(
        self,
        symbol: str,
        levels: List[OrderBookLevel],
        spread_bps: Decimal
    ) -> Optional[Dict[str, Any]]:
        """Evaluate bid levels for queue racing opportunity.

        Args:
            symbol: Trading symbol
            levels: Bid order book levels
            spread_bps: Current spread in basis points

        Returns:
            Signal dictionary if opportunity found, None otherwise
        """
        # Analyze best bid level
        best_bid = levels[0]

        # Check if level size is adequate
        if best_bid.quantity < self.min_level_size:
            return None

        # Estimate our queue position (pessimistic: assume we're at back)
        estimated_queue_pct = Decimal('100')

        # Only join if queue position would be acceptable
        if estimated_queue_pct > self.max_queue_position:
            # Try joining one tick better
            if len(levels) > 1:
                tick_size = self._calculate_tick_size(best_bid.price)
                improved_price = best_bid.price + tick_size

                # Verify improved price doesn't cross spread
                if spread_bps > self.min_spread_bps * Decimal('2'):
                    return self._create_signal(
                        symbol=symbol,
                        action='BUY',
                        price=improved_price,
                        reference_size=best_bid.quantity,
                        spread_bps=spread_bps,
                        level_index=0,
                        is_improved=True
                    )
            return None

        return self._create_signal(
            symbol=symbol,
            action='BUY',
            price=best_bid.price,
            reference_size=best_bid.quantity,
            spread_bps=spread_bps,
            level_index=0,
            is_improved=False
        )

    def _evaluate_ask_level(
        self,
        symbol: str,
        levels: List[OrderBookLevel],
        spread_bps: Decimal
    ) -> Optional[Dict[str, Any]]:
        """Evaluate ask levels for queue racing opportunity.

        Args:
            symbol: Trading symbol
            levels: Ask order book levels
            spread_bps: Current spread in basis points

        Returns:
            Signal dictionary if opportunity found, None otherwise
        """
        # Analyze best ask level
        best_ask = levels[0]

        # Check if level size is adequate
        if best_ask.quantity < self.min_level_size:
            return None

        # Estimate our queue position
        estimated_queue_pct = Decimal('100')

        # Only join if queue position would be acceptable
        if estimated_queue_pct > self.max_queue_position:
            # Try joining one tick better
            if len(levels) > 1:
                tick_size = self._calculate_tick_size(best_ask.price)
                improved_price = best_ask.price - tick_size

                # Verify improved price doesn't cross spread
                if spread_bps > self.min_spread_bps * Decimal('2'):
                    return self._create_signal(
                        symbol=symbol,
                        action='SELL',
                        price=improved_price,
                        reference_size=best_ask.quantity,
                        spread_bps=spread_bps,
                        level_index=0,
                        is_improved=True
                    )
            return None

        return self._create_signal(
            symbol=symbol,
            action='SELL',
            price=best_ask.price,
            reference_size=best_ask.quantity,
            spread_bps=spread_bps,
            level_index=0,
            is_improved=False
        )

    def _calculate_tick_size(self, price: Decimal) -> Decimal:
        """Calculate appropriate tick size for price level.

        Args:
            price: Price level

        Returns:
            Tick size in quote currency
        """
        # Standard crypto tick sizes
        if price >= Decimal('1000'):
            return Decimal('1.0')
        elif price >= Decimal('100'):
            return Decimal('0.1')
        elif price >= Decimal('10'):
            return Decimal('0.01')
        else:
            return Decimal('0.001')

    def _create_signal(
        self,
        symbol: str,
        action: str,
        price: Decimal,
        reference_size: Decimal,
        spread_bps: Decimal,
        level_index: int,
        is_improved: bool
    ) -> Dict[str, Any]:
        """Create trading signal dictionary.

        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            price: Order price
            reference_size: Reference size from order book
            spread_bps: Current spread in basis points
            level_index: Order book level index
            is_improved: Whether price is improved from best level

        Returns:
            Signal dictionary
        """
        # Calculate position size (small size for HFT)
        base_size = self.config.get('base_order_size', Decimal('100'))
        quantity = Decimal(str(base_size))

        # Calculate confidence based on spread and queue position
        confidence = min(
            spread_bps / Decimal('10'),  # Higher confidence for wider spreads
            Decimal('0.95')
        )

        # Estimate rebate capture
        rebate_bps = self.config.get('maker_rebate_bps', Decimal('0.2'))
        expected_rebate = quantity * price * rebate_bps / Decimal('10000')

        signal = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'strategy': 'queue_racing',
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'spread_bps': spread_bps,
                'level_index': level_index,
                'is_improved_price': is_improved,
                'reference_size': reference_size,
                'expected_rebate': expected_rebate,
                'quote_ttl_ms': self.quote_ttl_ms
            }
        }

        return signal

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate queue racing specific indicators.

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary of calculated indicators
        """
        if data.is_empty():
            return {}

        try:
            # Calculate average spread
            bid_prices = data.select(pl.col('bid_price_1'))
            ask_prices = data.select(pl.col('ask_price_1'))

            avg_spread_bps = Decimal('0')
            if not bid_prices.is_empty() and not ask_prices.is_empty():
                spreads = []
                for i in range(min(len(bid_prices), len(ask_prices))):
                    bid = Decimal(str(bid_prices[i, 0]))
                    ask = Decimal(str(ask_prices[i, 0]))
                    if bid > 0 and ask > 0:
                        spreads.append(self._calculate_spread_bps(bid, ask))

                if spreads:
                    avg_spread_bps = sum(spreads) / Decimal(str(len(spreads)))

            # Calculate average queue depth
            avg_bid_depth = Decimal('0')
            avg_ask_depth = Decimal('0')

            if 'bid_size_1' in data.columns:
                bid_sizes = data.select(pl.col('bid_size_1')).to_numpy()
                avg_bid_depth = Decimal(str(np.mean(bid_sizes)))

            if 'ask_size_1' in data.columns:
                ask_sizes = data.select(pl.col('ask_size_1')).to_numpy()
                avg_ask_depth = Decimal(str(np.mean(ask_sizes)))

            indicators = {
                'avg_spread_bps': avg_spread_bps,
                'avg_bid_depth': avg_bid_depth,
                'avg_ask_depth': avg_ask_depth,
                'total_fills': Decimal(str(self.fills_count)),
                'total_cancels': Decimal(str(self.cancels_count)),
                'rebates_earned': self.rebates_earned
            }

            return indicators

        except Exception as e:
            logger.error(
                "indicator_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return {}

    def update_queue_position(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        our_quantity: Decimal,
        ahead_quantity: Decimal,
        total_quantity: Decimal
    ) -> None:
        """Update tracked queue position for an order.

        Args:
            symbol: Trading symbol
            side: 'BID' or 'ASK'
            price: Price level
            our_quantity: Our order quantity
            ahead_quantity: Quantity ahead of us
            total_quantity: Total quantity at level
        """
        key = f"{symbol}_{side}_{price}"

        self.queue_positions[key] = QueuePosition(
            symbol=symbol,
            side=side,
            price=price,
            our_quantity=our_quantity,
            ahead_quantity=ahead_quantity,
            total_quantity=total_quantity,
            timestamp=datetime.now(timezone.utc)
        )

        # Log significant queue position changes
        if ahead_quantity / total_quantity < Decimal('0.1'):
            logger.info(
                "favorable_queue_position",
                symbol=symbol,
                side=side,
                price=float(price),
                queue_pct=float(ahead_quantity / total_quantity * 100)
            )

    async def on_fill(self, order_id: str, filled_quantity: Decimal, fill_price: Decimal) -> None:
        """Handle order fill event.

        Args:
            order_id: Filled order ID
            filled_quantity: Quantity filled
            fill_price: Fill price
        """
        self.fills_count += 1

        # Calculate rebate earned
        rebate_bps = self.config.get('maker_rebate_bps', Decimal('0.2'))
        rebate = filled_quantity * fill_price * rebate_bps / Decimal('10000')
        self.rebates_earned += rebate

        logger.info(
            "order_filled",
            order_id=order_id,
            quantity=float(filled_quantity),
            price=float(fill_price),
            rebate=float(rebate),
            total_rebates=float(self.rebates_earned)
        )

    async def on_cancel(self, order_id: str, reason: str) -> None:
        """Handle order cancellation event.

        Args:
            order_id: Cancelled order ID
            reason: Cancellation reason
        """
        self.cancels_count += 1

        logger.debug(
            "order_cancelled",
            order_id=order_id,
            reason=reason,
            total_cancels=self.cancels_count
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        total_orders = self.fills_count + self.cancels_count
        fill_rate = Decimal('0')
        if total_orders > 0:
            fill_rate = Decimal(str(self.fills_count)) / Decimal(str(total_orders))

        return {
            'total_fills': self.fills_count,
            'total_cancels': self.cancels_count,
            'fill_rate': float(fill_rate),
            'rebates_earned': float(self.rebates_earned),
            'active_positions': len(self.queue_positions)
        }
