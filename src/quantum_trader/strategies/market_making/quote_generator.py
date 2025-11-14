"""Market Making Quote Generator.

Generates two-sided quotes (bid/ask) for market making strategies,
optimizing for spread capture while managing inventory risk.

Performance Target: <10ms quote generation, continuous two-sided markets
Capital Allocation: Configurable via config
Risk: Inventory risk, adverse selection
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class InventorySkewMethod(Enum):
    """Methods for inventory skew calculation."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"


@dataclass
class Quote:
    """Two-sided market quote."""
    symbol: str
    bid_price: Decimal
    bid_quantity: Decimal
    ask_price: Decimal
    ask_quantity: Decimal
    mid_price: Decimal
    spread_bps: Decimal
    timestamp: datetime
    inventory_skew: Decimal
    confidence: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InventoryState:
    """Current inventory state for a symbol."""
    symbol: str
    position: Decimal  # Positive = long, negative = short
    target_position: Decimal
    max_position: Decimal
    avg_entry_price: Decimal
    unrealized_pnl: Decimal
    last_update: datetime

    @property
    def inventory_ratio(self) -> Decimal:
        """Calculate inventory as ratio of max (-1 to 1)."""
        if self.max_position == Decimal('0'):
            return Decimal('0')
        return self.position / self.max_position

    @property
    def is_long_biased(self) -> bool:
        """Check if inventory is long biased."""
        return self.position > self.target_position

    @property
    def is_short_biased(self) -> bool:
        """Check if inventory is short biased."""
        return self.position < self.target_position


class QuoteGenerator:
    """Market making quote generator.

    Generates optimal two-sided quotes based on:
    - Current market conditions
    - Inventory position
    - Volatility
    - Order flow toxicity
    - Competition in order book

    Key Features:
    - Dynamic spread adjustment
    - Inventory-based skewing
    - Volatility-adaptive sizing
    - Adverse selection protection
    - Multi-level quoting

    Attributes:
        config: Quote generator configuration
        risk_manager: Risk management instance
        inventory: Current inventory state per symbol
        quote_history: Historical quotes

    Example:
        >>> config = load_config('strategies.yaml')['market_making']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> generator = QuoteGenerator(config, risk_mgr)
        >>> quotes = await generator.generate_quotes(market_data, inventory)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize quote generator.

        Args:
            config: Quote generator configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Quote parameters
        self.base_spread_bps = Decimal(str(config.get('base_spread_bps', 10)))
        self.min_spread_bps = Decimal(str(config.get('min_spread_bps', 5)))
        self.max_spread_bps = Decimal(str(config.get('max_spread_bps', 50)))
        self.quote_size = Decimal(str(config.get('quote_size', 1000)))
        self.max_inventory_ratio = Decimal(str(config.get('max_inventory_ratio', 0.8)))
        self.inventory_skew_method = InventorySkewMethod(
            config.get('inventory_skew_method', 'linear')
        )

        # Multi-level quoting
        self.num_levels = config.get('num_quote_levels', 1)
        self.level_size_decay = Decimal(str(config.get('level_size_decay', 0.8)))
        self.level_spread_increment_bps = Decimal(
            str(config.get('level_spread_increment_bps', 5))
        )

        # Risk adjustments
        self.volatility_multiplier = Decimal(str(config.get('volatility_spread_multiplier', 1.5)))
        self.toxicity_threshold = Decimal(str(config.get('order_flow_toxicity_threshold', 0.7)))

        # State tracking
        self.inventory: Dict[str, InventoryState] = {}
        self.quote_history: List[Quote] = []
        self.quotes_generated: int = 0

        logger.info(
            "quote_generator_initialized",
            base_spread_bps=float(self.base_spread_bps),
            num_levels=self.num_levels,
            skew_method=self.inventory_skew_method.value
        )

    def _validate_config(self) -> None:
        """Validate quote generator configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = [
            'base_spread_bps',
            'min_spread_bps',
            'quote_size'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['min_spread_bps'] <= 0:
            raise ValueError("min_spread_bps must be positive")

        if self.config['base_spread_bps'] < self.config['min_spread_bps']:
            raise ValueError("base_spread_bps must be >= min_spread_bps")

    async def generate_quotes(
        self,
        market_data: pl.DataFrame,
        inventory: Dict[str, InventoryState]
    ) -> List[Quote]:
        """Generate market making quotes.

        Args:
            market_data: Polars DataFrame with columns:
                - symbol: str
                - mid_price: Decimal
                - bid_price: Decimal
                - ask_price: Decimal
                - bid_size: Decimal
                - ask_size: Decimal
                - volatility: Decimal (optional)
                - timestamp: datetime
            inventory: Current inventory state per symbol

        Returns:
            List of Quote objects

        Raises:
            ValueError: If market_data invalid
        """
        if market_data.is_empty():
            logger.warning("empty_market_data_received")
            return []

        try:
            self.inventory = inventory
            quotes = []

            # Process each symbol
            for row in market_data.iter_rows(named=True):
                symbol = row['symbol']

                # Get or create inventory state
                inv_state = inventory.get(symbol, self._create_default_inventory(symbol))

                # Generate quote for symbol
                quote = await self._generate_symbol_quote(row, inv_state)

                if quote:
                    quotes.append(quote)
                    self.quotes_generated += 1

            # Store in history (keep last 1000)
            self.quote_history.extend(quotes)
            if len(self.quote_history) > 1000:
                self.quote_history = self.quote_history[-1000:]

            logger.info(
                "quotes_generated",
                count=len(quotes),
                total_generated=self.quotes_generated
            )

            return quotes

        except Exception as e:
            logger.error(
                "quote_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _create_default_inventory(self, symbol: str) -> InventoryState:
        """Create default inventory state for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Default InventoryState
        """
        max_position = Decimal(str(self.config.get('max_position_size', 10000)))

        return InventoryState(
            symbol=symbol,
            position=Decimal('0'),
            target_position=Decimal('0'),
            max_position=max_position,
            avg_entry_price=Decimal('0'),
            unrealized_pnl=Decimal('0'),
            last_update=datetime.now(timezone.utc)
        )

    async def _generate_symbol_quote(
        self,
        market_row: Dict[str, Any],
        inventory: InventoryState
    ) -> Optional[Quote]:
        """Generate quote for a single symbol.

        Args:
            market_row: Market data row
            inventory: Current inventory state

        Returns:
            Quote object or None if unable to quote
        """
        try:
            symbol = market_row['symbol']

            # Extract market data
            mid_price = Decimal(str(market_row.get('mid_price', 0)))
            if mid_price == Decimal('0'):
                # Calculate from bid/ask if mid not provided
                bid = Decimal(str(market_row.get('bid_price', 0)))
                ask = Decimal(str(market_row.get('ask_price', 0)))
                if bid > 0 and ask > 0:
                    mid_price = (bid + ask) / Decimal('2')
                else:
                    logger.warning("invalid_market_data", symbol=symbol)
                    return None

            # Calculate base spread
            spread_bps = await self._calculate_optimal_spread(market_row, inventory)

            # Calculate inventory skew
            skew = self._calculate_inventory_skew(inventory)

            # Apply skew to quotes
            bid_offset, ask_offset = self._apply_inventory_skew(spread_bps, skew)

            # Calculate quote prices
            bid_price = mid_price * (Decimal('1') - bid_offset / Decimal('10000'))
            ask_price = mid_price * (Decimal('1') + ask_offset / Decimal('10000'))

            # Round to tick size
            tick_size = self._get_tick_size(mid_price)
            bid_price = self._round_price(bid_price, tick_size, ROUND_DOWN)
            ask_price = self._round_price(ask_price, tick_size, ROUND_UP)

            # Calculate quote sizes
            bid_quantity, ask_quantity = self._calculate_quote_sizes(inventory, skew)

            # Calculate actual spread
            actual_spread_bps = ((ask_price - bid_price) / mid_price) * Decimal('10000')

            # Calculate confidence
            confidence = self._calculate_quote_confidence(market_row, inventory)

            quote = Quote(
                symbol=symbol,
                bid_price=bid_price,
                bid_quantity=bid_quantity,
                ask_price=ask_price,
                ask_quantity=ask_quantity,
                mid_price=mid_price,
                spread_bps=actual_spread_bps,
                timestamp=datetime.now(timezone.utc),
                inventory_skew=skew,
                confidence=confidence,
                metadata={
                    'inventory_position': inventory.position,
                    'inventory_ratio': inventory.inventory_ratio,
                    'volatility': market_row.get('volatility', 0),
                    'num_levels': self.num_levels
                }
            )

            logger.debug(
                "quote_generated",
                symbol=symbol,
                bid=float(bid_price),
                ask=float(ask_price),
                spread_bps=float(actual_spread_bps),
                skew=float(skew)
            )

            return quote

        except Exception as e:
            logger.error(
                "symbol_quote_generation_failed",
                symbol=market_row.get('symbol', 'unknown'),
                error=str(e)
            )
            return None

    async def _calculate_optimal_spread(
        self,
        market_row: Dict[str, Any],
        inventory: InventoryState
    ) -> Decimal:
        """Calculate optimal spread based on market conditions.

        Args:
            market_row: Market data
            inventory: Inventory state

        Returns:
            Optimal spread in basis points
        """
        # Start with base spread
        spread = self.base_spread_bps

        # Adjust for volatility
        volatility = Decimal(str(market_row.get('volatility', 0)))
        if volatility > Decimal('0'):
            # Widen spread in high volatility
            vol_adjustment = volatility * self.volatility_multiplier
            spread = spread * (Decimal('1') + vol_adjustment)

        # Adjust for inventory risk
        inv_ratio = abs(inventory.inventory_ratio)
        if inv_ratio > Decimal('0.5'):
            # Widen spread when inventory is large
            inv_adjustment = (inv_ratio - Decimal('0.5')) * Decimal('2')
            spread = spread * (Decimal('1') + inv_adjustment)

        # Adjust for competition (bid-ask spread in market)
        market_bid = Decimal(str(market_row.get('bid_price', 0)))
        market_ask = Decimal(str(market_row.get('ask_price', 0)))
        if market_bid > 0 and market_ask > 0:
            market_mid = (market_bid + market_ask) / Decimal('2')
            market_spread_bps = ((market_ask - market_bid) / market_mid) * Decimal('10000')

            # Don't quote tighter than 80% of market spread
            min_competitive_spread = market_spread_bps * Decimal('0.8')
            spread = max(spread, min_competitive_spread)

        # Apply bounds
        spread = max(self.min_spread_bps, min(self.max_spread_bps, spread))

        return spread

    def _calculate_inventory_skew(self, inventory: InventoryState) -> Decimal:
        """Calculate inventory skew factor.

        Args:
            inventory: Current inventory state

        Returns:
            Skew factor (-1 to 1, negative = skew bids wider, positive = skew asks wider)
        """
        if inventory.max_position == Decimal('0'):
            return Decimal('0')

        # Get inventory ratio relative to target
        position_diff = inventory.position - inventory.target_position
        ratio = position_diff / inventory.max_position

        # Apply skew method
        if self.inventory_skew_method == InventorySkewMethod.LINEAR:
            skew = ratio

        elif self.inventory_skew_method == InventorySkewMethod.EXPONENTIAL:
            # Exponential skew for more aggressive inventory management
            sign = Decimal('1') if ratio >= 0 else Decimal('-1')
            skew = sign * (abs(ratio) ** Decimal('2'))

        else:  # SIGMOID
            # Sigmoid skew for smooth inventory management
            # tanh-like approximation using Decimal
            x = ratio * Decimal('3')  # Scale factor
            exp_2x = (x * Decimal('2')).exp()
            skew = (exp_2x - Decimal('1')) / (exp_2x + Decimal('1'))

        # Clamp to [-1, 1]
        return max(Decimal('-1'), min(Decimal('1'), skew))

    def _apply_inventory_skew(
        self,
        spread_bps: Decimal,
        skew: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Apply inventory skew to bid/ask offsets.

        Args:
            spread_bps: Base spread in basis points
            skew: Inventory skew factor (-1 to 1)

        Returns:
            Tuple of (bid_offset_bps, ask_offset_bps)
        """
        half_spread = spread_bps / Decimal('2')

        # Skew > 0 means long inventory: widen asks, tighten bids
        # Skew < 0 means short inventory: widen bids, tighten asks

        skew_adjustment = half_spread * skew * Decimal('0.5')  # Max 50% adjustment

        bid_offset = half_spread - skew_adjustment
        ask_offset = half_spread + skew_adjustment

        # Ensure non-negative
        bid_offset = max(Decimal('0'), bid_offset)
        ask_offset = max(Decimal('0'), ask_offset)

        return bid_offset, ask_offset

    def _calculate_quote_sizes(
        self,
        inventory: InventoryState,
        skew: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask quote sizes.

        Args:
            inventory: Current inventory state
            skew: Inventory skew factor

        Returns:
            Tuple of (bid_quantity, ask_quantity)
        """
        base_size = self.quote_size

        # Adjust sizes based on inventory
        # Long inventory: reduce bid size, increase ask size
        # Short inventory: increase bid size, reduce ask size

        size_skew_factor = Decimal('0.3')  # Max 30% size adjustment

        bid_multiplier = Decimal('1') - (skew * size_skew_factor)
        ask_multiplier = Decimal('1') + (skew * size_skew_factor)

        # Ensure positive
        bid_multiplier = max(Decimal('0.1'), bid_multiplier)
        ask_multiplier = max(Decimal('0.1'), ask_multiplier)

        bid_quantity = base_size * bid_multiplier
        ask_quantity = base_size * ask_multiplier

        # Round to reasonable precision
        bid_quantity = bid_quantity.quantize(Decimal('0.01'), rounding=ROUND_DOWN)
        ask_quantity = ask_quantity.quantize(Decimal('0.01'), rounding=ROUND_DOWN)

        return bid_quantity, ask_quantity

    def _calculate_quote_confidence(
        self,
        market_row: Dict[str, Any],
        inventory: InventoryState
    ) -> Decimal:
        """Calculate confidence in quote quality.

        Args:
            market_row: Market data
            inventory: Inventory state

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal('1.0')

        # Reduce confidence for large inventory
        inv_ratio = abs(inventory.inventory_ratio)
        if inv_ratio > Decimal('0.5'):
            confidence = confidence * (Decimal('1') - (inv_ratio - Decimal('0.5')))

        # Reduce confidence in high volatility
        volatility = Decimal(str(market_row.get('volatility', 0)))
        if volatility > Decimal('0.02'):  # 2% volatility threshold
            vol_penalty = min(volatility * Decimal('10'), Decimal('0.3'))
            confidence = confidence * (Decimal('1') - vol_penalty)

        # Reduce confidence for wide spreads
        market_bid = Decimal(str(market_row.get('bid_price', 0)))
        market_ask = Decimal(str(market_row.get('ask_price', 0)))
        if market_bid > 0 and market_ask > 0:
            mid = (market_bid + market_ask) / Decimal('2')
            spread_bps = ((market_ask - market_bid) / mid) * Decimal('10000')
            if spread_bps > Decimal('20'):
                spread_penalty = min((spread_bps - Decimal('20')) / Decimal('100'), Decimal('0.3'))
                confidence = confidence * (Decimal('1') - spread_penalty)

        return max(Decimal('0.1'), min(Decimal('1.0'), confidence))

    def _get_tick_size(self, price: Decimal) -> Decimal:
        """Get appropriate tick size for price level.

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
        elif price >= Decimal('1'):
            return Decimal('0.001')
        else:
            return Decimal('0.0001')

    def _round_price(self, price: Decimal, tick_size: Decimal, rounding: str) -> Decimal:
        """Round price to tick size.

        Args:
            price: Price to round
            tick_size: Tick size
            rounding: Rounding mode

        Returns:
            Rounded price
        """
        return (price / tick_size).quantize(Decimal('1'), rounding=rounding) * tick_size

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate quote generator performance indicators.

        Args:
            data: Historical quote data

        Returns:
            Dictionary of indicators
        """
        if not self.quote_history:
            return {}

        try:
            # Calculate average spread
            spreads = [q.spread_bps for q in self.quote_history]
            avg_spread = sum(spreads) / Decimal(str(len(spreads)))

            # Calculate average inventory skew
            skews = [abs(q.inventory_skew) for q in self.quote_history]
            avg_skew = sum(skews) / Decimal(str(len(skews)))

            # Calculate average confidence
            confidences = [q.confidence for q in self.quote_history]
            avg_confidence = sum(confidences) / Decimal(str(len(confidences)))

            return {
                'quotes_generated': Decimal(str(self.quotes_generated)),
                'avg_spread_bps': avg_spread,
                'avg_inventory_skew': avg_skew,
                'avg_confidence': avg_confidence
            }

        except Exception as e:
            logger.error("indicator_calculation_failed", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get quote generator performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return {
            'total_quotes_generated': self.quotes_generated,
            'active_symbols': len(self.inventory),
            'num_quote_levels': self.num_levels,
            'base_spread_bps': float(self.base_spread_bps)
        }
