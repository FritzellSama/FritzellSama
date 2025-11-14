"""Market Making Spread Calculator.

Calculates optimal bid-ask spreads for market making strategies based on
volatility, inventory, competition, and adverse selection risk.

Performance Target: <1ms calculation, real-time adjustment
Capital Allocation: N/A (supporting module)
Risk: Spread compression, adverse selection
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

import polars as pl
import numpy as np
import talib
from structlog import get_logger

logger = get_logger(__name__)


class SpreadModel(Enum):
    """Spread calculation models."""
    FIXED = "fixed"
    VOLATILITY_BASED = "volatility_based"
    INVENTORY_ADJUSTED = "inventory_adjusted"
    COMPETITIVE = "competitive"
    ADVERSE_SELECTION = "adverse_selection"
    HYBRID = "hybrid"


@dataclass
class SpreadParameters:
    """Calculated spread parameters."""
    symbol: str
    base_spread_bps: Decimal
    bid_spread_bps: Decimal
    ask_spread_bps: Decimal
    half_spread_bps: Decimal
    effective_spread_bps: Decimal
    timestamp: datetime
    model_used: SpreadModel
    confidence: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Current market conditions for spread calculation."""
    symbol: str
    volatility: Decimal
    volume: Decimal
    tick_size: Decimal
    competitive_spread_bps: Decimal
    order_flow_toxicity: Decimal
    timestamp: datetime


class SpreadCalculator:
    """Market making spread calculator.

    Calculates optimal bid-ask spreads using multiple models:
    - Fixed spread (baseline)
    - Volatility-based (Avellaneda-Stoikov)
    - Inventory-adjusted (skewing based on position)
    - Competitive (matching/improving market spreads)
    - Adverse selection protected
    - Hybrid (combining multiple models)

    Key Features:
    - Multiple calculation models
    - Real-time spread adjustment
    - Inventory skewing
    - Volatility adaptation
    - Adverse selection protection
    - Tick size awareness

    Attributes:
        config: Calculator configuration
        model: Spread calculation model to use
        min_spread_bps: Minimum allowed spread
        max_spread_bps: Maximum allowed spread
        spread_history: Historical spread calculations

    Example:
        >>> config = load_config('strategies.yaml')['market_making']['spread']
        >>> calculator = SpreadCalculator(config)
        >>> params = await calculator.calculate_spread(market_data, inventory)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize spread calculator.

        Args:
            config: Calculator configuration

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self._validate_config()

        # Spread model
        self.model = SpreadModel(config.get('model', 'hybrid'))

        # Spread bounds
        self.min_spread_bps = Decimal(str(config.get('min_spread_bps', 5)))
        self.max_spread_bps = Decimal(str(config.get('max_spread_bps', 100)))
        self.base_spread_bps = Decimal(str(config.get('base_spread_bps', 10)))

        # Model parameters
        self.volatility_multiplier = Decimal(str(config.get('volatility_multiplier', 2.0)))
        self.inventory_skew_max_bps = Decimal(str(config.get('inventory_skew_max_bps', 15)))
        self.competitive_improvement_bps = Decimal(str(config.get('competitive_improvement_bps', 1)))
        self.adverse_selection_factor = Decimal(str(config.get('adverse_selection_factor', 1.5)))

        # Inventory parameters
        self.max_inventory_ratio = Decimal(str(config.get('max_inventory_ratio', 0.8)))

        # State tracking
        self.spread_history: Dict[str, List[SpreadParameters]] = {}
        self.calculations_count: int = 0

        logger.info(
            "spread_calculator_initialized",
            model=self.model.value,
            base_spread_bps=float(self.base_spread_bps),
            min_spread_bps=float(self.min_spread_bps),
            max_spread_bps=float(self.max_spread_bps)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = ['model', 'min_spread_bps', 'base_spread_bps']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['min_spread_bps'] <= 0:
            raise ValueError("min_spread_bps must be positive")

        if self.config['base_spread_bps'] < self.config['min_spread_bps']:
            raise ValueError("base_spread_bps must be >= min_spread_bps")

    async def calculate_spread(
        self,
        market_data: pl.DataFrame,
        inventory: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SpreadParameters]:
        """Calculate optimal spreads for all symbols.

        Args:
            market_data: Polars DataFrame with market data:
                - symbol: str
                - bid_price: Decimal
                - ask_price: Decimal
                - last_price: Decimal
                - volume: Decimal
                - volatility: Decimal (optional)
                - timestamp: datetime
            inventory: Optional inventory state per symbol

        Returns:
            Dictionary of SpreadParameters by symbol

        Raises:
            ValueError: If market_data invalid
        """
        if market_data.is_empty():
            logger.warning("empty_market_data_received")
            return {}

        try:
            spreads = {}

            # Process each symbol
            for symbol in market_data['symbol'].unique():
                symbol_data = market_data.filter(pl.col('symbol') == symbol)

                # Extract market conditions
                conditions = self._extract_market_conditions(symbol_data)

                # Get inventory state if provided
                inv_state = inventory.get(symbol) if inventory else None

                # Calculate spread based on model
                spread_params = self._calculate_symbol_spread(conditions, inv_state)

                spreads[symbol] = spread_params

                # Store in history
                if symbol not in self.spread_history:
                    self.spread_history[symbol] = []
                self.spread_history[symbol].append(spread_params)

                # Keep last 100 calculations
                if len(self.spread_history[symbol]) > 100:
                    self.spread_history[symbol] = self.spread_history[symbol][-100:]

            self.calculations_count += len(spreads)

            logger.debug(
                "spreads_calculated",
                count=len(spreads),
                total_calculations=self.calculations_count
            )

            return spreads

        except Exception as e:
            logger.error(
                "spread_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _extract_market_conditions(self, data: pl.DataFrame) -> MarketConditions:
        """Extract market conditions from data.

        Args:
            data: Symbol market data

        Returns:
            MarketConditions object
        """
        symbol = data['symbol'][0]
        latest = data[-1]

        # Extract prices
        bid_price = Decimal(str(latest['bid_price'][0]))
        ask_price = Decimal(str(latest['ask_price'][0]))
        mid_price = (bid_price + ask_price) / Decimal('2')

        # Calculate competitive spread
        competitive_spread_bps = ((ask_price - bid_price) / mid_price) * Decimal('10000')

        # Get or calculate volatility
        if 'volatility' in data.columns:
            volatility = Decimal(str(latest['volatility'][0]))
        else:
            volatility = self._calculate_volatility(data)

        # Get volume
        volume = Decimal(str(data['volume'].sum()))

        # Calculate tick size
        tick_size = self._get_tick_size(mid_price)

        # Estimate order flow toxicity (simplified)
        order_flow_toxicity = self._estimate_toxicity(data)

        return MarketConditions(
            symbol=symbol,
            volatility=volatility,
            volume=volume,
            tick_size=tick_size,
            competitive_spread_bps=competitive_spread_bps,
            order_flow_toxicity=order_flow_toxicity,
            timestamp=datetime.now(timezone.utc)
        )

    def _calculate_volatility(self, data: pl.DataFrame) -> Decimal:
        """Calculate price volatility.

        Args:
            data: Market data

        Returns:
            Volatility (standard deviation of returns)
        """
        if data.height < 10:
            return Decimal('0.01')  # Default 1% volatility

        prices = data['last_price'].to_numpy().astype(float)

        # Calculate returns
        returns = np.diff(np.log(prices))

        # Calculate standard deviation
        volatility = np.std(returns)

        # Annualize (assuming 1-minute bars)
        # sqrt(365 * 24 * 60) for annualization
        annualized = volatility * np.sqrt(525600)

        return Decimal(str(annualized))

    def _estimate_toxicity(self, data: pl.DataFrame) -> Decimal:
        """Estimate order flow toxicity.

        Args:
            data: Market data

        Returns:
            Toxicity score (0-1)
        """
        if data.height < 5:
            return Decimal('0.5')

        # Simplified: use price volatility as proxy
        prices = data['last_price'].to_numpy().astype(float)
        price_changes = np.abs(np.diff(prices)) / prices[:-1]

        avg_change = np.mean(price_changes)

        # Normalize to 0-1 (assume 0.1% is high toxicity)
        toxicity = min(Decimal(str(avg_change)) / Decimal('0.001'), Decimal('1'))

        return toxicity

    def _calculate_symbol_spread(
        self,
        conditions: MarketConditions,
        inventory: Optional[Dict[str, Any]]
    ) -> SpreadParameters:
        """Calculate spread for a symbol based on model.

        Args:
            conditions: Market conditions
            inventory: Inventory state

        Returns:
            SpreadParameters
        """
        if self.model == SpreadModel.FIXED:
            return self._calculate_fixed_spread(conditions)

        elif self.model == SpreadModel.VOLATILITY_BASED:
            return self._calculate_volatility_spread(conditions)

        elif self.model == SpreadModel.INVENTORY_ADJUSTED:
            return self._calculate_inventory_spread(conditions, inventory)

        elif self.model == SpreadModel.COMPETITIVE:
            return self._calculate_competitive_spread(conditions)

        elif self.model == SpreadModel.ADVERSE_SELECTION:
            return self._calculate_adverse_selection_spread(conditions)

        else:  # HYBRID
            return self._calculate_hybrid_spread(conditions, inventory)

    def _calculate_fixed_spread(self, conditions: MarketConditions) -> SpreadParameters:
        """Calculate fixed spread.

        Args:
            conditions: Market conditions

        Returns:
            SpreadParameters
        """
        half_spread = self.base_spread_bps / Decimal('2')

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=self.base_spread_bps,
            bid_spread_bps=half_spread,
            ask_spread_bps=half_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=self.base_spread_bps,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.FIXED,
            confidence=Decimal('0.8'),
            metadata={'model': 'fixed'}
        )

    def _calculate_volatility_spread(self, conditions: MarketConditions) -> SpreadParameters:
        """Calculate volatility-based spread (Avellaneda-Stoikov).

        Args:
            conditions: Market conditions

        Returns:
            SpreadParameters
        """
        # Spread proportional to volatility
        vol_spread = self.base_spread_bps * (Decimal('1') + conditions.volatility * self.volatility_multiplier)

        # Apply bounds
        vol_spread = max(self.min_spread_bps, min(self.max_spread_bps, vol_spread))

        half_spread = vol_spread / Decimal('2')

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=vol_spread,
            bid_spread_bps=half_spread,
            ask_spread_bps=half_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=vol_spread,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.VOLATILITY_BASED,
            confidence=Decimal('0.85'),
            metadata={
                'model': 'volatility_based',
                'volatility': conditions.volatility
            }
        )

    def _calculate_inventory_spread(
        self,
        conditions: MarketConditions,
        inventory: Optional[Dict[str, Any]]
    ) -> SpreadParameters:
        """Calculate inventory-adjusted spread.

        Args:
            conditions: Market conditions
            inventory: Inventory state

        Returns:
            SpreadParameters
        """
        base_spread = self.base_spread_bps
        half_spread = base_spread / Decimal('2')

        # Default symmetric spread if no inventory
        if not inventory:
            return SpreadParameters(
                symbol=conditions.symbol,
                base_spread_bps=base_spread,
                bid_spread_bps=half_spread,
                ask_spread_bps=half_spread,
                half_spread_bps=half_spread,
                effective_spread_bps=base_spread,
                timestamp=datetime.now(timezone.utc),
                model_used=SpreadModel.INVENTORY_ADJUSTED,
                confidence=Decimal('0.7'),
                metadata={'model': 'inventory_adjusted', 'no_inventory': True}
            )

        # Calculate inventory skew
        position = Decimal(str(inventory.get('position', 0)))
        max_position = Decimal(str(inventory.get('max_position', 10000)))

        if max_position > 0:
            inventory_ratio = position / max_position
        else:
            inventory_ratio = Decimal('0')

        # Calculate skew (positive = long, negative = short)
        skew_bps = inventory_ratio * self.inventory_skew_max_bps

        # Apply skew to spreads
        bid_spread = half_spread - (skew_bps / Decimal('2'))
        ask_spread = half_spread + (skew_bps / Decimal('2'))

        # Ensure positive spreads
        bid_spread = max(Decimal('0.5'), bid_spread)
        ask_spread = max(Decimal('0.5'), ask_spread)

        effective_spread = bid_spread + ask_spread

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=base_spread,
            bid_spread_bps=bid_spread,
            ask_spread_bps=ask_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=effective_spread,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.INVENTORY_ADJUSTED,
            confidence=Decimal('0.9'),
            metadata={
                'model': 'inventory_adjusted',
                'inventory_ratio': inventory_ratio,
                'skew_bps': skew_bps
            }
        )

    def _calculate_competitive_spread(self, conditions: MarketConditions) -> SpreadParameters:
        """Calculate competitive spread (matching/improving market).

        Args:
            conditions: Market conditions

        Returns:
            SpreadParameters
        """
        # Match or improve competitive spread
        competitive = conditions.competitive_spread_bps

        # Improve by configured amount
        our_spread = max(
            self.min_spread_bps,
            competitive - self.competitive_improvement_bps
        )

        # Don't go wider than max
        our_spread = min(our_spread, self.max_spread_bps)

        half_spread = our_spread / Decimal('2')

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=our_spread,
            bid_spread_bps=half_spread,
            ask_spread_bps=half_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=our_spread,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.COMPETITIVE,
            confidence=Decimal('0.75'),
            metadata={
                'model': 'competitive',
                'market_spread_bps': competitive
            }
        )

    def _calculate_adverse_selection_spread(self, conditions: MarketConditions) -> SpreadParameters:
        """Calculate spread with adverse selection protection.

        Args:
            conditions: Market conditions

        Returns:
            SpreadParameters
        """
        # Widen spread based on toxicity
        base_spread = self.base_spread_bps

        toxicity_adjustment = Decimal('1') + (conditions.order_flow_toxicity * self.adverse_selection_factor)
        protected_spread = base_spread * toxicity_adjustment

        # Apply bounds
        protected_spread = max(self.min_spread_bps, min(self.max_spread_bps, protected_spread))

        half_spread = protected_spread / Decimal('2')

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=protected_spread,
            bid_spread_bps=half_spread,
            ask_spread_bps=half_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=protected_spread,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.ADVERSE_SELECTION,
            confidence=Decimal('0.88'),
            metadata={
                'model': 'adverse_selection',
                'toxicity': conditions.order_flow_toxicity,
                'toxicity_adjustment': toxicity_adjustment
            }
        )

    def _calculate_hybrid_spread(
        self,
        conditions: MarketConditions,
        inventory: Optional[Dict[str, Any]]
    ) -> SpreadParameters:
        """Calculate hybrid spread combining multiple models.

        Args:
            conditions: Market conditions
            inventory: Inventory state

        Returns:
            SpreadParameters
        """
        # Calculate using multiple models
        vol_params = self._calculate_volatility_spread(conditions)
        comp_params = self._calculate_competitive_spread(conditions)
        adv_params = self._calculate_adverse_selection_spread(conditions)

        # Weighted average (weights: volatility=40%, competitive=30%, adverse_selection=30%)
        base_spread = (
            vol_params.base_spread_bps * Decimal('0.4') +
            comp_params.base_spread_bps * Decimal('0.3') +
            adv_params.base_spread_bps * Decimal('0.3')
        )

        # Apply inventory skew if available
        half_spread = base_spread / Decimal('2')

        if inventory:
            position = Decimal(str(inventory.get('position', 0)))
            max_position = Decimal(str(inventory.get('max_position', 10000)))

            if max_position > 0:
                inventory_ratio = position / max_position
                skew_bps = inventory_ratio * self.inventory_skew_max_bps

                bid_spread = half_spread - (skew_bps / Decimal('2'))
                ask_spread = half_spread + (skew_bps / Decimal('2'))

                bid_spread = max(Decimal('0.5'), bid_spread)
                ask_spread = max(Decimal('0.5'), ask_spread)
            else:
                bid_spread = half_spread
                ask_spread = half_spread
        else:
            bid_spread = half_spread
            ask_spread = half_spread

        effective_spread = bid_spread + ask_spread

        return SpreadParameters(
            symbol=conditions.symbol,
            base_spread_bps=base_spread,
            bid_spread_bps=bid_spread,
            ask_spread_bps=ask_spread,
            half_spread_bps=half_spread,
            effective_spread_bps=effective_spread,
            timestamp=datetime.now(timezone.utc),
            model_used=SpreadModel.HYBRID,
            confidence=Decimal('0.95'),
            metadata={
                'model': 'hybrid',
                'volatility_component': vol_params.base_spread_bps,
                'competitive_component': comp_params.base_spread_bps,
                'adverse_selection_component': adv_params.base_spread_bps,
                'inventory_position': inventory.get('position', 0) if inventory else 0
            }
        )

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
        elif price >= Decimal('1'):
            return Decimal('0.001')
        else:
            return Decimal('0.0001')

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate performance indicators.

        Args:
            data: Historical data

        Returns:
            Dictionary of indicators
        """
        if not self.spread_history:
            return {}

        all_spreads = []
        for symbol_spreads in self.spread_history.values():
            all_spreads.extend([s.effective_spread_bps for s in symbol_spreads])

        if not all_spreads:
            return {}

        avg_spread = sum(all_spreads) / Decimal(str(len(all_spreads)))
        min_spread = min(all_spreads)
        max_spread = max(all_spreads)

        return {
            'calculations_count': Decimal(str(self.calculations_count)),
            'avg_spread_bps': avg_spread,
            'min_spread_bps': min_spread,
            'max_spread_bps': max_spread,
            'symbols_tracked': Decimal(str(len(self.spread_history)))
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'calculations_count': self.calculations_count,
            'symbols_tracked': len(self.spread_history),
            'model': self.model.value,
            'base_spread_bps': float(self.base_spread_bps)
        }
