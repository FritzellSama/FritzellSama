"""Options Spread Trading Strategy.

Trades multi-leg options strategies including vertical spreads, iron condors,
butterflies, and calendars for volatility and theta capture.

Performance Target: 65%+ win rate, moderate frequency (10-50 trades/day)
Capital Allocation: Configurable via config
Risk: Limited max loss per spread, volatility risk
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum

import polars as pl
import numpy as np
from scipy.stats import norm
from structlog import get_logger

logger = get_logger(__name__)


class SpreadType(Enum):
    """Types of option spreads."""
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    VERTICAL_SPREAD = "vertical_spread"
    DIAGONAL_SPREAD = "diagonal_spread"


class OptionType(Enum):
    """Option types."""
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class OptionLeg:
    """Single option leg in a spread."""
    symbol: str
    option_type: OptionType
    strike: Decimal
    expiry: datetime
    action: str  # 'BUY' or 'SELL'
    quantity: Decimal
    premium: Decimal
    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal
    implied_vol: Decimal


@dataclass
class OptionSpread:
    """Multi-leg options spread."""
    spread_type: SpreadType
    legs: List[OptionLeg]
    underlying_symbol: str
    underlying_price: Decimal
    net_premium: Decimal  # Credit or debit
    max_profit: Decimal
    max_loss: Decimal
    breakeven_prices: List[Decimal]
    probability_profit: Decimal
    expected_return: Decimal
    days_to_expiry: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptionsSpreadsStrategy:
    """Options spread trading strategy.

    Constructs and trades multi-leg options strategies to:
    - Capture theta decay (time value)
    - Trade volatility (vega)
    - Limit downside risk
    - Generate consistent income

    Key Features:
    - Multiple spread strategies
    - Greeks-based risk management
    - Implied volatility analysis
    - Probability-based entry
    - Dynamic adjustments

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        active_spreads: Currently open spreads
        spread_types: Enabled spread types

    Example:
        >>> config = load_config('strategies.yaml')['options']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = OptionsSpreadsStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(options_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize options spreads strategy.

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
        self.spread_types = [
            SpreadType(st) for st in config.get('spread_types', ['iron_condor', 'vertical_spread'])
        ]
        self.min_days_to_expiry = config.get('min_days_to_expiry', 7)
        self.max_days_to_expiry = config.get('max_days_to_expiry', 45)
        self.target_delta = Decimal(str(config.get('target_delta', 0.3)))
        self.min_credit = Decimal(str(config.get('min_credit_per_spread', 50)))

        # Risk parameters
        self.max_loss_per_spread = Decimal(str(config.get('max_loss_per_spread', 500)))
        self.min_prob_profit = Decimal(str(config.get('min_probability_profit', 0.6)))
        self.target_return_pct = Decimal(str(config.get('target_return_percent', 10)))

        # IV parameters
        self.min_iv_percentile = Decimal(str(config.get('min_iv_percentile', 30)))
        self.max_iv_percentile = Decimal(str(config.get('max_iv_percentile', 80)))

        # State tracking
        self.active_spreads: Dict[str, OptionSpread] = {}
        self.spreads_opened: int = 0
        self.spreads_closed: int = 0

        # Risk-free rate for pricing
        self.risk_free_rate = Decimal(str(config.get('risk_free_rate', 0.05)))

        logger.info(
            "options_spreads_strategy_initialized",
            spread_types=[st.value for st in self.spread_types],
            min_days_to_expiry=self.min_days_to_expiry,
            max_days_to_expiry=self.max_days_to_expiry
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = [
            'spread_types',
            'min_days_to_expiry',
            'max_days_to_expiry',
            'min_probability_profit'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate options spread signals.

        Args:
            market_data: Polars DataFrame with options chain data:
                - underlying_symbol: str
                - underlying_price: Decimal
                - option_symbol: str
                - strike: Decimal
                - expiry: datetime
                - option_type: str ('CALL' or 'PUT')
                - bid: Decimal
                - ask: Decimal
                - implied_vol: Decimal
                - delta: Decimal
                - gamma: Decimal
                - theta: Decimal
                - vega: Decimal
                - volume: Decimal
                - open_interest: Decimal

        Returns:
            List of signal dictionaries

        Raises:
            ValueError: If market_data invalid
        """
        if market_data.is_empty():
            logger.warning("empty_market_data_received")
            return []

        try:
            signals = []

            # Group by underlying
            for underlying in market_data['underlying_symbol'].unique():
                underlying_data = market_data.filter(
                    pl.col('underlying_symbol') == underlying
                )

                # Filter by days to expiry
                valid_expiries = self._filter_by_dte(underlying_data)

                for expiry in valid_expiries:
                    expiry_data = underlying_data.filter(pl.col('expiry') == expiry)

                    # Generate spreads for each enabled type
                    for spread_type in self.spread_types:
                        spread_signals = self._construct_spreads(
                            spread_type=spread_type,
                            options_data=expiry_data,
                            underlying=underlying
                        )
                        signals.extend(spread_signals)

            logger.info("options_spread_signals_generated", count=len(signals))
            return signals

        except Exception as e:
            logger.error(
                "signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _filter_by_dte(self, data: pl.DataFrame) -> List[datetime]:
        """Filter expiries by days to expiry.

        Args:
            data: Options data

        Returns:
            List of valid expiry dates
        """
        now = datetime.now(timezone.utc)
        valid_expiries = []

        for expiry in data['expiry'].unique():
            dte = (expiry - now).days

            if self.min_days_to_expiry <= dte <= self.max_days_to_expiry:
                valid_expiries.append(expiry)

        return valid_expiries

    def _construct_spreads(
        self,
        spread_type: SpreadType,
        options_data: pl.DataFrame,
        underlying: str
    ) -> List[Dict[str, Any]]:
        """Construct spreads of given type.

        Args:
            spread_type: Type of spread to construct
            options_data: Options chain data
            underlying: Underlying symbol

        Returns:
            List of spread signals
        """
        if spread_type == SpreadType.IRON_CONDOR:
            return self._construct_iron_condor(options_data, underlying)
        elif spread_type == SpreadType.VERTICAL_SPREAD:
            return self._construct_vertical_spread(options_data, underlying)
        elif spread_type == SpreadType.BUTTERFLY:
            return self._construct_butterfly(options_data, underlying)
        else:
            logger.debug("spread_type_not_implemented", spread_type=spread_type.value)
            return []

    def _construct_iron_condor(
        self,
        options_data: pl.DataFrame,
        underlying: str
    ) -> List[Dict[str, Any]]:
        """Construct iron condor spreads.

        Args:
            options_data: Options data
            underlying: Underlying symbol

        Returns:
            List of iron condor signals
        """
        signals = []

        if options_data.is_empty():
            return signals

        underlying_price = Decimal(str(options_data['underlying_price'][0]))
        expiry = options_data['expiry'][0]

        # Get puts and calls
        puts = options_data.filter(pl.col('option_type') == 'PUT')
        calls = options_data.filter(pl.col('option_type') == 'CALL')

        # Find put spread (sell higher strike, buy lower strike)
        put_sell_candidates = puts.filter(
            (pl.col('strike') < underlying_price) &
            (pl.col('delta').abs() >= float(self.target_delta) - 0.05) &
            (pl.col('delta').abs() <= float(self.target_delta) + 0.05)
        )

        # Find call spread (sell lower strike, buy higher strike)
        call_sell_candidates = calls.filter(
            (pl.col('strike') > underlying_price) &
            (pl.col('delta').abs() >= float(self.target_delta) - 0.05) &
            (pl.col('delta').abs() <= float(self.target_delta) + 0.05)
        )

        if put_sell_candidates.is_empty() or call_sell_candidates.is_empty():
            return signals

        # Select strikes for iron condor
        put_sell = put_sell_candidates[0]
        put_sell_strike = Decimal(str(put_sell['strike'][0]))

        call_sell = call_sell_candidates[0]
        call_sell_strike = Decimal(str(call_sell['strike'][0]))

        # Find buy strikes (further OTM)
        width = Decimal(str(self.config.get('spread_width', 5)))

        put_buy_strike = put_sell_strike - width
        call_buy_strike = call_sell_strike + width

        # Get options at buy strikes
        put_buy_data = puts.filter(pl.col('strike') == float(put_buy_strike))
        call_buy_data = calls.filter(pl.col('strike') == float(call_buy_strike))

        if put_buy_data.is_empty() or call_buy_data.is_empty():
            return signals

        # Create legs
        legs = []

        # Short put
        legs.append(self._create_option_leg(put_sell, 'SELL', OptionType.PUT))

        # Long put
        legs.append(self._create_option_leg(put_buy_data[0], 'BUY', OptionType.PUT))

        # Short call
        legs.append(self._create_option_leg(call_sell, 'SELL', OptionType.CALL))

        # Long call
        legs.append(self._create_option_leg(call_buy_data[0], 'BUY', OptionType.CALL))

        # Calculate spread economics
        net_premium = sum(
            leg.premium if leg.action == 'SELL' else -leg.premium
            for leg in legs
        )

        max_loss = width - net_premium
        max_profit = net_premium

        # Check if meets criteria
        if net_premium < self.min_credit:
            return signals

        if max_loss > self.max_loss_per_spread:
            return signals

        # Calculate probability of profit (simplified)
        prob_profit = self._calculate_prob_profit_iron_condor(
            underlying_price=underlying_price,
            put_strike=put_sell_strike,
            call_strike=call_sell_strike,
            implied_vol=Decimal(str(put_sell['implied_vol'][0])),
            days_to_expiry=(expiry - datetime.now(timezone.utc)).days
        )

        if prob_profit < self.min_prob_profit:
            return signals

        # Create spread
        spread = OptionSpread(
            spread_type=SpreadType.IRON_CONDOR,
            legs=legs,
            underlying_symbol=underlying,
            underlying_price=underlying_price,
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_prices=[put_sell_strike - net_premium, call_sell_strike + net_premium],
            probability_profit=prob_profit,
            expected_return=(max_profit / max_loss),
            days_to_expiry=(expiry - datetime.now(timezone.utc)).days,
            timestamp=datetime.now(timezone.utc),
            metadata={'expiry': expiry}
        )

        # Create signal
        signal = self._create_spread_signal(spread)
        signals.append(signal)

        return signals

    def _construct_vertical_spread(
        self,
        options_data: pl.DataFrame,
        underlying: str
    ) -> List[Dict[str, Any]]:
        """Construct vertical spreads.

        Args:
            options_data: Options data
            underlying: Underlying symbol

        Returns:
            List of vertical spread signals
        """
        signals = []

        if options_data.is_empty():
            return signals

        underlying_price = Decimal(str(options_data['underlying_price'][0]))
        expiry = options_data['expiry'][0]

        # Try bull call spread
        calls = options_data.filter(pl.col('option_type') == 'CALL')

        if not calls.is_empty():
            # Buy ATM or slightly OTM call
            buy_call = calls.filter(
                pl.col('strike') >= float(underlying_price * Decimal('0.98'))
            ).sort('strike')[0]

            buy_strike = Decimal(str(buy_call['strike'][0]))
            width = Decimal(str(self.config.get('spread_width', 5)))
            sell_strike = buy_strike + width

            # Sell higher strike call
            sell_call_data = calls.filter(pl.col('strike') == float(sell_strike))

            if not sell_call_data.is_empty():
                sell_call = sell_call_data[0]

                # Create legs
                buy_leg = self._create_option_leg(buy_call, 'BUY', OptionType.CALL)
                sell_leg = self._create_option_leg(sell_call, 'SELL', OptionType.CALL)

                # Calculate economics
                net_debit = buy_leg.premium - sell_leg.premium
                max_profit = width - net_debit
                max_loss = net_debit

                if max_loss <= self.max_loss_per_spread:
                    spread = OptionSpread(
                        spread_type=SpreadType.BULL_CALL_SPREAD,
                        legs=[buy_leg, sell_leg],
                        underlying_symbol=underlying,
                        underlying_price=underlying_price,
                        net_premium=-net_debit,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        breakeven_prices=[buy_strike + net_debit],
                        probability_profit=Decimal('0.5'),
                        expected_return=(max_profit / max_loss),
                        days_to_expiry=(expiry - datetime.now(timezone.utc)).days,
                        timestamp=datetime.now(timezone.utc),
                        metadata={'expiry': expiry, 'direction': 'bullish'}
                    )

                    signal = self._create_spread_signal(spread)
                    signals.append(signal)

        return signals

    def _construct_butterfly(
        self,
        options_data: pl.DataFrame,
        underlying: str
    ) -> List[Dict[str, Any]]:
        """Construct butterfly spreads.

        Args:
            options_data: Options data
            underlying: Underlying symbol

        Returns:
            List of butterfly signals
        """
        # Simplified butterfly construction
        # Full implementation would construct long butterfly (buy 1 low, sell 2 mid, buy 1 high)
        return []

    def _create_option_leg(
        self,
        option_data: pl.DataFrame,
        action: str,
        option_type: OptionType
    ) -> OptionLeg:
        """Create option leg from data.

        Args:
            option_data: Single option data row
            action: 'BUY' or 'SELL'
            option_type: Option type

        Returns:
            OptionLeg object
        """
        row = option_data[0]

        premium = Decimal(str(row['ask'][0])) if action == 'BUY' else Decimal(str(row['bid'][0]))

        return OptionLeg(
            symbol=str(row['option_symbol'][0]),
            option_type=option_type,
            strike=Decimal(str(row['strike'][0])),
            expiry=row['expiry'][0],
            action=action,
            quantity=Decimal('1'),
            premium=premium,
            delta=Decimal(str(row['delta'][0])),
            gamma=Decimal(str(row['gamma'][0])),
            theta=Decimal(str(row['theta'][0])),
            vega=Decimal(str(row['vega'][0])),
            implied_vol=Decimal(str(row['implied_vol'][0]))
        )

    def _calculate_prob_profit_iron_condor(
        self,
        underlying_price: Decimal,
        put_strike: Decimal,
        call_strike: Decimal,
        implied_vol: Decimal,
        days_to_expiry: int
    ) -> Decimal:
        """Calculate probability of profit for iron condor.

        Args:
            underlying_price: Current underlying price
            put_strike: Short put strike
            call_strike: Short call strike
            implied_vol: Implied volatility
            days_to_expiry: Days to expiration

        Returns:
            Probability of profit (0-1)
        """
        # Simplified calculation using normal distribution
        # Assume price follows normal distribution

        time_to_expiry = Decimal(str(days_to_expiry)) / Decimal('365')
        std_dev = underlying_price * implied_vol * (time_to_expiry.sqrt() if hasattr(time_to_expiry, 'sqrt') else Decimal(str(float(time_to_expiry) ** 0.5)))

        # Probability price stays between put and call strikes
        z_put = float((put_strike - underlying_price) / std_dev)
        z_call = float((call_strike - underlying_price) / std_dev)

        prob = norm.cdf(z_call) - norm.cdf(z_put)

        return Decimal(str(prob))

    def _create_spread_signal(self, spread: OptionSpread) -> Dict[str, Any]:
        """Create signal from spread.

        Args:
            spread: Option spread

        Returns:
            Signal dictionary
        """
        self.spreads_opened += 1

        # Calculate confidence based on probability of profit
        confidence = spread.probability_profit

        return {
            'symbol': spread.underlying_symbol,
            'action': 'OPEN_SPREAD',
            'spread_type': spread.spread_type.value,
            'legs': [
                {
                    'option_symbol': leg.symbol,
                    'action': leg.action,
                    'quantity': leg.quantity,
                    'strike': leg.strike,
                    'expiry': leg.expiry,
                    'option_type': leg.option_type.value
                }
                for leg in spread.legs
            ],
            'net_premium': spread.net_premium,
            'max_profit': spread.max_profit,
            'max_loss': spread.max_loss,
            'strategy': 'options_spreads',
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'underlying_price': spread.underlying_price,
                'probability_profit': spread.probability_profit,
                'expected_return': spread.expected_return,
                'days_to_expiry': spread.days_to_expiry,
                **spread.metadata
            }
        }

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate strategy indicators.

        Args:
            data: Historical data

        Returns:
            Dictionary of indicators
        """
        return {
            'spreads_opened': Decimal(str(self.spreads_opened)),
            'spreads_closed': Decimal(str(self.spreads_closed)),
            'active_spreads': Decimal(str(len(self.active_spreads)))
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'spreads_opened': self.spreads_opened,
            'spreads_closed': self.spreads_closed,
            'active_spreads': len(self.active_spreads),
            'enabled_spread_types': [st.value for st in self.spread_types]
        }
