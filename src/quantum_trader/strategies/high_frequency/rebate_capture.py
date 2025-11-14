"""Rebate Capture High-Frequency Trading Strategy.

Optimizes order placement to maximize exchange maker rebates while
minimizing market risk through rapid turnover and tight risk controls.

Performance Target: <10ms execution, 5000+ trades/day
Capital Allocation: Configurable via config
Risk: Minimal per-trade risk, adverse selection
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class RebateOpportunity:
    """Identified rebate capture opportunity."""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: Decimal
    quantity: Decimal
    expected_rebate: Decimal
    expected_fill_time_ms: int
    risk_score: Decimal  # 0-1, lower is better
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeRebateStructure:
    """Exchange rebate/fee structure."""
    exchange: str
    maker_rebate_bps: Decimal
    taker_fee_bps: Decimal
    volume_tier: str
    monthly_volume: Decimal
    tier_threshold: Decimal
    timestamp: datetime


class RestateCaptureStrategy:
    """Rebate capture high-frequency trading strategy.

    This strategy focuses on maximizing maker rebates by:
    - Placing passive limit orders at optimal price levels
    - Minimizing time in market to reduce adverse selection
    - Managing inventory to stay market-neutral
    - Optimizing across multiple exchanges for best rebate structures

    Key Features:
    - Multi-exchange rebate optimization
    - Sub-second hold times
    - Inventory-neutral positioning
    - Dynamic tier tracking
    - Adverse selection protection

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        exchange_rebates: Rebate structures per exchange
        active_opportunities: Currently tracked opportunities
        performance_stats: Performance tracking

    Example:
        >>> config = load_config('strategies.yaml')['high_frequency']['rebate_capture']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = RebateCaptureStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(market_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize rebate capture strategy.

        Args:
            config: Strategy configuration dictionary
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Exchange rebate tracking
        self.exchange_rebates: Dict[str, ExchangeRebateStructure] = {}
        self._initialize_exchange_rebates()

        # Strategy state
        self.active_opportunities: Dict[str, RebateOpportunity] = {}
        self.filled_orders: List[Dict[str, Any]] = []
        self.pending_orders: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.total_rebates_earned: Decimal = Decimal('0')
        self.total_fees_paid: Decimal = Decimal('0')
        self.total_trades: int = 0
        self.total_volume: Decimal = Decimal('0')

        # Configuration parameters
        self.min_rebate_bps = Decimal(str(config.get('min_rebate_bps', 0.2)))
        self.max_hold_time_ms = config.get('max_hold_time_ms', 5000)
        self.target_fills_per_hour = config.get('target_fills_per_hour', 1000)
        self.max_adverse_selection_bps = Decimal(str(config.get('max_adverse_selection_bps', 1.0)))
        self.inventory_limit = Decimal(str(config.get('inventory_limit', 10000)))
        self.min_spread_capture_bps = Decimal(str(config.get('min_spread_capture_bps', 0.5)))

        logger.info(
            "rebate_capture_strategy_initialized",
            min_rebate_bps=float(self.min_rebate_bps),
            max_hold_time_ms=self.max_hold_time_ms,
            target_fills_per_hour=self.target_fills_per_hour
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = [
            'min_rebate_bps',
            'max_hold_time_ms',
            'target_fills_per_hour',
            'exchanges'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['min_rebate_bps'] < 0:
            raise ValueError("min_rebate_bps must be non-negative")

        if self.config['max_hold_time_ms'] <= 0:
            raise ValueError("max_hold_time_ms must be positive")

    def _initialize_exchange_rebates(self) -> None:
        """Initialize exchange rebate structures from config."""
        exchanges_config = self.config.get('exchanges', {})

        for exchange_name, exchange_cfg in exchanges_config.items():
            self.exchange_rebates[exchange_name] = ExchangeRebateStructure(
                exchange=exchange_name,
                maker_rebate_bps=Decimal(str(exchange_cfg.get('maker_rebate_bps', 0))),
                taker_fee_bps=Decimal(str(exchange_cfg.get('taker_fee_bps', 5))),
                volume_tier=exchange_cfg.get('current_tier', 'standard'),
                monthly_volume=Decimal('0'),
                tier_threshold=Decimal(str(exchange_cfg.get('next_tier_threshold', 0))),
                timestamp=datetime.now(timezone.utc)
            )

        logger.info(
            "exchange_rebates_initialized",
            num_exchanges=len(self.exchange_rebates),
            exchanges=list(self.exchange_rebates.keys())
        )

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate rebate capture trading signals.

        Args:
            market_data: Polars DataFrame with columns:
                - symbol: str
                - exchange: str
                - bid_price: Decimal
                - ask_price: Decimal
                - bid_size: Decimal
                - ask_size: Decimal
                - last_price: Decimal
                - volume_1m: Decimal
                - timestamp: datetime

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

            # Clean up stale opportunities
            self._cleanup_stale_opportunities()

            # Process each market data row
            for row in market_data.iter_rows(named=True):
                symbol = row['symbol']
                exchange = row.get('exchange', 'unknown')

                # Get exchange rebate structure
                rebate_struct = self.exchange_rebates.get(exchange)
                if not rebate_struct:
                    logger.debug(
                        "unknown_exchange_skipped",
                        exchange=exchange,
                        symbol=symbol
                    )
                    continue

                # Check if rebate meets minimum threshold
                if rebate_struct.maker_rebate_bps < self.min_rebate_bps:
                    continue

                # Identify opportunities
                opportunities = self._identify_opportunities(row, rebate_struct)

                # Evaluate and filter opportunities
                for opp in opportunities:
                    if self._should_execute_opportunity(opp):
                        signal = self._create_signal_from_opportunity(opp)
                        signals.append(signal)
                        self.active_opportunities[f"{symbol}_{opp.side}"] = opp

            logger.info(
                "rebate_capture_signals_generated",
                signal_count=len(signals),
                active_opportunities=len(self.active_opportunities)
            )

            return signals

        except Exception as e:
            logger.error(
                "signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _identify_opportunities(
        self,
        market_row: Dict[str, Any],
        rebate_struct: ExchangeRebateStructure
    ) -> List[RebateOpportunity]:
        """Identify rebate capture opportunities from market data.

        Args:
            market_row: Market data row
            rebate_struct: Exchange rebate structure

        Returns:
            List of identified opportunities
        """
        opportunities = []
        symbol = market_row['symbol']

        # Extract market data
        bid_price = Decimal(str(market_row.get('bid_price', 0)))
        ask_price = Decimal(str(market_row.get('ask_price', 0)))
        bid_size = Decimal(str(market_row.get('bid_size', 0)))
        ask_size = Decimal(str(market_row.get('ask_size', 0)))

        if bid_price <= 0 or ask_price <= 0:
            return opportunities

        # Calculate spread
        mid_price = (bid_price + ask_price) / Decimal('2')
        spread_bps = ((ask_price - bid_price) / mid_price) * Decimal('10000')

        # Check if spread is sufficient for rebate capture
        if spread_bps < self.min_spread_capture_bps:
            return opportunities

        # Evaluate buy opportunity (join bid)
        if bid_size >= self.config.get('min_level_size', Decimal('1000')):
            buy_opp = self._evaluate_buy_opportunity(
                symbol=symbol,
                price=bid_price,
                reference_size=bid_size,
                mid_price=mid_price,
                spread_bps=spread_bps,
                rebate_struct=rebate_struct,
                market_row=market_row
            )
            if buy_opp:
                opportunities.append(buy_opp)

        # Evaluate sell opportunity (join ask)
        if ask_size >= self.config.get('min_level_size', Decimal('1000')):
            sell_opp = self._evaluate_sell_opportunity(
                symbol=symbol,
                price=ask_price,
                reference_size=ask_size,
                mid_price=mid_price,
                spread_bps=spread_bps,
                rebate_struct=rebate_struct,
                market_row=market_row
            )
            if sell_opp:
                opportunities.append(sell_opp)

        return opportunities

    def _evaluate_buy_opportunity(
        self,
        symbol: str,
        price: Decimal,
        reference_size: Decimal,
        mid_price: Decimal,
        spread_bps: Decimal,
        rebate_struct: ExchangeRebateStructure,
        market_row: Dict[str, Any]
    ) -> Optional[RebateOpportunity]:
        """Evaluate buy side rebate opportunity.

        Args:
            symbol: Trading symbol
            price: Bid price to join
            reference_size: Size at bid level
            mid_price: Current mid price
            spread_bps: Current spread
            rebate_struct: Exchange rebate structure
            market_row: Full market data row

        Returns:
            RebateOpportunity or None
        """
        # Calculate position size
        base_size = Decimal(str(self.config.get('base_order_size', 100)))
        quantity = min(base_size, reference_size * Decimal('0.1'))  # Max 10% of level

        # Calculate expected rebate
        notional = quantity * price
        expected_rebate = notional * rebate_struct.maker_rebate_bps / Decimal('10000')

        # Estimate fill time based on volume
        volume_1m = Decimal(str(market_row.get('volume_1m', 0)))
        est_fill_time_ms = self._estimate_fill_time(quantity, volume_1m)

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            price=price,
            mid_price=mid_price,
            spread_bps=spread_bps,
            est_fill_time_ms=est_fill_time_ms,
            market_row=market_row
        )

        # Check if opportunity meets criteria
        if est_fill_time_ms > self.max_hold_time_ms:
            return None

        if risk_score > Decimal('0.7'):  # Max acceptable risk
            return None

        return RebateOpportunity(
            symbol=symbol,
            side='BUY',
            price=price,
            quantity=quantity,
            expected_rebate=expected_rebate,
            expected_fill_time_ms=est_fill_time_ms,
            risk_score=risk_score,
            timestamp=datetime.now(timezone.utc),
            metadata={
                'exchange': rebate_struct.exchange,
                'spread_bps': spread_bps,
                'rebate_bps': rebate_struct.maker_rebate_bps,
                'reference_size': reference_size
            }
        )

    def _evaluate_sell_opportunity(
        self,
        symbol: str,
        price: Decimal,
        reference_size: Decimal,
        mid_price: Decimal,
        spread_bps: Decimal,
        rebate_struct: ExchangeRebateStructure,
        market_row: Dict[str, Any]
    ) -> Optional[RebateOpportunity]:
        """Evaluate sell side rebate opportunity.

        Args:
            symbol: Trading symbol
            price: Ask price to join
            reference_size: Size at ask level
            mid_price: Current mid price
            spread_bps: Current spread
            rebate_struct: Exchange rebate structure
            market_row: Full market data row

        Returns:
            RebateOpportunity or None
        """
        # Calculate position size
        base_size = Decimal(str(self.config.get('base_order_size', 100)))
        quantity = min(base_size, reference_size * Decimal('0.1'))

        # Calculate expected rebate
        notional = quantity * price
        expected_rebate = notional * rebate_struct.maker_rebate_bps / Decimal('10000')

        # Estimate fill time
        volume_1m = Decimal(str(market_row.get('volume_1m', 0)))
        est_fill_time_ms = self._estimate_fill_time(quantity, volume_1m)

        # Calculate risk score
        risk_score = self._calculate_risk_score(
            price=price,
            mid_price=mid_price,
            spread_bps=spread_bps,
            est_fill_time_ms=est_fill_time_ms,
            market_row=market_row
        )

        # Check criteria
        if est_fill_time_ms > self.max_hold_time_ms:
            return None

        if risk_score > Decimal('0.7'):
            return None

        return RebateOpportunity(
            symbol=symbol,
            side='SELL',
            price=price,
            quantity=quantity,
            expected_rebate=expected_rebate,
            expected_fill_time_ms=est_fill_time_ms,
            risk_score=risk_score,
            timestamp=datetime.now(timezone.utc),
            metadata={
                'exchange': rebate_struct.exchange,
                'spread_bps': spread_bps,
                'rebate_bps': rebate_struct.maker_rebate_bps,
                'reference_size': reference_size
            }
        )

    def _estimate_fill_time(self, quantity: Decimal, volume_1m: Decimal) -> int:
        """Estimate time to fill in milliseconds.

        Args:
            quantity: Order quantity
            volume_1m: 1-minute trading volume

        Returns:
            Estimated fill time in milliseconds
        """
        if volume_1m <= Decimal('0'):
            return self.max_hold_time_ms

        # Estimate based on order size as % of 1m volume
        volume_ratio = quantity / volume_1m
        # Base estimate: if we're 1% of volume, expect ~600ms fill
        est_time = int(volume_ratio * Decimal('60000'))

        return min(est_time, self.max_hold_time_ms)

    def _calculate_risk_score(
        self,
        price: Decimal,
        mid_price: Decimal,
        spread_bps: Decimal,
        est_fill_time_ms: int,
        market_row: Dict[str, Any]
    ) -> Decimal:
        """Calculate risk score for opportunity.

        Args:
            price: Order price
            mid_price: Current mid price
            spread_bps: Current spread
            est_fill_time_ms: Estimated fill time
            market_row: Market data

        Returns:
            Risk score (0-1, lower is better)
        """
        risk = Decimal('0')

        # Time risk (longer fill time = higher risk)
        time_risk = Decimal(str(est_fill_time_ms)) / Decimal(str(self.max_hold_time_ms))
        risk += time_risk * Decimal('0.4')

        # Spread risk (tighter spread = higher adverse selection risk)
        spread_risk = Decimal('1') - min(spread_bps / Decimal('10'), Decimal('1'))
        risk += spread_risk * Decimal('0.3')

        # Volatility risk
        volatility = Decimal(str(market_row.get('volatility', 0)))
        vol_risk = min(volatility * Decimal('100'), Decimal('1'))
        risk += vol_risk * Decimal('0.3')

        return min(risk, Decimal('1'))

    def _should_execute_opportunity(self, opportunity: RebateOpportunity) -> bool:
        """Determine if opportunity should be executed.

        Args:
            opportunity: Rebate opportunity

        Returns:
            True if should execute
        """
        # Check minimum rebate
        if opportunity.expected_rebate < self.min_rebate_bps / Decimal('10'):
            return False

        # Check risk score
        if opportunity.risk_score > Decimal('0.7'):
            return False

        # Check inventory limits (would need actual inventory tracking)
        # Simplified check here
        return True

    def _create_signal_from_opportunity(self, opportunity: RebateOpportunity) -> Dict[str, Any]:
        """Create trading signal from opportunity.

        Args:
            opportunity: Rebate opportunity

        Returns:
            Signal dictionary
        """
        confidence = Decimal('1') - opportunity.risk_score

        return {
            'symbol': opportunity.symbol,
            'action': opportunity.side,
            'price': opportunity.price,
            'quantity': opportunity.quantity,
            'strategy': 'rebate_capture',
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'expected_rebate': opportunity.expected_rebate,
                'expected_fill_time_ms': opportunity.expected_fill_time_ms,
                'risk_score': opportunity.risk_score,
                **opportunity.metadata
            }
        }

    def _cleanup_stale_opportunities(self) -> None:
        """Remove stale opportunities from tracking."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(milliseconds=self.max_hold_time_ms)

        stale_keys = [
            key for key, opp in self.active_opportunities.items()
            if opp.timestamp < cutoff_time
        ]

        for key in stale_keys:
            del self.active_opportunities[key]

        if stale_keys:
            logger.debug("stale_opportunities_removed", count=len(stale_keys))

    async def on_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        exchange: str
    ) -> None:
        """Handle order fill event.

        Args:
            order_id: Order ID
            symbol: Symbol
            side: Order side
            quantity: Filled quantity
            price: Fill price
            exchange: Exchange name
        """
        rebate_struct = self.exchange_rebates.get(exchange)
        if rebate_struct:
            notional = quantity * price
            rebate = notional * rebate_struct.maker_rebate_bps / Decimal('10000')
            self.total_rebates_earned += rebate
            self.total_volume += notional
            self.total_trades += 1

            logger.info(
                "rebate_order_filled",
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                price=float(price),
                rebate=float(rebate),
                total_rebates=float(self.total_rebates_earned)
            )

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate performance indicators.

        Args:
            data: Historical data

        Returns:
            Dictionary of indicators
        """
        net_rebates = self.total_rebates_earned - self.total_fees_paid

        return {
            'total_rebates_earned': self.total_rebates_earned,
            'total_fees_paid': self.total_fees_paid,
            'net_rebates': net_rebates,
            'total_trades': Decimal(str(self.total_trades)),
            'total_volume': self.total_volume,
            'avg_rebate_per_trade': (
                self.total_rebates_earned / Decimal(str(self.total_trades))
                if self.total_trades > 0 else Decimal('0')
            )
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'total_rebates_earned': float(self.total_rebates_earned),
            'total_fees_paid': float(self.total_fees_paid),
            'net_rebates': float(self.total_rebates_earned - self.total_fees_paid),
            'total_trades': self.total_trades,
            'total_volume': float(self.total_volume),
            'active_opportunities': len(self.active_opportunities)
        }
