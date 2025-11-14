"""Scalping High-Frequency Trading Strategy.

Ultra-short-term trading capturing small price movements with high frequency.
Targets small profits per trade with very high win rate and volume.

Performance Target: <5ms latency, 2000+ trades/day, 70%+ win rate
Capital Allocation: Configurable via config
Risk: Minimal per-trade, high transaction costs
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque

import polars as pl
import numpy as np
import talib
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ScalpingSignal:
    """Scalping trade signal."""
    symbol: str
    action: str
    entry_price: Decimal
    quantity: Decimal
    take_profit: Decimal
    stop_loss: Decimal
    expected_profit_bps: Decimal
    confidence: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMicrostructure:
    """Market microstructure analysis."""
    symbol: str
    bid_ask_spread_bps: Decimal
    order_flow_imbalance: Decimal  # -1 to 1
    volume_imbalance: Decimal
    momentum_score: Decimal
    volatility_regime: str  # 'low', 'medium', 'high'
    timestamp: datetime


class ScalpingStrategy:
    """High-frequency scalping trading strategy.

    Captures small price movements through rapid trading with:
    - Tight spreads (1-5 bps profit targets)
    - Sub-second holding periods
    - High win rate focus
    - Volume and order flow analysis
    - Microstructure edge detection

    Key Features:
    - Ultra-fast execution
    - Order flow analysis
    - Spread capture
    - Momentum micro-trends
    - Statistical edge detection

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        profit_target_bps: Target profit in basis points
        max_holding_time_ms: Maximum position hold time
        active_scalps: Currently active scalp trades

    Example:
        >>> config = load_config('strategies.yaml')['high_frequency']['scalping']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = ScalpingStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(market_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize scalping strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # Scalping parameters
        self.profit_target_bps = Decimal(str(config.get('profit_target_bps', 3)))
        self.stop_loss_bps = Decimal(str(config.get('stop_loss_bps', 2)))
        self.max_holding_time_ms = config.get('max_holding_time_ms', 2000)
        self.min_spread_bps = Decimal(str(config.get('min_spread_bps', 1)))
        self.max_spread_bps = Decimal(str(config.get('max_spread_bps', 8)))

        # Signal filters
        self.min_volume_ratio = Decimal(str(config.get('min_volume_ratio', 1.2)))
        self.min_momentum_score = Decimal(str(config.get('min_momentum_score', 0.6)))
        self.use_order_flow = config.get('use_order_flow_filter', True)
        self.order_flow_threshold = Decimal(str(config.get('order_flow_threshold', 0.3)))

        # Position sizing
        self.base_position_size = Decimal(str(config.get('base_position_size', 1000)))
        self.max_position_size = Decimal(str(config.get('max_position_size', 5000)))
        self.size_volatility_adjustment = config.get('size_volatility_adjustment', True)

        # State tracking
        self.active_scalps: Dict[str, Dict[str, Any]] = {}
        self.completed_scalps: deque = deque(maxlen=1000)
        self.market_microstructure: Dict[str, MarketMicrostructure] = {}

        # Performance tracking
        self.total_scalps: int = 0
        self.winning_scalps: int = 0
        self.total_profit: Decimal = Decimal('0')
        self.total_loss: Decimal = Decimal('0')

        logger.info(
            "scalping_strategy_initialized",
            profit_target_bps=float(self.profit_target_bps),
            stop_loss_bps=float(self.stop_loss_bps),
            max_holding_time_ms=self.max_holding_time_ms
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = [
            'profit_target_bps',
            'stop_loss_bps',
            'max_holding_time_ms'
        ]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if self.config['profit_target_bps'] <= 0:
            raise ValueError("profit_target_bps must be positive")

        if self.config['stop_loss_bps'] <= 0:
            raise ValueError("stop_loss_bps must be positive")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate scalping signals.

        Args:
            market_data: Polars DataFrame with tick data:
                - symbol: str
                - timestamp: datetime
                - bid_price: Decimal
                - ask_price: Decimal
                - bid_size: Decimal
                - ask_size: Decimal
                - last_price: Decimal
                - last_size: Decimal
                - volume: Decimal

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

            # Cleanup expired scalps
            self._cleanup_expired_scalps()

            # Process each symbol
            for symbol in market_data['symbol'].unique():
                symbol_data = market_data.filter(pl.col('symbol') == symbol)

                # Analyze microstructure
                microstructure = self._analyze_microstructure(symbol_data)
                self.market_microstructure[symbol] = microstructure

                # Check if market conditions suitable for scalping
                if not self._is_scalpable_market(microstructure):
                    logger.debug(
                        "market_not_scalpable",
                        symbol=symbol,
                        spread_bps=float(microstructure.bid_ask_spread_bps)
                    )
                    continue

                # Generate scalping signals
                symbol_signals = self._generate_scalp_signals(symbol, symbol_data, microstructure)
                signals.extend(symbol_signals)

            logger.info(
                "scalping_signals_generated",
                count=len(signals),
                active_scalps=len(self.active_scalps)
            )

            return signals

        except Exception as e:
            logger.error(
                "signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _analyze_microstructure(self, data: pl.DataFrame) -> MarketMicrostructure:
        """Analyze market microstructure.

        Args:
            data: Market tick data

        Returns:
            MarketMicrostructure analysis
        """
        symbol = data['symbol'][0]

        # Calculate bid-ask spread
        latest = data[-1]
        bid = Decimal(str(latest['bid_price'][0]))
        ask = Decimal(str(latest['ask_price'][0]))
        mid = (bid + ask) / Decimal('2')

        spread_bps = ((ask - bid) / mid) * Decimal('10000')

        # Calculate order flow imbalance
        bid_size = Decimal(str(latest['bid_size'][0]))
        ask_size = Decimal(str(latest['ask_size'][0]))
        total_size = bid_size + ask_size

        order_flow_imbalance = (
            (bid_size - ask_size) / total_size
            if total_size > 0 else Decimal('0')
        )

        # Calculate volume imbalance
        if data.height >= 10:
            recent_trades = data[-10:]
            buy_volume = Decimal('0')
            sell_volume = Decimal('0')

            for row in recent_trades.iter_rows(named=True):
                size = Decimal(str(row.get('last_size', 0)))
                # Classify trade as buy/sell based on price vs mid
                price = Decimal(str(row.get('last_price', mid)))
                if price >= mid:
                    buy_volume += size
                else:
                    sell_volume += size

            total_volume = buy_volume + sell_volume
            volume_imbalance = (
                (buy_volume - sell_volume) / total_volume
                if total_volume > 0 else Decimal('0')
            )
        else:
            volume_imbalance = Decimal('0')

        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(data)

        # Determine volatility regime
        volatility_regime = self._classify_volatility_regime(data, spread_bps)

        return MarketMicrostructure(
            symbol=symbol,
            bid_ask_spread_bps=spread_bps,
            order_flow_imbalance=order_flow_imbalance,
            volume_imbalance=volume_imbalance,
            momentum_score=momentum_score,
            volatility_regime=volatility_regime,
            timestamp=datetime.now(timezone.utc)
        )

    def _calculate_momentum_score(self, data: pl.DataFrame) -> Decimal:
        """Calculate short-term momentum score.

        Args:
            data: Market data

        Returns:
            Momentum score (0-1)
        """
        if data.height < 5:
            return Decimal('0.5')

        prices = data['last_price'].to_numpy().astype(float)

        # Simple momentum: price change over last 5 ticks
        price_change = (prices[-1] - prices[-5]) / prices[-5]

        # Normalize to 0-1 range
        # Assume +/- 0.1% is full momentum
        normalized = (Decimal(str(price_change)) * Decimal('1000') + Decimal('1')) / Decimal('2')

        return max(Decimal('0'), min(Decimal('1'), normalized))

    def _classify_volatility_regime(self, data: pl.DataFrame, spread_bps: Decimal) -> str:
        """Classify current volatility regime.

        Args:
            data: Market data
            spread_bps: Current spread

        Returns:
            Volatility regime: 'low', 'medium', 'high'
        """
        # Use spread as proxy for volatility
        if spread_bps < Decimal('3'):
            return 'low'
        elif spread_bps < Decimal('6'):
            return 'medium'
        else:
            return 'high'

    def _is_scalpable_market(self, microstructure: MarketMicrostructure) -> bool:
        """Check if market conditions suitable for scalping.

        Args:
            microstructure: Market microstructure analysis

        Returns:
            True if market is scalpable
        """
        # Spread must be within acceptable range
        if microstructure.bid_ask_spread_bps < self.min_spread_bps:
            return False  # Too tight, no edge

        if microstructure.bid_ask_spread_bps > self.max_spread_bps:
            return False  # Too wide, too risky

        # Avoid high volatility regimes
        if microstructure.volatility_regime == 'high':
            return False

        return True

    def _generate_scalp_signals(
        self,
        symbol: str,
        data: pl.DataFrame,
        microstructure: MarketMicrostructure
    ) -> List[Dict[str, Any]]:
        """Generate scalp signals for symbol.

        Args:
            symbol: Trading symbol
            data: Market data
            microstructure: Microstructure analysis

        Returns:
            List of signals
        """
        signals = []

        # Don't open new scalp if already have active position
        if symbol in self.active_scalps:
            return signals

        latest = data[-1]
        bid = Decimal(str(latest['bid_price'][0]))
        ask = Decimal(str(latest['ask_price'][0]))
        mid = (bid + ask) / Decimal('2')

        # Check for buy signal (positive order flow/volume imbalance)
        if self.use_order_flow:
            if (microstructure.order_flow_imbalance > self.order_flow_threshold and
                microstructure.volume_imbalance > self.order_flow_threshold):

                signal = self._create_scalp_signal(
                    symbol=symbol,
                    action='BUY',
                    entry_price=ask,  # Take liquidity
                    microstructure=microstructure
                )
                if signal:
                    signals.append(signal)
                    return signals  # One signal per symbol

        # Check for sell signal (negative order flow/volume imbalance)
        if self.use_order_flow:
            if (microstructure.order_flow_imbalance < -self.order_flow_threshold and
                microstructure.volume_imbalance < -self.order_flow_threshold):

                signal = self._create_scalp_signal(
                    symbol=symbol,
                    action='SELL',
                    entry_price=bid,  # Take liquidity
                    microstructure=microstructure
                )
                if signal:
                    signals.append(signal)

        return signals

    def _create_scalp_signal(
        self,
        symbol: str,
        action: str,
        entry_price: Decimal,
        microstructure: MarketMicrostructure
    ) -> Optional[Dict[str, Any]]:
        """Create scalp signal.

        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            entry_price: Entry price
            microstructure: Market microstructure

        Returns:
            Signal dictionary or None
        """
        # Calculate take profit and stop loss
        if action == 'BUY':
            take_profit = entry_price * (Decimal('1') + self.profit_target_bps / Decimal('10000'))
            stop_loss = entry_price * (Decimal('1') - self.stop_loss_bps / Decimal('10000'))
        else:
            take_profit = entry_price * (Decimal('1') - self.profit_target_bps / Decimal('10000'))
            stop_loss = entry_price * (Decimal('1') + self.stop_loss_bps / Decimal('10000'))

        # Calculate position size
        quantity = self._calculate_scalp_size(entry_price, microstructure)

        # Calculate confidence
        confidence = self._calculate_scalp_confidence(microstructure)

        # Calculate expected profit
        expected_profit_bps = self.profit_target_bps

        signal = {
            'symbol': symbol,
            'action': action,
            'price': entry_price,
            'quantity': quantity,
            'strategy': 'scalping',
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'expected_profit_bps': expected_profit_bps,
                'order_flow_imbalance': microstructure.order_flow_imbalance,
                'volume_imbalance': microstructure.volume_imbalance,
                'spread_bps': microstructure.bid_ask_spread_bps,
                'max_holding_time_ms': self.max_holding_time_ms
            }
        }

        self.total_scalps += 1

        # Track active scalp
        self.active_scalps[symbol] = {
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'quantity': quantity,
            'side': action,
            'entry_time': datetime.now(timezone.utc)
        }

        return signal

    def _calculate_scalp_size(
        self,
        price: Decimal,
        microstructure: MarketMicrostructure
    ) -> Decimal:
        """Calculate scalp position size.

        Args:
            price: Entry price
            microstructure: Market microstructure

        Returns:
            Position quantity
        """
        size = self.base_position_size

        # Adjust for volatility
        if self.size_volatility_adjustment:
            if microstructure.volatility_regime == 'low':
                size = size * Decimal('1.2')
            elif microstructure.volatility_regime == 'high':
                size = size * Decimal('0.8')

        # Cap at max
        size = min(size, self.max_position_size)

        return size

    def _calculate_scalp_confidence(self, microstructure: MarketMicrostructure) -> Decimal:
        """Calculate scalp signal confidence.

        Args:
            microstructure: Market microstructure

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal('0.5')

        # Boost for strong order flow
        if abs(microstructure.order_flow_imbalance) > Decimal('0.5'):
            confidence += Decimal('0.2')

        # Boost for aligned volume
        if abs(microstructure.volume_imbalance) > Decimal('0.5'):
            confidence += Decimal('0.2')

        # Boost for good momentum
        if microstructure.momentum_score > Decimal('0.7'):
            confidence += Decimal('0.1')

        return min(confidence, Decimal('1'))

    def _cleanup_expired_scalps(self) -> None:
        """Remove expired scalps from tracking."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(milliseconds=self.max_holding_time_ms)

        expired = [
            symbol for symbol, scalp in self.active_scalps.items()
            if scalp['entry_time'] < cutoff_time
        ]

        for symbol in expired:
            del self.active_scalps[symbol]

        if expired:
            logger.debug("expired_scalps_removed", count=len(expired))

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate performance indicators.

        Args:
            data: Historical data

        Returns:
            Dictionary of indicators
        """
        win_rate = (
            Decimal(str(self.winning_scalps)) / Decimal(str(self.total_scalps))
            if self.total_scalps > 0 else Decimal('0')
        )

        avg_profit = (
            self.total_profit / Decimal(str(self.winning_scalps))
            if self.winning_scalps > 0 else Decimal('0')
        )

        avg_loss = (
            self.total_loss / Decimal(str(self.total_scalps - self.winning_scalps))
            if (self.total_scalps - self.winning_scalps) > 0 else Decimal('0')
        )

        return {
            'total_scalps': Decimal(str(self.total_scalps)),
            'winning_scalps': Decimal(str(self.winning_scalps)),
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'total_scalps': self.total_scalps,
            'winning_scalps': self.winning_scalps,
            'win_rate': self.winning_scalps / self.total_scalps if self.total_scalps > 0 else 0,
            'total_profit': float(self.total_profit),
            'total_loss': float(self.total_loss),
            'active_scalps': len(self.active_scalps)
        }
