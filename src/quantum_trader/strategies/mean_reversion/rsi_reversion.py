"""RSI Mean Reversion Trading Strategy.

Identifies overbought/oversold conditions using RSI indicator and trades
mean reversion moves back to equilibrium levels.

Performance Target: High win rate (>60%), moderate frequency (50-200 trades/day)
Capital Allocation: Configurable via config
Risk: Mean reversion risk, trend continuation
"""

import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

import polars as pl
import numpy as np
import talib
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class RSISignal:
    """RSI-based trading signal."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
    rsi_value: Decimal
    price: Decimal
    quantity: Decimal
    confidence: Decimal
    entry_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class RSIReversionStrategy:
    """RSI mean reversion trading strategy.

    Uses RSI indicator to identify overbought (>70) and oversold (<30)
    conditions, entering positions expecting mean reversion.

    Key Features:
    - Multi-timeframe RSI analysis
    - Dynamic threshold adjustment
    - Divergence detection
    - Trend filter integration
    - Position sizing based on RSI extremity

    Attributes:
        config: Strategy configuration
        risk_manager: Risk management instance
        rsi_period: RSI calculation period
        oversold_threshold: RSI oversold level
        overbought_threshold: RSI overbought level
        active_positions: Currently held positions

    Example:
        >>> config = load_config('strategies.yaml')['mean_reversion']
        >>> risk_mgr = RiskManager(config['risk'], portfolio)
        >>> strategy = RSIReversionStrategy(config, risk_mgr)
        >>> signals = await strategy.generate_signals(market_data)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: Any) -> None:
        """Initialize RSI reversion strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.risk_manager = risk_manager
        self._validate_config()

        # RSI parameters
        self.rsi_period = config.get('rsi_period', 14)
        self.oversold_threshold = Decimal(str(config.get('oversold_threshold', 30)))
        self.overbought_threshold = Decimal(str(config.get('overbought_threshold', 70)))
        self.extreme_oversold = Decimal(str(config.get('extreme_oversold', 20)))
        self.extreme_overbought = Decimal(str(config.get('extreme_overbought', 80)))

        # Additional filters
        self.use_trend_filter = config.get('use_trend_filter', True)
        self.trend_ma_period = config.get('trend_ma_period', 50)
        self.use_divergence = config.get('use_divergence', True)
        self.min_divergence_bars = config.get('min_divergence_bars', 5)

        # Risk parameters
        self.position_size_pct = Decimal(str(config.get('position_size_percent', 1.0)))
        self.stop_loss_pct = Decimal(str(config.get('stop_loss_percent', 2.0)))
        self.take_profit_pct = Decimal(str(config.get('take_profit_percent', 3.0)))
        self.max_hold_hours = config.get('max_hold_hours', 12)

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.rsi_history: Dict[str, List[Decimal]] = {}
        self.price_history: Dict[str, List[Decimal]] = {}

        # Performance tracking
        self.total_signals: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0

        logger.info(
            "rsi_reversion_strategy_initialized",
            rsi_period=self.rsi_period,
            oversold=float(self.oversold_threshold),
            overbought=float(self.overbought_threshold)
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration invalid
        """
        required_fields = ['rsi_period', 'oversold_threshold', 'overbought_threshold']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        if not (0 < self.config['oversold_threshold'] < 50):
            raise ValueError("oversold_threshold must be between 0 and 50")

        if not (50 < self.config['overbought_threshold'] < 100):
            raise ValueError("overbought_threshold must be between 50 and 100")

    async def generate_signals(self, market_data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate RSI mean reversion signals.

        Args:
            market_data: Polars DataFrame with OHLCV data:
                - symbol: str
                - timestamp: datetime
                - open: Decimal
                - high: Decimal
                - low: Decimal
                - close: Decimal
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

            # Group by symbol
            for symbol in market_data['symbol'].unique():
                symbol_data = market_data.filter(pl.col('symbol') == symbol).sort('timestamp')

                if symbol_data.height < self.rsi_period + 10:
                    logger.debug(
                        "insufficient_data_for_rsi",
                        symbol=symbol,
                        rows=symbol_data.height,
                        required=self.rsi_period + 10
                    )
                    continue

                # Generate signals for symbol
                symbol_signals = await self._generate_symbol_signals(symbol_data)
                signals.extend(symbol_signals)

            logger.info("rsi_signals_generated", count=len(signals))
            return signals

        except Exception as e:
            logger.error(
                "signal_generation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _generate_symbol_signals(self, data: pl.DataFrame) -> List[Dict[str, Any]]:
        """Generate signals for a single symbol.

        Args:
            data: Symbol-specific OHLCV data

        Returns:
            List of signals
        """
        signals = []
        symbol = data['symbol'][0]

        # Calculate RSI
        close_prices = data['close'].to_numpy().astype(float)
        rsi_values = talib.RSI(close_prices, timeperiod=self.rsi_period)

        # Store RSI history
        if symbol not in self.rsi_history:
            self.rsi_history[symbol] = []
        self.rsi_history[symbol] = [Decimal(str(v)) for v in rsi_values[-100:]]

        # Store price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol] = [Decimal(str(p)) for p in close_prices[-100:]]

        # Get current values
        current_rsi = Decimal(str(rsi_values[-1]))
        current_price = Decimal(str(close_prices[-1]))

        # Check for existing position
        existing_position = self.active_positions.get(symbol)

        if existing_position:
            # Check exit conditions
            exit_signal = self._check_exit_conditions(
                symbol=symbol,
                current_rsi=current_rsi,
                current_price=current_price,
                position=existing_position,
                data=data
            )
            if exit_signal:
                signals.append(exit_signal)
        else:
            # Check entry conditions
            entry_signal = await self._check_entry_conditions(
                symbol=symbol,
                current_rsi=current_rsi,
                current_price=current_price,
                data=data,
                rsi_values=rsi_values
            )
            if entry_signal:
                signals.append(entry_signal)

        return signals

    async def _check_entry_conditions(
        self,
        symbol: str,
        current_rsi: Decimal,
        current_price: Decimal,
        data: pl.DataFrame,
        rsi_values: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Check for entry signal conditions.

        Args:
            symbol: Trading symbol
            current_rsi: Current RSI value
            current_price: Current price
            data: Market data
            rsi_values: RSI time series

        Returns:
            Signal dictionary or None
        """
        # Check oversold condition (potential buy)
        if current_rsi < self.oversold_threshold:
            # Apply trend filter if enabled
            if self.use_trend_filter:
                if not self._check_trend_filter(data, 'BUY'):
                    logger.debug(
                        "oversold_but_trend_unfavorable",
                        symbol=symbol,
                        rsi=float(current_rsi)
                    )
                    return None

            # Check for bullish divergence
            has_divergence = False
            if self.use_divergence:
                has_divergence = self._detect_bullish_divergence(
                    prices=self.price_history.get(symbol, []),
                    rsi_values=self.rsi_history.get(symbol, [])
                )

            # Calculate confidence (higher for extreme levels and divergence)
            confidence = self._calculate_entry_confidence(
                rsi=current_rsi,
                threshold=self.oversold_threshold,
                extreme=self.extreme_oversold,
                has_divergence=has_divergence
            )

            # Calculate position size
            balance = Decimal(str(self.config.get('account_balance', 100000)))
            quantity = self._calculate_position_size(
                price=current_price,
                balance=balance,
                confidence=confidence
            )

            # Calculate stop loss and take profit
            stop_loss = current_price * (Decimal('1') - self.stop_loss_pct / Decimal('100'))
            take_profit = current_price * (Decimal('1') + self.take_profit_pct / Decimal('100'))

            signal = {
                'symbol': symbol,
                'action': 'BUY',
                'price': current_price,
                'quantity': quantity,
                'strategy': 'rsi_reversion',
                'confidence': confidence,
                'timestamp': datetime.now(timezone.utc),
                'metadata': {
                    'rsi': current_rsi,
                    'threshold': self.oversold_threshold,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'has_divergence': has_divergence,
                    'entry_reason': 'oversold'
                }
            }

            self.total_signals += 1
            return signal

        # Check overbought condition (potential sell)
        elif current_rsi > self.overbought_threshold:
            # Apply trend filter
            if self.use_trend_filter:
                if not self._check_trend_filter(data, 'SELL'):
                    logger.debug(
                        "overbought_but_trend_unfavorable",
                        symbol=symbol,
                        rsi=float(current_rsi)
                    )
                    return None

            # Check for bearish divergence
            has_divergence = False
            if self.use_divergence:
                has_divergence = self._detect_bearish_divergence(
                    prices=self.price_history.get(symbol, []),
                    rsi_values=self.rsi_history.get(symbol, [])
                )

            # Calculate confidence
            confidence = self._calculate_entry_confidence(
                rsi=current_rsi,
                threshold=self.overbought_threshold,
                extreme=self.extreme_overbought,
                has_divergence=has_divergence
            )

            # Calculate position size
            balance = Decimal(str(self.config.get('account_balance', 100000)))
            quantity = self._calculate_position_size(
                price=current_price,
                balance=balance,
                confidence=confidence
            )

            # Calculate stop loss and take profit
            stop_loss = current_price * (Decimal('1') + self.stop_loss_pct / Decimal('100'))
            take_profit = current_price * (Decimal('1') - self.take_profit_pct / Decimal('100'))

            signal = {
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'quantity': quantity,
                'strategy': 'rsi_reversion',
                'confidence': confidence,
                'timestamp': datetime.now(timezone.utc),
                'metadata': {
                    'rsi': current_rsi,
                    'threshold': self.overbought_threshold,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'has_divergence': has_divergence,
                    'entry_reason': 'overbought'
                }
            }

            self.total_signals += 1
            return signal

        return None

    def _check_exit_conditions(
        self,
        symbol: str,
        current_rsi: Decimal,
        current_price: Decimal,
        position: Dict[str, Any],
        data: pl.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Check for position exit conditions.

        Args:
            symbol: Trading symbol
            current_rsi: Current RSI
            current_price: Current price
            position: Existing position
            data: Market data

        Returns:
            Exit signal or None
        """
        side = position['side']
        entry_price = position['entry_price']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        entry_time = position['entry_time']

        # Check stop loss
        if side == 'BUY' and current_price <= stop_loss:
            return self._create_exit_signal(symbol, 'SELL', current_price, position, 'stop_loss')
        elif side == 'SELL' and current_price >= stop_loss:
            return self._create_exit_signal(symbol, 'BUY', current_price, position, 'stop_loss')

        # Check take profit
        if side == 'BUY' and current_price >= take_profit:
            return self._create_exit_signal(symbol, 'SELL', current_price, position, 'take_profit')
        elif side == 'SELL' and current_price <= take_profit:
            return self._create_exit_signal(symbol, 'BUY', current_price, position, 'take_profit')

        # Check RSI mean reversion
        if side == 'BUY' and current_rsi >= Decimal('50'):
            return self._create_exit_signal(symbol, 'SELL', current_price, position, 'rsi_mean_reversion')
        elif side == 'SELL' and current_rsi <= Decimal('50'):
            return self._create_exit_signal(symbol, 'BUY', current_price, position, 'rsi_mean_reversion')

        # Check time-based exit
        if datetime.now(timezone.utc) - entry_time > timedelta(hours=self.max_hold_hours):
            exit_side = 'SELL' if side == 'BUY' else 'BUY'
            return self._create_exit_signal(symbol, exit_side, current_price, position, 'max_hold_time')

        return None

    def _create_exit_signal(
        self,
        symbol: str,
        action: str,
        price: Decimal,
        position: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """Create exit signal.

        Args:
            symbol: Trading symbol
            action: Exit action
            price: Current price
            position: Position to exit
            reason: Exit reason

        Returns:
            Exit signal dictionary
        """
        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']

        if position['side'] == 'BUY':
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity

        # Track win/loss
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        return {
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'strategy': 'rsi_reversion',
            'confidence': Decimal('0.9'),
            'timestamp': datetime.now(timezone.utc),
            'metadata': {
                'exit_reason': reason,
                'pnl': pnl,
                'entry_price': entry_price,
                'holding_time': (datetime.now(timezone.utc) - position['entry_time']).total_seconds()
            }
        }

    def _check_trend_filter(self, data: pl.DataFrame, action: str) -> bool:
        """Check if trend supports the action.

        Args:
            data: Market data
            action: Proposed action ('BUY' or 'SELL')

        Returns:
            True if trend is favorable
        """
        close_prices = data['close'].to_numpy().astype(float)

        if len(close_prices) < self.trend_ma_period:
            return True  # Not enough data, allow trade

        ma = talib.SMA(close_prices, timeperiod=self.trend_ma_period)
        current_price = close_prices[-1]
        current_ma = ma[-1]

        if action == 'BUY':
            return current_price > current_ma  # Only buy in uptrend
        else:
            return current_price < current_ma  # Only sell in downtrend

    def _detect_bullish_divergence(
        self,
        prices: List[Decimal],
        rsi_values: List[Decimal]
    ) -> bool:
        """Detect bullish RSI divergence.

        Args:
            prices: Price history
            rsi_values: RSI history

        Returns:
            True if bullish divergence detected
        """
        if len(prices) < self.min_divergence_bars or len(rsi_values) < self.min_divergence_bars:
            return False

        # Look for lower price lows but higher RSI lows
        recent_prices = prices[-self.min_divergence_bars:]
        recent_rsi = rsi_values[-self.min_divergence_bars:]

        price_trend = recent_prices[-1] < recent_prices[0]
        rsi_trend = recent_rsi[-1] > recent_rsi[0]

        return price_trend and rsi_trend

    def _detect_bearish_divergence(
        self,
        prices: List[Decimal],
        rsi_values: List[Decimal]
    ) -> bool:
        """Detect bearish RSI divergence.

        Args:
            prices: Price history
            rsi_values: RSI history

        Returns:
            True if bearish divergence detected
        """
        if len(prices) < self.min_divergence_bars or len(rsi_values) < self.min_divergence_bars:
            return False

        # Look for higher price highs but lower RSI highs
        recent_prices = prices[-self.min_divergence_bars:]
        recent_rsi = rsi_values[-self.min_divergence_bars:]

        price_trend = recent_prices[-1] > recent_prices[0]
        rsi_trend = recent_rsi[-1] < recent_rsi[0]

        return price_trend and rsi_trend

    def _calculate_entry_confidence(
        self,
        rsi: Decimal,
        threshold: Decimal,
        extreme: Decimal,
        has_divergence: bool
    ) -> Decimal:
        """Calculate entry confidence score.

        Args:
            rsi: Current RSI value
            threshold: RSI threshold (30 or 70)
            extreme: Extreme RSI level (20 or 80)
            has_divergence: Whether divergence detected

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from RSI distance to threshold
        if threshold < Decimal('50'):  # Oversold
            distance = threshold - rsi
            max_distance = threshold - extreme
        else:  # Overbought
            distance = rsi - threshold
            max_distance = extreme - threshold

        rsi_confidence = min(distance / max_distance, Decimal('1')) * Decimal('0.7')

        # Boost for divergence
        divergence_boost = Decimal('0.3') if has_divergence else Decimal('0')

        total_confidence = rsi_confidence + divergence_boost
        return min(total_confidence, Decimal('1'))

    def _calculate_position_size(
        self,
        price: Decimal,
        balance: Decimal,
        confidence: Decimal
    ) -> Decimal:
        """Calculate position size.

        Args:
            price: Entry price
            balance: Account balance
            confidence: Signal confidence

        Returns:
            Position quantity
        """
        # Base position size as percentage of balance
        base_value = balance * self.position_size_pct / Decimal('100')

        # Scale by confidence
        adjusted_value = base_value * confidence

        # Convert to quantity
        quantity = adjusted_value / price

        return quantity.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate strategy indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (
            Decimal(str(self.winning_trades)) / Decimal(str(total_trades))
            if total_trades > 0 else Decimal('0')
        )

        return {
            'total_signals': Decimal(str(self.total_signals)),
            'winning_trades': Decimal(str(self.winning_trades)),
            'losing_trades': Decimal(str(self.losing_trades)),
            'win_rate': win_rate,
            'active_positions': Decimal(str(len(self.active_positions)))
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Performance metrics dictionary
        """
        total_trades = self.winning_trades + self.losing_trades

        return {
            'total_signals': self.total_signals,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / total_trades if total_trades > 0 else 0,
            'active_positions': len(self.active_positions)
        }
