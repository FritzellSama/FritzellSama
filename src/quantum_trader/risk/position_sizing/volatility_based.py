"""
Quantum Trader AI - Volatility-Based Position Sizing
Production-grade volatility-adjusted position sizing

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import polars as pl
import numpy as np

from quantum_trader.models import Position
from quantum_trader.database.timeseries import TimeSeriesDB


logger = logging.getLogger(__name__)


class VolatilityBasedSizer:
    """
    Volatility-based position sizing system

    Features:
    - ATR-based sizing
    - Realized volatility adjustment
    - Volatility regime detection
    - Vol-normalized position limits
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize volatility-based sizer with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Position limits configuration
        position_limits = self.config.get("position_limits", {})
        self.max_position_size_percent = Decimal(str(position_limits.get("max_position_size_percent", 5.0)))
        self.max_leverage = Decimal(str(position_limits.get("max_leverage", 2.0)))
        self.min_position_size_usd = Decimal(str(position_limits.get("min_position_size_usd", 500)))

        # Market conditions configuration
        market_conditions = self.config.get("market_conditions", {})
        self.reduce_on_high_volatility = market_conditions.get("reduce_on_high_volatility", True)
        self.volatility_vix_threshold = Decimal(str(market_conditions.get("volatility_vix_threshold", 30.0)))
        self.volatility_position_multiplier = Decimal(str(market_conditions.get("volatility_position_multiplier", 0.5)))

        # Volatility calculation parameters
        self.atr_period = 14  # Standard ATR period
        self.volatility_window = 30  # Days for realized volatility
        self.volatility_long_window = 90  # Long-term volatility reference
        self.target_portfolio_volatility = Decimal('0.15')  # 15% annualized
        self.trading_days_per_year = Decimal('252')

        # Volatility regime thresholds
        self.low_vol_threshold = Decimal('0.10')  # 10% annualized
        self.high_vol_threshold = Decimal('0.25')  # 25% annualized

        # Position sizing parameters
        self.risk_per_trade = Decimal('0.02')  # 2% risk per trade
        self.vol_scalar = Decimal('1.0')  # Volatility scaling factor

        # Database connection
        self.db: Optional[TimeSeriesDB] = None

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("VolatilityBasedSizer initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()
            logger.info("VolatilityBasedSizer initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize VolatilityBasedSizer: {e}")
            raise

    async def calculate_atr(
        self,
        symbol: str,
        period: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Average True Range (ATR)

        ATR measures market volatility by decomposing the entire range of an asset.

        Args:
            symbol: Trading symbol
            period: ATR period (default from config)

        Returns:
            ATR value as Decimal
        """
        for attempt in range(self.retry_attempts):
            try:
                if not self.db:
                    raise RuntimeError("Database not initialized")

                atr_period = period if period is not None else self.atr_period

                # Fetch OHLC data
                lookback_date = datetime.utcnow() - timedelta(days=atr_period + 10)
                ohlc_df = await self.db.query_ohlcv(
                    symbol=symbol,
                    start_date=lookback_date,
                    end_date=datetime.utcnow(),
                    timeframe="1d"
                )

                if len(ohlc_df) < atr_period:
                    logger.warning(f"Insufficient data for ATR calculation: {len(ohlc_df)} < {atr_period}")
                    return Decimal('0')

                # Calculate True Range
                high = ohlc_df["high"].to_numpy()
                low = ohlc_df["low"].to_numpy()
                close = ohlc_df["close"].to_numpy()

                # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                tr_list = []
                for i in range(1, len(ohlc_df)):
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close[i-1])
                    tr3 = abs(low[i] - close[i-1])
                    tr = max(tr1, tr2, tr3)
                    tr_list.append(tr)

                # Calculate ATR as moving average of True Range
                if len(tr_list) >= atr_period:
                    atr_values = []
                    for i in range(atr_period - 1, len(tr_list)):
                        atr = np.mean(tr_list[i - atr_period + 1:i + 1])
                        atr_values.append(atr)

                    # Return most recent ATR
                    current_atr = Decimal(str(atr_values[-1]))
                    logger.info(f"ATR for {symbol}: {current_atr}")
                    return current_atr
                else:
                    return Decimal('0')

            except Exception as e:
                logger.error(f"Error calculating ATR (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay_ms / 1000)
                else:
                    return Decimal('0')

    async def calculate_realized_volatility(
        self,
        symbol: str,
        window: Optional[int] = None
    ) -> Decimal:
        """
        Calculate realized (historical) volatility

        Args:
            symbol: Trading symbol
            window: Lookback window in days

        Returns:
            Annualized volatility as Decimal
        """
        for attempt in range(self.retry_attempts):
            try:
                if not self.db:
                    raise RuntimeError("Database not initialized")

                vol_window = window if window is not None else self.volatility_window

                # Fetch price data
                lookback_date = datetime.utcnow() - timedelta(days=vol_window + 10)
                price_df = await self.db.query_ohlcv(
                    symbol=symbol,
                    start_date=lookback_date,
                    end_date=datetime.utcnow(),
                    timeframe="1d"
                )

                if len(price_df) < vol_window:
                    logger.warning(f"Insufficient data for volatility calculation: {len(price_df)} < {vol_window}")
                    return Decimal('0')

                # Calculate log returns
                close_prices = price_df["close"].to_numpy()
                log_returns = np.log(close_prices[1:] / close_prices[:-1])

                # Calculate standard deviation
                std_dev = np.std(log_returns, ddof=1)

                # Annualize: multiply by sqrt(252)
                annualized_vol = Decimal(str(std_dev)) * self.trading_days_per_year.sqrt()

                logger.info(f"Realized volatility for {symbol}: {annualized_vol:.4f}")
                return annualized_vol

            except Exception as e:
                logger.error(f"Error calculating realized volatility (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay_ms / 1000)
                else:
                    return Decimal('0')

    async def detect_volatility_regime(
        self,
        symbol: str
    ) -> str:
        """
        Detect current volatility regime

        Args:
            symbol: Trading symbol

        Returns:
            Regime string: 'LOW', 'NORMAL', 'HIGH', 'EXTREME'
        """
        try:
            # Calculate short-term and long-term volatility
            short_vol = await self.calculate_realized_volatility(symbol, self.volatility_window)
            long_vol = await self.calculate_realized_volatility(symbol, self.volatility_long_window)

            if short_vol == Decimal('0') or long_vol == Decimal('0'):
                return 'UNKNOWN'

            # Calculate volatility ratio
            vol_ratio = short_vol / long_vol

            # Classify regime
            if short_vol < self.low_vol_threshold:
                regime = 'LOW'
            elif short_vol > self.high_vol_threshold:
                if vol_ratio > Decimal('1.5'):
                    regime = 'EXTREME'  # Volatility spike
                else:
                    regime = 'HIGH'
            else:
                regime = 'NORMAL'

            logger.info(f"Volatility regime for {symbol}: {regime} (vol: {short_vol:.4f}, ratio: {vol_ratio:.2f})")
            return regime

        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return 'UNKNOWN'

    async def calculate_atr_position_size(
        self,
        symbol: str,
        account_value: Decimal,
        entry_price: Decimal,
        stop_loss_distance: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size based on ATR

        Position Size = (Account Value * Risk %) / (ATR * ATR Multiplier)

        Args:
            symbol: Trading symbol
            account_value: Account value
            entry_price: Entry price
            stop_loss_distance: Custom stop loss distance (uses ATR if None)

        Returns:
            Position size in USD
        """
        try:
            # Calculate ATR
            atr = await self.calculate_atr(symbol)

            if atr == Decimal('0'):
                logger.warning(f"ATR is zero for {symbol}, using fallback sizing")
                return self._calculate_fallback_size(account_value)

            # Determine stop loss distance
            if stop_loss_distance is not None:
                stop_distance = stop_loss_distance
            else:
                # Use 2x ATR as stop loss distance
                stop_distance = atr * Decimal('2')

            # Calculate position size
            # Risk amount = account_value * risk_per_trade
            risk_amount = account_value * self.risk_per_trade

            # Position size = risk_amount / stop_distance
            position_size = (risk_amount / stop_distance) * entry_price

            # Apply constraints
            position_size = self._apply_size_constraints(position_size, account_value)

            logger.info(f"ATR-based position size for {symbol}: ${position_size}")
            return position_size

        except Exception as e:
            logger.error(f"Error calculating ATR position size: {e}")
            return self._calculate_fallback_size(account_value)

    async def calculate_volatility_adjusted_size(
        self,
        symbol: str,
        account_value: Decimal,
        base_position_size: Decimal
    ) -> Decimal:
        """
        Adjust position size based on current volatility

        Args:
            symbol: Trading symbol
            account_value: Account value
            base_position_size: Base position size before adjustment

        Returns:
            Adjusted position size
        """
        try:
            # Get volatility regime
            regime = await self.detect_volatility_regime(symbol)

            # Calculate realized volatility
            current_vol = await self.calculate_realized_volatility(symbol)

            if current_vol == Decimal('0'):
                return base_position_size

            # Target volatility scaling
            # If current vol > target, reduce size; if current vol < target, increase size
            vol_scalar = self.target_portfolio_volatility / current_vol

            # Clamp scaling factor
            vol_scalar = min(max(vol_scalar, Decimal('0.5')), Decimal('2.0'))

            # Apply regime-based adjustments
            if regime == 'LOW':
                regime_multiplier = Decimal('1.2')  # Increase size in low vol
            elif regime == 'HIGH':
                regime_multiplier = Decimal('0.8')  # Reduce size in high vol
            elif regime == 'EXTREME':
                regime_multiplier = Decimal('0.5')  # Significantly reduce in extreme vol
            else:
                regime_multiplier = Decimal('1.0')

            # Calculate adjusted size
            adjusted_size = base_position_size * vol_scalar * regime_multiplier

            # Apply constraints
            adjusted_size = self._apply_size_constraints(adjusted_size, account_value)

            logger.info(f"Volatility-adjusted size for {symbol}: ${adjusted_size} (regime: {regime}, scalar: {vol_scalar:.2f})")
            return adjusted_size

        except Exception as e:
            logger.error(f"Error calculating volatility-adjusted size: {e}")
            return base_position_size

    async def calculate_volatility_normalized_size(
        self,
        symbol: str,
        account_value: Decimal,
        target_volatility: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate position size normalized to target volatility

        Position Size = (Account Value * Target Vol) / (Asset Vol)

        Args:
            symbol: Trading symbol
            account_value: Account value
            target_volatility: Target portfolio volatility (uses config default if None)

        Returns:
            Volatility-normalized position size
        """
        try:
            target_vol = target_volatility if target_volatility is not None else self.target_portfolio_volatility

            # Calculate asset volatility
            asset_vol = await self.calculate_realized_volatility(symbol)

            if asset_vol == Decimal('0'):
                logger.warning(f"Asset volatility is zero for {symbol}")
                return self._calculate_fallback_size(account_value)

            # Normalize position size
            # This ensures each position contributes equally to portfolio volatility
            normalized_size = (account_value * target_vol) / asset_vol

            # Apply constraints
            normalized_size = self._apply_size_constraints(normalized_size, account_value)

            logger.info(f"Volatility-normalized size for {symbol}: ${normalized_size}")
            return normalized_size

        except Exception as e:
            logger.error(f"Error calculating volatility-normalized size: {e}")
            return self._calculate_fallback_size(account_value)

    def _apply_size_constraints(
        self,
        position_size: Decimal,
        account_value: Decimal
    ) -> Decimal:
        """Apply position size constraints"""
        # Apply minimum size
        if position_size < self.min_position_size_usd:
            return Decimal('0')

        # Apply maximum size (percentage of account)
        max_size = account_value * (self.max_position_size_percent / Decimal('100'))
        if position_size > max_size:
            position_size = max_size

        # Apply leverage limit
        max_leveraged_size = account_value * self.max_leverage
        if position_size > max_leveraged_size:
            position_size = max_leveraged_size

        return position_size

    def _calculate_fallback_size(self, account_value: Decimal) -> Decimal:
        """Calculate fallback position size when volatility data unavailable"""
        # Use fixed percentage of account
        fallback_size = account_value * self.risk_per_trade

        # Apply constraints
        return self._apply_size_constraints(fallback_size, account_value)

    async def calculate_portfolio_volatility(
        self,
        positions: List[Position]
    ) -> Decimal:
        """
        Calculate current portfolio volatility

        Args:
            positions: List of current positions

        Returns:
            Portfolio volatility (annualized)
        """
        try:
            if not positions:
                return Decimal('0')

            # Calculate portfolio value
            portfolio_value = sum(abs(p.quantity * p.current_price) for p in positions)

            if portfolio_value == Decimal('0'):
                return Decimal('0')

            # Calculate weighted volatility
            weighted_vol = Decimal('0')

            for position in positions:
                position_value = abs(position.quantity * position.current_price)
                weight = position_value / portfolio_value

                # Get position volatility
                position_vol = await self.calculate_realized_volatility(position.symbol)

                # Add weighted contribution
                weighted_vol += weight * position_vol

            logger.info(f"Portfolio volatility: {weighted_vol:.4f}")
            return weighted_vol

        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return Decimal('0')

    async def get_volatility_sizing_report(
        self,
        symbol: str,
        account_value: Decimal,
        entry_price: Decimal
    ) -> Dict:
        """
        Generate comprehensive volatility sizing report

        Args:
            symbol: Trading symbol
            account_value: Account value
            entry_price: Entry price

        Returns:
            Comprehensive report dictionary
        """
        try:
            # Calculate various position sizes
            atr = await self.calculate_atr(symbol)
            realized_vol = await self.calculate_realized_volatility(symbol)
            regime = await self.detect_volatility_regime(symbol)

            atr_size = await self.calculate_atr_position_size(symbol, account_value, entry_price)
            vol_norm_size = await self.calculate_volatility_normalized_size(symbol, account_value)

            # Calculate base size and adjust
            base_size = account_value * self.risk_per_trade
            adjusted_size = await self.calculate_volatility_adjusted_size(symbol, account_value, base_size)

            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'account_value': float(account_value),
                'entry_price': float(entry_price),
                'volatility_metrics': {
                    'atr': float(atr),
                    'realized_volatility': float(realized_vol),
                    'volatility_regime': regime,
                    'target_volatility': float(self.target_portfolio_volatility)
                },
                'position_sizes': {
                    'atr_based': float(atr_size),
                    'volatility_normalized': float(vol_norm_size),
                    'volatility_adjusted': float(adjusted_size),
                    'base_size': float(base_size)
                },
                'recommended_size': float(atr_size),  # ATR-based is most conservative
                'recommendation': self._get_sizing_recommendation(regime, realized_vol)
            }

            return report

        except Exception as e:
            logger.error(f"Error generating volatility sizing report: {e}")
            return {'error': str(e)}

    def _get_sizing_recommendation(self, regime: str, volatility: Decimal) -> str:
        """Get sizing recommendation based on volatility"""
        if regime == 'EXTREME':
            return "REDUCE: Extreme volatility detected - reduce position sizes significantly"
        elif regime == 'HIGH':
            return "CAUTION: High volatility - use conservative position sizing"
        elif regime == 'LOW':
            return "INCREASE: Low volatility environment - can consider larger positions"
        else:
            return "NORMAL: Standard position sizing appropriate"

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()

        logger.info("VolatilityBasedSizer cleanup complete")
