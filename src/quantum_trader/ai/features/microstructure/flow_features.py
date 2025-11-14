"""Order flow and market microstructure features.

This module computes advanced market microstructure features from order flow data,
including order book imbalance, trade flow toxicity, price impact, and liquidity metrics.
Critical for high-frequency and institutional trading strategies.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import polars as pl
import numpy as np
from structlog import get_logger
from collections import deque

logger = get_logger(__name__)


class FlowFeaturesError(Exception):
    """Base exception for flow features computation."""
    pass


class OrderFlowFeatures:
    """Compute order flow and market microstructure features.

    Analyzes order book dynamics, trade flow, and market impact to generate
    features that capture market microstructure effects. Essential for
    understanding short-term price movements and liquidity conditions.

    Attributes:
        config: Configuration dictionary
        lookback_periods: Time periods for feature computation
        cache: Cache for rolling computations

    Example:
        >>> config = {
        ...     'lookback_periods': [10, 30, 60],
        ...     'depth_levels': 5,
        ...     'vpin_bucket_size': 50,
        ...     'trade_classification': 'tick_rule'
        ... }
        >>> flow_features = OrderFlowFeatures(config)
        >>> features = await flow_features.compute(market_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize order flow features computer.

        Args:
            config: Configuration dictionary with keys:
                - lookback_periods: List of lookback periods in ticks
                - depth_levels: Number of order book levels to analyze
                - vpin_bucket_size: Bucket size for VPIN calculation
                - trade_classification: Method for trade classification
                - min_trade_size: Minimum trade size to consider
                - impact_decay: Price impact decay factor

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_periods = self.config.get('lookback_periods', [10, 30, 60])
        self.depth_levels = self.config.get('depth_levels', 5)
        self.vpin_bucket_size = self.config.get('vpin_bucket_size', 50)
        self.trade_classification = self.config.get('trade_classification', 'tick_rule')
        self.min_trade_size = Decimal(str(self.config.get('min_trade_size', 0.0)))
        self.impact_decay = Decimal(str(self.config.get('impact_decay', 0.9)))

        # Cache for rolling computations
        self.cache: Dict[str, deque] = {
            'trades': deque(maxlen=max(self.lookback_periods)),
            'prices': deque(maxlen=max(self.lookback_periods)),
            'volumes': deque(maxlen=max(self.lookback_periods)),
            'order_imbalances': deque(maxlen=max(self.lookback_periods))
        }

        self._lock = asyncio.Lock()

        logger.info(
            "order_flow_features_initialized",
            lookback_periods=self.lookback_periods,
            depth_levels=self.depth_levels
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if 'lookback_periods' in self.config:
            periods = self.config['lookback_periods']
            if not all(p > 0 for p in periods):
                raise ValueError("All lookback_periods must be positive")

        if 'depth_levels' in self.config and self.config['depth_levels'] < 1:
            raise ValueError("depth_levels must be positive")

        if 'vpin_bucket_size' in self.config and self.config['vpin_bucket_size'] < 1:
            raise ValueError("vpin_bucket_size must be positive")

    async def compute(
        self,
        market_data: pl.DataFrame,
        orderbook_data: Optional[pl.DataFrame] = None,
        trade_data: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Compute order flow features from market data.

        Args:
            market_data: Main market data with OHLCV
            orderbook_data: Optional order book snapshots
            trade_data: Optional trade-by-trade data

        Returns:
            DataFrame with computed flow features

        Raises:
            FlowFeaturesError: If computation fails
            ValueError: If input data is invalid

        Example:
            >>> features_df = await flow_features.compute(
            ...     market_data=ohlcv_df,
            ...     orderbook_data=book_df,
            ...     trade_data=trades_df
            ... )
        """
        try:
            if market_data.is_empty():
                raise ValueError("market_data cannot be empty")

            result = market_data.clone()

            # Compute basic flow features from OHLCV
            result = await self._compute_volume_features(result)
            result = await self._compute_price_impact_features(result)
            result = await self._compute_spread_features(result)

            # Compute order book features if available
            if orderbook_data is not None and not orderbook_data.is_empty():
                book_features = await self._compute_orderbook_features(orderbook_data)
                result = self._merge_features(result, book_features)

            # Compute trade flow features if available
            if trade_data is not None and not trade_data.is_empty():
                trade_features = await self._compute_trade_flow_features(trade_data)
                result = self._merge_features(result, trade_features)

            logger.info(
                "flow_features_computed",
                row_count=result.height,
                feature_count=len(result.columns)
            )

            return result

        except Exception as e:
            logger.error("flow_features_computation_failed", error=str(e))
            raise FlowFeaturesError(f"Failed to compute flow features: {str(e)}") from e

    async def _compute_volume_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute volume-based flow features.

        Args:
            data: Market data with volume information

        Returns:
            DataFrame with volume features added
        """
        result = data

        try:
            # Volume rate of change
            for period in self.lookback_periods:
                result = result.with_columns([
                    (pl.col('volume') / pl.col('volume').shift(period))
                    .fill_null(Decimal('1.0'))
                    .alias(f'volume_roc_{period}')
                ])

            # Volume moving average ratio
            for period in self.lookback_periods:
                result = result.with_columns([
                    (pl.col('volume') / pl.col('volume').rolling_mean(window_size=period))
                    .fill_null(Decimal('1.0'))
                    .alias(f'volume_ma_ratio_{period}')
                ])

            # Volume standard deviation
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('volume').rolling_std(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'volume_std_{period}')
                ])

            # Cumulative volume delta (approximation from OHLCV)
            result = result.with_columns([
                pl.when(pl.col('close') > pl.col('open'))
                .then(pl.col('volume'))
                .when(pl.col('close') < pl.col('open'))
                .then(-pl.col('volume'))
                .otherwise(Decimal('0.0'))
                .alias('volume_delta')
            ])

            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('volume_delta').rolling_sum(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'cumulative_volume_delta_{period}')
                ])

            logger.debug("volume_features_computed")

        except Exception as e:
            logger.warning("volume_features_computation_failed", error=str(e))

        return result

    async def _compute_price_impact_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute price impact and market impact features.

        Args:
            data: Market data with price and volume

        Returns:
            DataFrame with price impact features
        """
        result = data

        try:
            # Price-volume correlation
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('close').rolling_corr(pl.col('volume'), window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'price_volume_corr_{period}')
                ])

            # Amihud illiquidity measure: |return| / volume
            result = result.with_columns([
                ((pl.col('close') - pl.col('close').shift(1)).abs() /
                 (pl.col('volume') + Decimal('1e-10')))
                .fill_null(Decimal('0.0'))
                .alias('amihud_illiquidity')
            ])

            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('amihud_illiquidity').rolling_mean(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'amihud_illiquidity_{period}')
                ])

            # Kyle's lambda: price impact per unit volume
            for period in self.lookback_periods:
                returns = (pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1)
                signed_volume = pl.col('volume_delta') if 'volume_delta' in result.columns else pl.col('volume')

                result = result.with_columns([
                    (returns / (signed_volume + Decimal('1e-10')))
                    .rolling_mean(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'kyle_lambda_{period}')
                ])

            # Realized volatility per unit volume
            for period in self.lookback_periods:
                result = result.with_columns([
                    (pl.col('close').pct_change().rolling_std(window_size=period) /
                     (pl.col('volume').rolling_mean(window_size=period) + Decimal('1e-10')))
                    .fill_null(Decimal('0.0'))
                    .alias(f'vol_per_volume_{period}')
                ])

            logger.debug("price_impact_features_computed")

        except Exception as e:
            logger.warning("price_impact_features_computation_failed", error=str(e))

        return result

    async def _compute_spread_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """Compute bid-ask spread proxy features from OHLCV.

        Args:
            data: Market data with OHLCV

        Returns:
            DataFrame with spread features
        """
        result = data

        try:
            # High-Low spread proxy
            result = result.with_columns([
                ((pl.col('high') - pl.col('low')) / pl.col('close'))
                .fill_null(Decimal('0.0'))
                .alias('hl_spread_ratio')
            ])

            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('hl_spread_ratio').rolling_mean(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'avg_spread_ratio_{period}')
                ])

            # Roll spread estimator
            for period in [2, 5, 10]:
                if period <= max(self.lookback_periods):
                    result = result.with_columns([
                        (Decimal('2.0') *
                         (-(pl.col('close') - pl.col('close').shift(1)) *
                          (pl.col('close').shift(1) - pl.col('close').shift(2)))
                         .rolling_mean(window_size=period).sqrt())
                        .fill_null(Decimal('0.0'))
                        .alias(f'roll_spread_{period}')
                    ])

            # Relative spread volatility
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('hl_spread_ratio').rolling_std(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'spread_volatility_{period}')
                ])

            logger.debug("spread_features_computed")

        except Exception as e:
            logger.warning("spread_features_computation_failed", error=str(e))

        return result

    async def _compute_orderbook_features(
        self,
        orderbook_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Compute features from order book snapshots.

        Args:
            orderbook_data: Order book data with bid/ask levels

        Returns:
            DataFrame with order book features
        """
        try:
            result = orderbook_data.clone()

            # Order book imbalance (OBI)
            for level in range(1, min(self.depth_levels + 1, 6)):
                bid_col = f'bid_size_{level}' if f'bid_size_{level}' in result.columns else 'bid_size'
                ask_col = f'ask_size_{level}' if f'ask_size_{level}' in result.columns else 'ask_size'

                if bid_col in result.columns and ask_col in result.columns:
                    result = result.with_columns([
                        ((pl.col(bid_col) - pl.col(ask_col)) /
                         (pl.col(bid_col) + pl.col(ask_col) + Decimal('1e-10')))
                        .fill_null(Decimal('0.0'))
                        .alias(f'obi_level_{level}')
                    ])

            # Weighted order book imbalance
            if all(f'bid_size_{i}' in result.columns for i in range(1, self.depth_levels + 1)):
                total_bid_volume = sum(
                    pl.col(f'bid_size_{i}') for i in range(1, self.depth_levels + 1)
                )
                total_ask_volume = sum(
                    pl.col(f'ask_size_{i}') for i in range(1, self.depth_levels + 1)
                )

                result = result.with_columns([
                    ((total_bid_volume - total_ask_volume) /
                     (total_bid_volume + total_ask_volume + Decimal('1e-10')))
                    .alias('weighted_obi')
                ])

            # Depth imbalance ratio
            if 'bid_size' in result.columns and 'ask_size' in result.columns:
                result = result.with_columns([
                    (pl.col('bid_size') / (pl.col('ask_size') + Decimal('1e-10')))
                    .fill_null(Decimal('1.0'))
                    .alias('depth_ratio')
                ])

                for period in self.lookback_periods:
                    result = result.with_columns([
                        pl.col('depth_ratio').rolling_mean(window_size=period)
                        .fill_null(Decimal('1.0'))
                        .alias(f'avg_depth_ratio_{period}')
                    ])

            # Mid-price movement
            if 'bid_price' in result.columns and 'ask_price' in result.columns:
                result = result.with_columns([
                    ((pl.col('bid_price') + pl.col('ask_price')) / Decimal('2.0'))
                    .alias('mid_price')
                ])

                result = result.with_columns([
                    pl.col('mid_price').pct_change()
                    .fill_null(Decimal('0.0'))
                    .alias('mid_price_return')
                ])

            logger.debug(
                "orderbook_features_computed",
                feature_count=len(result.columns) - len(orderbook_data.columns)
            )

            return result

        except Exception as e:
            logger.error("orderbook_features_computation_failed", error=str(e))
            raise FlowFeaturesError(
                f"Failed to compute order book features: {str(e)}"
            ) from e

    async def _compute_trade_flow_features(
        self,
        trade_data: pl.DataFrame
    ) -> pl.DataFrame:
        """Compute features from trade flow data.

        Args:
            trade_data: Trade-by-trade data

        Returns:
            DataFrame with trade flow features
        """
        try:
            result = trade_data.clone()

            # Trade direction classification
            if 'side' not in result.columns:
                result = await self._classify_trades(result)

            # Buy/sell volume imbalance
            result = result.with_columns([
                pl.when(pl.col('side') == 'buy')
                .then(pl.col('size'))
                .otherwise(Decimal('0.0'))
                .alias('buy_volume')
            ])

            result = result.with_columns([
                pl.when(pl.col('side') == 'sell')
                .then(pl.col('size'))
                .otherwise(Decimal('0.0'))
                .alias('sell_volume')
            ])

            for period in self.lookback_periods:
                result = result.with_columns([
                    ((pl.col('buy_volume').rolling_sum(window_size=period) -
                      pl.col('sell_volume').rolling_sum(window_size=period)) /
                     (pl.col('buy_volume').rolling_sum(window_size=period) +
                      pl.col('sell_volume').rolling_sum(window_size=period) + Decimal('1e-10')))
                    .fill_null(Decimal('0.0'))
                    .alias(f'trade_imbalance_{period}')
                ])

            # Trade intensity (number of trades)
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.lit(1).rolling_sum(window_size=period)
                    .alias(f'trade_count_{period}')
                ])

            # VPIN (Volume-Synchronized Probability of Informed Trading)
            if self.vpin_bucket_size > 0:
                result = await self._compute_vpin(result)

            # Average trade size
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('size').rolling_mean(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'avg_trade_size_{period}')
                ])

            # Trade size standard deviation
            for period in self.lookback_periods:
                result = result.with_columns([
                    pl.col('size').rolling_std(window_size=period)
                    .fill_null(Decimal('0.0'))
                    .alias(f'trade_size_std_{period}')
                ])

            logger.debug(
                "trade_flow_features_computed",
                feature_count=len(result.columns) - len(trade_data.columns)
            )

            return result

        except Exception as e:
            logger.error("trade_flow_features_computation_failed", error=str(e))
            raise FlowFeaturesError(
                f"Failed to compute trade flow features: {str(e)}"
            ) from e

    async def _classify_trades(self, trade_data: pl.DataFrame) -> pl.DataFrame:
        """Classify trades as buy or sell using tick rule or other methods.

        Args:
            trade_data: Trade data with price information

        Returns:
            DataFrame with 'side' column added
        """
        result = trade_data

        try:
            if self.trade_classification == 'tick_rule':
                # Classify based on price change
                result = result.with_columns([
                    pl.when(pl.col('price') > pl.col('price').shift(1))
                    .then(pl.lit('buy'))
                    .when(pl.col('price') < pl.col('price').shift(1))
                    .then(pl.lit('sell'))
                    .otherwise(pl.lit('unknown'))
                    .alias('side')
                ])

            logger.debug("trades_classified", method=self.trade_classification)

        except Exception as e:
            logger.warning("trade_classification_failed", error=str(e))

        return result

    async def _compute_vpin(self, trade_data: pl.DataFrame) -> pl.DataFrame:
        """Compute Volume-Synchronized Probability of Informed Trading (VPIN).

        Args:
            trade_data: Trade data with buy/sell classification

        Returns:
            DataFrame with VPIN feature
        """
        result = trade_data

        try:
            # Create volume buckets
            cumsum_volume = result['size'].cum_sum()
            bucket_id = (cumsum_volume / Decimal(str(self.vpin_bucket_size))).cast(pl.Int64)

            result = result.with_columns(bucket_id.alias('bucket_id'))

            # Compute VPIN per bucket
            # VPIN = |Buy Volume - Sell Volume| / Total Volume
            vpin_by_bucket = (
                result.group_by('bucket_id')
                .agg([
                    (pl.col('buy_volume').sum() - pl.col('sell_volume').sum()).abs()
                    .truediv(pl.col('size').sum() + Decimal('1e-10'))
                    .alias('vpin_bucket')
                ])
            )

            # Join back to original data
            result = result.join(vpin_by_bucket, on='bucket_id', how='left')

            # Rolling average of VPIN
            result = result.with_columns([
                pl.col('vpin_bucket')
                .fill_null(Decimal('0.0'))
                .alias('vpin')
            ])

            logger.debug("vpin_computed", bucket_size=self.vpin_bucket_size)

        except Exception as e:
            logger.warning("vpin_computation_failed", error=str(e))
            # Add default VPIN column
            result = result.with_columns([
                pl.lit(Decimal('0.0')).alias('vpin')
            ])

        return result

    def _merge_features(
        self,
        base: pl.DataFrame,
        features: pl.DataFrame
    ) -> pl.DataFrame:
        """Merge feature DataFrames on timestamp.

        Args:
            base: Base DataFrame
            features: Features DataFrame to merge

        Returns:
            Merged DataFrame
        """
        try:
            if 'timestamp' in base.columns and 'timestamp' in features.columns:
                result = base.join(features, on='timestamp', how='left')
            else:
                # Concatenate columns if same length
                if base.height == features.height:
                    result = pl.concat([base, features], how='horizontal')
                else:
                    logger.warning(
                        "feature_merge_skipped",
                        reason="length_mismatch",
                        base_height=base.height,
                        features_height=features.height
                    )
                    result = base

            return result

        except Exception as e:
            logger.warning("feature_merge_failed", error=str(e))
            return base

    async def compute_realtime(
        self,
        tick: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Compute flow features for a single real-time tick.

        Args:
            tick: Real-time tick data dictionary

        Returns:
            Dictionary of computed features

        Example:
            >>> tick = {
            ...     'price': Decimal('50000.0'),
            ...     'volume': Decimal('1.5'),
            ...     'timestamp': datetime.now(timezone.utc)
            ... }
            >>> features = await flow_features.compute_realtime(tick)
        """
        try:
            async with self._lock:
                # Update cache
                self.cache['prices'].append(tick.get('price', Decimal('0.0')))
                self.cache['volumes'].append(tick.get('volume', Decimal('0.0')))

                # Compute features from cache
                features = {}

                # Volume features
                if len(self.cache['volumes']) > 0:
                    recent_volumes = list(self.cache['volumes'])
                    features['volume_current'] = recent_volumes[-1]

                    if len(recent_volumes) >= 10:
                        features['volume_ma_10'] = Decimal(str(np.mean([float(v) for v in recent_volumes[-10:]])))

                # Price features
                if len(self.cache['prices']) > 1:
                    recent_prices = list(self.cache['prices'])
                    price_change = recent_prices[-1] - recent_prices[-2]
                    features['price_change'] = price_change

                return features

        except Exception as e:
            logger.error("realtime_computation_failed", error=str(e))
            return {}
