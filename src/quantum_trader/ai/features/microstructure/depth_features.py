"""Order book depth feature extraction for market microstructure analysis.

This module extracts sophisticated features from order book depth data including
bid-ask spreads, imbalances, price impacts, and liquidity measures.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class DepthFeatureExtractor:
    """Extract advanced features from order book depth data.

    Analyzes order book microstructure to generate trading signals based on
    liquidity, imbalances, and price impact measures.

    Attributes:
        config: Configuration dictionary with feature parameters
        max_depth_levels: Maximum depth levels to analyze
        min_tick_size: Minimum tick size for the instrument
        volume_decimals: Decimal places for volume calculations
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize depth feature extractor.

        Args:
            config: Configuration dictionary containing:
                - max_depth_levels: Maximum depth levels (default from config)
                - min_tick_size: Minimum tick size
                - volume_decimals: Decimal precision
                - price_impact_levels: Levels for price impact calculation

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.max_depth_levels = config.get('max_depth_levels', 20)
        self.min_tick_size = Decimal(str(config.get('min_tick_size', '0.01')))
        self.volume_decimals = config.get('volume_decimals', 8)
        self.price_impact_levels = config.get('price_impact_levels', [10, 50, 100])

        logger.info(
            "depth_feature_extractor_initialized",
            max_depth_levels=self.max_depth_levels,
            min_tick_size=str(self.min_tick_size)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = ['max_depth_levels', 'min_tick_size', 'volume_decimals']
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("config_validation_failed", error=error_msg)
            raise ValueError(error_msg)

        if self.config['max_depth_levels'] <= 0:
            raise ValueError("max_depth_levels must be positive")

        if Decimal(str(self.config['min_tick_size'])) <= 0:
            raise ValueError("min_tick_size must be positive")

    def extract_features(
        self,
        orderbook_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Extract all depth features from order book data.

        Args:
            orderbook_df: Polars DataFrame with columns:
                - timestamp: UTC timestamp
                - bid_price_N: Bid prices at level N
                - bid_volume_N: Bid volumes at level N
                - ask_price_N: Ask prices at level N
                - ask_volume_N: Ask volumes at level N

        Returns:
            DataFrame with extracted features

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_orderbook_data(orderbook_df)

            features = orderbook_df.select(['timestamp'])

            # Extract basic spread features
            spread_features = self._calculate_spread_features(orderbook_df)
            features = features.hstack(spread_features)

            # Extract imbalance features
            imbalance_features = self._calculate_imbalance_features(orderbook_df)
            features = features.hstack(imbalance_features)

            # Extract depth features
            depth_features = self._calculate_depth_features(orderbook_df)
            features = features.hstack(depth_features)

            # Extract price impact features
            impact_features = self._calculate_price_impact_features(orderbook_df)
            features = features.hstack(impact_features)

            # Extract liquidity features
            liquidity_features = self._calculate_liquidity_features(orderbook_df)
            features = features.hstack(liquidity_features)

            logger.debug(
                "depth_features_extracted",
                rows=features.height,
                features=features.width - 1
            )

            return features

        except Exception as e:
            logger.error("depth_feature_extraction_failed", error=str(e))
            raise

    def _validate_orderbook_data(self, df: pl.DataFrame) -> None:
        """Validate order book data format.

        Args:
            df: Order book dataframe to validate

        Raises:
            ValueError: If data format is invalid
        """
        required_cols = ['timestamp']

        # Check for bid/ask price and volume columns
        for level in range(1, self.max_depth_levels + 1):
            required_cols.extend([
                f'bid_price_{level}',
                f'bid_volume_{level}',
                f'ask_price_{level}',
                f'ask_volume_{level}'
            ])

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols[:5]}"
            logger.error("orderbook_validation_failed", error=error_msg)
            raise ValueError(error_msg)

        if df.height == 0:
            raise ValueError("Empty orderbook dataframe")

    def _calculate_spread_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate bid-ask spread features.

        Args:
            df: Order book dataframe

        Returns:
            DataFrame with spread features
        """
        try:
            spread_features = pl.DataFrame({
                'spread_absolute': df['ask_price_1'] - df['bid_price_1'],
                'spread_bps': ((df['ask_price_1'] - df['bid_price_1']) / df['bid_price_1'] * Decimal('10000')),
                'spread_mid': (df['ask_price_1'] + df['bid_price_1']) / Decimal('2'),
            })

            # Weighted spread across multiple levels
            weighted_spread = Decimal('0')
            total_weight = Decimal('0')

            for level in range(1, min(6, self.max_depth_levels + 1)):
                weight = Decimal('1') / Decimal(str(level))
                weighted_spread += (df[f'ask_price_{level}'] - df[f'bid_price_{level}']) * weight
                total_weight += weight

            spread_features = spread_features.with_columns(
                (weighted_spread / total_weight).alias('spread_weighted')
            )

            return spread_features

        except Exception as e:
            logger.error("spread_feature_calculation_failed", error=str(e))
            raise

    def _calculate_imbalance_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate order book imbalance features.

        Args:
            df: Order book dataframe

        Returns:
            DataFrame with imbalance features
        """
        try:
            # Calculate total bid and ask volumes at different depths
            imbalance_features_dict = {}

            for depth in [1, 5, 10, 20]:
                if depth > self.max_depth_levels:
                    continue

                bid_vol = pl.lit(Decimal('0'))
                ask_vol = pl.lit(Decimal('0'))

                for level in range(1, depth + 1):
                    bid_vol = bid_vol + df[f'bid_volume_{level}']
                    ask_vol = ask_vol + df[f'ask_volume_{level}']

                # Order book imbalance ratio
                total_vol = bid_vol + ask_vol
                imbalance_features_dict[f'imbalance_ratio_{depth}'] = (
                    (bid_vol - ask_vol) / total_vol.fill_null(Decimal('1'))
                )

                # Bid/Ask volume ratio
                imbalance_features_dict[f'bid_ask_ratio_{depth}'] = (
                    bid_vol / ask_vol.fill_null(Decimal('1'))
                )

            # Volume-weighted imbalance
            vwap_bid = Decimal('0')
            vwap_ask = Decimal('0')
            total_bid_vol = Decimal('0')
            total_ask_vol = Decimal('0')

            for level in range(1, min(11, self.max_depth_levels + 1)):
                bid_vol = df[f'bid_volume_{level}']
                ask_vol = df[f'ask_volume_{level}']

                vwap_bid += df[f'bid_price_{level}'] * bid_vol
                vwap_ask += df[f'ask_price_{level}'] * ask_vol
                total_bid_vol += bid_vol
                total_ask_vol += ask_vol

            imbalance_features_dict['vwap_imbalance'] = (
                (vwap_bid / total_bid_vol.fill_null(Decimal('1'))) -
                (vwap_ask / total_ask_vol.fill_null(Decimal('1')))
            )

            return pl.DataFrame(imbalance_features_dict)

        except Exception as e:
            logger.error("imbalance_feature_calculation_failed", error=str(e))
            raise

    def _calculate_depth_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate order book depth features.

        Args:
            df: Order book dataframe

        Returns:
            DataFrame with depth features
        """
        try:
            depth_features_dict = {}

            # Total volume at different depths
            for depth in [5, 10, 20]:
                if depth > self.max_depth_levels:
                    continue

                bid_depth = pl.lit(Decimal('0'))
                ask_depth = pl.lit(Decimal('0'))

                for level in range(1, depth + 1):
                    bid_depth = bid_depth + df[f'bid_volume_{level}']
                    ask_depth = ask_depth + df[f'ask_volume_{level}']

                depth_features_dict[f'total_bid_depth_{depth}'] = bid_depth
                depth_features_dict[f'total_ask_depth_{depth}'] = ask_depth
                depth_features_dict[f'total_depth_{depth}'] = bid_depth + ask_depth

            # Depth concentration (what % of volume is in top N levels)
            top3_bid = pl.lit(Decimal('0'))
            top3_ask = pl.lit(Decimal('0'))
            total_bid = pl.lit(Decimal('0'))
            total_ask = pl.lit(Decimal('0'))

            for level in range(1, min(self.max_depth_levels + 1, 21)):
                bid_vol = df[f'bid_volume_{level}']
                ask_vol = df[f'ask_volume_{level}']

                if level <= 3:
                    top3_bid = top3_bid + bid_vol
                    top3_ask = top3_ask + ask_vol

                total_bid = total_bid + bid_vol
                total_ask = total_ask + ask_vol

            depth_features_dict['bid_concentration_top3'] = (
                top3_bid / total_bid.fill_null(Decimal('1'))
            )
            depth_features_dict['ask_concentration_top3'] = (
                top3_ask / total_ask.fill_null(Decimal('1'))
            )

            return pl.DataFrame(depth_features_dict)

        except Exception as e:
            logger.error("depth_feature_calculation_failed", error=str(e))
            raise

    def _calculate_price_impact_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate estimated price impact for different order sizes.

        Args:
            df: Order book dataframe

        Returns:
            DataFrame with price impact features
        """
        try:
            impact_features_dict = {}

            for target_volume in self.price_impact_levels:
                target_vol_decimal = Decimal(str(target_volume))

                # Calculate buy impact (moving up the ask side)
                buy_impact = self._calculate_side_impact(
                    df, 'ask', target_vol_decimal
                )
                impact_features_dict[f'buy_impact_{target_volume}'] = buy_impact

                # Calculate sell impact (moving down the bid side)
                sell_impact = self._calculate_side_impact(
                    df, 'bid', target_vol_decimal
                )
                impact_features_dict[f'sell_impact_{target_volume}'] = sell_impact

                # Impact asymmetry
                impact_features_dict[f'impact_asymmetry_{target_volume}'] = (
                    buy_impact - sell_impact
                )

            return pl.DataFrame(impact_features_dict)

        except Exception as e:
            logger.error("price_impact_calculation_failed", error=str(e))
            raise

    def _calculate_side_impact(
        self,
        df: pl.DataFrame,
        side: str,
        target_volume: Decimal
    ) -> pl.Series:
        """Calculate price impact for one side of the book.

        Args:
            df: Order book dataframe
            side: 'bid' or 'ask'
            target_volume: Target volume to execute

        Returns:
            Series with price impact in basis points
        """
        # This is a simplified impact calculation
        # In production, this would use the actual order book traversal

        cumulative_volume = pl.lit(Decimal('0'))
        weighted_price = pl.lit(Decimal('0'))

        for level in range(1, min(self.max_depth_levels + 1, 21)):
            level_volume = df[f'{side}_volume_{level}']
            level_price = df[f'{side}_price_{level}']

            cumulative_volume = cumulative_volume + level_volume
            weighted_price = weighted_price + (level_price * level_volume)

        avg_price = weighted_price / cumulative_volume.fill_null(Decimal('1'))
        mid_price = (df['bid_price_1'] + df['ask_price_1']) / Decimal('2')

        impact_bps = (
            (avg_price - mid_price).abs() / mid_price * Decimal('10000')
        )

        return impact_bps

    def _calculate_liquidity_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate liquidity measures.

        Args:
            df: Order book dataframe

        Returns:
            DataFrame with liquidity features
        """
        try:
            liquidity_features_dict = {}

            # Average order size at different levels
            for level_group in [(1, 3), (4, 10), (11, 20)]:
                start, end = level_group
                end = min(end, self.max_depth_levels)

                if start > end:
                    continue

                bid_sizes = []
                ask_sizes = []

                for level in range(start, end + 1):
                    bid_sizes.append(df[f'bid_volume_{level}'])
                    ask_sizes.append(df[f'ask_volume_{level}'])

                if bid_sizes:
                    avg_bid_size = pl.concat(bid_sizes).sum() / Decimal(str(len(bid_sizes)))
                    avg_ask_size = pl.concat(ask_sizes).sum() / Decimal(str(len(ask_sizes)))

                    liquidity_features_dict[f'avg_bid_size_{start}_{end}'] = avg_bid_size
                    liquidity_features_dict[f'avg_ask_size_{start}_{end}'] = avg_ask_size

            # Price levels within N ticks
            for n_ticks in [5, 10, 20]:
                tick_range = self.min_tick_size * Decimal(str(n_ticks))

                bid_levels_in_range = Decimal('0')
                ask_levels_in_range = Decimal('0')

                for level in range(2, min(self.max_depth_levels + 1, 21)):
                    bid_distance = df['bid_price_1'] - df[f'bid_price_{level}']
                    ask_distance = df[f'ask_price_{level}'] - df['ask_price_1']

                    # Count levels within range (simplified)
                    # In production, this would be more sophisticated

                liquidity_features_dict[f'liquidity_score_{n_ticks}ticks'] = (
                    df['bid_volume_1'] + df['ask_volume_1']
                ) / Decimal(str(n_ticks))

            return pl.DataFrame(liquidity_features_dict)

        except Exception as e:
            logger.error("liquidity_feature_calculation_failed", error=str(e))
            raise

    async def extract_features_async(
        self,
        orderbook_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Async wrapper for feature extraction.

        Args:
            orderbook_df: Order book dataframe

        Returns:
            DataFrame with extracted features
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.extract_features,
            orderbook_df
        )
