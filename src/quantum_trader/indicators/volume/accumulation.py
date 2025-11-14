"""
Volume Accumulation Indicators - On-Balance Volume, Accumulation/Distribution, etc.

Implements volume-based accumulation indicators for identifying buying and selling pressure.
Uses Polars for high-performance calculation on large datasets.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class VolumeAccumulationIndicators:
    """
    Production-grade volume accumulation indicators.

    Calculates various volume-based accumulation metrics including:
    - On-Balance Volume (OBV)
    - Accumulation/Distribution Line (A/D)
    - Money Flow Index (MFI)
    - Chaikin Money Flow (CMF)
    - Volume-Weighted Average Price (VWAP)

    All calculations use Decimal precision and Polars for performance.

    Example:
        >>> config = {
        ...     'mfi_period': 14,
        ...     'cmf_period': 20
        ... }
        >>> indicators = VolumeAccumulationIndicators(config)
        >>> df = indicators.calculate_all(ohlcv_data)
        >>> obv = df['obv']
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize volume accumulation indicators.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Periods for various indicators
        self.mfi_period = int(config.get('mfi_period', 14))
        self.cmf_period = int(config.get('cmf_period', 20))
        self.vwap_period = config.get('vwap_period')  # None = session VWAP

        logger.info(
            "volume_accumulation_indicators_initialized",
            mfi_period=self.mfi_period,
            cmf_period=self.cmf_period
        )

    def calculate_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all volume accumulation indicators.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with added indicator columns

        Raises:
            ValueError: If required columns missing
        """
        self._validate_dataframe(df)

        logger.debug(
            "calculating_volume_accumulation_indicators",
            rows=len(df)
        )

        result = df.clone()

        # Calculate each indicator
        result = self.calculate_obv(result)
        result = self.calculate_ad_line(result)
        result = self.calculate_mfi(result)
        result = self.calculate_cmf(result)
        result = self.calculate_vwap(result)
        result = self.calculate_pvt(result)

        return result

    def calculate_obv(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate On-Balance Volume (OBV).

        OBV = Previous OBV + (Volume if Close > Previous Close, else -Volume)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'obv' column
        """
        result = df.with_columns([
            # Price change direction
            (
                pl.when(pl.col('close') > pl.col('close').shift(1))
                .then(pl.col('volume'))
                .when(pl.col('close') < pl.col('close').shift(1))
                .then(-pl.col('volume'))
                .otherwise(pl.lit(Decimal("0")))
            ).alias('volume_direction')
        ])

        # Cumulative sum
        result = result.with_columns([
            pl.col('volume_direction').cum_sum().alias('obv')
        ])

        # Drop temporary column
        result = result.drop('volume_direction')

        return result

    def calculate_ad_line(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Accumulation/Distribution Line.

        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume
        A/D Line = Previous A/D Line + Money Flow Volume

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'ad_line' column
        """
        result = df.with_columns([
            # Money Flow Multiplier
            (
                ((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) /
                (pl.col('high') - pl.col('low'))
            ).fill_nan(Decimal("0")).alias('mf_multiplier'),
        ])

        # Money Flow Volume
        result = result.with_columns([
            (pl.col('mf_multiplier') * pl.col('volume')).alias('mf_volume')
        ])

        # Cumulative A/D Line
        result = result.with_columns([
            pl.col('mf_volume').cum_sum().alias('ad_line')
        ])

        # Drop temporary columns
        result = result.drop(['mf_multiplier', 'mf_volume'])

        return result

    def calculate_mfi(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Money Flow Index (MFI).

        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price * Volume
        Money Ratio = Positive Money Flow / Negative Money Flow
        MFI = 100 - (100 / (1 + Money Ratio))

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'mfi' column
        """
        result = df.with_columns([
            # Typical Price
            ((pl.col('high') + pl.col('low') + pl.col('close')) / Decimal("3")).alias('typical_price')
        ])

        # Raw Money Flow
        result = result.with_columns([
            (pl.col('typical_price') * pl.col('volume')).alias('raw_money_flow')
        ])

        # Price direction
        result = result.with_columns([
            (pl.col('typical_price') > pl.col('typical_price').shift(1)).alias('price_up')
        ])

        # Positive and Negative Money Flow
        result = result.with_columns([
            pl.when(pl.col('price_up'))
            .then(pl.col('raw_money_flow'))
            .otherwise(Decimal("0"))
            .alias('positive_mf'),

            pl.when(~pl.col('price_up'))
            .then(pl.col('raw_money_flow'))
            .otherwise(Decimal("0"))
            .alias('negative_mf')
        ])

        # Rolling sums
        result = result.with_columns([
            pl.col('positive_mf').rolling_sum(window_size=self.mfi_period).alias('positive_mf_sum'),
            pl.col('negative_mf').rolling_sum(window_size=self.mfi_period).alias('negative_mf_sum')
        ])

        # Money Ratio and MFI
        result = result.with_columns([
            (
                Decimal("100") - (
                    Decimal("100") / (
                        Decimal("1") + (pl.col('positive_mf_sum') / pl.col('negative_mf_sum'))
                    )
                )
            ).fill_nan(Decimal("50")).alias('mfi')
        ])

        # Drop temporary columns
        result = result.drop([
            'typical_price', 'raw_money_flow', 'price_up',
            'positive_mf', 'negative_mf', 'positive_mf_sum', 'negative_mf_sum'
        ])

        return result

    def calculate_cmf(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Chaikin Money Flow (CMF).

        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Money Flow Multiplier * Volume
        CMF = Sum(Money Flow Volume, N) / Sum(Volume, N)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'cmf' column
        """
        result = df.with_columns([
            # Money Flow Multiplier
            (
                ((pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))) /
                (pl.col('high') - pl.col('low'))
            ).fill_nan(Decimal("0")).alias('mf_multiplier')
        ])

        # Money Flow Volume
        result = result.with_columns([
            (pl.col('mf_multiplier') * pl.col('volume')).alias('mf_volume')
        ])

        # Rolling sums
        result = result.with_columns([
            pl.col('mf_volume').rolling_sum(window_size=self.cmf_period).alias('mf_volume_sum'),
            pl.col('volume').rolling_sum(window_size=self.cmf_period).alias('volume_sum')
        ])

        # CMF
        result = result.with_columns([
            (pl.col('mf_volume_sum') / pl.col('volume_sum')).fill_nan(Decimal("0")).alias('cmf')
        ])

        # Drop temporary columns
        result = result.drop([
            'mf_multiplier', 'mf_volume', 'mf_volume_sum', 'volume_sum'
        ])

        return result

    def calculate_vwap(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Volume-Weighted Average Price (VWAP).

        Typical Price = (High + Low + Close) / 3
        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'vwap' column
        """
        result = df.with_columns([
            # Typical Price
            ((pl.col('high') + pl.col('low') + pl.col('close')) / Decimal("3")).alias('typical_price')
        ])

        # Typical Price * Volume
        result = result.with_columns([
            (pl.col('typical_price') * pl.col('volume')).alias('tp_volume')
        ])

        if self.vwap_period:
            # Rolling VWAP
            result = result.with_columns([
                pl.col('tp_volume').rolling_sum(window_size=self.vwap_period).alias('tp_volume_sum'),
                pl.col('volume').rolling_sum(window_size=self.vwap_period).alias('volume_sum')
            ])

            result = result.with_columns([
                (pl.col('tp_volume_sum') / pl.col('volume_sum')).alias('vwap')
            ])

            result = result.drop(['tp_volume_sum', 'volume_sum'])

        else:
            # Session VWAP (cumulative)
            result = result.with_columns([
                (
                    pl.col('tp_volume').cum_sum() / pl.col('volume').cum_sum()
                ).alias('vwap')
            ])

        # Drop temporary columns
        result = result.drop(['typical_price', 'tp_volume'])

        return result

    def calculate_pvt(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Price-Volume Trend (PVT).

        PVT = Previous PVT + (Volume * ((Close - Previous Close) / Previous Close))

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with 'pvt' column
        """
        result = df.with_columns([
            # Price change percentage
            (
                (pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1)
            ).fill_nan(Decimal("0")).alias('price_change_pct')
        ])

        # PVT increment
        result = result.with_columns([
            (pl.col('volume') * pl.col('price_change_pct')).alias('pvt_increment')
        ])

        # Cumulative PVT
        result = result.with_columns([
            pl.col('pvt_increment').cum_sum().alias('pvt')
        ])

        # Drop temporary columns
        result = result.drop(['price_change_pct', 'pvt_increment'])

        return result

    def calculate_volume_oscillator(
        self,
        df: pl.DataFrame,
        short_period: Optional[int] = None,
        long_period: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Calculate Volume Oscillator.

        Volume Oscillator = ((Short MA - Long MA) / Long MA) * 100

        Args:
            df: Input DataFrame
            short_period: Short moving average period
            long_period: Long moving average period

        Returns:
            DataFrame with 'volume_oscillator' column
        """
        short_period = short_period or int(self.config.get('vo_short_period', 5))
        long_period = long_period or int(self.config.get('vo_long_period', 10))

        result = df.with_columns([
            pl.col('volume').rolling_mean(window_size=short_period).alias('volume_ma_short'),
            pl.col('volume').rolling_mean(window_size=long_period).alias('volume_ma_long')
        ])

        result = result.with_columns([
            (
                ((pl.col('volume_ma_short') - pl.col('volume_ma_long')) / pl.col('volume_ma_long')) * Decimal("100")
            ).fill_nan(Decimal("0")).alias('volume_oscillator')
        ])

        result = result.drop(['volume_ma_short', 'volume_ma_long'])

        return result

    def calculate_volume_ratio(
        self,
        df: pl.DataFrame,
        period: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Calculate Volume Ratio (current volume vs average volume).

        Volume Ratio = Current Volume / Average Volume

        Args:
            df: Input DataFrame
            period: Lookback period for average

        Returns:
            DataFrame with 'volume_ratio' column
        """
        period = period or int(self.config.get('volume_ratio_period', 20))

        result = df.with_columns([
            pl.col('volume').rolling_mean(window_size=period).alias('volume_avg')
        ])

        result = result.with_columns([
            (pl.col('volume') / pl.col('volume_avg')).fill_nan(Decimal("1")).alias('volume_ratio')
        ])

        result = result.drop('volume_avg')

        return result

    def _validate_dataframe(self, df: pl.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(df) == 0:
            raise ValueError("DataFrame is empty")

    def get_signals(
        self,
        df: pl.DataFrame,
        thresholds: Optional[Dict[str, Decimal]] = None
    ) -> pl.DataFrame:
        """
        Generate trading signals from volume accumulation indicators.

        Args:
            df: DataFrame with calculated indicators
            thresholds: Signal thresholds

        Returns:
            DataFrame with signal columns
        """
        thresholds = thresholds or {}

        mfi_oversold = thresholds.get('mfi_oversold', Decimal("20"))
        mfi_overbought = thresholds.get('mfi_overbought', Decimal("80"))
        cmf_threshold = thresholds.get('cmf_threshold', Decimal("0"))

        result = df.with_columns([
            # MFI signals
            (pl.col('mfi') < mfi_oversold).alias('mfi_oversold'),
            (pl.col('mfi') > mfi_overbought).alias('mfi_overbought'),

            # CMF signals
            (pl.col('cmf') > cmf_threshold).alias('cmf_bullish'),
            (pl.col('cmf') < -cmf_threshold).alias('cmf_bearish'),

            # OBV divergence (simplified)
            (
                (pl.col('close') > pl.col('close').shift(1)) &
                (pl.col('obv') < pl.col('obv').shift(1))
            ).alias('obv_bearish_divergence'),

            (
                (pl.col('close') < pl.col('close').shift(1)) &
                (pl.col('obv') > pl.col('obv').shift(1))
            ).alias('obv_bullish_divergence')
        ])

        return result
