"""
Volume Profile Indicator.

This module implements Volume Profile for analyzing volume distribution
across price levels in institutional trading systems.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class VolumeProfileError(Exception):
    """Base exception for Volume Profile indicator errors."""
    pass


class VolumeProfileValidator:
    """Validates Volume Profile configuration and input data."""

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate Volume Profile configuration.

        Args:
            config: Configuration dictionary

        Raises:
            VolumeProfileError: If configuration is invalid
        """
        if 'num_bins' not in config:
            raise VolumeProfileError("Missing required config key: num_bins")

        try:
            num_bins = int(config['num_bins'])
            if num_bins < 2:
                raise VolumeProfileError("num_bins must be at least 2")

        except (ValueError, TypeError) as e:
            raise VolumeProfileError(f"Invalid configuration values: {e}")

    @staticmethod
    def validate_dataframe(df: pl.DataFrame) -> None:
        """Validate input dataframe structure.

        Args:
            df: Input dataframe

        Raises:
            VolumeProfileError: If dataframe is invalid
        """
        required_columns = ['high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise VolumeProfileError(f"Missing required columns: {missing}")

        if len(df) < 1:
            raise VolumeProfileError("Dataframe must have at least 1 row")


class VolumeProfile:
    """Volume Profile indicator implementation.

    Volume Profile shows the distribution of volume traded at different
    price levels, helping identify key support/resistance and value areas.

    Attributes:
        config: Configuration dictionary
        num_bins: Number of price bins for volume distribution
        lookback: Lookback period for rolling profile

    Example:
        >>> config = {
        ...     'num_bins': 24,
        ...     'lookback': 100
        ... }
        >>> vp = VolumeProfile(config)
        >>> result = await vp.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Volume Profile indicator.

        Args:
            config: Configuration dictionary

        Raises:
            VolumeProfileError: If configuration is invalid
        """
        VolumeProfileValidator.validate_config(config)

        self.config = config
        self.num_bins = int(config['num_bins'])
        self.lookback = int(config.get('lookback', 100))

        logger.info(
            "VolumeProfile initialized",
            num_bins=self.num_bins,
            lookback=self.lookback
        )

    async def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Volume Profile metrics.

        Args:
            df: Input dataframe with columns: high, low, close, volume

        Returns:
            DataFrame with additional columns:
                - vp_poc: Point of Control (price with highest volume)
                - vp_vah: Value Area High
                - vp_val: Value Area Low
                - vp_volume_at_price: Volume traded at current price level

        Raises:
            VolumeProfileError: If calculation fails

        Example:
            >>> df = pl.DataFrame({
            ...     'high': [102.0, 103.0, 104.0],
            ...     'low': [99.0, 100.0, 101.0],
            ...     'close': [101.0, 102.0, 103.0],
            ...     'volume': [1000000, 1200000, 950000]
            ... })
            >>> result = await vp.calculate(df)
        """
        try:
            VolumeProfileValidator.validate_dataframe(df)

            logger.debug("Calculating Volume Profile", rows=len(df))

            highs = [Decimal(str(x)) for x in df['high'].to_list()]
            lows = [Decimal(str(x)) for x in df['low'].to_list()]
            closes = [Decimal(str(x)) for x in df['close'].to_list()]
            volumes = [Decimal(str(x)) for x in df['volume'].to_list()]

            # Calculate rolling volume profile
            poc_values = []
            vah_values = []
            val_values = []
            volume_at_price = []

            for i in range(len(closes)):
                if i < self.lookback - 1:
                    # Not enough data yet
                    poc_values.append(None)
                    vah_values.append(None)
                    val_values.append(None)
                    volume_at_price.append(None)
                else:
                    # Get lookback window
                    start_idx = i - self.lookback + 1
                    window_highs = highs[start_idx:i + 1]
                    window_lows = lows[start_idx:i + 1]
                    window_volumes = volumes[start_idx:i + 1]

                    # Calculate volume profile for window
                    profile = self._calculate_profile(
                        window_highs,
                        window_lows,
                        window_volumes
                    )

                    # Extract metrics
                    poc = profile['poc']
                    vah = profile['vah']
                    val = profile['val']

                    # Get volume at current close price
                    vol_at_close = self._get_volume_at_price(
                        closes[i],
                        profile['bins']
                    )

                    poc_values.append(float(poc) if poc is not None else None)
                    vah_values.append(float(vah) if vah is not None else None)
                    val_values.append(float(val) if val is not None else None)
                    volume_at_price.append(float(vol_at_close) if vol_at_close is not None else None)

            # Add columns to dataframe
            result = df.with_columns([
                pl.Series('vp_poc', poc_values),
                pl.Series('vp_vah', vah_values),
                pl.Series('vp_val', val_values),
                pl.Series('vp_volume_at_price', volume_at_price)
            ])

            logger.info("Volume Profile calculated successfully", rows=len(result))

            return result

        except VolumeProfileError:
            raise
        except Exception as e:
            logger.error("Volume Profile calculation failed", error=str(e))
            raise VolumeProfileError(f"Calculation failed: {e}")

    def _calculate_profile(
        self,
        highs: List[Decimal],
        lows: List[Decimal],
        volumes: List[Decimal]
    ) -> Dict[str, Any]:
        """Calculate volume profile for a price range.

        Args:
            highs: List of high prices
            lows: List of low prices
            volumes: List of volumes

        Returns:
            Dictionary with profile data including POC, VAH, VAL
        """
        # Find price range
        min_price = min(lows)
        max_price = max(highs)

        if max_price <= min_price:
            return {
                'poc': None,
                'vah': None,
                'val': None,
                'bins': []
            }

        # Calculate bin size
        price_range = max_price - min_price
        bin_size = price_range / Decimal(str(self.num_bins))

        # Initialize bins
        bins = []
        for i in range(self.num_bins):
            bin_low = min_price + (bin_size * Decimal(str(i)))
            bin_high = bin_low + bin_size
            bins.append({
                'low': bin_low,
                'high': bin_high,
                'volume': Decimal('0')
            })

        # Distribute volume across bins
        for high, low, volume in zip(highs, lows, volumes):
            # Find which bins this bar overlaps
            for bin_data in bins:
                # Check if bar overlaps with bin
                if high >= bin_data['low'] and low <= bin_data['high']:
                    # Calculate overlap percentage
                    overlap_low = max(low, bin_data['low'])
                    overlap_high = min(high, bin_data['high'])
                    overlap = overlap_high - overlap_low
                    bar_range = high - low

                    if bar_range > Decimal('0'):
                        # Distribute volume proportionally
                        overlap_pct = overlap / bar_range
                        bin_data['volume'] += volume * overlap_pct
                    else:
                        # Bar has no range, distribute evenly
                        bin_data['volume'] += volume / Decimal(str(self.num_bins))

        # Find Point of Control (highest volume bin)
        poc_bin = max(bins, key=lambda b: b['volume'])
        poc = (poc_bin['low'] + poc_bin['high']) / Decimal('2')

        # Calculate Value Area (70% of total volume)
        total_volume = sum(b['volume'] for b in bins)
        value_area_volume = total_volume * Decimal('0.70')

        # Find value area starting from POC
        poc_idx = bins.index(poc_bin)
        value_area_bins = [poc_idx]
        current_volume = poc_bin['volume']

        # Expand value area
        lower_idx = poc_idx - 1
        upper_idx = poc_idx + 1

        while current_volume < value_area_volume:
            # Determine which direction to expand
            lower_vol = bins[lower_idx]['volume'] if lower_idx >= 0 else Decimal('0')
            upper_vol = bins[upper_idx]['volume'] if upper_idx < len(bins) else Decimal('0')

            if lower_vol > upper_vol and lower_idx >= 0:
                value_area_bins.append(lower_idx)
                current_volume += lower_vol
                lower_idx -= 1
            elif upper_idx < len(bins):
                value_area_bins.append(upper_idx)
                current_volume += upper_vol
                upper_idx += 1
            else:
                break

        # Calculate VAH and VAL
        if value_area_bins:
            val_idx = min(value_area_bins)
            vah_idx = max(value_area_bins)
            val = bins[val_idx]['low']
            vah = bins[vah_idx]['high']
        else:
            val = None
            vah = None

        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'bins': bins
        }

    def _get_volume_at_price(
        self,
        price: Decimal,
        bins: List[Dict[str, Any]]
    ) -> Optional[Decimal]:
        """Get volume at a specific price level.

        Args:
            price: Price to query
            bins: Volume profile bins

        Returns:
            Volume at price level or None
        """
        for bin_data in bins:
            if bin_data['low'] <= price <= bin_data['high']:
                return bin_data['volume']
        return None

    def get_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from Volume Profile.

        Args:
            df: DataFrame with Volume Profile calculations

        Returns:
            DataFrame with signal columns:
                - vp_signal: 1 for buy at VAL, -1 for sell at VAH
                - vp_at_poc: True when price near POC
                - vp_in_value_area: True when price in value area

        Raises:
            VolumeProfileError: If required columns missing
        """
        try:
            required = ['close', 'vp_poc', 'vp_vah', 'vp_val']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise VolumeProfileError(f"Missing required columns: {missing}")

            closes = df['close'].to_list()
            poc_values = df['vp_poc'].to_list()
            vah_values = df['vp_vah'].to_list()
            val_values = df['vp_val'].to_list()

            signals = []
            at_poc_flags = []
            in_value_area_flags = []

            # Tolerance for POC detection (0.2% of price)
            poc_tolerance = Decimal('0.002')

            for close, poc, vah, val in zip(closes, poc_values, vah_values, val_values):
                if poc is None or vah is None or val is None:
                    signals.append(0)
                    at_poc_flags.append(False)
                    in_value_area_flags.append(False)
                else:
                    close_dec = Decimal(str(close))
                    poc_dec = Decimal(str(poc))
                    vah_dec = Decimal(str(vah))
                    val_dec = Decimal(str(val))

                    # Signal: buy near VAL, sell near VAH
                    tolerance = close_dec * poc_tolerance

                    if abs(close_dec - val_dec) < tolerance:
                        signals.append(1)
                    elif abs(close_dec - vah_dec) < tolerance:
                        signals.append(-1)
                    else:
                        signals.append(0)

                    # Check if at POC
                    at_poc = abs(close_dec - poc_dec) < tolerance
                    at_poc_flags.append(at_poc)

                    # Check if in value area
                    in_value_area = val_dec <= close_dec <= vah_dec
                    in_value_area_flags.append(in_value_area)

            result = df.with_columns([
                pl.Series('vp_signal', signals),
                pl.Series('vp_at_poc', at_poc_flags),
                pl.Series('vp_in_value_area', in_value_area_flags)
            ])

            logger.debug(
                "Volume Profile signals generated",
                buy_signals=sum(1 for s in signals if s == 1),
                sell_signals=sum(1 for s in signals if s == -1),
                at_poc_count=sum(at_poc_flags),
                in_value_area_count=sum(in_value_area_flags)
            )

            return result

        except Exception as e:
            logger.error("Signal generation failed", error=str(e))
            raise VolumeProfileError(f"Signal generation failed: {e}")


async def calculate_volume_profile(
    df: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate Volume Profile.

    Args:
        df: Input dataframe with OHLCV data
        config: Configuration dictionary

    Returns:
        DataFrame with Volume Profile calculations

    Example:
        >>> config = {
        ...     'num_bins': 24,
        ...     'lookback': 100
        ... }
        >>> result = await calculate_volume_profile(df, config)
    """
    vp = VolumeProfile(config)
    return await vp.calculate(df)
