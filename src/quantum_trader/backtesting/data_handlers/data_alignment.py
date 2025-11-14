"""
Data Alignment - Synchronize and align multi-source market data.

This module provides tools to align market data from multiple sources and timeframes,
handling missing data, timezone conversions, and timestamp synchronization.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class DataAlignmentHandler:
    """Handles alignment and synchronization of market data.

    Attributes:
        config: Alignment configuration from environment
        alignment_method: Method for handling misaligned data
        fill_method: Method for filling missing values
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize data alignment handler.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.alignment_method: str = self.config['alignment_method']
        self.fill_method: str = self.config['fill_method']

        logger.info("DataAlignmentHandler initialized", alignment_method=self.alignment_method)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'alignment_method': os.getenv('ALIGNMENT_METHOD', 'inner'),
                'fill_method': os.getenv('FILL_METHOD', 'forward'),
                'max_gap_seconds': int(os.getenv('MAX_GAP_SECONDS', '300')),
                'tolerance_seconds': int(os.getenv('TOLERANCE_SECONDS', '1')),
                'require_complete_data': os.getenv('REQUIRE_COMPLETE_DATA', 'false').lower() == 'true',
            }

            logger.debug("Data alignment config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    async def align_multiple_sources(
        self,
        data_sources: Dict[str, pl.DataFrame],
        timestamp_column: str = 'timestamp'
    ) -> pl.DataFrame:
        """Align data from multiple sources by timestamp.

        Args:
            data_sources: Dictionary mapping source name to DataFrame
            timestamp_column: Name of timestamp column

        Returns:
            Aligned DataFrame with data from all sources

        Example:
            >>> aligned = await handler.align_multiple_sources({
            ...     'exchange1': df1,
            ...     'exchange2': df2
            ... })
            >>> aligned.columns
            ['timestamp', 'exchange1_price', 'exchange2_price', ...]
        """
        try:
            logger.info(
                "Aligning multiple data sources",
                source_count=len(data_sources),
                method=self.alignment_method
            )

            if not data_sources:
                raise ValueError("No data sources provided")

            # Validate all sources have timestamp column
            for source_name, df in data_sources.items():
                if timestamp_column not in df.columns:
                    raise ValueError(f"Source '{source_name}' missing timestamp column")

            # Start with first source
            source_names = list(data_sources.keys())
            aligned = data_sources[source_names[0]].clone()

            # Rename columns to include source prefix
            aligned = self._prefix_columns(aligned, source_names[0], timestamp_column)

            # Join remaining sources
            for source_name in source_names[1:]:
                df = data_sources[source_name]
                df_prefixed = self._prefix_columns(df, source_name, timestamp_column)

                aligned = self._join_dataframes(
                    aligned,
                    df_prefixed,
                    timestamp_column,
                    self.alignment_method
                )

            # Handle missing data
            aligned = await self._handle_missing_data(aligned, timestamp_column)

            # Sort by timestamp
            aligned = aligned.sort(timestamp_column)

            logger.info(
                "Data sources aligned",
                rows=aligned.height,
                columns=len(aligned.columns)
            )

            return aligned

        except Exception as e:
            logger.error("Failed to align data sources", error=str(e))
            raise

    async def align_timeframes(
        self,
        high_freq_data: pl.DataFrame,
        low_freq_data: pl.DataFrame,
        high_freq_interval: str,
        low_freq_interval: str
    ) -> pl.DataFrame:
        """Align data from different timeframes.

        Args:
            high_freq_data: Higher frequency data (e.g., 1m)
            low_freq_data: Lower frequency data (e.g., 1h)
            high_freq_interval: High frequency interval
            low_freq_interval: Low frequency interval

        Returns:
            Aligned DataFrame at high frequency with low freq data filled forward

        Example:
            >>> aligned = await handler.align_timeframes(
            ...     minute_data, hour_data, '1m', '1h'
            ... )
        """
        try:
            logger.info(
                "Aligning timeframes",
                high_freq=high_freq_interval,
                low_freq=low_freq_interval
            )

            # Validate data
            if high_freq_data.height == 0 or low_freq_data.height == 0:
                raise ValueError("Empty data provided")

            # Join with as-of join (forward fill)
            aligned = high_freq_data.join_asof(
                low_freq_data,
                on='timestamp',
                strategy='forward',
                suffix='_low_freq'
            )

            logger.info("Timeframes aligned", rows=aligned.height)

            return aligned

        except Exception as e:
            logger.error("Failed to align timeframes", error=str(e))
            raise

    async def synchronize_timestamps(
        self,
        data: pl.DataFrame,
        target_interval: str,
        timestamp_column: str = 'timestamp'
    ) -> pl.DataFrame:
        """Synchronize timestamps to regular intervals.

        Args:
            data: Input data with irregular timestamps
            target_interval: Target interval (e.g., '1m', '5m', '1h')
            timestamp_column: Name of timestamp column

        Returns:
            DataFrame with synchronized timestamps

        Example:
            >>> synced = await handler.synchronize_timestamps(
            ...     irregular_data, '1m'
            ... )
        """
        try:
            logger.info(
                "Synchronizing timestamps",
                target_interval=target_interval,
                input_rows=data.height
            )

            if data.height == 0:
                return data

            # Parse interval
            interval_seconds = self._parse_interval_to_seconds(target_interval)

            # Get time range
            min_ts = data[timestamp_column].min()
            max_ts = data[timestamp_column].max()

            # Generate regular timestamp grid
            regular_timestamps = self._generate_timestamp_grid(
                min_ts,
                max_ts,
                interval_seconds
            )

            # Create DataFrame with regular timestamps
            regular_df = pl.DataFrame({
                timestamp_column: regular_timestamps
            })

            # Join with original data using nearest timestamp
            synchronized = regular_df.join_asof(
                data,
                on=timestamp_column,
                strategy='nearest',
                tolerance=f"{self.config['tolerance_seconds']}s"
            )

            # Fill missing values
            synchronized = await self._handle_missing_data(synchronized, timestamp_column)

            logger.info(
                "Timestamps synchronized",
                output_rows=synchronized.height
            )

            return synchronized

        except Exception as e:
            logger.error("Failed to synchronize timestamps", error=str(e))
            raise

    async def fill_gaps(
        self,
        data: pl.DataFrame,
        timestamp_column: str = 'timestamp',
        fill_method: Optional[str] = None
    ) -> pl.DataFrame:
        """Fill gaps in time series data.

        Args:
            data: Input data with gaps
            timestamp_column: Name of timestamp column
            fill_method: Fill method ('forward', 'backward', 'interpolate', 'zero')

        Returns:
            DataFrame with gaps filled

        Example:
            >>> filled = await handler.fill_gaps(data, fill_method='forward')
        """
        try:
            fill_method = fill_method or self.fill_method

            logger.info(
                "Filling gaps",
                method=fill_method,
                input_rows=data.height
            )

            if data.height == 0:
                return data

            if fill_method == 'forward':
                filled = data.fill_null(strategy='forward')

            elif fill_method == 'backward':
                filled = data.fill_null(strategy='backward')

            elif fill_method == 'zero':
                filled = data.fill_null(0)

            elif fill_method == 'interpolate':
                # Linear interpolation for numeric columns
                filled = data.clone()
                for col in filled.columns:
                    if col != timestamp_column and filled[col].dtype in [pl.Float64, pl.Float32]:
                        filled = filled.with_columns(
                            pl.col(col).interpolate()
                        )

            else:
                raise ValueError(f"Unknown fill method: {fill_method}")

            logger.info("Gaps filled", output_rows=filled.height)

            return filled

        except Exception as e:
            logger.error("Failed to fill gaps", error=str(e))
            raise

    async def detect_gaps(
        self,
        data: pl.DataFrame,
        timestamp_column: str = 'timestamp',
        expected_interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect gaps in time series data.

        Args:
            data: Input data
            timestamp_column: Name of timestamp column
            expected_interval: Expected interval between timestamps

        Returns:
            List of gap information dictionaries

        Example:
            >>> gaps = await handler.detect_gaps(data, expected_interval='1m')
            >>> len(gaps)
            3
        """
        try:
            logger.info("Detecting gaps", expected_interval=expected_interval)

            if data.height < 2:
                return []

            gaps: List[Dict[str, Any]] = []

            # Calculate time differences
            timestamps = data[timestamp_column].to_list()

            if expected_interval:
                expected_seconds = self._parse_interval_to_seconds(expected_interval)
                max_gap = expected_seconds * 2  # Allow some tolerance
            else:
                max_gap = self.config['max_gap_seconds']

            for i in range(1, len(timestamps)):
                prev_ts = timestamps[i - 1]
                curr_ts = timestamps[i]

                if isinstance(prev_ts, datetime) and isinstance(curr_ts, datetime):
                    gap_seconds = (curr_ts - prev_ts).total_seconds()
                else:
                    # Handle timestamp as integer (unix timestamp)
                    gap_seconds = float(curr_ts - prev_ts)

                if gap_seconds > max_gap:
                    gaps.append({
                        'start_timestamp': prev_ts,
                        'end_timestamp': curr_ts,
                        'gap_seconds': gap_seconds,
                        'row_index': i
                    })

            logger.info("Gaps detected", gap_count=len(gaps))

            return gaps

        except Exception as e:
            logger.error("Failed to detect gaps", error=str(e))
            raise

    async def resample_data(
        self,
        data: pl.DataFrame,
        source_interval: str,
        target_interval: str,
        aggregation: str = 'ohlc'
    ) -> pl.DataFrame:
        """Resample data to different timeframe.

        Args:
            data: Input data
            source_interval: Source data interval
            target_interval: Target interval
            aggregation: Aggregation method ('ohlc', 'mean', 'last')

        Returns:
            Resampled DataFrame

        Example:
            >>> resampled = await handler.resample_data(
            ...     minute_data, '1m', '5m', aggregation='ohlc'
            ... )
        """
        try:
            logger.info(
                "Resampling data",
                source=source_interval,
                target=target_interval,
                aggregation=aggregation
            )

            if data.height == 0:
                return data

            # Parse intervals
            target_seconds = self._parse_interval_to_seconds(target_interval)
            target_duration = f"{target_seconds}s"

            if aggregation == 'ohlc':
                # OHLC aggregation
                resampled = data.group_by_dynamic(
                    'timestamp',
                    every=target_duration
                ).agg([
                    pl.col('open').first().alias('open'),
                    pl.col('high').max().alias('high'),
                    pl.col('low').min().alias('low'),
                    pl.col('close').last().alias('close'),
                    pl.col('volume').sum().alias('volume') if 'volume' in data.columns else pl.lit(0).alias('volume')
                ])

            elif aggregation == 'mean':
                # Mean aggregation for all numeric columns
                agg_exprs = []
                for col in data.columns:
                    if col != 'timestamp' and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                        agg_exprs.append(pl.col(col).mean().alias(col))

                resampled = data.group_by_dynamic(
                    'timestamp',
                    every=target_duration
                ).agg(agg_exprs)

            elif aggregation == 'last':
                # Last value aggregation
                agg_exprs = [pl.col(col).last().alias(col) for col in data.columns if col != 'timestamp']
                resampled = data.group_by_dynamic(
                    'timestamp',
                    every=target_duration
                ).agg(agg_exprs)

            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")

            logger.info("Data resampled", output_rows=resampled.height)

            return resampled

        except Exception as e:
            logger.error("Failed to resample data", error=str(e))
            raise

    def _prefix_columns(
        self,
        df: pl.DataFrame,
        prefix: str,
        exclude_column: str
    ) -> pl.DataFrame:
        """Add prefix to all columns except the excluded one.

        Args:
            df: Input DataFrame
            prefix: Prefix to add
            exclude_column: Column to exclude from prefixing

        Returns:
            DataFrame with prefixed columns
        """
        try:
            rename_map = {
                col: f"{prefix}_{col}"
                for col in df.columns
                if col != exclude_column
            }

            return df.rename(rename_map)

        except Exception as e:
            logger.error("Failed to prefix columns", error=str(e))
            raise

    def _join_dataframes(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        on_column: str,
        how: str
    ) -> pl.DataFrame:
        """Join two DataFrames.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on_column: Join column
            how: Join method ('inner', 'left', 'outer')

        Returns:
            Joined DataFrame
        """
        try:
            return left.join(right, on=on_column, how=how)

        except Exception as e:
            logger.error("Failed to join dataframes", error=str(e))
            raise

    async def _handle_missing_data(
        self,
        data: pl.DataFrame,
        timestamp_column: str
    ) -> pl.DataFrame:
        """Handle missing data based on configuration.

        Args:
            data: Input data
            timestamp_column: Timestamp column name

        Returns:
            DataFrame with missing data handled
        """
        try:
            if self.config['require_complete_data']:
                # Drop rows with any null values
                return data.drop_nulls()
            else:
                # Fill using configured method
                return await self.fill_gaps(data, timestamp_column, self.fill_method)

        except Exception as e:
            logger.error("Failed to handle missing data", error=str(e))
            raise

    def _parse_interval_to_seconds(self, interval: str) -> int:
        """Parse interval string to seconds.

        Args:
            interval: Interval string (e.g., '1m', '5m', '1h', '1d')

        Returns:
            Interval in seconds

        Raises:
            ValueError: If interval format invalid
        """
        try:
            if interval.endswith('s'):
                return int(interval[:-1])
            elif interval.endswith('m'):
                return int(interval[:-1]) * 60
            elif interval.endswith('h'):
                return int(interval[:-1]) * 3600
            elif interval.endswith('d'):
                return int(interval[:-1]) * 86400
            else:
                raise ValueError(f"Invalid interval format: {interval}")

        except Exception as e:
            logger.error("Failed to parse interval", error=str(e))
            raise ValueError(f"Invalid interval: {interval}")

    def _generate_timestamp_grid(
        self,
        start: datetime,
        end: datetime,
        interval_seconds: int
    ) -> List[datetime]:
        """Generate regular timestamp grid.

        Args:
            start: Start timestamp
            end: End timestamp
            interval_seconds: Interval in seconds

        Returns:
            List of timestamps
        """
        try:
            timestamps: List[datetime] = []
            current = start

            while current <= end:
                timestamps.append(current)
                current += timedelta(seconds=interval_seconds)

            return timestamps

        except Exception as e:
            logger.error("Failed to generate timestamp grid", error=str(e))
            raise
