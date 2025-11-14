"""
Data Validator - Validate market data quality and integrity.

This module provides comprehensive validation of market data to ensure
data quality, detect anomalies, and prevent backtest errors from bad data.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class DataValidator:
    """Validates market data quality and integrity.

    Attributes:
        config: Validation configuration from environment
        validation_rules: Configured validation rules
        error_threshold: Maximum allowed error rate
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize data validator.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.validation_rules: Dict[str, Any] = self._load_validation_rules()
        self.error_threshold: Decimal = self.config['error_threshold']

        logger.info("DataValidator initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'error_threshold': Decimal(os.getenv('DATA_ERROR_THRESHOLD', '0.01')),
                'strict_mode': os.getenv('DATA_STRICT_MODE', 'true').lower() == 'true',
                'check_price_spikes': os.getenv('CHECK_PRICE_SPIKES', 'true').lower() == 'true',
                'max_price_change_percent': Decimal(os.getenv('MAX_PRICE_CHANGE_PERCENT', '20.0')),
                'check_volume_anomalies': os.getenv('CHECK_VOLUME_ANOMALIES', 'true').lower() == 'true',
                'min_volume': Decimal(os.getenv('MIN_VOLUME', '0.0')),
            }

            logger.debug("Data validator config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules.

        Returns:
            Validation rules dictionary
        """
        return {
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'numeric_columns': ['open', 'high', 'low', 'close', 'volume'],
            'positive_columns': ['open', 'high', 'low', 'close', 'volume'],
            'price_columns': ['open', 'high', 'low', 'close'],
        }

    async def validate_ohlcv_data(
        self,
        data: pl.DataFrame
    ) -> Tuple[bool, List[str]]:
        """Validate OHLCV market data.

        Args:
            data: OHLCV data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)

        Example:
            >>> is_valid, errors = await validator.validate_ohlcv_data(df)
            >>> if not is_valid:
            ...     print(errors)
        """
        try:
            logger.info("Validating OHLCV data", rows=data.height)

            errors: List[str] = []

            # Check schema
            schema_errors = await self._validate_schema(data)
            errors.extend(schema_errors)

            # Check for null values
            null_errors = await self._validate_no_nulls(data)
            errors.extend(null_errors)

            # Check price relationships (high >= low, etc.)
            price_errors = await self._validate_price_relationships(data)
            errors.extend(price_errors)

            # Check for price spikes
            if self.config['check_price_spikes']:
                spike_errors = await self._validate_price_spikes(data)
                errors.extend(spike_errors)

            # Check volume anomalies
            if self.config['check_volume_anomalies']:
                volume_errors = await self._validate_volume(data)
                errors.extend(volume_errors)

            # Check timestamp continuity
            timestamp_errors = await self._validate_timestamps(data)
            errors.extend(timestamp_errors)

            # Determine if valid based on error threshold
            error_rate = Decimal(len(errors)) / Decimal(data.height) if data.height > 0 else Decimal('0')
            is_valid = error_rate <= self.error_threshold

            if not is_valid:
                logger.warning(
                    "Data validation failed",
                    error_count=len(errors),
                    error_rate=str(error_rate)
                )
            else:
                logger.info("Data validation passed", error_count=len(errors))

            return is_valid, errors

        except Exception as e:
            logger.error("Validation failed", error=str(e))
            raise

    async def validate_trade_data(
        self,
        data: pl.DataFrame
    ) -> Tuple[bool, List[str]]:
        """Validate trade execution data.

        Args:
            data: Trade data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            logger.info("Validating trade data", rows=data.height)

            errors: List[str] = []

            # Check required columns for trades
            required_trade_columns = ['timestamp', 'symbol', 'side', 'quantity', 'price']
            for col in required_trade_columns:
                if col not in data.columns:
                    errors.append(f"Missing required column: {col}")

            if errors:
                return False, errors

            # Check for null values in critical columns
            for col in required_trade_columns:
                null_count = data[col].null_count()
                if null_count > 0:
                    errors.append(f"Null values in {col}: {null_count} rows")

            # Validate quantities are positive
            if 'quantity' in data.columns:
                negative_qty = data.filter(pl.col('quantity') <= 0).height
                if negative_qty > 0:
                    errors.append(f"Negative or zero quantities: {negative_qty} trades")

            # Validate prices are positive
            if 'price' in data.columns:
                negative_price = data.filter(pl.col('price') <= 0).height
                if negative_price > 0:
                    errors.append(f"Negative or zero prices: {negative_price} trades")

            # Check order sides are valid
            if 'side' in data.columns:
                valid_sides = ['BUY', 'SELL']
                invalid_sides = data.filter(~pl.col('side').is_in(valid_sides)).height
                if invalid_sides > 0:
                    errors.append(f"Invalid order sides: {invalid_sides} trades")

            is_valid = len(errors) == 0
            logger.info("Trade validation complete", is_valid=is_valid, errors=len(errors))

            return is_valid, errors

        except Exception as e:
            logger.error("Trade validation failed", error=str(e))
            raise

    async def check_data_completeness(
        self,
        data: pl.DataFrame,
        expected_interval: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if data is complete for the expected interval.

        Args:
            data: Data to check
            expected_interval: Expected interval between rows

        Returns:
            Tuple of (is_complete, stats_dict)

        Example:
            >>> is_complete, stats = await validator.check_data_completeness(df, '1m')
            >>> print(f"Missing {stats['missing_count']} intervals")
        """
        try:
            logger.info("Checking data completeness", expected_interval=expected_interval)

            if data.height < 2:
                return True, {'missing_count': 0, 'total_expected': data.height}

            # Parse interval to seconds
            interval_seconds = self._parse_interval(expected_interval)

            # Get timestamp range
            timestamps = data['timestamp'].to_list()
            start_ts = timestamps[0]
            end_ts = timestamps[-1]

            # Calculate expected number of intervals
            if isinstance(start_ts, datetime) and isinstance(end_ts, datetime):
                time_diff = (end_ts - start_ts).total_seconds()
            else:
                time_diff = float(end_ts - start_ts)

            expected_count = int(time_diff / interval_seconds) + 1
            actual_count = data.height
            missing_count = max(0, expected_count - actual_count)

            completeness_ratio = Decimal(actual_count) / Decimal(expected_count) if expected_count > 0 else Decimal('1')

            stats = {
                'expected_count': expected_count,
                'actual_count': actual_count,
                'missing_count': missing_count,
                'completeness_ratio': completeness_ratio
            }

            is_complete = missing_count == 0

            logger.info(
                "Completeness check done",
                is_complete=is_complete,
                missing=missing_count,
                completeness=str(completeness_ratio)
            )

            return is_complete, stats

        except Exception as e:
            logger.error("Completeness check failed", error=str(e))
            raise

    async def detect_outliers(
        self,
        data: pl.DataFrame,
        column: str,
        method: str = 'iqr'
    ) -> pl.DataFrame:
        """Detect outliers in a data column.

        Args:
            data: Input data
            column: Column to check for outliers
            method: Detection method ('iqr', 'zscore', 'isolation')

        Returns:
            DataFrame with outlier rows

        Example:
            >>> outliers = await validator.detect_outliers(df, 'volume', method='iqr')
            >>> print(f"Found {outliers.height} outliers")
        """
        try:
            logger.info("Detecting outliers", column=column, method=method)

            if column not in data.columns:
                raise ValueError(f"Column {column} not found")

            if method == 'iqr':
                outliers = await self._detect_outliers_iqr(data, column)

            elif method == 'zscore':
                outliers = await self._detect_outliers_zscore(data, column)

            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            logger.info("Outlier detection complete", outlier_count=outliers.height)

            return outliers

        except Exception as e:
            logger.error("Outlier detection failed", error=str(e))
            raise

    async def _validate_schema(self, data: pl.DataFrame) -> List[str]:
        """Validate data schema."""
        errors: List[str] = []

        # Check required columns
        for col in self.validation_rules['required_columns']:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")

        return errors

    async def _validate_no_nulls(self, data: pl.DataFrame) -> List[str]:
        """Validate no null values in critical columns."""
        errors: List[str] = []

        for col in self.validation_rules['required_columns']:
            if col in data.columns:
                null_count = data[col].null_count()
                if null_count > 0:
                    errors.append(f"Null values in {col}: {null_count} rows")

        return errors

    async def _validate_price_relationships(self, data: pl.DataFrame) -> List[str]:
        """Validate OHLC price relationships."""
        errors: List[str] = []

        try:
            # High should be >= Low
            invalid = data.filter(pl.col('high') < pl.col('low')).height
            if invalid > 0:
                errors.append(f"High < Low in {invalid} rows")

            # High should be >= Open
            invalid = data.filter(pl.col('high') < pl.col('open')).height
            if invalid > 0:
                errors.append(f"High < Open in {invalid} rows")

            # High should be >= Close
            invalid = data.filter(pl.col('high') < pl.col('close')).height
            if invalid > 0:
                errors.append(f"High < Close in {invalid} rows")

            # Low should be <= Open
            invalid = data.filter(pl.col('low') > pl.col('open')).height
            if invalid > 0:
                errors.append(f"Low > Open in {invalid} rows")

            # Low should be <= Close
            invalid = data.filter(pl.col('low') > pl.col('close')).height
            if invalid > 0:
                errors.append(f"Low > Close in {invalid} rows")

        except Exception as e:
            errors.append(f"Price relationship validation error: {e}")

        return errors

    async def _validate_price_spikes(self, data: pl.DataFrame) -> List[str]:
        """Validate no extreme price spikes."""
        errors: List[str] = []

        try:
            if data.height < 2:
                return errors

            # Calculate price changes
            price_changes = data.with_columns(
                ((pl.col('close') - pl.col('close').shift(1)) / pl.col('close').shift(1) * 100).alias('pct_change')
            )

            # Check for spikes
            max_change = self.config['max_price_change_percent']
            spikes = price_changes.filter(
                (pl.col('pct_change').abs() > float(max_change)) & pl.col('pct_change').is_not_null()
            ).height

            if spikes > 0:
                errors.append(f"Price spikes exceeding {max_change}%: {spikes} occurrences")

        except Exception as e:
            errors.append(f"Price spike validation error: {e}")

        return errors

    async def _validate_volume(self, data: pl.DataFrame) -> List[str]:
        """Validate volume data."""
        errors: List[str] = []

        try:
            # Check for negative volumes
            negative_vol = data.filter(pl.col('volume') < 0).height
            if negative_vol > 0:
                errors.append(f"Negative volumes: {negative_vol} rows")

            # Check for zero volumes (optional based on config)
            min_vol = self.config['min_volume']
            if min_vol > Decimal('0'):
                low_vol = data.filter(pl.col('volume') < float(min_vol)).height
                if low_vol > 0:
                    errors.append(f"Volumes below minimum {min_vol}: {low_vol} rows")

        except Exception as e:
            errors.append(f"Volume validation error: {e}")

        return errors

    async def _validate_timestamps(self, data: pl.DataFrame) -> List[str]:
        """Validate timestamp continuity."""
        errors: List[str] = []

        try:
            # Check for duplicate timestamps
            duplicates = data.height - data.select('timestamp').n_unique()
            if duplicates > 0:
                errors.append(f"Duplicate timestamps: {duplicates} rows")

            # Check timestamps are sorted
            timestamps = data['timestamp'].to_list()
            if timestamps != sorted(timestamps):
                errors.append("Timestamps are not in ascending order")

        except Exception as e:
            errors.append(f"Timestamp validation error: {e}")

        return errors

    async def _detect_outliers_iqr(
        self,
        data: pl.DataFrame,
        column: str
    ) -> pl.DataFrame:
        """Detect outliers using IQR method."""
        try:
            # Calculate quartiles
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1

            # Calculate bounds
            lower_bound = q1 - Decimal('1.5') * Decimal(str(iqr))
            upper_bound = q3 + Decimal('1.5') * Decimal(str(iqr))

            # Filter outliers
            outliers = data.filter(
                (pl.col(column) < float(lower_bound)) | (pl.col(column) > float(upper_bound))
            )

            return outliers

        except Exception as e:
            logger.error("IQR outlier detection failed", error=str(e))
            raise

    async def _detect_outliers_zscore(
        self,
        data: pl.DataFrame,
        column: str,
        threshold: Decimal = Decimal('3.0')
    ) -> pl.DataFrame:
        """Detect outliers using Z-score method."""
        try:
            # Calculate mean and std
            mean = data[column].mean()
            std = data[column].std()

            if std == 0:
                return data.head(0)  # No outliers if no variance

            # Calculate z-scores
            data_with_zscore = data.with_columns(
                ((pl.col(column) - mean) / std).abs().alias('zscore')
            )

            # Filter outliers
            outliers = data_with_zscore.filter(pl.col('zscore') > float(threshold))

            return outliers.drop('zscore')

        except Exception as e:
            logger.error("Z-score outlier detection failed", error=str(e))
            raise

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds."""
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
            raise
