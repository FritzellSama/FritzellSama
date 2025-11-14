"""
Data validator for comprehensive market data validation.

This module provides validation logic for all types of market data with
anomaly detection, schema validation, and data quality checks.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validator."""

    enable_schema_validation: bool
    enable_range_validation: bool
    enable_anomaly_detection: bool
    enable_consistency_checks: bool
    max_price_deviation_percent: Decimal
    max_volume_deviation_percent: Decimal
    max_timestamp_drift_seconds: int
    min_price: Decimal
    max_price: Decimal
    min_volume: Decimal
    max_volume: Decimal


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


@dataclass
class ValidationResult:
    """Result of data validation."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, int]


class DataValidator:
    """
    Comprehensive data validator for market data.

    Validates market data with schema checks, range validation, anomaly
    detection, and consistency checks.

    Attributes:
        config: Validator configuration

    Example:
        ```python
        config = {
            'enable_schema_validation': True,
            'enable_range_validation': True,
            'enable_anomaly_detection': True,
            'enable_consistency_checks': True,
            'max_price_deviation_percent': '10.0',
            'max_volume_deviation_percent': '500.0',
            'max_timestamp_drift_seconds': 60,
            'min_price': '0.00000001',
            'max_price': '1000000000',
            'min_volume': '0',
            'max_volume': '1000000000'
        }

        validator = DataValidator(config)

        # Validate OHLCV data
        result = validator.validate_ohlcv(ohlcv_df)
        if not result.valid:
            print(f"Validation failed: {result.errors}")

        # Validate trade data
        result = validator.validate_trades(trades_df)

        # Validate orderbook
        result = validator.validate_orderbook(orderbook_data)
        ```
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize data validator.

        Args:
            config: Validator configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)

        self.config = ValidationConfig(
            enable_schema_validation=config.get('enable_schema_validation', True),
            enable_range_validation=config.get('enable_range_validation', True),
            enable_anomaly_detection=config.get('enable_anomaly_detection', True),
            enable_consistency_checks=config.get('enable_consistency_checks', True),
            max_price_deviation_percent=Decimal(str(config.get('max_price_deviation_percent', '10.0'))),
            max_volume_deviation_percent=Decimal(str(config.get('max_volume_deviation_percent', '500.0'))),
            max_timestamp_drift_seconds=config.get('max_timestamp_drift_seconds', 60),
            min_price=Decimal(str(config.get('min_price', '0.00000001'))),
            max_price=Decimal(str(config.get('max_price', '1000000000'))),
            min_volume=Decimal(str(config.get('min_volume', '0'))),
            max_volume=Decimal(str(config.get('max_volume', '1000000000')))
        )

        self._stats: Dict[str, int] = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'anomalies_detected': 0
        }

        logger.info("Data validator initialized", config=config)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # All fields are optional with defaults
        pass

    def validate_ohlcv(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate OHLCV candle data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, int] = {
            'rows_checked': len(df),
            'schema_errors': 0,
            'range_errors': 0,
            'anomalies': 0,
            'consistency_errors': 0
        }

        try:
            self._stats['total_validations'] += 1

            # Schema validation
            if self.config.enable_schema_validation:
                schema_errors = self._validate_ohlcv_schema(df)
                errors.extend(schema_errors)
                stats['schema_errors'] = len(schema_errors)

            if errors:
                self._stats['failed_validations'] += 1
                return ValidationResult(False, errors, warnings, stats)

            # Range validation
            if self.config.enable_range_validation:
                range_errors = self._validate_ohlcv_ranges(df)
                errors.extend(range_errors)
                stats['range_errors'] = len(range_errors)

            # OHLC consistency checks
            if self.config.enable_consistency_checks:
                consistency_errors = self._validate_ohlcv_consistency(df)
                errors.extend(consistency_errors)
                stats['consistency_errors'] = len(consistency_errors)

            # Anomaly detection
            if self.config.enable_anomaly_detection:
                anomalies = self._detect_ohlcv_anomalies(df)
                warnings.extend(anomalies)
                stats['anomalies'] = len(anomalies)
                self._stats['anomalies_detected'] += len(anomalies)

            valid = len(errors) == 0

            if valid:
                self._stats['successful_validations'] += 1
            else:
                self._stats['failed_validations'] += 1

            return ValidationResult(valid, errors, warnings, stats)

        except Exception as e:
            logger.error("OHLCV validation error", error=str(e))
            self._stats['failed_validations'] += 1
            return ValidationResult(
                False,
                [f"Validation error: {e}"],
                warnings,
                stats
            )

    def validate_trades(self, df: pl.DataFrame) -> ValidationResult:
        """
        Validate trade data.

        Args:
            df: DataFrame with trade data

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, int] = {
            'rows_checked': len(df),
            'schema_errors': 0,
            'range_errors': 0,
            'anomalies': 0
        }

        try:
            self._stats['total_validations'] += 1

            # Schema validation
            if self.config.enable_schema_validation:
                schema_errors = self._validate_trades_schema(df)
                errors.extend(schema_errors)
                stats['schema_errors'] = len(schema_errors)

            if errors:
                self._stats['failed_validations'] += 1
                return ValidationResult(False, errors, warnings, stats)

            # Range validation
            if self.config.enable_range_validation:
                range_errors = self._validate_trades_ranges(df)
                errors.extend(range_errors)
                stats['range_errors'] = len(range_errors)

            # Anomaly detection
            if self.config.enable_anomaly_detection:
                anomalies = self._detect_trade_anomalies(df)
                warnings.extend(anomalies)
                stats['anomalies'] = len(anomalies)
                self._stats['anomalies_detected'] += len(anomalies)

            valid = len(errors) == 0

            if valid:
                self._stats['successful_validations'] += 1
            else:
                self._stats['failed_validations'] += 1

            return ValidationResult(valid, errors, warnings, stats)

        except Exception as e:
            logger.error("Trade validation error", error=str(e))
            self._stats['failed_validations'] += 1
            return ValidationResult(
                False,
                [f"Validation error: {e}"],
                warnings,
                stats
            )

    def validate_orderbook(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate orderbook data.

        Args:
            data: Orderbook dictionary

        Returns:
            ValidationResult with validation outcome
        """
        errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, int] = {
            'bid_levels': 0,
            'ask_levels': 0,
            'schema_errors': 0,
            'range_errors': 0,
            'consistency_errors': 0
        }

        try:
            self._stats['total_validations'] += 1

            # Schema validation
            if self.config.enable_schema_validation:
                schema_errors = self._validate_orderbook_schema(data)
                errors.extend(schema_errors)
                stats['schema_errors'] = len(schema_errors)

            if errors:
                self._stats['failed_validations'] += 1
                return ValidationResult(False, errors, warnings, stats)

            stats['bid_levels'] = len(data.get('bids', []))
            stats['ask_levels'] = len(data.get('asks', []))

            # Range validation
            if self.config.enable_range_validation:
                range_errors = self._validate_orderbook_ranges(data)
                errors.extend(range_errors)
                stats['range_errors'] = len(range_errors)

            # Consistency checks
            if self.config.enable_consistency_checks:
                consistency_errors = self._validate_orderbook_consistency(data)
                errors.extend(consistency_errors)
                stats['consistency_errors'] = len(consistency_errors)

            valid = len(errors) == 0

            if valid:
                self._stats['successful_validations'] += 1
            else:
                self._stats['failed_validations'] += 1

            return ValidationResult(valid, errors, warnings, stats)

        except Exception as e:
            logger.error("Orderbook validation error", error=str(e))
            self._stats['failed_validations'] += 1
            return ValidationResult(
                False,
                [f"Validation error: {e}"],
                warnings,
                stats
            )

    def _validate_ohlcv_schema(self, df: pl.DataFrame) -> List[str]:
        """Validate OHLCV schema."""
        errors = []

        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if errors:
            return errors

        # Check for null values
        null_counts = df.null_count()
        for col in required_columns:
            if null_counts[col][0] > 0:
                errors.append(f"Column {col} contains {null_counts[col][0]} null values")

        return errors

    def _validate_ohlcv_ranges(self, df: pl.DataFrame) -> List[str]:
        """Validate OHLCV value ranges."""
        errors = []

        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # Check minimum price
            min_val = df[col].min()
            if min_val is not None:
                min_decimal = Decimal(str(min_val))
                if min_decimal < self.config.min_price:
                    errors.append(
                        f"{col} has value {min_decimal} below minimum {self.config.min_price}"
                    )

                # Check maximum price
                max_val = df[col].max()
                if max_val is not None:
                    max_decimal = Decimal(str(max_val))
                    if max_decimal > self.config.max_price:
                        errors.append(
                            f"{col} has value {max_decimal} above maximum {self.config.max_price}"
                        )

        # Check volume
        if 'volume' in df.columns:
            min_vol = df['volume'].min()
            if min_vol is not None and Decimal(str(min_vol)) < self.config.min_volume:
                errors.append(f"Volume below minimum: {min_vol}")

            max_vol = df['volume'].max()
            if max_vol is not None and Decimal(str(max_vol)) > self.config.max_volume:
                errors.append(f"Volume above maximum: {max_vol}")

        return errors

    def _validate_ohlcv_consistency(self, df: pl.DataFrame) -> List[str]:
        """Validate OHLCV consistency (high >= low, etc.)."""
        errors = []

        # Check high >= low
        invalid_high_low = df.filter(pl.col('high') < pl.col('low'))
        if len(invalid_high_low) > 0:
            errors.append(f"{len(invalid_high_low)} rows with high < low")

        # Check high >= open/close
        invalid_high_open = df.filter(pl.col('high') < pl.col('open'))
        if len(invalid_high_open) > 0:
            errors.append(f"{len(invalid_high_open)} rows with high < open")

        invalid_high_close = df.filter(pl.col('high') < pl.col('close'))
        if len(invalid_high_close) > 0:
            errors.append(f"{len(invalid_high_close)} rows with high < close")

        # Check low <= open/close
        invalid_low_open = df.filter(pl.col('low') > pl.col('open'))
        if len(invalid_low_open) > 0:
            errors.append(f"{len(invalid_low_open)} rows with low > open")

        invalid_low_close = df.filter(pl.col('low') > pl.col('close'))
        if len(invalid_low_close) > 0:
            errors.append(f"{len(invalid_low_close)} rows with low > close")

        return errors

    def _detect_ohlcv_anomalies(self, df: pl.DataFrame) -> List[str]:
        """Detect anomalies in OHLCV data."""
        warnings = []

        if len(df) < 2:
            return warnings

        # Check for large price jumps
        df_sorted = df.sort('timestamp')

        for price_col in ['close']:
            if price_col not in df_sorted.columns:
                continue

            prices = df_sorted[price_col].to_list()

            for i in range(1, len(prices)):
                if prices[i - 1] is None or prices[i] is None:
                    continue

                prev_price = Decimal(str(prices[i - 1]))
                curr_price = Decimal(str(prices[i]))

                if prev_price == 0:
                    continue

                change_percent = abs((curr_price - prev_price) / prev_price * 100)

                if change_percent > self.config.max_price_deviation_percent:
                    warnings.append(
                        f"Large price jump detected: {change_percent:.2f}% at index {i}"
                    )

        return warnings

    def _validate_trades_schema(self, df: pl.DataFrame) -> List[str]:
        """Validate trades schema."""
        errors = []

        required_columns = ['timestamp', 'symbol', 'price', 'volume']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        return errors

    def _validate_trades_ranges(self, df: pl.DataFrame) -> List[str]:
        """Validate trade value ranges."""
        errors = []

        # Check price range
        if 'price' in df.columns:
            min_price = df['price'].min()
            if min_price is not None and Decimal(str(min_price)) < self.config.min_price:
                errors.append(f"Trade price below minimum: {min_price}")

            max_price = df['price'].max()
            if max_price is not None and Decimal(str(max_price)) > self.config.max_price:
                errors.append(f"Trade price above maximum: {max_price}")

        # Check volume range
        if 'volume' in df.columns:
            min_vol = df['volume'].min()
            if min_vol is not None and Decimal(str(min_vol)) < self.config.min_volume:
                errors.append(f"Trade volume below minimum: {min_vol}")

        return errors

    def _detect_trade_anomalies(self, df: pl.DataFrame) -> List[str]:
        """Detect anomalies in trade data."""
        warnings = []

        # Check for duplicate trades
        if 'trade_id' in df.columns:
            duplicates = df.filter(pl.col('trade_id').is_duplicated())
            if len(duplicates) > 0:
                warnings.append(f"{len(duplicates)} duplicate trade IDs detected")

        return warnings

    def _validate_orderbook_schema(self, data: Dict[str, Any]) -> List[str]:
        """Validate orderbook schema."""
        errors = []

        required_fields = ['symbol', 'timestamp', 'bids', 'asks']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if 'bids' in data and not isinstance(data['bids'], list):
            errors.append("bids must be a list")

        if 'asks' in data and not isinstance(data['asks'], list):
            errors.append("asks must be a list")

        return errors

    def _validate_orderbook_ranges(self, data: Dict[str, Any]) -> List[str]:
        """Validate orderbook value ranges."""
        errors = []

        for side in ['bids', 'asks']:
            if side not in data:
                continue

            for i, level in enumerate(data[side]):
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    errors.append(f"{side}[{i}] has invalid format")
                    continue

                price, size = Decimal(str(level[0])), Decimal(str(level[1]))

                if price < self.config.min_price or price > self.config.max_price:
                    errors.append(f"{side}[{i}] price {price} out of range")

                if size < self.config.min_volume or size > self.config.max_volume:
                    errors.append(f"{side}[{i}] size {size} out of range")

        return errors

    def _validate_orderbook_consistency(self, data: Dict[str, Any]) -> List[str]:
        """Validate orderbook consistency."""
        errors = []

        # Check bid/ask spread
        if data.get('bids') and data.get('asks'):
            best_bid = Decimal(str(data['bids'][0][0]))
            best_ask = Decimal(str(data['asks'][0][0]))

            if best_bid >= best_ask:
                errors.append(f"Invalid spread: best_bid {best_bid} >= best_ask {best_ask}")

        # Check bid ordering (descending)
        if data.get('bids'):
            for i in range(len(data['bids']) - 1):
                curr_price = Decimal(str(data['bids'][i][0]))
                next_price = Decimal(str(data['bids'][i + 1][0]))

                if curr_price < next_price:
                    errors.append(f"Bids not sorted descending at index {i}")
                    break

        # Check ask ordering (ascending)
        if data.get('asks'):
            for i in range(len(data['asks']) - 1):
                curr_price = Decimal(str(data['asks'][i][0]))
                next_price = Decimal(str(data['asks'][i + 1][0]))

                if curr_price > next_price:
                    errors.append(f"Asks not sorted ascending at index {i}")
                    break

        return errors

    def get_stats(self) -> Dict[str, int]:
        """
        Get validator statistics.

        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()
