"""Feature validation and sanitization for ML features.

This module provides comprehensive validation and sanitization capabilities
for machine learning features, ensuring data quality and consistency before
model training and inference.
"""

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from datetime import datetime, timezone, timedelta
import polars as pl
import numpy as np
from structlog import get_logger
from enum import Enum

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    pass


class ValueValidationError(ValidationError):
    """Raised when value validation fails."""
    pass


class FeatureValidator:
    """Production-ready feature validator.

    Validates feature data for completeness, correctness, and consistency.
    Supports schema validation, range checks, null handling, outlier detection,
    and custom validation rules.

    Attributes:
        config: Configuration dictionary
        validation_level: Strictness level for validation
        schemas: Registered feature schemas
        rules: Custom validation rules

    Example:
        >>> config = {
        ...     'validation_level': 'strict',
        ...     'max_null_ratio': 0.1,
        ...     'outlier_std_threshold': 3.0,
        ...     'enable_type_coercion': True
        ... }
        >>> validator = FeatureValidator(config)
        >>> result = await validator.validate(features_df)
        >>> if result['is_valid']:
        ...     print("Features are valid!")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature validator.

        Args:
            config: Configuration dictionary with keys:
                - validation_level: Strictness level (strict/moderate/lenient)
                - max_null_ratio: Maximum allowed null ratio per column
                - outlier_std_threshold: Std deviations for outlier detection
                - enable_type_coercion: Allow automatic type coercion
                - min_rows: Minimum required rows
                - max_rows: Maximum allowed rows
                - required_columns: List of required column names

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        level_str = self.config.get('validation_level', 'moderate')
        self.validation_level = ValidationLevel(level_str)

        self.max_null_ratio = Decimal(str(self.config.get('max_null_ratio', 0.1)))
        self.outlier_std_threshold = Decimal(str(self.config.get('outlier_std_threshold', 3.0)))
        self.enable_type_coercion = self.config.get('enable_type_coercion', True)
        self.min_rows = self.config.get('min_rows', 1)
        self.max_rows = self.config.get('max_rows', 1_000_000)
        self.required_columns = self.config.get('required_columns', [])

        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            "feature_validator_initialized",
            validation_level=self.validation_level.value,
            max_null_ratio=str(self.max_null_ratio)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if 'validation_level' in self.config:
            level = self.config['validation_level']
            if level not in ['strict', 'moderate', 'lenient']:
                raise ValueError(f"Invalid validation_level: {level}")

        if 'max_null_ratio' in self.config:
            ratio = self.config['max_null_ratio']
            if not 0 <= ratio <= 1:
                raise ValueError("max_null_ratio must be between 0 and 1")

        if 'outlier_std_threshold' in self.config:
            threshold = self.config['outlier_std_threshold']
            if threshold < 0:
                raise ValueError("outlier_std_threshold must be non-negative")

    async def validate(
        self,
        features: pl.DataFrame,
        schema_name: Optional[str] = None,
        raise_on_error: bool = True
    ) -> Dict[str, Any]:
        """Validate feature DataFrame.

        Args:
            features: Polars DataFrame to validate
            schema_name: Optional schema to validate against
            raise_on_error: Raise exception on validation failure

        Returns:
            Dictionary containing validation results:
                - is_valid: Boolean indicating if validation passed
                - errors: List of validation errors
                - warnings: List of validation warnings
                - statistics: Feature statistics

        Raises:
            ValidationError: If validation fails and raise_on_error is True

        Example:
            >>> result = await validator.validate(df, schema_name='technical')
            >>> if not result['is_valid']:
            ...     print(f"Errors: {result['errors']}")
        """
        errors = []
        warnings = []

        try:
            # Basic structure validation
            struct_errors = await self._validate_structure(features)
            errors.extend(struct_errors)

            # Schema validation
            if schema_name and schema_name in self.schemas:
                schema_errors = await self._validate_schema(features, schema_name)
                errors.extend(schema_errors)

            # Null value validation
            null_errors, null_warnings = await self._validate_nulls(features)
            errors.extend(null_errors)
            warnings.extend(null_warnings)

            # Data type validation
            type_errors = await self._validate_types(features)
            errors.extend(type_errors)

            # Range validation
            range_errors = await self._validate_ranges(features)
            errors.extend(range_errors)

            # Outlier detection
            outlier_warnings = await self._detect_outliers(features)
            warnings.extend(outlier_warnings)

            # Temporal validation (if timestamp column exists)
            if 'timestamp' in features.columns:
                temporal_errors = await self._validate_temporal(features)
                errors.extend(temporal_errors)

            # Custom rules validation
            if schema_name and schema_name in self.rules:
                rule_errors = await self._validate_rules(features, schema_name)
                errors.extend(rule_errors)

            # Calculate statistics
            statistics = await self._calculate_statistics(features)

            is_valid = len(errors) == 0

            result = {
                'is_valid': is_valid,
                'errors': errors,
                'warnings': warnings,
                'statistics': statistics,
                'validation_level': self.validation_level.value,
                'timestamp': datetime.now(timezone.utc)
            }

            if not is_valid:
                logger.warning(
                    "feature_validation_failed",
                    error_count=len(errors),
                    warning_count=len(warnings)
                )

                if raise_on_error:
                    error_msg = "; ".join(errors[:5])  # First 5 errors
                    raise ValidationError(
                        f"Feature validation failed: {error_msg}"
                    )
            else:
                logger.info(
                    "feature_validation_passed",
                    row_count=features.height,
                    column_count=len(features.columns),
                    warning_count=len(warnings)
                )

            return result

        except ValidationError:
            raise
        except Exception as e:
            logger.error("feature_validation_error", error=str(e))
            raise ValidationError(f"Validation failed: {str(e)}") from e

    async def _validate_structure(self, features: pl.DataFrame) -> List[str]:
        """Validate basic DataFrame structure.

        Args:
            features: DataFrame to validate

        Returns:
            List of error messages
        """
        errors = []

        # Check if empty
        if features.is_empty():
            errors.append("DataFrame is empty")
            return errors

        # Check row count
        if features.height < self.min_rows:
            errors.append(
                f"Too few rows: {features.height} < {self.min_rows}"
            )

        if features.height > self.max_rows:
            errors.append(
                f"Too many rows: {features.height} > {self.max_rows}"
            )

        # Check required columns
        missing_cols = set(self.required_columns) - set(features.columns)
        if missing_cols:
            errors.append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )

        # Check for duplicate columns
        if len(features.columns) != len(set(features.columns)):
            errors.append("Duplicate column names detected")

        return errors

    async def _validate_schema(
        self,
        features: pl.DataFrame,
        schema_name: str
    ) -> List[str]:
        """Validate DataFrame against registered schema.

        Args:
            features: DataFrame to validate
            schema_name: Name of schema to validate against

        Returns:
            List of error messages
        """
        errors = []
        schema = self.schemas[schema_name]

        # Check columns match
        expected_cols = set(schema.get('columns', []))
        actual_cols = set(features.columns)

        missing = expected_cols - actual_cols
        if missing:
            errors.append(
                f"Missing schema columns: {', '.join(missing)}"
            )

        # Check data types
        if 'dtypes' in schema:
            for col, expected_dtype in schema['dtypes'].items():
                if col in features.columns:
                    actual_dtype = str(features.schema[col])
                    if actual_dtype != expected_dtype:
                        errors.append(
                            f"Column '{col}' type mismatch: "
                            f"expected {expected_dtype}, got {actual_dtype}"
                        )

        return errors

    async def _validate_nulls(
        self,
        features: pl.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Validate null values in DataFrame.

        Args:
            features: DataFrame to validate

        Returns:
            Tuple of (error messages, warning messages)
        """
        errors = []
        warnings = []

        for col in features.columns:
            null_count = features[col].null_count()
            null_ratio = Decimal(str(null_count)) / Decimal(str(features.height))

            if null_ratio > self.max_null_ratio:
                msg = (
                    f"Column '{col}' has high null ratio: "
                    f"{float(null_ratio):.2%}"
                )

                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        return errors, warnings

    async def _validate_types(self, features: pl.DataFrame) -> List[str]:
        """Validate data types are appropriate.

        Args:
            features: DataFrame to validate

        Returns:
            List of error messages
        """
        errors = []

        for col in features.columns:
            dtype = features.schema[col]

            # Check for object/string types that should be numeric
            if dtype == pl.Utf8:
                # Try to infer if should be numeric
                try:
                    sample = features[col].drop_nulls().head(100)
                    if not sample.is_empty():
                        # Check if all values are numeric strings
                        is_numeric = all(
                            self._is_numeric_string(str(val))
                            for val in sample.to_list()
                        )

                        if is_numeric and self.enable_type_coercion:
                            warnings_msg = (
                                f"Column '{col}' contains numeric strings "
                                f"that could be coerced to Decimal"
                            )
                            if self.validation_level == ValidationLevel.STRICT:
                                errors.append(warnings_msg)

                except Exception as e:
                    logger.debug(f"Type inference failed for {col}", error=str(e))

        return errors

    async def _validate_ranges(self, features: pl.DataFrame) -> List[str]:
        """Validate numeric values are in reasonable ranges.

        Args:
            features: DataFrame to validate

        Returns:
            List of error messages
        """
        errors = []

        numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8}

        for col in features.columns:
            dtype = features.schema[col]

            if dtype in numeric_types:
                try:
                    # Check for infinite values
                    if dtype in {pl.Float64, pl.Float32}:
                        inf_count = features.filter(
                            pl.col(col).is_infinite()
                        ).height

                        if inf_count > 0:
                            errors.append(
                                f"Column '{col}' contains {inf_count} infinite values"
                            )

                    # Check for NaN values
                    if dtype in {pl.Float64, pl.Float32}:
                        nan_count = features.filter(
                            pl.col(col).is_nan()
                        ).height

                        if nan_count > 0:
                            errors.append(
                                f"Column '{col}' contains {nan_count} NaN values"
                            )

                except Exception as e:
                    logger.debug(f"Range validation failed for {col}", error=str(e))

        return errors

    async def _detect_outliers(self, features: pl.DataFrame) -> List[str]:
        """Detect outliers in numeric columns.

        Args:
            features: DataFrame to check

        Returns:
            List of warning messages
        """
        warnings = []

        numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8}

        for col in features.columns:
            dtype = features.schema[col]

            if dtype in numeric_types:
                try:
                    mean = features[col].mean()
                    std = features[col].std()

                    if mean is not None and std is not None and std > 0:
                        threshold = float(self.outlier_std_threshold)

                        outliers = features.filter(
                            (pl.col(col) - mean).abs() > (std * threshold)
                        )

                        outlier_count = outliers.height
                        outlier_ratio = Decimal(str(outlier_count)) / Decimal(str(features.height))

                        if outlier_ratio > Decimal('0.01'):  # More than 1%
                            warnings.append(
                                f"Column '{col}' has {outlier_count} outliers "
                                f"({float(outlier_ratio):.2%})"
                            )

                except Exception as e:
                    logger.debug(f"Outlier detection failed for {col}", error=str(e))

        return warnings

    async def _validate_temporal(self, features: pl.DataFrame) -> List[str]:
        """Validate temporal consistency.

        Args:
            features: DataFrame with timestamp column

        Returns:
            List of error messages
        """
        errors = []

        try:
            timestamps = features['timestamp']

            # Check for null timestamps
            if timestamps.null_count() > 0:
                errors.append("Timestamp column contains null values")

            # Check for duplicate timestamps
            unique_count = timestamps.n_unique()
            if unique_count < features.height:
                dup_count = features.height - unique_count
                msg = f"Found {dup_count} duplicate timestamps"

                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(msg)

            # Check if sorted
            if features.height > 1:
                is_sorted = (timestamps[1:] >= timestamps[:-1]).all()
                if not is_sorted:
                    msg = "Timestamps are not sorted"

                    if self.validation_level == ValidationLevel.STRICT:
                        errors.append(msg)

            # Check for future timestamps
            now = datetime.now(timezone.utc)
            future_count = features.filter(
                pl.col('timestamp') > now
            ).height

            if future_count > 0:
                errors.append(
                    f"Found {future_count} timestamps in the future"
                )

        except Exception as e:
            logger.debug("Temporal validation failed", error=str(e))

        return errors

    async def _validate_rules(
        self,
        features: pl.DataFrame,
        schema_name: str
    ) -> List[str]:
        """Validate custom rules.

        Args:
            features: DataFrame to validate
            schema_name: Name of schema with rules

        Returns:
            List of error messages
        """
        errors = []

        try:
            rules = self.rules.get(schema_name, [])

            for rule in rules:
                try:
                    rule_result = rule(features)
                    if isinstance(rule_result, str):
                        errors.append(rule_result)
                    elif rule_result is False:
                        errors.append(f"Custom rule '{rule.__name__}' failed")

                except Exception as e:
                    errors.append(
                        f"Rule '{rule.__name__}' execution failed: {str(e)}"
                    )

        except Exception as e:
            logger.error("Custom rule validation failed", error=str(e))

        return errors

    async def _calculate_statistics(
        self,
        features: pl.DataFrame
    ) -> Dict[str, Any]:
        """Calculate feature statistics.

        Args:
            features: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        try:
            stats = {
                'row_count': features.height,
                'column_count': len(features.columns),
                'total_nulls': sum(col.null_count() for col in features),
                'memory_bytes': features.estimated_size(),
                'columns': {}
            }

            numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8}

            for col in features.columns:
                dtype = features.schema[col]
                col_stats = {
                    'dtype': str(dtype),
                    'null_count': features[col].null_count(),
                    'unique_count': features[col].n_unique()
                }

                if dtype in numeric_types:
                    try:
                        col_stats['mean'] = features[col].mean()
                        col_stats['std'] = features[col].std()
                        col_stats['min'] = features[col].min()
                        col_stats['max'] = features[col].max()
                    except:
                        pass

                stats['columns'][col] = col_stats

            return stats

        except Exception as e:
            logger.error("Statistics calculation failed", error=str(e))
            return {}

    def register_schema(
        self,
        name: str,
        columns: List[str],
        dtypes: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a feature schema.

        Args:
            name: Schema name
            columns: List of expected column names
            dtypes: Optional dictionary of column -> dtype mappings
            metadata: Optional metadata

        Example:
            >>> validator.register_schema(
            ...     'technical',
            ...     columns=['timestamp', 'symbol', 'rsi', 'macd'],
            ...     dtypes={'rsi': 'Float64', 'macd': 'Float64'}
            ... )
        """
        self.schemas[name] = {
            'columns': columns,
            'dtypes': dtypes or {},
            'metadata': metadata or {}
        }

        logger.info(
            "schema_registered",
            name=name,
            column_count=len(columns)
        )

    def register_rule(
        self,
        schema_name: str,
        rule: Callable[[pl.DataFrame], bool]
    ) -> None:
        """Register a custom validation rule.

        Args:
            schema_name: Schema to apply rule to
            rule: Callable that takes DataFrame and returns bool or error string

        Example:
            >>> def check_positive_values(df: pl.DataFrame) -> bool:
            ...     return (df['price'] > 0).all()
            >>> validator.register_rule('market_data', check_positive_values)
        """
        if schema_name not in self.rules:
            self.rules[schema_name] = []

        self.rules[schema_name].append(rule)

        logger.info(
            "validation_rule_registered",
            schema=schema_name,
            rule=rule.__name__
        )

    @staticmethod
    def _is_numeric_string(s: str) -> bool:
        """Check if string represents a numeric value.

        Args:
            s: String to check

        Returns:
            True if string is numeric
        """
        try:
            Decimal(s)
            return True
        except (InvalidOperation, ValueError):
            return False

    async def sanitize(
        self,
        features: pl.DataFrame,
        drop_nulls: bool = False,
        fill_method: Optional[str] = None,
        remove_outliers: bool = False
    ) -> pl.DataFrame:
        """Sanitize features by handling nulls and outliers.

        Args:
            features: DataFrame to sanitize
            drop_nulls: Drop rows with null values
            fill_method: Method to fill nulls ('forward', 'backward', 'mean', 'zero')
            remove_outliers: Remove outlier rows

        Returns:
            Sanitized DataFrame

        Raises:
            ValidationError: If sanitization fails

        Example:
            >>> clean_df = await validator.sanitize(
            ...     df,
            ...     fill_method='mean',
            ...     remove_outliers=True
            ... )
        """
        try:
            result = features

            # Handle nulls
            if drop_nulls:
                result = result.drop_nulls()
                logger.info(
                    "nulls_dropped",
                    rows_removed=features.height - result.height
                )

            elif fill_method:
                if fill_method == 'zero':
                    result = result.fill_null(0)
                elif fill_method == 'forward':
                    result = result.fill_null(strategy='forward')
                elif fill_method == 'backward':
                    result = result.fill_null(strategy='backward')
                elif fill_method == 'mean':
                    for col in result.columns:
                        dtype = result.schema[col]
                        if dtype in {pl.Float64, pl.Float32, pl.Int64, pl.Int32}:
                            mean_val = result[col].mean()
                            if mean_val is not None:
                                result = result.with_columns(
                                    pl.col(col).fill_null(mean_val)
                                )

                logger.info("nulls_filled", method=fill_method)

            # Remove outliers
            if remove_outliers:
                original_height = result.height

                numeric_types = {pl.Float64, pl.Float32, pl.Int64, pl.Int32}

                for col in result.columns:
                    dtype = result.schema[col]

                    if dtype in numeric_types:
                        mean = result[col].mean()
                        std = result[col].std()

                        if mean is not None and std is not None and std > 0:
                            threshold = float(self.outlier_std_threshold)

                            result = result.filter(
                                (pl.col(col) - mean).abs() <= (std * threshold)
                            )

                logger.info(
                    "outliers_removed",
                    rows_removed=original_height - result.height
                )

            return result

        except Exception as e:
            logger.error("sanitization_failed", error=str(e))
            raise ValidationError(f"Sanitization failed: {str(e)}") from e
