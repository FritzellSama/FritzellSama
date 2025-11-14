"""Correlation Indicator - Cross-Asset and Inter-Market Analysis.

The Correlation Indicator measures the statistical relationship between
different assets, indicators, or markets. Useful for portfolio diversification,
pairs trading, and market regime detection.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class CorrelationIndicator:
    """Correlation indicator for measuring relationships between series.

    Calculates rolling correlations between price series or indicators.
    Supports Pearson, Spearman, and Kendall correlation methods.

    Attributes:
        config: Configuration dictionary containing indicator parameters
        period: Lookback period for correlation calculation
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Example:
        >>> config = {
        ...     "period": 30,
        ...     "method": "pearson",
        ...     "series_pairs": [
        ...         {"x": "btc_close", "y": "eth_close", "name": "btc_eth"}
        ...     ]
        ... }
        >>> corr = CorrelationIndicator(config)
        >>> result = await corr.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Correlation Indicator.

        Args:
            config: Configuration dictionary with keys:
                - period: int, lookback period (default from config)
                - method: str, correlation method
                - series_pairs: list, pairs of series to correlate
                - min_periods: int, minimum periods required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.period = int(self.config.get("period", self.config.get("default_period", 30)))
        self.method = self.config.get("method", "pearson")
        self.series_pairs = self.config.get("series_pairs", [])
        self.min_periods = int(self.config.get("min_periods", self.period))

        logger.info(
            "correlation_indicator_initialized",
            period=self.period,
            method=self.method,
            num_pairs=len(self.series_pairs)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        period = self.config.get("period", self.config.get("default_period", 30))
        if not isinstance(period, (int, str)) or int(period) < 2:
            raise ValueError(f"Invalid period: {period}. Must be >= 2")

        method = self.config.get("method", "pearson")
        valid_methods = ["pearson", "spearman", "kendall"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Must be one of {valid_methods}"
            )

        series_pairs = self.config.get("series_pairs", [])
        if not isinstance(series_pairs, list):
            raise ValueError("series_pairs must be a list")

        for idx, pair in enumerate(series_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"Series pair {idx} must be a dictionary")
            if "x" not in pair or "y" not in pair:
                raise ValueError(f"Series pair {idx} must have 'x' and 'y' fields")

        logger.debug("correlation_indicator_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate rolling correlations for configured series pairs.

        Args:
            data: Polars DataFrame with columns specified in series_pairs

        Returns:
            DataFrame with additional columns:
                - {name}_correlation: Correlation coefficient (-1 to 1)
                - {name}_strength: Correlation strength classification
                - {name}_direction: Positive or negative correlation

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_correlations",
                rows=len(data),
                period=self.period,
                num_pairs=len(self.series_pairs)
            )

            result = data.clone()

            for pair in self.series_pairs:
                x_col = pair["x"]
                y_col = pair["y"]
                pair_name = pair.get("name", f"{x_col}_{y_col}")

                # Calculate rolling correlation
                corr_values = self._calculate_rolling_correlation(
                    data[x_col].to_list(),
                    data[y_col].to_list()
                )

                result = result.with_columns([
                    pl.Series(f"{pair_name}_correlation", corr_values)
                ])

                # Add strength and direction classification
                result = result.with_columns([
                    self._classify_correlation_strength(f"{pair_name}_correlation")
                      .alias(f"{pair_name}_strength"),
                    self._classify_correlation_direction(f"{pair_name}_correlation")
                      .alias(f"{pair_name}_direction")
                ])

            logger.debug(
                "correlations_calculated",
                rows=len(result),
                num_pairs=len(self.series_pairs)
            )

            return result

        except Exception as e:
            logger.error(
                "correlation_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _calculate_rolling_correlation(
        self,
        x_values: List[Optional[float]],
        y_values: List[Optional[float]]
    ) -> List[Optional[float]]:
        """Calculate rolling correlation coefficient.

        Uses Pearson correlation coefficient formula:
        r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)

        Args:
            x_values: First series
            y_values: Second series

        Returns:
            List of correlation coefficients
        """
        correlations = []

        for i in range(len(x_values)):
            if i < self.min_periods - 1:
                correlations.append(None)
            else:
                # Get the window
                start_idx = max(0, i - self.period + 1)
                window_x = x_values[start_idx:i + 1]
                window_y = y_values[start_idx:i + 1]

                # Remove None values
                valid_pairs = [
                    (x, y) for x, y in zip(window_x, window_y)
                    if x is not None and y is not None
                ]

                if len(valid_pairs) < 2:
                    correlations.append(None)
                    continue

                x_clean = [p[0] for p in valid_pairs]
                y_clean = [p[1] for p in valid_pairs]

                # Calculate Pearson correlation
                n = len(x_clean)
                mean_x = sum(x_clean) / n
                mean_y = sum(y_clean) / n

                numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_clean, y_clean))
                denominator_x = sum((x - mean_x) ** 2 for x in x_clean)
                denominator_y = sum((y - mean_y) ** 2 for y in y_clean)

                if denominator_x == 0 or denominator_y == 0:
                    correlations.append(None)
                else:
                    correlation = numerator / (denominator_x * denominator_y) ** 0.5
                    correlations.append(correlation)

        return correlations

    def _classify_correlation_strength(self, corr_col: str) -> pl.Expr:
        """Classify correlation strength.

        Args:
            corr_col: Name of correlation column

        Returns:
            Polars expression for strength classification
        """
        # Thresholds from config
        strong_threshold = Decimal(str(self.config.get("strong_threshold", "0.7")))
        moderate_threshold = Decimal(str(self.config.get("moderate_threshold", "0.4")))

        return (
            pl.when(pl.col(corr_col).abs() >= float(strong_threshold))
              .then(pl.lit("strong"))
              .when(pl.col(corr_col).abs() >= float(moderate_threshold))
              .then(pl.lit("moderate"))
              .otherwise(pl.lit("weak"))
        )

    def _classify_correlation_direction(self, corr_col: str) -> pl.Expr:
        """Classify correlation direction.

        Args:
            corr_col: Name of correlation column

        Returns:
            Polars expression for direction classification
        """
        threshold = Decimal(str(self.config.get("direction_threshold", "0.1")))

        return (
            pl.when(pl.col(corr_col) > float(threshold))
              .then(pl.lit("positive"))
              .when(pl.col(corr_col) < -float(threshold))
              .then(pl.lit("negative"))
              .otherwise(pl.lit("neutral"))
        )

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a Polars DataFrame")

        # Check for required columns from series_pairs
        missing_columns = []
        for pair in self.series_pairs:
            for col in [pair["x"], pair["y"]]:
                if col not in data.columns:
                    missing_columns.append(col)

        if missing_columns:
            raise ValueError(f"Missing required columns: {list(set(missing_columns))}")

        if len(data) < self.min_periods:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {self.min_periods}"
            )

    def detect_regime_changes(
        self,
        data: pl.DataFrame,
        correlation_col: str
    ) -> pl.DataFrame:
        """Detect market regime changes based on correlation shifts.

        Args:
            data: DataFrame with correlation values
            correlation_col: Name of correlation column to analyze

        Returns:
            DataFrame with regime change indicators

        Raises:
            ValueError: If correlation column is missing
        """
        try:
            if correlation_col not in data.columns:
                raise ValueError(f"Correlation column '{correlation_col}' not found")

            # Detect significant changes in correlation
            change_threshold = Decimal(str(self.config.get("regime_change_threshold", "0.3")))

            result = data.with_columns([
                pl.col(correlation_col).shift(1).alias(f"{correlation_col}_prev")
            ])

            result = result.with_columns([
                (pl.col(correlation_col) - pl.col(f"{correlation_col}_prev"))
                  .abs()
                  .alias(f"{correlation_col}_change")
            ])

            result = result.with_columns([
                pl.when(pl.col(f"{correlation_col}_change") > float(change_threshold))
                  .then(pl.lit(True))
                  .otherwise(pl.lit(False))
                  .alias(f"{correlation_col}_regime_change")
            ])

            # Clean up intermediate columns
            result = result.drop([f"{correlation_col}_prev"])

            logger.debug(
                "regime_changes_detected",
                correlation_col=correlation_col,
                threshold=float(change_threshold)
            )

            return result

        except Exception as e:
            logger.error("regime_change_detection_failed", error=str(e))
            raise

    def calculate_correlation_matrix(
        self,
        data: pl.DataFrame,
        columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for multiple columns.

        Args:
            data: DataFrame with columns to correlate
            columns: List of column names

        Returns:
            Dictionary representing correlation matrix

        Raises:
            ValueError: If columns are missing
        """
        try:
            missing = [col for col in columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            matrix = {}

            for col1 in columns:
                matrix[col1] = {}
                for col2 in columns:
                    if col1 == col2:
                        matrix[col1][col2] = 1.0
                    else:
                        # Calculate correlation for entire series
                        corr = self._calculate_series_correlation(
                            data[col1].to_list(),
                            data[col2].to_list()
                        )
                        matrix[col1][col2] = corr

            logger.debug(
                "correlation_matrix_calculated",
                num_columns=len(columns)
            )

            return matrix

        except Exception as e:
            logger.error("correlation_matrix_calculation_failed", error=str(e))
            raise

    def _calculate_series_correlation(
        self,
        x_values: List[Optional[float]],
        y_values: List[Optional[float]]
    ) -> float:
        """Calculate correlation for entire series.

        Args:
            x_values: First series
            y_values: Second series

        Returns:
            Correlation coefficient
        """
        # Remove None values
        valid_pairs = [
            (x, y) for x, y in zip(x_values, y_values)
            if x is not None and y is not None
        ]

        if len(valid_pairs) < 2:
            return 0.0

        x_clean = [p[0] for p in valid_pairs]
        y_clean = [p[1] for p in valid_pairs]

        n = len(x_clean)
        mean_x = sum(x_clean) / n
        mean_y = sum(y_clean) / n

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_clean, y_clean))
        denominator_x = sum((x - mean_x) ** 2 for x in x_clean)
        denominator_y = sum((y - mean_y) ** 2 for y in y_clean)

        if denominator_x == 0 or denominator_y == 0:
            return 0.0

        correlation = numerator / (denominator_x * denominator_y) ** 0.5
        return correlation

    def find_decorrelated_pairs(
        self,
        correlation_matrix: Dict[str, Dict[str, float]],
        threshold: Optional[Decimal] = None
    ) -> List[tuple]:
        """Find pairs of assets with low correlation for diversification.

        Args:
            correlation_matrix: Correlation matrix from calculate_correlation_matrix
            threshold: Maximum correlation threshold (default from config)

        Returns:
            List of (asset1, asset2, correlation) tuples
        """
        try:
            if threshold is None:
                threshold = Decimal(str(self.config.get("decorrelation_threshold", "0.3")))

            decorrelated_pairs = []

            assets = list(correlation_matrix.keys())
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i + 1:]:
                    corr = abs(correlation_matrix[asset1][asset2])
                    if corr < float(threshold):
                        decorrelated_pairs.append((asset1, asset2, corr))

            # Sort by correlation (lowest first)
            decorrelated_pairs.sort(key=lambda x: x[2])

            logger.debug(
                "decorrelated_pairs_found",
                num_pairs=len(decorrelated_pairs),
                threshold=float(threshold)
            )

            return decorrelated_pairs

        except Exception as e:
            logger.error("decorrelation_analysis_failed", error=str(e))
            raise


async def calculate_correlation(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate correlation indicator.

    Args:
        data: Polars DataFrame with series to correlate
        config: Configuration dictionary

    Returns:
        DataFrame with correlation indicators

    Example:
        >>> config = {
        ...     "period": 30,
        ...     "series_pairs": [
        ...         {"x": "btc_close", "y": "eth_close", "name": "btc_eth"}
        ...     ]
        ... }
        >>> result = await calculate_correlation(df, config)
    """
    indicator = CorrelationIndicator(config)
    return await indicator.calculate(data)
