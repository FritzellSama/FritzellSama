"""Composite Indicator - Multi-Indicator Aggregation Framework.

The Composite Indicator combines multiple technical indicators into a single
unified signal, using weighted scoring or machine learning techniques.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class CompositeIndicator:
    """Composite indicator combining multiple technical indicators.

    This indicator aggregates signals from multiple technical indicators
    to produce a unified trading signal. It supports:
    - Weighted averaging
    - Majority voting
    - Threshold-based scoring
    - Custom aggregation functions

    Attributes:
        config: Configuration dictionary containing indicator parameters
        indicators: List of indicator configurations to combine
        aggregation_method: Method for combining signals

    Example:
        >>> config = {
        ...     "indicators": [
        ...         {"name": "rsi", "weight": 0.3, "column": "rsi"},
        ...         {"name": "macd", "weight": 0.4, "column": "macd_signal"},
        ...         {"name": "bb", "weight": 0.3, "column": "bb_percent"}
        ...     ],
        ...     "aggregation_method": "weighted_average"
        ... }
        >>> composite = CompositeIndicator(config)
        >>> result = await composite.calculate(df)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Composite Indicator.

        Args:
            config: Configuration dictionary with keys:
                - indicators: List of indicator configurations
                - aggregation_method: str, aggregation method
                - threshold: Decimal, signal threshold
                - min_indicators: int, minimum indicators required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()
        self.indicators = self.config.get("indicators", [])
        self.aggregation_method = self.config.get("aggregation_method", "weighted_average")
        self.threshold = Decimal(str(self.config.get("threshold", "0.5")))
        self.min_indicators = int(self.config.get("min_indicators", 1))

        logger.info(
            "composite_indicator_initialized",
            num_indicators=len(self.indicators),
            aggregation_method=self.aggregation_method,
            threshold=float(self.threshold)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        indicators = self.config.get("indicators", [])
        if not isinstance(indicators, list):
            raise ValueError("indicators must be a list")

        if len(indicators) == 0:
            logger.warning("no_indicators_configured", message="No indicators in composite")

        # Validate each indicator configuration
        for idx, ind in enumerate(indicators):
            if not isinstance(ind, dict):
                raise ValueError(f"Indicator {idx} must be a dictionary")
            if "column" not in ind:
                raise ValueError(f"Indicator {idx} missing required 'column' field")

        aggregation_method = self.config.get("aggregation_method", "weighted_average")
        valid_methods = ["weighted_average", "majority_vote", "threshold", "custom"]
        if aggregation_method not in valid_methods:
            raise ValueError(
                f"Invalid aggregation_method: {aggregation_method}. "
                f"Must be one of {valid_methods}"
            )

        logger.debug("composite_indicator_config_validated", config=self.config)

    async def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite indicator signal.

        Args:
            data: Polars DataFrame with indicator columns specified in config

        Returns:
            DataFrame with additional columns:
                - composite_score: Aggregated score (-1 to 1)
                - composite_signal: "buy", "sell", or "neutral"
                - composite_strength: Confidence level (0 to 1)
                - composite_components: JSON string of component scores

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        try:
            self._validate_data(data)

            logger.debug(
                "calculating_composite_indicator",
                rows=len(data),
                num_indicators=len(self.indicators),
                method=self.aggregation_method
            )

            if self.aggregation_method == "weighted_average":
                result = self._calculate_weighted_average(data)
            elif self.aggregation_method == "majority_vote":
                result = self._calculate_majority_vote(data)
            elif self.aggregation_method == "threshold":
                result = self._calculate_threshold_based(data)
            else:  # custom
                result = self._calculate_custom(data)

            # Generate signals from composite score
            result = self._generate_signals(result)

            logger.debug(
                "composite_indicator_calculated",
                rows=len(result),
                avg_score=result["composite_score"].mean() if "composite_score" in result.columns else None
            )

            return result

        except Exception as e:
            logger.error(
                "composite_indicator_calculation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _calculate_weighted_average(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite using weighted average.

        Args:
            data: Input DataFrame with indicator columns

        Returns:
            DataFrame with composite_score column
        """
        # Normalize and weight each indicator
        scores = []
        total_weight = Decimal("0")

        for ind in self.indicators:
            column = ind["column"]
            weight = Decimal(str(ind.get("weight", "1.0")))
            total_weight += weight

            # Normalize indicator to -1 to 1 range if needed
            min_val = Decimal(str(ind.get("min_value", "-100")))
            max_val = Decimal(str(ind.get("max_value", "100")))

            normalized_col = f"{column}_normalized"

            # Normalize: ((value - min) / (max - min)) * 2 - 1
            data = data.with_columns([
                (((pl.col(column) - float(min_val)) / (float(max_val) - float(min_val))) * 2 - 1)
                  .alias(normalized_col)
            ])

            scores.append((normalized_col, float(weight)))

        # Calculate weighted sum
        if total_weight > Decimal("0"):
            composite_expr = pl.lit(0.0)
            for col, weight in scores:
                composite_expr = composite_expr + (pl.col(col) * weight)

            data = data.with_columns([
                (composite_expr / float(total_weight)).alias("composite_score")
            ])

            # Clean up normalized columns
            normalized_cols = [s[0] for s in scores]
            data = data.drop(normalized_cols)
        else:
            data = data.with_columns([pl.lit(0.0).alias("composite_score")])

        return data

    def _calculate_majority_vote(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite using majority voting.

        Each indicator votes -1 (sell), 0 (neutral), or 1 (buy).
        Composite score is the sum of votes.

        Args:
            data: Input DataFrame with indicator columns

        Returns:
            DataFrame with composite_score column
        """
        votes_expr = pl.lit(0)

        for ind in self.indicators:
            column = ind["column"]
            buy_threshold = Decimal(str(ind.get("buy_threshold", "0")))
            sell_threshold = Decimal(str(ind.get("sell_threshold", "0")))

            # Determine vote for this indicator
            vote = (
                pl.when(pl.col(column) > float(buy_threshold))
                  .then(pl.lit(1))
                  .when(pl.col(column) < float(sell_threshold))
                  .then(pl.lit(-1))
                  .otherwise(pl.lit(0))
            )

            votes_expr = votes_expr + vote

        # Normalize by number of indicators
        if len(self.indicators) > 0:
            data = data.with_columns([
                (votes_expr / len(self.indicators)).alias("composite_score")
            ])
        else:
            data = data.with_columns([pl.lit(0.0).alias("composite_score")])

        return data

    def _calculate_threshold_based(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite using threshold-based scoring.

        Each indicator that exceeds its threshold adds to the score.

        Args:
            data: Input DataFrame with indicator columns

        Returns:
            DataFrame with composite_score column
        """
        score_expr = pl.lit(0)

        for ind in self.indicators:
            column = ind["column"]
            threshold = Decimal(str(ind.get("threshold", "0")))
            direction = ind.get("direction", "above")  # "above" or "below"

            if direction == "above":
                condition = pl.col(column) > float(threshold)
            else:
                condition = pl.col(column) < float(threshold)

            score_expr = score_expr + pl.when(condition).then(pl.lit(1)).otherwise(pl.lit(0))

        # Normalize by number of indicators
        if len(self.indicators) > 0:
            data = data.with_columns([
                (score_expr / len(self.indicators)).alias("composite_score")
            ])
        else:
            data = data.with_columns([pl.lit(0.0).alias("composite_score")])

        return data

    def _calculate_custom(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite using custom logic.

        This is a placeholder for custom aggregation logic.
        Can be extended based on specific requirements.

        Args:
            data: Input DataFrame with indicator columns

        Returns:
            DataFrame with composite_score column
        """
        # Default to weighted average
        logger.info("using_custom_aggregation", message="Using weighted average as default")
        return self._calculate_weighted_average(data)

    def _generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate trading signals from composite score.

        Args:
            data: DataFrame with composite_score column

        Returns:
            DataFrame with signal columns
        """
        buy_threshold = Decimal(str(self.config.get("buy_threshold", "0.3")))
        sell_threshold = Decimal(str(self.config.get("sell_threshold", "-0.3")))

        data = data.with_columns([
            # Generate signal
            pl.when(pl.col("composite_score") > float(buy_threshold))
              .then(pl.lit("buy"))
              .when(pl.col("composite_score") < float(sell_threshold))
              .then(pl.lit("sell"))
              .otherwise(pl.lit("neutral"))
              .alias("composite_signal"),

            # Calculate strength (absolute value of score)
            pl.col("composite_score").abs().alias("composite_strength")
        ])

        return data

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input DataFrame

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError("Data must be a Polars DataFrame")

        # Check for required indicator columns
        missing_columns = []
        for ind in self.indicators:
            column = ind["column"]
            if column not in data.columns:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(f"Missing required indicator columns: {missing_columns}")

        if len(data) < 1:
            raise ValueError("Insufficient data: DataFrame is empty")

    def get_component_contributions(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate individual contributions of each indicator.

        Args:
            data: DataFrame with indicator and composite values

        Returns:
            DataFrame with contribution columns for each indicator

        Raises:
            ValueError: If required columns are missing
        """
        try:
            if "composite_score" not in data.columns:
                raise ValueError("composite_score not found. Calculate composite first.")

            result = data.clone()

            for ind in self.indicators:
                column = ind["column"]
                weight = Decimal(str(ind.get("weight", "1.0")))

                # Calculate contribution (simplified)
                contribution_col = f"{column}_contribution"
                result = result.with_columns([
                    (pl.col(column) * float(weight)).alias(contribution_col)
                ])

            logger.debug("component_contributions_calculated", num_components=len(self.indicators))

            return result

        except Exception as e:
            logger.error("component_contribution_calculation_failed", error=str(e))
            raise


async def calculate_composite(
    data: pl.DataFrame,
    config: Dict[str, Any]
) -> pl.DataFrame:
    """Convenience function to calculate composite indicator.

    Args:
        data: Polars DataFrame with indicator columns
        config: Configuration dictionary

    Returns:
        DataFrame with composite indicator

    Example:
        >>> config = {
        ...     "indicators": [
        ...         {"column": "rsi", "weight": 0.5},
        ...         {"column": "macd", "weight": 0.5}
        ...     ],
        ...     "aggregation_method": "weighted_average"
        ... }
        >>> result = await calculate_composite(df, config)
    """
    indicator = CompositeIndicator(config)
    return await indicator.calculate(data)
