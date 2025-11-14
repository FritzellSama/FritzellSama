"""Feature interaction engineering module.

This module creates polynomial and custom interaction features between
existing features to capture non-linear relationships in market data.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from itertools import combinations
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class InteractionFeatureEngineer:
    """Engineer interaction features from base features.

    Creates polynomial, ratio, and custom interaction features
    to capture complex relationships in market data.

    Attributes:
        config: Configuration dictionary
        degree: Polynomial degree for interactions
        feature_names: List of feature names

    Example:
        >>> config = {
        ...     "degree": 2,
        ...     "include_bias": False,
        ...     "interaction_types": ["multiply", "divide", "diff"]
        ... }
        >>> engineer = InteractionFeatureEngineer(config)
        >>> features = await engineer.create_interactions(data, ["price", "volume"])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize interaction feature engineer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.degree = config.get("degree", 2)
        self.include_bias = config.get("include_bias", False)
        self.interaction_types = config.get(
            "interaction_types",
            ["multiply", "divide", "add", "subtract"]
        )

        self.max_interactions = config.get("max_interactions", 100)
        self.min_correlation = Decimal(str(config.get("min_correlation", "0.01")))

        logger.info(
            "Interaction feature engineer initialized",
            degree=self.degree,
            types=self.interaction_types
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        valid_types = ["multiply", "divide", "add", "subtract", "power", "log_ratio"]

        types = self.config.get("interaction_types", [])
        invalid = set(types) - set(valid_types)

        if invalid:
            raise ValueError(f"Invalid interaction types: {invalid}")

    async def create_interactions(
        self,
        data: pl.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None
    ) -> pl.DataFrame:
        """Create interaction features.

        Args:
            data: Input DataFrame
            feature_cols: Columns to create interactions from
            target_col: Optional target column for correlation filtering

        Returns:
            DataFrame with added interaction features

        Raises:
            ValueError: If feature columns not found
        """
        try:
            # Validate columns
            missing = set(feature_cols) - set(data.columns)
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")

            logger.info(
                "Creating interaction features",
                num_features=len(feature_cols),
                types=self.interaction_types
            )

            result = data.clone()

            # Generate pairwise interactions
            interactions = await self._generate_pairwise_interactions(
                result,
                feature_cols
            )

            # Add polynomial features if degree > 2
            if self.degree > 2:
                interactions = await self._add_polynomial_features(
                    interactions,
                    feature_cols
                )

            # Filter by correlation if target provided
            if target_col and target_col in interactions.columns:
                interactions = await self._filter_by_correlation(
                    interactions,
                    target_col
                )

            logger.info(
                "Interaction features created",
                original_features=len(feature_cols),
                new_features=len(interactions.columns) - len(data.columns)
            )

            return interactions

        except Exception as e:
            logger.error("Failed to create interaction features", error=str(e))
            raise

    async def _generate_pairwise_interactions(
        self,
        data: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """Generate pairwise interaction features.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns

        Returns:
            DataFrame with pairwise interactions
        """
        try:
            result = data.clone()
            interactions_added = 0

            # Generate all pairs
            for feat1, feat2 in combinations(feature_cols, 2):
                if interactions_added >= self.max_interactions:
                    break

                # Multiplication
                if "multiply" in self.interaction_types:
                    result = result.with_columns(
                        (pl.col(feat1) * pl.col(feat2))
                        .alias(f"{feat1}_x_{feat2}")
                    )
                    interactions_added += 1

                # Division (with zero protection)
                if "divide" in self.interaction_types:
                    result = result.with_columns(
                        (pl.col(feat1) / (pl.col(feat2) + Decimal("0.0001")))
                        .alias(f"{feat1}_div_{feat2}")
                    )
                    interactions_added += 1

                    result = result.with_columns(
                        (pl.col(feat2) / (pl.col(feat1) + Decimal("0.0001")))
                        .alias(f"{feat2}_div_{feat1}")
                    )
                    interactions_added += 1

                # Addition
                if "add" in self.interaction_types:
                    result = result.with_columns(
                        (pl.col(feat1) + pl.col(feat2))
                        .alias(f"{feat1}_plus_{feat2}")
                    )
                    interactions_added += 1

                # Subtraction
                if "subtract" in self.interaction_types:
                    result = result.with_columns([
                        (pl.col(feat1) - pl.col(feat2))
                        .alias(f"{feat1}_minus_{feat2}"),

                        (pl.col(feat2) - pl.col(feat1))
                        .alias(f"{feat2}_minus_{feat1}")
                    ])
                    interactions_added += 2

                # Log ratio
                if "log_ratio" in self.interaction_types:
                    result = result.with_columns(
                        (pl.col(feat1) / (pl.col(feat2) + Decimal("0.0001")))
                        .log()
                        .alias(f"log_{feat1}_div_{feat2}")
                    )
                    interactions_added += 1

            return result

        except Exception as e:
            logger.error("Failed to generate pairwise interactions", error=str(e))
            raise

    async def _add_polynomial_features(
        self,
        data: pl.DataFrame,
        feature_cols: List[str]
    ) -> pl.DataFrame:
        """Add polynomial features.

        Args:
            data: Input DataFrame
            feature_cols: Feature columns

        Returns:
            DataFrame with polynomial features
        """
        try:
            result = data.clone()

            if "power" not in self.interaction_types:
                return result

            # Add powers up to degree
            for col in feature_cols:
                for degree in range(2, self.degree + 1):
                    result = result.with_columns(
                        (pl.col(col) ** degree)
                        .alias(f"{col}_pow{degree}")
                    )

            # Add square roots (if configured)
            if self.config.get("include_sqrt", False):
                for col in feature_cols:
                    result = result.with_columns(
                        pl.col(col).abs().sqrt()
                        .alias(f"{col}_sqrt")
                    )

            return result

        except Exception as e:
            logger.error("Failed to add polynomial features", error=str(e))
            raise

    async def _filter_by_correlation(
        self,
        data: pl.DataFrame,
        target_col: str
    ) -> pl.DataFrame:
        """Filter features by correlation with target.

        Args:
            data: DataFrame with features
            target_col: Target column

        Returns:
            DataFrame with filtered features
        """
        try:
            # Get interaction columns (those not in original data)
            all_cols = set(data.columns)
            interaction_cols = [
                col for col in all_cols
                if any(op in col for op in ["_x_", "_div_", "_plus_", "_minus_", "_pow", "_sqrt", "log_"])
            ]

            if not interaction_cols:
                return data

            # Calculate correlations
            correlations = {}

            for col in interaction_cols:
                # Simple correlation calculation
                try:
                    col_data = data[col].to_numpy()
                    target_data = data[target_col].to_numpy()

                    # Handle NaN/inf
                    mask = np.isfinite(col_data) & np.isfinite(target_data)

                    if mask.sum() > 0:
                        corr = np.corrcoef(
                            col_data[mask],
                            target_data[mask]
                        )[0, 1]

                        correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
                    else:
                        correlations[col] = 0.0

                except Exception:
                    correlations[col] = 0.0

            # Filter low correlation features
            cols_to_keep = [
                col for col, corr in correlations.items()
                if Decimal(str(corr)) >= self.min_correlation
            ]

            # Keep original columns + high correlation interactions
            original_cols = set(data.columns) - set(interaction_cols)
            final_cols = list(original_cols) + cols_to_keep

            logger.info(
                "Filtered features by correlation",
                original=len(interaction_cols),
                kept=len(cols_to_keep),
                threshold=float(self.min_correlation)
            )

            return data.select(final_cols)

        except Exception as e:
            logger.error("Failed to filter by correlation", error=str(e))
            raise

    async def create_custom_interactions(
        self,
        data: pl.DataFrame,
        custom_interactions: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """Create custom interaction features.

        Args:
            data: Input DataFrame
            custom_interactions: List of custom interaction definitions
                Each dict should have keys: 'name', 'features', 'operation'

        Returns:
            DataFrame with custom interactions

        Example:
            >>> custom = [
            ...     {
            ...         "name": "price_volume_momentum",
            ...         "features": ["price", "volume"],
            ...         "operation": "multiply"
            ...     }
            ... ]
            >>> result = await engineer.create_custom_interactions(data, custom)
        """
        try:
            result = data.clone()

            for interaction in custom_interactions:
                name = interaction["name"]
                features = interaction["features"]
                operation = interaction["operation"]

                # Validate features exist
                missing = set(features) - set(data.columns)
                if missing:
                    logger.warning(
                        "Skipping interaction - missing features",
                        name=name,
                        missing=missing
                    )
                    continue

                # Apply operation
                if operation == "multiply":
                    expr = pl.col(features[0])
                    for feat in features[1:]:
                        expr = expr * pl.col(feat)

                    result = result.with_columns(expr.alias(name))

                elif operation == "sum":
                    expr = pl.col(features[0])
                    for feat in features[1:]:
                        expr = expr + pl.col(feat)

                    result = result.with_columns(expr.alias(name))

                elif operation == "mean":
                    expr = sum(pl.col(feat) for feat in features) / len(features)
                    result = result.with_columns(expr.alias(name))

                elif operation == "max":
                    expr = pl.max_horizontal(*[pl.col(feat) for feat in features])
                    result = result.with_columns(expr.alias(name))

                elif operation == "min":
                    expr = pl.min_horizontal(*[pl.col(feat) for feat in features])
                    result = result.with_columns(expr.alias(name))

            logger.info(
                "Custom interactions created",
                count=len(custom_interactions)
            )

            return result

        except Exception as e:
            logger.error("Failed to create custom interactions", error=str(e))
            raise

    async def create_ratio_features(
        self,
        data: pl.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pl.DataFrame:
        """Create ratio features between columns.

        Args:
            data: Input DataFrame
            numerator_cols: Numerator columns
            denominator_cols: Denominator columns

        Returns:
            DataFrame with ratio features
        """
        try:
            result = data.clone()

            for num_col in numerator_cols:
                for den_col in denominator_cols:
                    if num_col == den_col:
                        continue

                    # Create ratio
                    result = result.with_columns(
                        (pl.col(num_col) / (pl.col(den_col) + Decimal("0.0001")))
                        .alias(f"ratio_{num_col}_to_{den_col}")
                    )

            logger.info(
                "Ratio features created",
                numerators=len(numerator_cols),
                denominators=len(denominator_cols)
            )

            return result

        except Exception as e:
            logger.error("Failed to create ratio features", error=str(e))
            raise

    def get_feature_importance(
        self,
        data: pl.DataFrame,
        target_col: str,
        top_k: int = 20
    ) -> pl.DataFrame:
        """Get top interaction features by correlation.

        Args:
            data: DataFrame with features
            target_col: Target column
            top_k: Number of top features to return

        Returns:
            DataFrame with top features and correlations
        """
        try:
            # Get interaction columns
            interaction_cols = [
                col for col in data.columns
                if any(op in col for op in ["_x_", "_div_", "_plus_", "_minus_", "_pow", "ratio_"])
                and col != target_col
            ]

            if not interaction_cols:
                return pl.DataFrame()

            # Calculate correlations
            correlations = []

            for col in interaction_cols:
                try:
                    col_data = data[col].to_numpy()
                    target_data = data[target_col].to_numpy()

                    mask = np.isfinite(col_data) & np.isfinite(target_data)

                    if mask.sum() > 0:
                        corr = np.corrcoef(
                            col_data[mask],
                            target_data[mask]
                        )[0, 1]

                        if not np.isnan(corr):
                            correlations.append({
                                "feature": col,
                                "correlation": Decimal(str(abs(corr)))
                            })

                except Exception:
                    continue

            # Create DataFrame and sort
            if correlations:
                importance_df = pl.DataFrame(correlations)
                importance_df = importance_df.sort("correlation", descending=True)
                importance_df = importance_df.head(top_k)

                logger.info(
                    "Feature importance calculated",
                    total_features=len(interaction_cols),
                    top_k=top_k
                )

                return importance_df
            else:
                return pl.DataFrame()

        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            raise
