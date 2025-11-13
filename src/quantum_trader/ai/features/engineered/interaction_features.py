"""
Interaction Feature Engineering for Trading Signals.

This module creates interaction features by combining multiple input
features to capture complex relationships in market data.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl
from structlog import get_logger

from quantum_trader.exceptions import ValidationError, FeatureError

logger = get_logger(__name__)


@dataclass
class InteractionConfig:
    """Configuration for interaction feature generation.

    Attributes:
        feature_pairs: List of feature pairs to interact
        interaction_types: Types of interactions to compute
        polynomial_degree: Degree for polynomial interactions
        enable_all_pairs: Whether to compute all pairwise interactions
        exclude_columns: Columns to exclude from interactions
        prefix: Prefix for generated feature names
    """
    feature_pairs: List[Tuple[str, str]]
    interaction_types: List[str] = None
    polynomial_degree: int = 2
    enable_all_pairs: bool = False
    exclude_columns: List[str] = None
    prefix: str = "inter"

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.interaction_types is None:
            self.interaction_types = ["multiply", "add", "divide", "ratio"]
        if self.exclude_columns is None:
            self.exclude_columns = []


class InteractionFeatureGenerator:
    """Generate interaction features from market data.

    Creates various types of interaction features to capture complex
    relationships between different market indicators and signals.

    Attributes:
        config: Feature generation configuration
        generated_features: Names of generated features

    Example:
        >>> config = InteractionConfig(
        ...     feature_pairs=[("price", "volume"), ("rsi", "macd")],
        ...     interaction_types=["multiply", "ratio"],
        ...     polynomial_degree=2
        ... )
        >>> generator = InteractionFeatureGenerator(config)
        >>> df_with_interactions = generator.generate_features(df)
    """

    def __init__(self, config: InteractionConfig) -> None:
        """Initialize interaction feature generator.

        Args:
            config: Feature generation configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.generated_features: List[str] = []

        logger.info(
            "Interaction feature generator initialized",
            num_pairs=len(config.feature_pairs),
            interaction_types=config.interaction_types,
            polynomial_degree=config.polynomial_degree
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.config.feature_pairs and not self.config.enable_all_pairs:
            raise ValidationError("Must specify feature_pairs or enable_all_pairs")

        if self.config.polynomial_degree < 1:
            raise ValidationError("polynomial_degree must be >= 1")

        valid_types = ["multiply", "add", "subtract", "divide", "ratio", "diff"]
        for interaction_type in self.config.interaction_types:
            if interaction_type not in valid_types:
                raise ValidationError(f"Invalid interaction type: {interaction_type}")

    def generate_features(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate interaction features.

        Args:
            data: Input dataframe with base features

        Returns:
            Dataframe with interaction features added

        Raises:
            ValidationError: If data is invalid
            FeatureError: If feature generation fails
        """
        try:
            self._validate_data(data)

            logger.info("Generating interaction features", num_rows=len(data))

            result = data.clone()

            # Generate pairwise interactions
            if self.config.enable_all_pairs:
                feature_cols = [
                    col for col in data.columns
                    if col not in self.config.exclude_columns
                    and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
                ]
                pairs = [(f1, f2) for i, f1 in enumerate(feature_cols)
                         for f2 in feature_cols[i + 1:]]
            else:
                pairs = self.config.feature_pairs

            # Generate interactions for each pair
            for feat1, feat2 in pairs:
                if feat1 not in data.columns or feat2 not in data.columns:
                    logger.warning(
                        f"Feature pair not found in data",
                        feat1=feat1,
                        feat2=feat2
                    )
                    continue

                result = self._generate_pair_interactions(result, feat1, feat2)

            # Generate polynomial features
            if self.config.polynomial_degree > 1:
                result = self._generate_polynomial_features(result, data.columns)

            logger.info(
                "Interaction features generated",
                num_features=len(self.generated_features),
                total_columns=len(result.columns)
            )

            return result

        except Exception as e:
            logger.error("Feature generation failed", error=str(e))
            raise FeatureError(f"Feature generation failed: {e}") from e

    def _generate_pair_interactions(
        self,
        data: pl.DataFrame,
        feat1: str,
        feat2: str
    ) -> pl.DataFrame:
        """Generate interactions for a feature pair.

        Args:
            data: Input dataframe
            feat1: First feature name
            feat2: Second feature name

        Returns:
            Dataframe with interactions added
        """
        result = data.clone()

        for interaction_type in self.config.interaction_types:
            try:
                feature_name = f"{self.config.prefix}_{feat1}_{interaction_type}_{feat2}"

                if interaction_type == "multiply":
                    result = result.with_columns(
                        (pl.col(feat1) * pl.col(feat2)).alias(feature_name)
                    )

                elif interaction_type == "add":
                    result = result.with_columns(
                        (pl.col(feat1) + pl.col(feat2)).alias(feature_name)
                    )

                elif interaction_type == "subtract":
                    result = result.with_columns(
                        (pl.col(feat1) - pl.col(feat2)).alias(feature_name)
                    )

                elif interaction_type == "divide":
                    # Safe division - replace inf/nan with 0
                    result = result.with_columns(
                        (pl.col(feat1) / (pl.col(feat2) + 1e-10)).alias(feature_name)
                    )
                    result = result.with_columns(
                        pl.col(feature_name).fill_nan(0).fill_null(0)
                    )

                elif interaction_type == "ratio":
                    # Ratio: (f1 - f2) / (f1 + f2)
                    result = result.with_columns(
                        ((pl.col(feat1) - pl.col(feat2)) /
                         (pl.col(feat1) + pl.col(feat2) + 1e-10)).alias(feature_name)
                    )
                    result = result.with_columns(
                        pl.col(feature_name).fill_nan(0).fill_null(0)
                    )

                elif interaction_type == "diff":
                    result = result.with_columns(
                        pl.col(feat1).diff().alias(f"{feature_name}_1")
                    )
                    result = result.with_columns(
                        pl.col(feat2).diff().alias(f"{feature_name}_2")
                    )

                self.generated_features.append(feature_name)

            except Exception as e:
                logger.warning(
                    f"Failed to generate interaction",
                    interaction_type=interaction_type,
                    feat1=feat1,
                    feat2=feat2,
                    error=str(e)
                )

        return result

    def _generate_polynomial_features(
        self,
        data: pl.DataFrame,
        base_columns: List[str]
    ) -> pl.DataFrame:
        """Generate polynomial features.

        Args:
            data: Input dataframe
            base_columns: Base feature columns

        Returns:
            Dataframe with polynomial features added
        """
        result = data.clone()

        numeric_cols = [
            col for col in base_columns
            if col not in self.config.exclude_columns
            and data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]

        for col in numeric_cols:
            for degree in range(2, self.config.polynomial_degree + 1):
                try:
                    feature_name = f"{self.config.prefix}_{col}_pow{degree}"
                    result = result.with_columns(
                        (pl.col(col) ** degree).alias(feature_name)
                    )
                    result = result.with_columns(
                        pl.col(feature_name).fill_nan(0).fill_null(0)
                    )
                    self.generated_features.append(feature_name)

                except Exception as e:
                    logger.warning(
                        f"Failed to generate polynomial feature",
                        column=col,
                        degree=degree,
                        error=str(e)
                    )

        return result

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data.

        Args:
            data: Input dataframe

        Raises:
            ValidationError: If data is invalid
        """
        if data is None or len(data) == 0:
            raise ValidationError("Data cannot be empty")

        if len(data.columns) == 0:
            raise ValidationError("Data must have at least one column")

    async def generate_features_async(
        self,
        data: pl.DataFrame
    ) -> pl.DataFrame:
        """Generate features asynchronously.

        Args:
            data: Input dataframe

        Returns:
            Dataframe with interaction features added
        """
        return await asyncio.to_thread(self.generate_features, data)

    def get_feature_names(self) -> List[str]:
        """Get names of generated features.

        Returns:
            List of feature names
        """
        return self.generated_features.copy()

    def get_feature_importance(
        self,
        data: pl.DataFrame,
        target_col: str
    ) -> Dict[str, Decimal]:
        """Calculate feature importance using correlation.

        Args:
            data: Dataframe with features and target
            target_col: Target column name

        Returns:
            Dictionary of feature importances

        Raises:
            ValidationError: If data is invalid
        """
        try:
            if target_col not in data.columns:
                raise ValidationError(f"Target column not found: {target_col}")

            importances = {}

            for feature in self.generated_features:
                if feature not in data.columns:
                    continue

                # Calculate correlation
                corr = data.select([
                    pl.corr(feature, target_col).alias("correlation")
                ]).item()

                if corr is not None and not np.isnan(corr):
                    importances[feature] = Decimal(str(abs(corr)))
                else:
                    importances[feature] = Decimal("0")

            logger.info(
                "Feature importance calculated",
                num_features=len(importances)
            )

            return importances

        except Exception as e:
            logger.error("Feature importance calculation failed", error=str(e))
            raise FeatureError(f"Feature importance calculation failed: {e}") from e

    def select_top_features(
        self,
        data: pl.DataFrame,
        target_col: str,
        top_k: int
    ) -> List[str]:
        """Select top-k most important features.

        Args:
            data: Dataframe with features and target
            target_col: Target column name
            top_k: Number of top features to select

        Returns:
            List of top feature names

        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if top_k < 1:
                raise ValidationError("top_k must be >= 1")

            importances = self.get_feature_importance(data, target_col)

            # Sort by importance
            sorted_features = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )

            top_features = [feat for feat, _ in sorted_features[:top_k]]

            logger.info(
                "Top features selected",
                top_k=top_k,
                num_selected=len(top_features)
            )

            return top_features

        except Exception as e:
            logger.error("Feature selection failed", error=str(e))
            raise FeatureError(f"Feature selection failed: {e}") from e

    def transform(
        self,
        data: pl.DataFrame,
        selected_features: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """Transform data keeping only selected features.

        Args:
            data: Input dataframe
            selected_features: Features to keep (all generated if None)

        Returns:
            Transformed dataframe

        Raises:
            ValidationError: If data is invalid
        """
        try:
            if selected_features is None:
                selected_features = self.generated_features

            # Keep only selected features plus original columns
            original_cols = [
                col for col in data.columns
                if col not in self.generated_features
            ]

            keep_cols = original_cols + [
                feat for feat in selected_features
                if feat in data.columns
            ]

            result = data.select(keep_cols)

            logger.info(
                "Data transformed",
                original_columns=len(data.columns),
                transformed_columns=len(result.columns)
            )

            return result

        except Exception as e:
            logger.error("Transform failed", error=str(e))
            raise FeatureError(f"Transform failed: {e}") from e

    def reset(self) -> None:
        """Reset generated features list."""
        self.generated_features.clear()
        logger.info("Feature generator reset")
