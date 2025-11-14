"""
Polynomial and Interaction Feature Engineering.

This module creates polynomial and interaction features from raw
market data for enhanced predictive modeling.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

import numpy as np
import polars as pl
from sklearn.preprocessing import PolynomialFeatures as SKLearnPolynomialFeatures
from structlog import get_logger

logger = get_logger(__name__)


class PolynomialFeatures:
    """Production-ready polynomial feature generator.

    Creates polynomial and interaction features from market data
    for machine learning models.

    Attributes:
        config: Configuration dictionary
        degree: Polynomial degree
        include_bias: Whether to include bias term
        interaction_only: Whether to only include interaction terms
        feature_names: Names of input features
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize polynomial feature generator.

        Args:
            config: Configuration dictionary containing:
                - features.polynomial.degree
                - features.polynomial.include_bias
                - features.polynomial.interaction_only
                - features.polynomial.max_features
        """
        self.config = config
        self._validate_config()

        poly_config = self.config["features"]["polynomial"]
        self.degree = poly_config["degree"]
        self.include_bias = poly_config["include_bias"]
        self.interaction_only = poly_config["interaction_only"]
        self.max_features = poly_config.get("max_features", 1000)

        self.feature_names: Optional[List[str]] = None
        self.poly_transformer: Optional[SKLearnPolynomialFeatures] = None

        logger.info("polynomial_features_initialized",
                   degree=self.degree,
                   interaction_only=self.interaction_only)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "features.polynomial.degree",
            "features.polynomial.include_bias",
            "features.polynomial.interaction_only"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

        if self.config["features"]["polynomial"]["degree"] < 1:
            raise ValueError("Polynomial degree must be at least 1")

    def fit(self, feature_names: List[str]) -> None:
        """Fit polynomial transformer with feature names.

        Args:
            feature_names: List of input feature names
        """
        self.feature_names = feature_names

        # Initialize sklearn polynomial features
        self.poly_transformer = SKLearnPolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )

        # Fit with dummy data to get feature names
        dummy_data = np.zeros((1, len(feature_names)))
        self.poly_transformer.fit(dummy_data)

        logger.info("polynomial_transformer_fitted",
                   input_features=len(feature_names),
                   output_features=self.poly_transformer.n_output_features_)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform DataFrame with polynomial features.

        Args:
            df: Polars DataFrame with input features

        Returns:
            DataFrame with polynomial features added

        Raises:
            RuntimeError: If transformer not fitted
            ValueError: If required columns are missing
        """
        if self.poly_transformer is None or self.feature_names is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        try:
            # Validate columns
            for col in self.feature_names:
                if col not in df.columns:
                    raise ValueError(f"Required column {col} not found in DataFrame")

            logger.debug("transforming_polynomial_features", rows=len(df))

            # Extract feature columns
            feature_data = df.select(self.feature_names).to_numpy()

            # Transform
            poly_features = self.poly_transformer.transform(feature_data)

            # Get feature names
            poly_feature_names = self.poly_transformer.get_feature_names_out(self.feature_names)

            # Limit features if necessary
            if poly_features.shape[1] > self.max_features:
                logger.warning("feature_count_exceeds_max",
                             total_features=poly_features.shape[1],
                             max_features=self.max_features)
                poly_features = poly_features[:, :self.max_features]
                poly_feature_names = poly_feature_names[:self.max_features]

            # Create DataFrame with polynomial features
            poly_df = pl.DataFrame(
                poly_features,
                schema=poly_feature_names.tolist()
            )

            # Combine with original DataFrame
            result_df = pl.concat([df, poly_df], how="horizontal")

            logger.debug("polynomial_features_transformed",
                        output_features=len(poly_feature_names))

            return result_df

        except Exception as e:
            logger.error("polynomial_transform_failed", error=str(e))
            raise

    def fit_transform(self, df: pl.DataFrame, feature_names: List[str]) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Polars DataFrame with input features
            feature_names: List of feature names to use

        Returns:
            DataFrame with polynomial features
        """
        self.fit(feature_names)
        return self.transform(df)

    async def create_interaction_features(
        self,
        df: pl.DataFrame,
        feature_pairs: List[Tuple[str, str]]
    ) -> pl.DataFrame:
        """Create specific interaction features.

        Args:
            df: Polars DataFrame
            feature_pairs: List of (feature1, feature2) tuples to interact

        Returns:
            DataFrame with interaction features added

        Raises:
            ValueError: If features not found
        """
        try:
            logger.debug("creating_interaction_features",
                        pair_count=len(feature_pairs))

            result_df = df.clone()

            for feat1, feat2 in feature_pairs:
                if feat1 not in df.columns or feat2 not in df.columns:
                    logger.warning("interaction_feature_not_found",
                                 feature1=feat1,
                                 feature2=feat2)
                    continue

                # Create interaction feature name
                interaction_name = f"{feat1}_x_{feat2}"

                # Calculate interaction (multiplication)
                result_df = result_df.with_columns(
                    (pl.col(feat1) * pl.col(feat2)).alias(interaction_name)
                )

            logger.debug("interaction_features_created",
                        interactions=len(feature_pairs))

            return result_df

        except Exception as e:
            logger.error("interaction_feature_creation_failed", error=str(e))
            raise

    async def create_ratio_features(
        self,
        df: pl.DataFrame,
        numerator_features: List[str],
        denominator_features: List[str]
    ) -> pl.DataFrame:
        """Create ratio features.

        Args:
            df: Polars DataFrame
            numerator_features: Features to use as numerators
            denominator_features: Features to use as denominators

        Returns:
            DataFrame with ratio features added

        Raises:
            ValueError: If features not found
        """
        try:
            logger.debug("creating_ratio_features",
                        numerators=len(numerator_features),
                        denominators=len(denominator_features))

            result_df = df.clone()

            for num_feat in numerator_features:
                for den_feat in denominator_features:
                    if num_feat == den_feat:
                        continue

                    if num_feat not in df.columns or den_feat not in df.columns:
                        continue

                    # Create ratio feature name
                    ratio_name = f"{num_feat}_div_{den_feat}"

                    # Calculate ratio (with zero division protection)
                    result_df = result_df.with_columns(
                        pl.when(pl.col(den_feat) != 0)
                        .then(pl.col(num_feat) / pl.col(den_feat))
                        .otherwise(0)
                        .alias(ratio_name)
                    )

            logger.debug("ratio_features_created")

            return result_df

        except Exception as e:
            logger.error("ratio_feature_creation_failed", error=str(e))
            raise

    async def create_log_features(
        self,
        df: pl.DataFrame,
        features: List[str]
    ) -> pl.DataFrame:
        """Create logarithmic features.

        Args:
            df: Polars DataFrame
            features: Features to transform

        Returns:
            DataFrame with log features added

        Raises:
            ValueError: If features not found
        """
        try:
            logger.debug("creating_log_features", feature_count=len(features))

            result_df = df.clone()

            for feat in features:
                if feat not in df.columns:
                    logger.warning("log_feature_not_found", feature=feat)
                    continue

                # Create log feature name
                log_name = f"log_{feat}"

                # Calculate log (with protection for non-positive values)
                result_df = result_df.with_columns(
                    pl.when(pl.col(feat) > 0)
                    .then(pl.col(feat).log())
                    .otherwise(0)
                    .alias(log_name)
                )

            logger.debug("log_features_created", feature_count=len(features))

            return result_df

        except Exception as e:
            logger.error("log_feature_creation_failed", error=str(e))
            raise

    async def create_power_features(
        self,
        df: pl.DataFrame,
        features: List[str],
        powers: List[int]
    ) -> pl.DataFrame:
        """Create power features.

        Args:
            df: Polars DataFrame
            features: Features to transform
            powers: List of powers to apply

        Returns:
            DataFrame with power features added

        Raises:
            ValueError: If features not found
        """
        try:
            logger.debug("creating_power_features",
                        feature_count=len(features),
                        powers=powers)

            result_df = df.clone()

            for feat in features:
                if feat not in df.columns:
                    logger.warning("power_feature_not_found", feature=feat)
                    continue

                for power in powers:
                    if power == 1:
                        continue

                    # Create power feature name
                    power_name = f"{feat}_pow{power}"

                    # Calculate power
                    result_df = result_df.with_columns(
                        (pl.col(feat) ** power).alias(power_name)
                    )

            logger.debug("power_features_created")

            return result_df

        except Exception as e:
            logger.error("power_feature_creation_failed", error=str(e))
            raise

    async def create_all_engineered_features(
        self,
        df: pl.DataFrame,
        base_features: List[str]
    ) -> pl.DataFrame:
        """Create all engineered features.

        Args:
            df: Polars DataFrame
            base_features: Base features to engineer from

        Returns:
            DataFrame with all engineered features

        Raises:
            ValueError: If input is invalid
        """
        try:
            logger.info("creating_all_engineered_features",
                       rows=len(df),
                       base_features=len(base_features))

            # Validate base features exist
            for feat in base_features:
                if feat not in df.columns:
                    raise ValueError(f"Base feature {feat} not found in DataFrame")

            result_df = df.clone()

            # Create polynomial features if enabled
            if self.config["features"]["polynomial"].get("enabled", True):
                result_df = self.fit_transform(result_df, base_features)

            # Create specific interactions
            if self.config["features"].get("interactions", {}).get("enabled", True):
                # Create pairs for most important features (limit to avoid explosion)
                max_interaction_features = self.config["features"].get("interactions", {}).get("max_feature_pairs", 10)
                important_features = base_features[:min(max_interaction_features, len(base_features))]

                interaction_pairs = list(combinations(important_features, 2))
                result_df = await self.create_interaction_features(result_df, interaction_pairs)

            # Create ratio features
            if self.config["features"].get("ratios", {}).get("enabled", True):
                max_ratio_features = self.config["features"].get("ratios", {}).get("max_features", 5)
                ratio_features = base_features[:min(max_ratio_features, len(base_features))]
                result_df = await self.create_ratio_features(result_df, ratio_features, ratio_features)

            # Create log features for positive-valued features
            if self.config["features"].get("logarithmic", {}).get("enabled", True):
                result_df = await self.create_log_features(result_df, base_features)

            # Create power features
            if self.config["features"].get("powers", {}).get("enabled", True):
                powers = self.config["features"].get("powers", {}).get("degrees", [2, 3])
                max_power_features = self.config["features"].get("powers", {}).get("max_features", 5)
                power_features = base_features[:min(max_power_features, len(base_features))]
                result_df = await self.create_power_features(result_df, power_features, powers)

            logger.info("all_engineered_features_created",
                       input_features=len(df.columns),
                       output_features=len(result_df.columns),
                       features_added=len(result_df.columns) - len(df.columns))

            return result_df

        except Exception as e:
            logger.error("engineered_feature_creation_failed", error=str(e))
            raise

    def get_feature_importance_rank(
        self,
        feature_importance: Dict[str, Decimal]
    ) -> List[str]:
        """Rank features by importance.

        Args:
            feature_importance: Dictionary of feature -> importance score

        Returns:
            List of feature names sorted by importance
        """
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [feat for feat, _ in sorted_features]
