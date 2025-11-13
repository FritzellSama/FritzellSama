"""Feature Selection for Reducing Dimensionality and Improving Model Performance.

This module implements production-ready feature selection methods for trading ML models,
including variance-based, correlation-based, and model-based selection.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Set, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from structlog import get_logger

logger = get_logger(__name__)


class BaseFeatureSelector(ABC):
    """Abstract base class for feature selectors.

    Attributes:
        config: Configuration dictionary
        selected_features: List of selected feature names
        fitted: Whether selector has been fitted
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.selected_features: List[str] = []
        self.fitted: bool = False

    @abstractmethod
    def fit(self, data: pl.DataFrame, labels: Optional[np.ndarray] = None) -> None:
        """Fit selector to data.

        Args:
            data: Polars DataFrame with features
            labels: Optional target labels for supervised selection
        """
        pass

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform data by selecting features.

        Args:
            data: Polars DataFrame with features

        Returns:
            DataFrame with selected features only
        """
        pass

    def fit_transform(
        self,
        data: pl.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> pl.DataFrame:
        """Fit selector and transform data in one step.

        Args:
            data: Polars DataFrame with features
            labels: Optional target labels

        Returns:
            DataFrame with selected features
        """
        self.fit(data, labels)
        return self.transform(data)

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names.

        Returns:
            List of selected feature names

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")
        return self.selected_features.copy()


class VarianceFeatureSelector(BaseFeatureSelector):
    """Select features based on variance threshold.

    Removes features with low variance (nearly constant features).

    Attributes:
        config: Configuration dictionary
        threshold: Variance threshold
        variances: Calculated variances for each feature
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize variance-based feature selector.

        Args:
            config: Configuration with keys:
                - threshold: Variance threshold (default 0.01)
                - exclude_columns: Columns to exclude from selection

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.threshold: Decimal = Decimal(str(config.get("threshold", 0.01)))
        self.exclude_columns: Set[str] = set(config.get("exclude_columns", []))
        self.variances: Dict[str, Decimal] = {}

        logger.debug("variance_selector_initialized", threshold=float(self.threshold))

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        threshold = self.config.get("threshold", 0.01)
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("threshold must be non-negative number")

    def fit(self, data: pl.DataFrame, labels: Optional[np.ndarray] = None) -> None:
        """Fit selector by calculating feature variances.

        Args:
            data: Polars DataFrame with features
            labels: Not used (variance selection is unsupervised)

        Raises:
            ValueError: If data is invalid
        """
        try:
            if not isinstance(data, pl.DataFrame):
                raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

            logger.debug("fitting_variance_selector", n_features=len(data.columns))

            # Calculate variance for numeric columns
            numeric_cols = self._get_numeric_columns(data)
            selectable_cols = [
                col for col in numeric_cols
                if col not in self.exclude_columns
            ]

            for col in selectable_cols:
                variance = data[col].var()
                if variance is not None:
                    self.variances[col] = Decimal(str(variance))

            # Select features above threshold
            self.selected_features = [
                col for col, var in self.variances.items()
                if var >= self.threshold
            ]

            # Add back excluded columns that exist in data
            for col in self.exclude_columns:
                if col in data.columns and col not in self.selected_features:
                    self.selected_features.append(col)

            self.fitted = True

            logger.info(
                "variance_selector_fitted",
                total_features=len(data.columns),
                selected_features=len(self.selected_features),
                removed_features=len(data.columns) - len(self.selected_features)
            )

        except Exception as e:
            logger.error("variance_selector_fit_failed", error=str(e))
            raise

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Select features based on variance.

        Args:
            data: Polars DataFrame with features

        Returns:
            DataFrame with selected features only

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        # Keep only selected features that exist in data
        features_to_keep = [
            col for col in self.selected_features
            if col in data.columns
        ]

        return data.select(features_to_keep)

    def _get_numeric_columns(self, data: pl.DataFrame) -> List[str]:
        """Get list of numeric columns.

        Args:
            data: Polars DataFrame

        Returns:
            List of numeric column names
        """
        numeric_types = [
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64
        ]

        return [
            col for col in data.columns
            if data[col].dtype in numeric_types
        ]

    def get_variance_report(self) -> pl.DataFrame:
        """Get report of feature variances.

        Returns:
            DataFrame with feature names and variances

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        df = pl.DataFrame({
            "feature": list(self.variances.keys()),
            "variance": [float(v) for v in self.variances.values()],
            "selected": [f in self.selected_features for f in self.variances.keys()]
        })

        return df.sort("variance", descending=True)


class CorrelationFeatureSelector(BaseFeatureSelector):
    """Select features by removing highly correlated ones.

    Removes redundant features that are highly correlated with other features.

    Attributes:
        config: Configuration dictionary
        correlation_threshold: Correlation threshold for removal
        correlation_matrix: Calculated correlation matrix
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize correlation-based feature selector.

        Args:
            config: Configuration with keys:
                - correlation_threshold: Threshold for high correlation (default 0.95)
                - method: Correlation method ('pearson' or 'spearman')
                - exclude_columns: Columns to exclude from removal

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.correlation_threshold: Decimal = Decimal(
            str(config.get("correlation_threshold", 0.95))
        )
        self.method: str = config.get("method", "pearson")
        self.exclude_columns: Set[str] = set(config.get("exclude_columns", []))
        self.correlation_matrix: Optional[pl.DataFrame] = None
        self.removed_features: List[Tuple[str, str, Decimal]] = []

        logger.debug(
            "correlation_selector_initialized",
            threshold=float(self.correlation_threshold),
            method=self.method
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        threshold = self.config.get("correlation_threshold", 0.95)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")

        method = self.config.get("method", "pearson")
        if method not in ["pearson", "spearman"]:
            raise ValueError("method must be 'pearson' or 'spearman'")

    def fit(self, data: pl.DataFrame, labels: Optional[np.ndarray] = None) -> None:
        """Fit selector by identifying correlated features.

        Args:
            data: Polars DataFrame with features
            labels: Not used (correlation selection is unsupervised)

        Raises:
            ValueError: If data is invalid
        """
        try:
            if not isinstance(data, pl.DataFrame):
                raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

            logger.debug("fitting_correlation_selector", n_features=len(data.columns))

            # Get numeric columns
            numeric_cols = [
                col for col in data.columns
                if data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]

            # Calculate correlation matrix (convert to pandas for corr calculation)
            numeric_data = data.select(numeric_cols).to_pandas()

            if self.method == "pearson":
                corr_matrix = numeric_data.corr(method="pearson")
            else:
                corr_matrix = numeric_data.corr(method="spearman")

            # Find highly correlated feature pairs
            features_to_remove = set()

            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    correlation = abs(corr_matrix.iloc[i, j])

                    if correlation >= float(self.correlation_threshold):
                        # Remove one of the correlated features
                        # Prefer to keep features not in exclude list
                        if col_i in self.exclude_columns and col_j not in self.exclude_columns:
                            features_to_remove.add(col_j)
                            self.removed_features.append((col_i, col_j, Decimal(str(correlation))))
                        elif col_j in self.exclude_columns and col_i not in self.exclude_columns:
                            features_to_remove.add(col_i)
                            self.removed_features.append((col_i, col_j, Decimal(str(correlation))))
                        else:
                            # Remove the second one arbitrarily
                            features_to_remove.add(col_j)
                            self.removed_features.append((col_i, col_j, Decimal(str(correlation))))

            # Selected features are those not removed
            self.selected_features = [
                col for col in numeric_cols
                if col not in features_to_remove
            ]

            # Add non-numeric columns
            non_numeric_cols = [
                col for col in data.columns
                if col not in numeric_cols
            ]
            self.selected_features.extend(non_numeric_cols)

            self.fitted = True

            logger.info(
                "correlation_selector_fitted",
                total_features=len(data.columns),
                selected_features=len(self.selected_features),
                removed_features=len(features_to_remove)
            )

        except Exception as e:
            logger.error("correlation_selector_fit_failed", error=str(e))
            raise

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Select features by removing correlated ones.

        Args:
            data: Polars DataFrame with features

        Returns:
            DataFrame with selected features only

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        features_to_keep = [
            col for col in self.selected_features
            if col in data.columns
        ]

        return data.select(features_to_keep)

    def get_removed_features_report(self) -> pl.DataFrame:
        """Get report of removed features and their correlations.

        Returns:
            DataFrame with correlated feature pairs

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        if not self.removed_features:
            return pl.DataFrame({
                "feature_1": [],
                "feature_2": [],
                "correlation": []
            })

        df = pl.DataFrame({
            "feature_1": [f1 for f1, _, _ in self.removed_features],
            "feature_2": [f2 for _, f2, _ in self.removed_features],
            "correlation": [float(corr) for _, _, corr in self.removed_features]
        })

        return df.sort("correlation", descending=True)


class ModelBasedFeatureSelector(BaseFeatureSelector):
    """Select features based on model importance.

    Uses a tree-based model to determine feature importance and selects
    the top K features.

    Attributes:
        config: Configuration dictionary
        k: Number of features to select
        model: Trained model for importance calculation
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model-based feature selector.

        Args:
            config: Configuration with keys:
                - k: Number of features to select (or 'auto')
                - model_type: 'classifier' or 'regressor'
                - n_estimators: Number of trees for RF
                - importance_threshold: Minimum importance threshold

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.k: Any = config.get("k", "auto")
        self.model_type: str = config.get("model_type", "classifier")
        self.n_estimators: int = config.get("n_estimators", 100)
        self.importance_threshold: Optional[Decimal] = None

        if "importance_threshold" in config:
            self.importance_threshold = Decimal(str(config["importance_threshold"]))

        self.model: Optional[Any] = None
        self.feature_importances: Dict[str, Decimal] = {}

        logger.debug(
            "model_selector_initialized",
            k=self.k,
            model_type=self.model_type
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        if "model_type" in self.config:
            valid_types = {"classifier", "regressor"}
            model_type = self.config["model_type"]
            if model_type not in valid_types:
                raise ValueError(f"model_type must be one of {valid_types}")

        k = self.config.get("k", "auto")
        if k != "auto" and (not isinstance(k, int) or k <= 0):
            raise ValueError("k must be 'auto' or positive integer")

    def fit(self, data: pl.DataFrame, labels: Optional[np.ndarray] = None) -> None:
        """Fit selector by training model and calculating importances.

        Args:
            data: Polars DataFrame with features
            labels: Target labels (required for model-based selection)

        Raises:
            ValueError: If data or labels are invalid
        """
        try:
            if not isinstance(data, pl.DataFrame):
                raise ValueError(f"data must be Polars DataFrame, got {type(data)}")

            if labels is None:
                raise ValueError("labels required for model-based feature selection")

            logger.debug("fitting_model_selector", n_features=len(data.columns))

            # Convert to numpy for sklearn
            numeric_cols = [
                col for col in data.columns
                if data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]

            X = data.select(numeric_cols).to_numpy()
            feature_names = numeric_cols

            # Train model
            if self.model_type == "classifier":
                self.model = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=42,
                    n_jobs=-1
                )

            self.model.fit(X, labels)

            # Get feature importances
            importances = self.model.feature_importances_
            self.feature_importances = {
                name: Decimal(str(imp))
                for name, imp in zip(feature_names, importances)
            }

            # Select features
            if self.k == "auto":
                # Select features above threshold
                if self.importance_threshold:
                    self.selected_features = [
                        name for name, imp in self.feature_importances.items()
                        if imp >= self.importance_threshold
                    ]
                else:
                    # Select top 50% by default
                    k_auto = max(1, len(feature_names) // 2)
                    sorted_features = sorted(
                        self.feature_importances.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    self.selected_features = [name for name, _ in sorted_features[:k_auto]]
            else:
                # Select top K features
                sorted_features = sorted(
                    self.feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                self.selected_features = [name for name, _ in sorted_features[:self.k]]

            # Add non-numeric columns
            non_numeric_cols = [col for col in data.columns if col not in numeric_cols]
            self.selected_features.extend(non_numeric_cols)

            self.fitted = True

            logger.info(
                "model_selector_fitted",
                total_features=len(data.columns),
                selected_features=len(self.selected_features)
            )

        except Exception as e:
            logger.error("model_selector_fit_failed", error=str(e))
            raise

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Select top K features based on importance.

        Args:
            data: Polars DataFrame with features

        Returns:
            DataFrame with selected features only

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        features_to_keep = [
            col for col in self.selected_features
            if col in data.columns
        ]

        return data.select(features_to_keep)

    def get_importance_report(self) -> pl.DataFrame:
        """Get report of feature importances.

        Returns:
            DataFrame with feature names and importances

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        df = pl.DataFrame({
            "feature": list(self.feature_importances.keys()),
            "importance": [float(imp) for imp in self.feature_importances.values()],
            "selected": [f in self.selected_features for f in self.feature_importances.keys()]
        })

        return df.sort("importance", descending=True)


class CompositeFeatureSelector(BaseFeatureSelector):
    """Composite feature selector combining multiple selection methods.

    Applies multiple selectors in sequence or combines their results.

    Attributes:
        config: Configuration dictionary
        selectors: List of feature selectors
        combination_method: How to combine selector results
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize composite feature selector.

        Args:
            config: Configuration with keys:
                - selectors: List of selector configs
                - combination_method: 'union' or 'intersection'

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.selectors: List[BaseFeatureSelector] = []
        self.combination_method: str = config.get("combination_method", "intersection")

        # Initialize selectors
        for selector_config in config.get("selectors", []):
            selector_type = selector_config.get("type")

            if selector_type == "variance":
                self.selectors.append(VarianceFeatureSelector(selector_config))
            elif selector_type == "correlation":
                self.selectors.append(CorrelationFeatureSelector(selector_config))
            elif selector_type == "model":
                self.selectors.append(ModelBasedFeatureSelector(selector_config))

        logger.info(
            "composite_selector_initialized",
            n_selectors=len(self.selectors),
            method=self.combination_method
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        if "selectors" not in self.config or not self.config["selectors"]:
            raise ValueError("At least one selector config required")

        method = self.config.get("combination_method", "intersection")
        if method not in ["union", "intersection"]:
            raise ValueError("combination_method must be 'union' or 'intersection'")

    def fit(self, data: pl.DataFrame, labels: Optional[np.ndarray] = None) -> None:
        """Fit all selectors.

        Args:
            data: Polars DataFrame with features
            labels: Optional target labels

        Raises:
            ValueError: If data is invalid
        """
        try:
            logger.debug("fitting_composite_selector", n_selectors=len(self.selectors))

            # Fit each selector
            for i, selector in enumerate(self.selectors):
                logger.debug("fitting_selector", index=i, type=type(selector).__name__)
                selector.fit(data, labels)

            # Combine selected features
            feature_sets = [set(s.get_selected_features()) for s in self.selectors]

            if self.combination_method == "intersection":
                # Features selected by ALL selectors
                combined = set.intersection(*feature_sets)
            else:  # union
                # Features selected by ANY selector
                combined = set.union(*feature_sets)

            self.selected_features = list(combined)
            self.fitted = True

            logger.info(
                "composite_selector_fitted",
                selected_features=len(self.selected_features),
                method=self.combination_method
            )

        except Exception as e:
            logger.error("composite_selector_fit_failed", error=str(e))
            raise

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Select features based on combined selection.

        Args:
            data: Polars DataFrame with features

        Returns:
            DataFrame with selected features only

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        features_to_keep = [
            col for col in self.selected_features
            if col in data.columns
        ]

        return data.select(features_to_keep)

    def get_selector_agreement_report(self) -> pl.DataFrame:
        """Get report showing agreement between selectors.

        Returns:
            DataFrame showing which selectors selected which features

        Raises:
            ValueError: If selector not fitted
        """
        if not self.fitted:
            raise ValueError("Selector must be fitted first")

        # Get all unique features from all selectors
        all_features = set()
        for selector in self.selectors:
            all_features.update(selector.get_selected_features())

        # Build agreement matrix
        data_dict = {"feature": list(all_features)}

        for i, selector in enumerate(self.selectors):
            selector_name = f"selector_{i}_{type(selector).__name__}"
            selected = set(selector.get_selected_features())
            data_dict[selector_name] = [f in selected for f in all_features]

        # Add final selection column
        data_dict["final_selected"] = [
            f in self.selected_features for f in all_features
        ]

        return pl.DataFrame(data_dict)
