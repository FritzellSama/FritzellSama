"""Feature Importance Analysis for ML Model Explainability.

This module implements production-ready feature importance calculation and analysis
for understanding ML model predictions in trading systems.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from structlog import get_logger

logger = get_logger(__name__)


class BaseFeatureImportance(ABC):
    """Abstract base class for feature importance calculators.

    Attributes:
        config: Configuration dictionary
        feature_names: List of feature names
        importance_scores: Calculated importance scores
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature importance calculator.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.feature_names: List[str] = []
        self.importance_scores: Optional[np.ndarray] = None

    @abstractmethod
    def calculate(
        self,
        model: Any,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Decimal]:
        """Calculate feature importance.

        Args:
            model: Trained model
            features: Feature matrix
            labels: Target labels
            feature_names: Names of features

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    @abstractmethod
    def get_top_features(self, n: int) -> List[Tuple[str, Decimal]]:
        """Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples
        """
        pass


class TreeBasedFeatureImportance(BaseFeatureImportance):
    """Feature importance using tree-based model (Random Forest).

    Calculates importance using:
    - Mean decrease in impurity (Gini importance)
    - Permutation importance
    - SHAP values (optional)

    Attributes:
        config: Configuration dictionary
        method: Importance calculation method
        n_estimators: Number of trees for RF
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize tree-based feature importance.

        Args:
            config: Configuration with keys:
                - method: 'gini', 'permutation', or 'both'
                - n_estimators: Number of trees (default 100)
                - max_depth: Maximum tree depth
                - random_state: Random seed
                - n_repeats: Number of permutation repeats

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)
        self._validate_config()

        self.method: str = config.get("method", "both")
        self.n_estimators: int = config.get("n_estimators", 100)
        self.max_depth: Optional[int] = config.get("max_depth", None)
        self.random_state: int = config.get("random_state", 42)
        self.n_repeats: int = config.get("n_repeats", 10)

        # Storage for different importance types
        self.gini_importance: Optional[Dict[str, Decimal]] = None
        self.permutation_importance: Optional[Dict[str, Decimal]] = None

        logger.info(
            "tree_feature_importance_initialized",
            method=self.method,
            n_estimators=self.n_estimators
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If config invalid
        """
        valid_methods = {"gini", "permutation", "both"}
        method = self.config.get("method", "both")

        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")

        n_estimators = self.config.get("n_estimators", 100)
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be positive integer")

    def calculate(
        self,
        model: Any,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Decimal]:
        """Calculate feature importance using tree-based methods.

        Args:
            model: Trained sklearn model (or None to train RF)
            features: Feature matrix (shape: n_samples x n_features)
            labels: Target labels
            feature_names: Names of features

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            self._validate_inputs(features, labels, feature_names)
            self.feature_names = feature_names

            logger.debug(
                "calculating_feature_importance",
                n_features=len(feature_names),
                n_samples=len(features),
                method=self.method
            )

            # Train RF model if not provided
            if model is None:
                model = self._train_random_forest(features, labels)

            # Calculate importance based on method
            if self.method in ["gini", "both"]:
                self.gini_importance = self._calculate_gini_importance(model, feature_names)

            if self.method in ["permutation", "both"]:
                self.permutation_importance = self._calculate_permutation_importance(
                    model, features, labels, feature_names
                )

            # Combine or select primary importance
            if self.method == "both":
                importance_dict = self._combine_importance_scores()
            elif self.method == "gini":
                importance_dict = self.gini_importance
            else:
                importance_dict = self.permutation_importance

            self.importance_scores = np.array(
                [float(importance_dict[name]) for name in feature_names],
                dtype=np.float64
            )

            logger.info(
                "feature_importance_calculated",
                top_feature=max(importance_dict.items(), key=lambda x: x[1])[0]
            )

            return importance_dict

        except Exception as e:
            logger.error("feature_importance_calculation_failed", error=str(e))
            raise

    def _validate_inputs(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """Validate inputs.

        Args:
            features: Feature matrix
            labels: Target labels
            feature_names: Feature names

        Raises:
            ValueError: If inputs invalid
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"features must be numpy array, got {type(features)}")

        if not isinstance(labels, np.ndarray):
            raise ValueError(f"labels must be numpy array, got {type(labels)}")

        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have same length: {len(features)} != {len(labels)}"
            )

        if features.shape[1] != len(feature_names):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) must match "
                f"number of features ({features.shape[1]})"
            )

    def _train_random_forest(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Any:
        """Train Random Forest model.

        Args:
            features: Feature matrix
            labels: Target labels

        Returns:
            Trained RandomForest model
        """
        # Determine if classification or regression
        unique_labels = np.unique(labels)

        if len(unique_labels) <= 10:  # Assume classification
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:  # Regression
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )

        logger.debug("training_random_forest", model_type=type(model).__name__)
        model.fit(features, labels)

        return model

    def _calculate_gini_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, Decimal]:
        """Calculate Gini (mean decrease impurity) importance.

        Args:
            model: Trained tree-based model
            feature_names: Feature names

        Returns:
            Dictionary of feature importances
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = model.feature_importances_

        # Convert to Decimal and normalize
        total = Decimal(str(np.sum(importances)))
        importance_dict = {
            name: Decimal(str(imp)) / total
            for name, imp in zip(feature_names, importances)
        }

        logger.debug("gini_importance_calculated", n_features=len(importance_dict))

        return importance_dict

    def _calculate_permutation_importance(
        self,
        model: Any,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Decimal]:
        """Calculate permutation importance.

        Args:
            model: Trained model
            features: Feature matrix
            labels: Target labels
            feature_names: Feature names

        Returns:
            Dictionary of feature importances
        """
        logger.debug("calculating_permutation_importance", n_repeats=self.n_repeats)

        result = permutation_importance(
            model,
            features,
            labels,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )

        importances = result.importances_mean

        # Convert to Decimal and normalize
        total = Decimal(str(np.sum(importances)))
        if total > 0:
            importance_dict = {
                name: Decimal(str(imp)) / total
                for name, imp in zip(feature_names, importances)
            }
        else:
            # All zero importance - distribute equally
            importance_dict = {
                name: Decimal("1") / Decimal(str(len(feature_names)))
                for name in feature_names
            }

        logger.debug("permutation_importance_calculated", n_features=len(importance_dict))

        return importance_dict

    def _combine_importance_scores(self) -> Dict[str, Decimal]:
        """Combine Gini and permutation importance scores.

        Returns:
            Dictionary of combined importance scores
        """
        if self.gini_importance is None or self.permutation_importance is None:
            raise ValueError("Both importance types must be calculated first")

        # Average of both methods
        combined = {}
        for name in self.feature_names:
            gini = self.gini_importance.get(name, Decimal("0"))
            perm = self.permutation_importance.get(name, Decimal("0"))
            combined[name] = (gini + perm) / Decimal("2")

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {name: score / total for name, score in combined.items()}

        return combined

    def get_top_features(self, n: int = 10) -> List[Tuple[str, Decimal]]:
        """Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance_score) tuples sorted by importance

        Raises:
            ValueError: If importance not calculated yet
        """
        if self.importance_scores is None:
            raise ValueError("Must call calculate() first")

        # Get primary importance dict
        if self.method == "both":
            importance_dict = self._combine_importance_scores()
        elif self.method == "gini":
            importance_dict = self.gini_importance
        else:
            importance_dict = self.permutation_importance

        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_features[:n]

    def get_feature_ranking(self) -> List[str]:
        """Get all features ranked by importance.

        Returns:
            List of feature names sorted by importance (descending)

        Raises:
            ValueError: If importance not calculated yet
        """
        top_features = self.get_top_features(n=len(self.feature_names))
        return [name for name, _ in top_features]

    def get_importance_statistics(self) -> Dict[str, Any]:
        """Get statistics about feature importance.

        Returns:
            Dictionary with importance statistics

        Raises:
            ValueError: If importance not calculated yet
        """
        if self.importance_scores is None:
            raise ValueError("Must call calculate() first")

        # Get primary importance dict
        if self.method == "both":
            importance_dict = self._combine_importance_scores()
        elif self.method == "gini":
            importance_dict = self.gini_importance
        else:
            importance_dict = self.permutation_importance

        scores = [float(score) for score in importance_dict.values()]

        return {
            "n_features": len(self.feature_names),
            "mean_importance": float(np.mean(scores)),
            "std_importance": float(np.std(scores)),
            "max_importance": float(np.max(scores)),
            "min_importance": float(np.min(scores)),
            "top_feature": max(importance_dict.items(), key=lambda x: x[1])[0],
            "method": self.method,
        }

    def export_to_dataframe(self) -> pl.DataFrame:
        """Export feature importance to Polars DataFrame.

        Returns:
            DataFrame with feature names and importance scores

        Raises:
            ValueError: If importance not calculated yet
        """
        if self.importance_scores is None:
            raise ValueError("Must call calculate() first")

        # Get all importance types
        data = {"feature_name": self.feature_names}

        if self.gini_importance:
            data["gini_importance"] = [
                float(self.gini_importance[name]) for name in self.feature_names
            ]

        if self.permutation_importance:
            data["permutation_importance"] = [
                float(self.permutation_importance[name]) for name in self.feature_names
            ]

        if self.method == "both":
            combined = self._combine_importance_scores()
            data["combined_importance"] = [
                float(combined[name]) for name in self.feature_names
            ]

        df = pl.DataFrame(data)

        # Sort by primary importance
        sort_col = "combined_importance" if self.method == "both" else f"{self.method}_importance"
        df = df.sort(sort_col, descending=True)

        return df

    def visualize_importance(self, top_n: int = 20) -> Dict[str, Any]:
        """Generate data for importance visualization.

        Args:
            top_n: Number of top features to include

        Returns:
            Dictionary with visualization data

        Raises:
            ValueError: If importance not calculated yet
        """
        if self.importance_scores is None:
            raise ValueError("Must call calculate() first")

        top_features = self.get_top_features(n=top_n)

        return {
            "feature_names": [name for name, _ in top_features],
            "importance_scores": [float(score) for _, score in top_features],
            "title": f"Top {top_n} Feature Importances ({self.method.title()})",
            "xlabel": "Importance Score",
            "ylabel": "Features",
        }


class FeatureImportanceAnalyzer:
    """High-level feature importance analyzer with multiple methods.

    Attributes:
        config: Configuration dictionary
        calculators: Dictionary of importance calculators
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize feature importance analyzer.

        Args:
            config: Configuration with keys:
                - methods: List of methods to use ['tree', 'permutation']
                - tree_config: Config for tree-based importance
        """
        self.config = config
        self.calculators: Dict[str, BaseFeatureImportance] = {}

        # Initialize calculators
        if "tree" in config.get("methods", ["tree"]):
            tree_config = config.get("tree_config", {})
            tree_config.setdefault("method", "both")
            self.calculators["tree"] = TreeBasedFeatureImportance(tree_config)

        logger.info(
            "feature_importance_analyzer_initialized",
            methods=list(self.calculators.keys())
        )

    def analyze(
        self,
        model: Any,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict[str, Decimal]]:
        """Analyze feature importance using all configured methods.

        Args:
            model: Trained model
            features: Feature matrix
            labels: Target labels
            feature_names: Feature names

        Returns:
            Dictionary mapping method names to importance dictionaries
        """
        results = {}

        for method_name, calculator in self.calculators.items():
            logger.debug("analyzing_with_method", method=method_name)
            try:
                importance = calculator.calculate(model, features, labels, feature_names)
                results[method_name] = importance
            except Exception as e:
                logger.error(
                    "importance_calculation_failed",
                    method=method_name,
                    error=str(e)
                )

        return results

    def get_consensus_ranking(self) -> List[str]:
        """Get consensus feature ranking across all methods.

        Returns:
            List of feature names sorted by average rank across methods
        """
        if not self.calculators:
            return []

        # Get rankings from each calculator
        rankings = {}
        for method_name, calculator in self.calculators.items():
            try:
                ranking = calculator.get_feature_ranking()
                rankings[method_name] = {
                    name: idx for idx, name in enumerate(ranking)
                }
            except ValueError:
                logger.warning("calculator_not_ready", method=method_name)

        if not rankings:
            return []

        # Calculate average rank for each feature
        all_features = list(rankings[list(rankings.keys())[0]].keys())
        avg_ranks = {}

        for feature in all_features:
            ranks = [rankings[method][feature] for method in rankings]
            avg_ranks[feature] = np.mean(ranks)

        # Sort by average rank
        sorted_features = sorted(avg_ranks.items(), key=lambda x: x[1])

        return [name for name, _ in sorted_features]
