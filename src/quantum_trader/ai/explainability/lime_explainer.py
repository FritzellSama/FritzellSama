"""LIME explainer for model interpretability.

This module implements Local Interpretable Model-agnostic Explanations (LIME)
for understanding model predictions on individual samples.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from structlog import get_logger

logger = get_logger(__name__)


class LIMEExplainer:
    """LIME explainer for model interpretability.

    Generates local explanations for individual predictions by fitting
    interpretable models around the prediction of interest.

    Attributes:
        config: Configuration dictionary
        model: Model to explain
        feature_names: List of feature names

    Example:
        >>> config = {
        ...     "num_samples": 5000,
        ...     "kernel_width": "0.75",
        ...     "feature_selection": "auto",
        ...     "num_features": 10
        ... }
        >>> explainer = LIMEExplainer(config, model)
        >>> explanation = await explainer.explain_instance(
        ...     instance,
        ...     feature_names=["price", "volume", "rsi"]
        ... )
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Initialize LIME explainer.

        Args:
            config: Configuration dictionary
            model: Model to explain (must have predict method)
            feature_names: Optional feature names

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self.model = model
        self.feature_names = feature_names
        self._validate_config()

        # LIME parameters
        self.num_samples = config.get("num_samples", 5000)
        self.kernel_width = Decimal(str(config.get("kernel_width", "0.75")))
        self.feature_selection = config.get("feature_selection", "auto")
        self.num_features = config.get("num_features", 10)

        # Sampling parameters
        self.sample_around_instance = config.get("sample_around_instance", True)
        self.random_state = config.get("random_state", 42)

        # Ridge regression for local model
        self.ridge_alpha = Decimal(str(config.get("ridge_alpha", "1.0")))

        np.random.seed(self.random_state)

        logger.info(
            "LIME explainer initialized",
            num_samples=self.num_samples,
            kernel_width=float(self.kernel_width),
            num_features=self.num_features
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must have predict method")

        valid_selection = ["auto", "forward_selection", "lasso_path", "none"]
        selection = self.config.get("feature_selection", "auto")

        if selection not in valid_selection:
            raise ValueError(
                f"Invalid feature_selection '{selection}'. "
                f"Must be one of {valid_selection}"
            )

    async def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Explain a single prediction.

        Args:
            instance: Instance to explain [feature_dim]
            feature_names: Optional feature names
            labels: Optional labels to explain (for classification)

        Returns:
            Dictionary containing explanation

        Raises:
            ValueError: If instance shape is invalid
        """
        try:
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)

            if feature_names is not None:
                self.feature_names = feature_names

            logger.info(
                "Explaining instance",
                num_features=instance.shape[1],
                num_samples=self.num_samples
            )

            # Generate neighborhood samples
            neighborhood_samples = self._generate_neighborhood(instance)

            # Get predictions for neighborhood
            neighborhood_predictions = await self._predict_neighborhood(
                neighborhood_samples
            )

            # Calculate sample weights (kernel)
            sample_weights = self._kernel_fn(
                instance,
                neighborhood_samples
            )

            # Fit local interpretable model
            local_model, used_features = self._fit_local_model(
                neighborhood_samples,
                neighborhood_predictions,
                sample_weights
            )

            # Extract feature contributions
            feature_weights = self._extract_feature_weights(
                local_model,
                used_features
            )

            # Generate explanation
            explanation = {
                "instance": instance.flatten().tolist(),
                "prediction": float(self.model.predict(instance)[0]),
                "feature_weights": feature_weights,
                "local_r2": self._calculate_local_r2(
                    local_model,
                    neighborhood_samples,
                    neighborhood_predictions,
                    sample_weights
                ),
                "num_features": len(used_features),
                "timestamp": datetime.utcnow()
            }

            logger.info(
                "Explanation generated",
                prediction=explanation["prediction"],
                local_r2=explanation["local_r2"],
                top_features=list(feature_weights.keys())[:5]
            )

            return explanation

        except Exception as e:
            logger.error("Failed to explain instance", error=str(e))
            raise

    def _generate_neighborhood(
        self,
        instance: np.ndarray
    ) -> np.ndarray:
        """Generate neighborhood samples around instance.

        Args:
            instance: Instance to sample around

        Returns:
            Neighborhood samples [num_samples, feature_dim]
        """
        feature_dim = instance.shape[1]

        if self.sample_around_instance:
            # Sample around instance with perturbations
            # Calculate feature std from instance (simplified)
            std_scale = float(self.config.get("perturbation_std", "0.1"))

            perturbations = np.random.normal(
                0,
                std_scale,
                size=(self.num_samples, feature_dim)
            )

            neighborhood = instance + perturbations

        else:
            # Sample from training distribution (simplified - uniform)
            neighborhood = np.random.uniform(
                -1, 1,
                size=(self.num_samples, feature_dim)
            )

        return neighborhood

    async def _predict_neighborhood(
        self,
        neighborhood: np.ndarray
    ) -> np.ndarray:
        """Get model predictions for neighborhood.

        Args:
            neighborhood: Neighborhood samples

        Returns:
            Predictions
        """
        if asyncio.iscoroutinefunction(self.model.predict):
            predictions = await self.model.predict(neighborhood)
        else:
            predictions = self.model.predict(neighborhood)

        return predictions.flatten()

    def _kernel_fn(
        self,
        instance: np.ndarray,
        samples: np.ndarray
    ) -> np.ndarray:
        """Calculate kernel weights for samples.

        Args:
            instance: Instance to explain
            samples: Neighborhood samples

        Returns:
            Sample weights
        """
        # Euclidean distance
        distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))

        # Exponential kernel
        kernel_width = float(self.kernel_width) * np.sqrt(instance.shape[1])
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        return weights

    def _fit_local_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> tuple[Ridge, List[int]]:
        """Fit local interpretable model.

        Args:
            samples: Neighborhood samples
            predictions: Model predictions
            weights: Sample weights

        Returns:
            Tuple of (local_model, used_feature_indices)
        """
        # Feature selection
        if self.feature_selection == "auto" or self.feature_selection == "none":
            # Use top-k features by correlation
            correlations = np.array([
                np.abs(np.corrcoef(samples[:, i], predictions)[0, 1])
                for i in range(samples.shape[1])
            ])

            # Handle NaN
            correlations = np.nan_to_num(correlations, 0)

            # Select top features
            top_k = min(self.num_features, samples.shape[1])
            used_features = np.argsort(correlations)[-top_k:].tolist()

        elif self.feature_selection == "forward_selection":
            used_features = self._forward_feature_selection(
                samples,
                predictions,
                weights
            )

        else:
            # Use all features
            used_features = list(range(samples.shape[1]))

        # Fit Ridge regression
        local_model = Ridge(
            alpha=float(self.ridge_alpha),
            fit_intercept=True
        )

        local_model.fit(
            samples[:, used_features],
            predictions,
            sample_weight=weights
        )

        return local_model, used_features

    def _forward_feature_selection(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> List[int]:
        """Forward feature selection.

        Args:
            samples: Neighborhood samples
            predictions: Model predictions
            weights: Sample weights

        Returns:
            Selected feature indices
        """
        selected_features = []
        remaining_features = list(range(samples.shape[1]))

        for _ in range(min(self.num_features, samples.shape[1])):
            best_feature = None
            best_score = float('-inf')

            for feat in remaining_features:
                # Try adding this feature
                test_features = selected_features + [feat]

                # Fit model
                model = Ridge(alpha=float(self.ridge_alpha))
                model.fit(
                    samples[:, test_features],
                    predictions,
                    sample_weight=weights
                )

                # Calculate weighted R2
                score = model.score(
                    samples[:, test_features],
                    predictions,
                    sample_weight=weights
                )

                if score > best_score:
                    best_score = score
                    best_feature = feat

            if best_feature is not None:
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)

        return selected_features

    def _extract_feature_weights(
        self,
        local_model: Ridge,
        used_features: List[int]
    ) -> Dict[str, Decimal]:
        """Extract feature weights from local model.

        Args:
            local_model: Fitted local model
            used_features: Feature indices used

        Returns:
            Dictionary mapping feature names to weights
        """
        weights = {}

        for i, feat_idx in enumerate(used_features):
            coef = Decimal(str(local_model.coef_[i]))

            if self.feature_names and feat_idx < len(self.feature_names):
                feat_name = self.feature_names[feat_idx]
            else:
                feat_name = f"feature_{feat_idx}"

            weights[feat_name] = coef

        # Sort by absolute value
        weights = dict(
            sorted(
                weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )

        return weights

    def _calculate_local_r2(
        self,
        local_model: Ridge,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Decimal:
        """Calculate local R-squared.

        Args:
            local_model: Local model
            samples: Samples
            predictions: True predictions
            weights: Sample weights

        Returns:
            R-squared score
        """
        # Get feature indices from model
        n_features = local_model.coef_.shape[0]

        # Calculate R2
        r2 = local_model.score(
            samples[:, :n_features],
            predictions,
            sample_weight=weights
        )

        return Decimal(str(r2))

    async def explain_batch(
        self,
        instances: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Explain multiple instances.

        Args:
            instances: Instances to explain [num_instances, feature_dim]
            feature_names: Optional feature names

        Returns:
            List of explanations
        """
        try:
            logger.info("Explaining batch", num_instances=len(instances))

            explanations = []

            for i in range(len(instances)):
                explanation = await self.explain_instance(
                    instances[i],
                    feature_names=feature_names
                )
                explanations.append(explanation)

            logger.info("Batch explanation completed", count=len(explanations))

            return explanations

        except Exception as e:
            logger.error("Failed to explain batch", error=str(e))
            raise

    def get_feature_importance(
        self,
        explanations: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """Aggregate feature importance from multiple explanations.

        Args:
            explanations: List of explanations

        Returns:
            DataFrame with aggregated feature importance
        """
        try:
            # Collect all feature weights
            feature_weights = {}

            for explanation in explanations:
                for feat_name, weight in explanation["feature_weights"].items():
                    if feat_name not in feature_weights:
                        feature_weights[feat_name] = []

                    feature_weights[feat_name].append(abs(float(weight)))

            # Calculate statistics
            importance_data = []

            for feat_name, weights in feature_weights.items():
                importance_data.append({
                    "feature": feat_name,
                    "mean_abs_weight": Decimal(str(np.mean(weights))),
                    "std_abs_weight": Decimal(str(np.std(weights))),
                    "max_abs_weight": Decimal(str(np.max(weights))),
                    "frequency": len(weights)
                })

            # Create DataFrame
            df = pl.DataFrame(importance_data)
            df = df.sort("mean_abs_weight", descending=True)

            logger.info(
                "Feature importance aggregated",
                num_features=len(importance_data),
                num_explanations=len(explanations)
            )

            return df

        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            raise
