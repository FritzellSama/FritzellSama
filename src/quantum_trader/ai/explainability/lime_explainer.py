"""
LIME Explainer for Model Interpretability.

This module provides Local Interpretable Model-agnostic Explanations (LIME)
for understanding ML model predictions in trading contexts.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import polars as pl
from structlog import get_logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from quantum_trader.exceptions import ExplainerError, ValidationError

logger = get_logger(__name__)


@dataclass
class Explanation:
    """Explanation for a single prediction.

    Attributes:
        instance_id: Identifier for the explained instance
        prediction: Model prediction value
        feature_weights: Feature contribution weights
        local_prediction: LIME local model prediction
        score: R-squared score of local model
        intercept: Intercept of local model
        timestamp: Explanation timestamp
        metadata: Additional metadata
    """
    instance_id: str
    prediction: Decimal
    feature_weights: Dict[str, Decimal]
    local_prediction: Decimal
    score: Decimal
    intercept: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LIMEConfig:
    """Configuration for LIME explainer.

    Attributes:
        num_samples: Number of perturbed samples to generate
        num_features: Number of top features to include in explanation
        kernel_width: Width of the exponential kernel
        feature_selection: Method for feature selection
        discretize_continuous: Whether to discretize continuous features
        distance_metric: Distance metric for weighting samples
        model_regressor: Regressor for local model
        random_state: Random seed for reproducibility
    """
    num_samples: int = 5000
    num_features: int = 10
    kernel_width: Decimal = Decimal("0.25")
    feature_selection: str = "auto"
    discretize_continuous: bool = False
    distance_metric: str = "euclidean"
    model_regressor: Optional[Any] = None
    random_state: int = 42


class LIMEExplainer:
    """LIME explainer for model interpretability.

    Provides local explanations for individual predictions by fitting
    interpretable models on perturbed samples around the instance.

    Attributes:
        config: Explainer configuration
        feature_names: Names of input features
        scaler: Feature scaler for normalization

    Example:
        >>> config = LIMEConfig(
        ...     num_samples=5000,
        ...     num_features=10,
        ...     kernel_width=Decimal("0.25")
        ... )
        >>> explainer = LIMEExplainer(config, feature_names)
        >>> explanation = explainer.explain_instance(
        ...     instance,
        ...     model.predict,
        ...     instance_id="test_1"
        ... )
        >>> print(explanation.feature_weights)
    """

    def __init__(
        self,
        config: LIMEConfig,
        feature_names: List[str]
    ) -> None:
        """Initialize LIME explainer.

        Args:
            config: Explainer configuration
            feature_names: Names of input features

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self.feature_names = feature_names
        self._validate_config()

        # Initialize components
        self.scaler = StandardScaler()
        self.np_random = np.random.RandomState(config.random_state)

        if config.model_regressor is None:
            self.model_regressor = Ridge(alpha=1.0, random_state=config.random_state)
        else:
            self.model_regressor = config.model_regressor

        logger.info(
            "LIME explainer initialized",
            num_features=len(feature_names),
            num_samples=config.num_samples,
            kernel_width=float(config.kernel_width)
        )

    def _validate_config(self) -> None:
        """Validate explainer configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if self.config.num_samples < 1:
            raise ValidationError("num_samples must be >= 1")

        if self.config.num_features < 1:
            raise ValidationError("num_features must be >= 1")

        if self.config.kernel_width <= 0:
            raise ValidationError("kernel_width must be > 0")

        if len(self.feature_names) == 0:
            raise ValidationError("feature_names cannot be empty")

        valid_feature_selection = ["auto", "forward_selection", "lasso_path", "none"]
        if self.config.feature_selection not in valid_feature_selection:
            raise ValidationError(f"Invalid feature_selection: {self.config.feature_selection}")

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance_id: Optional[str] = None,
        num_samples: Optional[int] = None
    ) -> Explanation:
        """Explain a single prediction.

        Args:
            instance: Instance to explain [num_features,]
            predict_fn: Model prediction function
            instance_id: Optional instance identifier
            num_samples: Number of samples (overrides config)

        Returns:
            Explanation object

        Raises:
            ExplainerError: If explanation fails
            ValidationError: If instance is invalid
        """
        try:
            # Validate instance
            self._validate_instance(instance)

            if instance_id is None:
                instance_id = f"instance_{datetime.now(timezone.utc).timestamp()}"

            if num_samples is None:
                num_samples = self.config.num_samples

            logger.debug(
                "Generating explanation",
                instance_id=instance_id,
                num_samples=num_samples
            )

            # Get original prediction
            original_prediction = predict_fn(instance.reshape(1, -1))[0]

            # Generate perturbed samples
            perturbed_samples = self._generate_samples(instance, num_samples)

            # Get predictions for perturbed samples
            predictions = predict_fn(perturbed_samples)

            # Calculate sample weights based on distance
            distances = self._calculate_distances(instance, perturbed_samples)
            sample_weights = self._kernel_function(distances)

            # Fit local linear model
            local_model = self._fit_local_model(
                perturbed_samples,
                predictions,
                sample_weights
            )

            # Extract feature weights
            feature_weights = {}
            coefficients = local_model.coef_
            for i, feat_name in enumerate(self.feature_names):
                feature_weights[feat_name] = Decimal(str(coefficients[i]))

            # Get local prediction
            local_prediction = local_model.predict(instance.reshape(1, -1))[0]

            # Calculate model score
            score = local_model.score(
                perturbed_samples,
                predictions,
                sample_weight=sample_weights
            )

            # Create explanation
            explanation = Explanation(
                instance_id=instance_id,
                prediction=Decimal(str(original_prediction)),
                feature_weights=feature_weights,
                local_prediction=Decimal(str(local_prediction)),
                score=Decimal(str(score)),
                intercept=Decimal(str(local_model.intercept_)),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "num_samples": num_samples,
                    "num_features": len(self.feature_names)
                }
            )

            logger.info(
                "Explanation generated",
                instance_id=instance_id,
                score=float(score),
                top_feature=max(feature_weights, key=lambda k: abs(feature_weights[k]))
            )

            return explanation

        except Exception as e:
            logger.error("Explanation failed", instance_id=instance_id, error=str(e))
            raise ExplainerError(f"Explanation failed: {e}") from e

    def explain_batch(
        self,
        instances: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance_ids: Optional[List[str]] = None
    ) -> List[Explanation]:
        """Explain multiple predictions.

        Args:
            instances: Instances to explain [N, num_features]
            predict_fn: Model prediction function
            instance_ids: Optional instance identifiers

        Returns:
            List of explanation objects

        Raises:
            ExplainerError: If explanation fails
        """
        try:
            if instance_ids is None:
                instance_ids = [f"instance_{i}" for i in range(len(instances))]

            if len(instances) != len(instance_ids):
                raise ValidationError(
                    f"Instance and ID count mismatch: {len(instances)} != {len(instance_ids)}"
                )

            logger.info("Generating batch explanations", num_instances=len(instances))

            explanations = []
            for instance, instance_id in zip(instances, instance_ids):
                explanation = self.explain_instance(instance, predict_fn, instance_id)
                explanations.append(explanation)

            logger.info("Batch explanations completed", num_explanations=len(explanations))

            return explanations

        except Exception as e:
            logger.error("Batch explanation failed", error=str(e))
            raise ExplainerError(f"Batch explanation failed: {e}") from e

    async def explain_instance_async(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        instance_id: Optional[str] = None,
        num_samples: Optional[int] = None
    ) -> Explanation:
        """Explain instance asynchronously.

        Args:
            instance: Instance to explain
            predict_fn: Model prediction function
            instance_id: Optional instance identifier
            num_samples: Number of samples (overrides config)

        Returns:
            Explanation object
        """
        return await asyncio.to_thread(
            self.explain_instance,
            instance,
            predict_fn,
            instance_id,
            num_samples
        )

    def _generate_samples(
        self,
        instance: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """Generate perturbed samples around the instance.

        Args:
            instance: Original instance
            num_samples: Number of samples to generate

        Returns:
            Perturbed samples [num_samples, num_features]
        """
        # Generate samples from normal distribution
        samples = self.np_random.normal(
            loc=instance,
            scale=np.std(instance) * 0.1,  # 10% of std as perturbation
            size=(num_samples, len(instance))
        )

        return samples

    def _calculate_distances(
        self,
        instance: np.ndarray,
        samples: np.ndarray
    ) -> np.ndarray:
        """Calculate distances between instance and samples.

        Args:
            instance: Original instance
            samples: Perturbed samples

        Returns:
            Distance array [num_samples,]
        """
        if self.config.distance_metric == "euclidean":
            distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))
        elif self.config.distance_metric == "manhattan":
            distances = np.sum(np.abs(samples - instance), axis=1)
        elif self.config.distance_metric == "cosine":
            dot_product = np.dot(samples, instance)
            norm_samples = np.linalg.norm(samples, axis=1)
            norm_instance = np.linalg.norm(instance)
            distances = 1 - (dot_product / (norm_samples * norm_instance + 1e-10))
        else:
            # Default to euclidean
            distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))

        return distances

    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        """Calculate sample weights using kernel function.

        Args:
            distances: Distance array

        Returns:
            Weight array [num_samples,]
        """
        # Exponential kernel
        kernel_width = float(self.config.kernel_width)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        return weights

    def _fit_local_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Any:
        """Fit local linear model.

        Args:
            samples: Perturbed samples
            predictions: Model predictions
            weights: Sample weights

        Returns:
            Fitted local model
        """
        # Fit ridge regression
        model = Ridge(alpha=1.0, random_state=self.config.random_state)
        model.fit(samples, predictions, sample_weight=weights)

        return model

    def _validate_instance(self, instance: np.ndarray) -> None:
        """Validate instance.

        Args:
            instance: Instance to validate

        Raises:
            ValidationError: If instance is invalid
        """
        if instance is None or len(instance) == 0:
            raise ValidationError("Instance cannot be empty")

        if instance.ndim != 1:
            raise ValidationError(f"Instance must be 1D array, got {instance.ndim}D")

        if len(instance) != len(self.feature_names):
            raise ValidationError(
                f"Instance dimension mismatch: expected {len(self.feature_names)}, "
                f"got {len(instance)}"
            )

        if np.any(np.isnan(instance)):
            raise ValidationError("Instance contains NaN values")

        if np.any(np.isinf(instance)):
            raise ValidationError("Instance contains Inf values")

    def get_top_features(
        self,
        explanation: Explanation,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, Decimal]]:
        """Get top contributing features.

        Args:
            explanation: Explanation object
            top_k: Number of top features (uses config if None)

        Returns:
            List of (feature_name, weight) tuples
        """
        if top_k is None:
            top_k = self.config.num_features

        # Sort by absolute weight
        sorted_features = sorted(
            explanation.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_features[:top_k]

    def visualize_explanation(
        self,
        explanation: Explanation,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create visualization data for explanation.

        Args:
            explanation: Explanation object
            top_k: Number of top features to include

        Returns:
            Visualization data dictionary
        """
        top_features = self.get_top_features(explanation, top_k)

        viz_data = {
            "instance_id": explanation.instance_id,
            "prediction": float(explanation.prediction),
            "local_prediction": float(explanation.local_prediction),
            "score": float(explanation.score),
            "features": [
                {
                    "name": feat,
                    "weight": float(weight),
                    "direction": "positive" if weight > 0 else "negative"
                }
                for feat, weight in top_features
            ],
            "timestamp": explanation.timestamp.isoformat()
        }

        return viz_data

    def aggregate_explanations(
        self,
        explanations: List[Explanation]
    ) -> Dict[str, Decimal]:
        """Aggregate multiple explanations to find global patterns.

        Args:
            explanations: List of explanation objects

        Returns:
            Aggregated feature importance scores

        Raises:
            ValidationError: If explanations list is empty
        """
        try:
            if not explanations:
                raise ValidationError("Explanations list cannot be empty")

            # Aggregate feature weights
            aggregated_weights: Dict[str, List[Decimal]] = {
                feat: [] for feat in self.feature_names
            }

            for explanation in explanations:
                for feat, weight in explanation.feature_weights.items():
                    if feat in aggregated_weights:
                        aggregated_weights[feat].append(abs(weight))

            # Calculate mean absolute weight for each feature
            mean_weights = {}
            for feat, weights in aggregated_weights.items():
                if weights:
                    mean_weights[feat] = sum(weights) / Decimal(str(len(weights)))
                else:
                    mean_weights[feat] = Decimal("0")

            logger.info(
                "Explanations aggregated",
                num_explanations=len(explanations),
                num_features=len(mean_weights)
            )

            return mean_weights

        except Exception as e:
            logger.error("Aggregation failed", error=str(e))
            raise ExplainerError(f"Aggregation failed: {e}") from e
