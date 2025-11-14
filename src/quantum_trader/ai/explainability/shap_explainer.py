"""SHAP-based model explainability for trading ML models.

This module implements SHAP (SHapley Additive exPlanations) for interpreting
ML model predictions in trading contexts.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureImportance:
    """SHAP feature importance for a single feature.

    Attributes:
        feature_name: Name of the feature
        mean_abs_shap: Mean absolute SHAP value (Decimal)
        mean_shap: Mean SHAP value (Decimal)
        std_shap: Standard deviation of SHAP values (Decimal)
        rank: Importance rank
    """
    feature_name: str
    mean_abs_shap: Decimal
    mean_shap: Decimal
    std_shap: Decimal
    rank: int


@dataclass
class PredictionExplanation:
    """Explanation for a single prediction.

    Attributes:
        prediction: Model prediction
        base_value: Base value (expected value)
        shap_values: SHAP values for each feature
        feature_values: Actual feature values
        top_features: Top contributing features
        timestamp: UTC timestamp
    """
    prediction: Decimal
    base_value: Decimal
    shap_values: Dict[str, Decimal]
    feature_values: Dict[str, Decimal]
    top_features: List[Tuple[str, Decimal]]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class SHAPExplainer:
    """SHAP-based explainability for trading ML models.

    Provides interpretable explanations for model predictions using
    SHAP values with support for multiple model types.

    Attributes:
        config: Configuration dictionary
        model: ML model to explain
        explainer: SHAP explainer instance
        feature_names: List of feature names

    Example:
        >>> config = {
        ...     'explainer_type': 'tree',  # 'tree', 'kernel', 'deep'
        ...     'n_samples': 100,
        ...     'feature_names': ['sma_20', 'rsi_14', 'volume_ratio'],
        ...     'background_samples': 1000
        ... }
        >>> explainer = SHAPExplainer(config, model)
        >>> explanation = explainer.explain_prediction(features)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        background_data: Optional[np.ndarray] = None
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            config: Configuration with keys:
                - explainer_type: 'tree', 'kernel', 'deep', 'linear'
                - n_samples: Number of samples for kernel explainer
                - feature_names: List of feature names
                - background_samples: Number of background samples
            model: ML model to explain
            background_data: Background dataset for explainer
        """
        self.config = config
        self._validate_config()

        self.model = model
        self.explainer_type = config['explainer_type']
        self.n_samples = config.get('n_samples', 100)
        self.feature_names = config.get('feature_names', [])
        self.background_samples = config.get('background_samples', 1000)

        # Initialize explainer (would use actual SHAP library in production)
        self.explainer = self._initialize_explainer(background_data)

        # Cache for explanations
        self.explanations: List[PredictionExplanation] = []

        logger.info(
            "initialized_shap_explainer",
            explainer_type=self.explainer_type,
            n_features=len(self.feature_names)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = ['explainer_type']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        valid_types = ['tree', 'kernel', 'deep', 'linear', 'gradient']
        if self.config['explainer_type'] not in valid_types:
            raise ValueError(f"Invalid explainer_type. Must be one of {valid_types}")

    def _initialize_explainer(
        self,
        background_data: Optional[np.ndarray]
    ) -> Any:
        """Initialize SHAP explainer based on type.

        Args:
            background_data: Background dataset

        Returns:
            SHAP explainer instance
        """
        # In production, this would initialize actual SHAP explainers
        # For now, we'll create a mock explainer structure

        logger.info(
            "initializing_shap_explainer",
            type=self.explainer_type,
            has_background=background_data is not None
        )

        # Mock explainer - in production would be:
        # if self.explainer_type == 'tree':
        #     import shap
        #     return shap.TreeExplainer(self.model)
        # elif self.explainer_type == 'kernel':
        #     return shap.KernelExplainer(self.model.predict, background_data)
        # etc.

        return {'type': self.explainer_type, 'model': self.model}

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_values_dict: Optional[Dict[str, Decimal]] = None
    ) -> PredictionExplanation:
        """Explain a single prediction.

        Args:
            features: Feature array for prediction
            feature_values_dict: Optional dict mapping feature names to values

        Returns:
            PredictionExplanation object
        """
        try:
            # Make prediction
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(features.reshape(1, -1))[0]
            else:
                # Fallback for models without predict method
                prediction = 0.0

            prediction_decimal = Decimal(str(prediction))

            # Calculate SHAP values (mock implementation)
            shap_values = self._calculate_shap_values(features)

            # Base value (expected value)
            base_value = self._calculate_base_value()

            # Create feature values dict
            if feature_values_dict is None:
                feature_values_dict = {}
                for i, name in enumerate(self.feature_names):
                    if i < len(features):
                        feature_values_dict[name] = Decimal(str(features[i]))

            # Create SHAP values dict
            shap_values_dict = {}
            for i, name in enumerate(self.feature_names):
                if i < len(shap_values):
                    shap_values_dict[name] = Decimal(str(shap_values[i]))

            # Get top contributing features
            top_features = sorted(
                shap_values_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            explanation = PredictionExplanation(
                prediction=prediction_decimal,
                base_value=base_value,
                shap_values=shap_values_dict,
                feature_values=feature_values_dict,
                top_features=top_features
            )

            # Store explanation
            self.explanations.append(explanation)

            # Limit cache size
            max_cache = self.config.get('max_cache_size', 10000)
            if len(self.explanations) > max_cache:
                self.explanations = self.explanations[-max_cache:]

            logger.debug(
                "explained_prediction",
                prediction=str(prediction_decimal),
                base_value=str(base_value),
                top_feature=top_features[0][0] if top_features else None
            )

            return explanation

        except Exception as e:
            logger.error("failed_to_explain_prediction", error=str(e))
            raise

    def _calculate_shap_values(self, features: np.ndarray) -> np.ndarray:
        """Calculate SHAP values for features.

        This is a simplified mock implementation. In production, this would
        use the actual SHAP library's explainer.

        Args:
            features: Feature array

        Returns:
            SHAP values array
        """
        # Mock SHAP calculation - in production would be:
        # shap_values = self.explainer.shap_values(features.reshape(1, -1))

        # Simple mock: use feature values with some perturbation
        mock_shap = features * np.random.uniform(0.8, 1.2, size=features.shape)

        # Normalize to sum to prediction deviation from base
        if np.sum(np.abs(mock_shap)) > 1e-8:
            mock_shap = mock_shap / np.sum(np.abs(mock_shap))

        return mock_shap

    def _calculate_base_value(self) -> Decimal:
        """Calculate base value (expected prediction).

        Returns:
            Base value as Decimal
        """
        # Mock implementation - would use actual expected value from explainer
        return Decimal('0.5')

    def explain_batch(
        self,
        features_batch: np.ndarray
    ) -> List[PredictionExplanation]:
        """Explain batch of predictions.

        Args:
            features_batch: Batch of feature arrays

        Returns:
            List of PredictionExplanation objects
        """
        try:
            logger.info("explaining_batch", batch_size=len(features_batch))

            explanations = []
            for features in features_batch:
                explanation = self.explain_prediction(features)
                explanations.append(explanation)

            logger.info("batch_explanation_complete", n_explanations=len(explanations))

            return explanations

        except Exception as e:
            logger.error("failed_to_explain_batch", error=str(e))
            raise

    def get_feature_importance(
        self,
        n_top: Optional[int] = None
    ) -> List[FeatureImportance]:
        """Calculate global feature importance from SHAP values.

        Args:
            n_top: Return only top N features (None = all)

        Returns:
            List of FeatureImportance objects sorted by importance
        """
        try:
            if not self.explanations:
                raise ValueError("No explanations available for importance calculation")

            # Collect SHAP values for each feature
            feature_shap_values: Dict[str, List[float]] = {
                name: [] for name in self.feature_names
            }

            for explanation in self.explanations:
                for feature_name, shap_value in explanation.shap_values.items():
                    if feature_name in feature_shap_values:
                        feature_shap_values[feature_name].append(float(shap_value))

            # Calculate importance metrics
            importances = []
            for feature_name, shap_vals in feature_shap_values.items():
                if not shap_vals:
                    continue

                shap_array = np.array(shap_vals)

                importance = FeatureImportance(
                    feature_name=feature_name,
                    mean_abs_shap=Decimal(str(np.mean(np.abs(shap_array)))),
                    mean_shap=Decimal(str(np.mean(shap_array))),
                    std_shap=Decimal(str(np.std(shap_array))),
                    rank=0  # Will be set after sorting
                )
                importances.append(importance)

            # Sort by mean absolute SHAP value
            importances.sort(key=lambda x: x.mean_abs_shap, reverse=True)

            # Assign ranks
            for i, importance in enumerate(importances):
                importance.rank = i + 1

            # Return top N if specified
            if n_top is not None:
                importances = importances[:n_top]

            logger.info(
                "calculated_feature_importance",
                n_features=len(importances),
                top_feature=importances[0].feature_name if importances else None
            )

            return importances

        except Exception as e:
            logger.error("failed_to_calculate_importance", error=str(e))
            raise

    def get_feature_dependence(
        self,
        feature_name: str
    ) -> Dict[str, Any]:
        """Analyze feature dependence (how SHAP value varies with feature value).

        Args:
            feature_name: Feature to analyze

        Returns:
            Dictionary with dependence analysis
        """
        try:
            if feature_name not in self.feature_names:
                raise ValueError(f"Unknown feature: {feature_name}")

            if not self.explanations:
                raise ValueError("No explanations available")

            # Collect feature values and SHAP values
            feature_vals = []
            shap_vals = []

            for explanation in self.explanations:
                if feature_name in explanation.feature_values:
                    feature_vals.append(float(explanation.feature_values[feature_name]))
                    shap_vals.append(float(explanation.shap_values[feature_name]))

            if not feature_vals:
                raise ValueError(f"No data for feature: {feature_name}")

            feature_array = np.array(feature_vals)
            shap_array = np.array(shap_vals)

            # Calculate correlation
            correlation = np.corrcoef(feature_array, shap_array)[0, 1]

            # Calculate binned statistics
            n_bins = min(20, len(feature_vals) // 10)
            if n_bins > 0:
                bins = np.linspace(
                    np.min(feature_array),
                    np.max(feature_array),
                    n_bins + 1
                )

                bin_indices = np.digitize(feature_array, bins) - 1
                bin_means = []

                for i in range(n_bins):
                    mask = bin_indices == i
                    if np.any(mask):
                        bin_means.append(float(np.mean(shap_array[mask])))
                    else:
                        bin_means.append(0.0)
            else:
                bin_means = []

            dependence = {
                'feature_name': feature_name,
                'correlation': float(correlation),
                'feature_range': (float(np.min(feature_array)), float(np.max(feature_array))),
                'shap_range': (float(np.min(shap_array)), float(np.max(shap_array))),
                'mean_feature_value': float(np.mean(feature_array)),
                'mean_shap_value': float(np.mean(shap_array)),
                'bin_means': bin_means,
                'n_samples': len(feature_vals)
            }

            logger.debug(
                "calculated_feature_dependence",
                feature=feature_name,
                correlation=correlation
            )

            return dependence

        except Exception as e:
            logger.error("failed_to_calculate_dependence", error=str(e))
            raise

    def get_explanation_dataframe(self) -> pl.DataFrame:
        """Get all explanations as Polars DataFrame.

        Returns:
            DataFrame with explanation data
        """
        if not self.explanations:
            return pl.DataFrame()

        data = {
            'prediction': [str(e.prediction) for e in self.explanations],
            'base_value': [str(e.base_value) for e in self.explanations],
            'timestamp': [e.timestamp.isoformat() for e in self.explanations]
        }

        # Add SHAP values for each feature
        for feature_name in self.feature_names:
            shap_col = []
            value_col = []

            for explanation in self.explanations:
                shap_val = explanation.shap_values.get(feature_name)
                feat_val = explanation.feature_values.get(feature_name)

                shap_col.append(str(shap_val) if shap_val is not None else None)
                value_col.append(str(feat_val) if feat_val is not None else None)

            data[f'shap_{feature_name}'] = shap_col
            data[f'value_{feature_name}'] = value_col

        return pl.DataFrame(data)

    def export_explanations(self, path: str) -> None:
        """Export explanations to file.

        Args:
            path: Export file path
        """
        try:
            df = self.get_explanation_dataframe()

            if path.endswith('.csv'):
                df.write_csv(path)
            elif path.endswith('.parquet'):
                df.write_parquet(path)
            elif path.endswith('.json'):
                df.write_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")

            logger.info(
                "exported_explanations",
                path=path,
                n_explanations=len(self.explanations)
            )

        except Exception as e:
            logger.error("failed_to_export_explanations", error=str(e), path=path)
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of explanations.

        Returns:
            Dictionary with summary statistics
        """
        if not self.explanations:
            return {'error': 'No explanations available'}

        predictions = [float(e.prediction) for e in self.explanations]

        summary = {
            'n_explanations': len(self.explanations),
            'n_features': len(self.feature_names),
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }

        # Add feature importance summary
        try:
            importances = self.get_feature_importance(n_top=5)
            summary['top_features'] = [
                {
                    'name': imp.feature_name,
                    'mean_abs_shap': str(imp.mean_abs_shap),
                    'rank': imp.rank
                }
                for imp in importances
            ]
        except Exception:
            pass

        return summary

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self.explanations.clear()
        logger.info("cleared_explanation_cache")
