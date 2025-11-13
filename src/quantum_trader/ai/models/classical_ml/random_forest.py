"""Random Forest model for trading prediction.

This module implements a Random Forest classifier/regressor optimized for trading
with proper Decimal handling and comprehensive evaluation metrics.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import joblib

logger = structlog.get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base for all ML models."""

    def __init__(self, config: Dict) -> None:
        """Initialize base model.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass


@dataclass
class FeatureImportance:
    """Feature importance information.

    Attributes:
        feature_idx: Feature index
        importance: Importance score (Decimal)
        feature_name: Optional feature name
    """
    feature_idx: int
    importance: Decimal
    feature_name: Optional[str] = None


class RandomForestModel(BaseMLModel):
    """Random Forest model for trading predictions.

    Supports both classification and regression tasks with comprehensive
    feature importance analysis and evaluation metrics.

    Attributes:
        config: Configuration dictionary
        task_type: 'classification' or 'regression'
        model: Sklearn Random Forest model
        feature_names: Optional feature names

    Example:
        >>> config = {
        ...     'task_type': 'classification',
        ...     'n_estimators': 100,
        ...     'max_depth': 10,
        ...     'min_samples_split': 5,
        ...     'min_samples_leaf': 2,
        ...     'max_features': 'sqrt',
        ...     'random_state': 42,
        ...     'n_jobs': -1,
        ...     'class_weight': 'balanced'
        ... }
        >>> model = RandomForestModel(config)
        >>> model.train(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Random Forest model.

        Args:
            config: Configuration with keys:
                - task_type: 'classification' or 'regression'
                - n_estimators: Number of trees
                - max_depth: Maximum tree depth
                - min_samples_split: Minimum samples to split
                - min_samples_leaf: Minimum samples per leaf
                - max_features: Features to consider for splits
                - random_state: Random seed
                - n_jobs: Number of parallel jobs
                - class_weight: Class weights (classification only)
        """
        super().__init__(config)
        self._validate_config()

        self.task_type = config['task_type']
        self.feature_names: Optional[List[str]] = None
        self.model: Optional[Any] = None
        self.is_trained = False

        # Initialize appropriate model
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config.get('max_depth'),
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                max_features=config['max_features'],
                random_state=config['random_state'],
                n_jobs=config['n_jobs'],
                class_weight=config.get('class_weight', None),
                bootstrap=config.get('bootstrap', True),
                oob_score=config.get('oob_score', False),
                warm_start=config.get('warm_start', False),
                verbose=0
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=config['n_estimators'],
                max_depth=config.get('max_depth'),
                min_samples_split=config['min_samples_split'],
                min_samples_leaf=config['min_samples_leaf'],
                max_features=config['max_features'],
                random_state=config['random_state'],
                n_jobs=config['n_jobs'],
                bootstrap=config.get('bootstrap', True),
                oob_score=config.get('oob_score', False),
                warm_start=config.get('warm_start', False),
                verbose=0
            )

        logger.info(
            "initialized_random_forest",
            task_type=self.task_type,
            n_estimators=config['n_estimators'],
            max_depth=config.get('max_depth')
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'task_type', 'n_estimators', 'min_samples_split',
            'min_samples_leaf', 'max_features', 'random_state', 'n_jobs'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['task_type'] not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")

        if self.config['n_estimators'] <= 0:
            raise ValueError("n_estimators must be positive")

        if self.config['min_samples_split'] < 2:
            raise ValueError("min_samples_split must be >= 2")

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """Train Random Forest model.

        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (n_samples,)
            feature_names: Optional feature names

        Raises:
            ValueError: If input shapes are invalid
        """
        try:
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Feature and label count mismatch: {features.shape[0]} != {labels.shape[0]}"
                )

            if features.shape[0] == 0:
                raise ValueError("Cannot train on empty dataset")

            logger.info(
                "starting_training",
                n_samples=features.shape[0],
                n_features=features.shape[1],
                task_type=self.task_type
            )

            # Store feature names
            if feature_names is not None:
                if len(feature_names) != features.shape[1]:
                    raise ValueError("Feature names count mismatch")
                self.feature_names = feature_names

            # Train model
            self.model.fit(features, labels)
            self.is_trained = True

            # Log training results
            train_score = self.model.score(features, labels)

            logger.info(
                "training_complete",
                train_score=float(train_score),
                n_trees=len(self.model.estimators_),
                oob_score=float(self.model.oob_score_) if hasattr(self.model, 'oob_score_') else None
            )

        except Exception as e:
            logger.error("failed_to_train", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained or invalid input
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if features.shape[0] == 0:
                raise ValueError("Cannot predict on empty dataset")

            predictions = self.model.predict(features)

            logger.debug(
                "made_predictions",
                n_samples=features.shape[0],
                prediction_shape=predictions.shape
            )

            return predictions

        except Exception as e:
            logger.error("failed_to_predict", error=str(e))
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only).

        Args:
            features: Input features

        Returns:
            Class probabilities (n_samples, n_classes)

        Raises:
            ValueError: If not a classification model
        """
        try:
            if self.task_type != 'classification':
                raise ValueError("predict_proba only available for classification")

            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            probabilities = self.model.predict_proba(features)

            logger.debug(
                "predicted_probabilities",
                n_samples=features.shape[0],
                n_classes=probabilities.shape[1]
            )

            return probabilities

        except Exception as e:
            logger.error("failed_to_predict_proba", error=str(e))
            raise

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If model not trained
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            predictions = self.predict(features)

            metrics: Dict[str, float] = {}

            if self.task_type == 'classification':
                # Classification metrics
                metrics['accuracy'] = accuracy_score(labels, predictions)
                metrics['precision'] = precision_score(
                    labels, predictions, average='weighted', zero_division=0
                )
                metrics['recall'] = recall_score(
                    labels, predictions, average='weighted', zero_division=0
                )
                metrics['f1'] = f1_score(
                    labels, predictions, average='weighted', zero_division=0
                )

                # Per-class metrics
                unique_classes = np.unique(labels)
                for cls in unique_classes:
                    binary_labels = (labels == cls).astype(int)
                    binary_preds = (predictions == cls).astype(int)
                    metrics[f'precision_class_{cls}'] = precision_score(
                        binary_labels, binary_preds, zero_division=0
                    )
                    metrics[f'recall_class_{cls}'] = recall_score(
                        binary_labels, binary_preds, zero_division=0
                    )

            else:  # regression
                # Regression metrics
                metrics['mse'] = mean_squared_error(labels, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(labels, predictions)
                metrics['r2'] = r2_score(labels, predictions)

                # Additional metrics
                residuals = labels - predictions
                metrics['mean_residual'] = float(np.mean(residuals))
                metrics['std_residual'] = float(np.std(residuals))

            logger.info(
                "evaluation_complete",
                task_type=self.task_type,
                metrics={k: float(v) for k, v in metrics.items()}
            )

            return metrics

        except Exception as e:
            logger.error("failed_to_evaluate", error=str(e))
            raise

    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> List[FeatureImportance]:
        """Get feature importance scores.

        Args:
            top_n: Return only top N features (None = all)

        Returns:
            List of FeatureImportance objects sorted by importance

        Raises:
            ValueError: If model not trained
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained to get feature importance")

            importances = self.model.feature_importances_

            # Create FeatureImportance objects
            feature_importance_list = []
            for idx, importance in enumerate(importances):
                feature_name = None
                if self.feature_names is not None:
                    feature_name = self.feature_names[idx]

                feature_importance_list.append(
                    FeatureImportance(
                        feature_idx=idx,
                        importance=Decimal(str(importance)),
                        feature_name=feature_name
                    )
                )

            # Sort by importance
            feature_importance_list.sort(
                key=lambda x: x.importance,
                reverse=True
            )

            # Return top N if specified
            if top_n is not None:
                feature_importance_list = feature_importance_list[:top_n]

            logger.debug(
                "retrieved_feature_importance",
                total_features=len(importances),
                top_n=top_n
            )

            return feature_importance_list

        except Exception as e:
            logger.error("failed_to_get_feature_importance", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Save path
        """
        try:
            if not self.is_trained:
                logger.warning("saving_untrained_model", path=path)

            model_data = {
                'model': self.model,
                'config': self.config,
                'task_type': self.task_type,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }

            joblib.dump(model_data, path)

            logger.info("saved_model", path=path)

        except Exception as e:
            logger.error("failed_to_save_model", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Load path
        """
        try:
            model_data = joblib.load(path)

            self.model = model_data['model']
            self.config = model_data['config']
            self.task_type = model_data['task_type']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']

            logger.info(
                "loaded_model",
                path=path,
                task_type=self.task_type,
                is_trained=self.is_trained
            )

        except Exception as e:
            logger.error("failed_to_load_model", error=str(e), path=path)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary of model information
        """
        info = {
            'task_type': self.task_type,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names
        }

        if self.is_trained:
            info['n_estimators'] = len(self.model.estimators_)
            info['max_depth'] = self.model.max_depth

            if hasattr(self.model, 'oob_score_'):
                info['oob_score'] = float(self.model.oob_score_)

            if self.task_type == 'classification':
                info['n_classes'] = len(self.model.classes_)
                info['classes'] = self.model.classes_.tolist()

        return info
