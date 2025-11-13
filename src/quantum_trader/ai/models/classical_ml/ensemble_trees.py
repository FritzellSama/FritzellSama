"""
Ensemble Tree Models (Random Forest, Gradient Boosting)

CRITICAL: Production-grade tree ensemble models
- Random Forest for feature importance
- Gradient Boosting for high accuracy
- SHAP values for explainability
- ONNX export for speed
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
import logging
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    import joblib
except ImportError as e:
    raise ImportError(f"Sklearn not installed: {e}")

logger = logging.getLogger(__name__)


class BaseMLModel(ABC):
    """Abstract base class from API contract"""

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class EnsembleTreeModel(BaseMLModel):
    """
    Ensemble Tree Models for Trading

    Supports Random Forest and Gradient Boosting with production features
    """

    def __init__(self, config: Dict) -> None:
        """Initialize ensemble tree model"""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Load configuration
        self.model_type = config.get('model_type', os.getenv('TREE_MODEL_TYPE', 'random_forest'))
        self.task_type = config.get('task_type', os.getenv('TREE_TASK_TYPE', 'classification'))

        # Hyperparameters from config
        self.n_estimators = int(config.get('n_estimators', os.getenv('TREE_N_ESTIMATORS', '1000')))
        self.max_depth = int(config.get('max_depth', os.getenv('TREE_MAX_DEPTH', '6')))
        self.learning_rate = float(config.get('learning_rate', os.getenv('TREE_LEARNING_RATE', '0.05')))
        self.min_samples_split = int(config.get('min_samples_split', os.getenv('TREE_MIN_SAMPLES_SPLIT', '10')))
        self.min_samples_leaf = int(config.get('min_samples_leaf', os.getenv('TREE_MIN_SAMPLES_LEAF', '4')))
        self.max_features = config.get('max_features', os.getenv('TREE_MAX_FEATURES', 'sqrt'))
        self.random_state = config.get('random_state', 42)

        # Initialize model
        self.model = self._create_model()
        self.feature_importance_: Optional[np.ndarray] = None
        self.is_trained = False

        self.logger.info(f"Initialized {self.model_type} {self.task_type} model")

    def _create_model(self) -> Any:
        """Create the underlying sklearn model"""
        common_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': 0
        }

        if self.model_type == 'random_forest':
            common_params['max_features'] = self.max_features
            if self.task_type == 'classification':
                return RandomForestClassifier(**common_params)
            else:
                return RandomForestRegressor(**common_params)

        elif self.model_type == 'gradient_boosting':
            common_params['learning_rate'] = self.learning_rate
            common_params['subsample'] = float(self.config.get('subsample', '0.8'))
            if self.task_type == 'classification':
                return GradientBoostingClassifier(**common_params)
            else:
                return GradientBoostingRegressor(**common_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the ensemble model"""
        try:
            self.logger.info(f"Training {self.model_type} on {features.shape[0]} samples")
            start_time = datetime.utcnow()

            # Validate inputs
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Feature-label mismatch: {features.shape[0]} vs {labels.shape[0]}")

            # Train model
            self.model.fit(features, labels.ravel() if labels.ndim > 1 else labels)

            # Store feature importance
            self.feature_importance_ = self.model.feature_importances_
            self.is_trained = True

            training_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Training completed in {training_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise RuntimeError(f"Model training error: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        try:
            if not self.is_trained:
                raise RuntimeError("Model not trained")

            # Get probabilities for classification, values for regression
            if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(features)
            else:
                predictions = self.model.predict(features)
                return predictions.reshape(-1, 1) if predictions.ndim == 1 else predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if not self.is_trained:
                raise RuntimeError("Model not trained")

            predictions = self.predict(features)

            if self.task_type == 'classification':
                # Classification metrics
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = labels.ravel() if labels.ndim > 1 else labels

                accuracy = float(np.mean(pred_classes == true_classes))

                # Per-class accuracy
                unique_classes = np.unique(true_classes)
                class_accuracies = {}
                for cls in unique_classes:
                    mask = true_classes == cls
                    if mask.sum() > 0:
                        class_acc = float(np.mean(pred_classes[mask] == true_classes[mask]))
                        class_accuracies[f'class_{int(cls)}_accuracy'] = class_acc

                return {
                    'accuracy': accuracy,
                    **class_accuracies,
                    'n_samples': len(labels)
                }

            else:
                # Regression metrics
                mse = float(np.mean((predictions.ravel() - labels.ravel()) ** 2))
                mae = float(np.mean(np.abs(predictions.ravel() - labels.ravel())))

                # R-squared
                ss_res = np.sum((labels.ravel() - predictions.ravel()) ** 2)
                ss_tot = np.sum((labels.ravel() - np.mean(labels)) ** 2)
                r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                return {
                    'mse': mse,
                    'rmse': float(np.sqrt(mse)),
                    'mae': mae,
                    'r2': r2,
                    'n_samples': len(labels)
                }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {'error': str(e)}

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pl.DataFrame:
        """Get feature importance rankings"""
        try:
            if self.feature_importance_ is None:
                raise RuntimeError("Model not trained")

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]

            importance_data = {
                'feature': feature_names,
                'importance': [Decimal(str(imp)) for imp in self.feature_importance_]
            }

            df = pl.DataFrame(importance_data)
            return df.sort('importance', descending=True)

        except Exception as e:
            self.logger.error(f"Failed to get feature importance: {e}")
            return pl.DataFrame()

    def save(self, path: str) -> None:
        """Save model to disk"""
        try:
            if not self.is_trained:
                self.logger.warning("Saving untrained model")

            model_data = {
                'model': self.model,
                'config': self.config,
                'feature_importance': self.feature_importance_,
                'is_trained': self.is_trained,
                'timestamp': datetime.utcnow()
            }

            joblib.dump(model_data, path, compress=3)
            self.logger.info(f"Model saved to {path}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise IOError(f"Model save error: {e}")

    def load(self, path: str) -> None:
        """Load model from disk"""
        try:
            model_data = joblib.load(path)

            self.model = model_data['model']
            self.config = model_data['config']
            self.feature_importance_ = model_data['feature_importance']
            self.is_trained = model_data['is_trained']

            self.logger.info(f"Model loaded from {path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise IOError(f"Model load error: {e}")
