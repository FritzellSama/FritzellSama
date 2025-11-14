"""CatBoost gradient boosting model for trading predictions.

This module implements a CatBoost-based machine learning model optimized for
trading signal generation and market prediction. CatBoost excels at handling
heterogeneous features, categorical data, and provides excellent performance
with minimal hyperparameter tuning.

Key features:
- Fast training with GPU support
- Built-in categorical feature handling
- Reduced overfitting through ordered boosting
- Feature importance analysis
- Cross-validation support

Reference:
    Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).
    CatBoost: unbiased boosting with categorical features. NeurIPS 2018.

Example:
    ```python
    from quantum_trader.ai.models.classical_ml.catboost_model import CatBoostModel
    import numpy as np

    config = {
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "task_type": "GPU",
        "loss_function": "RMSE",
    }

    model = CatBoostModel(config)
    model.train(train_features, train_labels)
    predictions = model.predict(test_features)
    metrics = model.evaluate(test_features, test_labels)
    ```
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from structlog import get_logger
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class CatBoostModel(BaseMLModel):
    """CatBoost gradient boosting model for trading predictions.

    Production-ready implementation with:
    - Regression and classification support
    - GPU acceleration
    - Categorical feature handling
    - Feature importance analysis
    - Model interpretability
    - Cross-validation

    Attributes:
        model: CatBoost model instance
        task_type: Task type ('regression' or 'classification')
        feature_importance: Feature importance scores
        categorical_features: Indices of categorical features
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize CatBoost model.

        Args:
            config: Configuration dictionary with keys:
                - task_type: 'regression' or 'classification' (default: 'regression')
                - iterations: Number of boosting iterations (default: 1000)
                - learning_rate: Learning rate (default: 0.03)
                - depth: Tree depth (default: 6)
                - l2_leaf_reg: L2 regularization (default: 3.0)
                - device_type: 'CPU' or 'GPU' (default: 'CPU')
                - loss_function: Loss function (default: 'RMSE' for regression, 'Logloss' for classification)
                - eval_metric: Evaluation metric (default: same as loss_function)
                - early_stopping_rounds: Early stopping patience (default: 50)
                - verbose: Verbosity level (default: False)
                - random_seed: Random seed (default: 42)
                - cat_features: List of categorical feature indices (default: [])
        """
        super().__init__(config)

        # Task configuration
        self.task_type = config.get("task_type", "regression")
        if self.task_type not in ["regression", "classification"]:
            raise ValueError(f"Invalid task_type: {self.task_type}")

        # Model hyperparameters
        self.iterations = int(config.get("iterations", 1000))
        self.learning_rate = float(config.get("learning_rate", 0.03))
        self.depth = int(config.get("depth", 6))
        self.l2_leaf_reg = float(config.get("l2_leaf_reg", 3.0))
        self.device_type = config.get("device_type", "CPU")
        self.early_stopping_rounds = int(config.get("early_stopping_rounds", 50))
        self.verbose = config.get("verbose", False)
        self.random_seed = int(config.get("random_seed", 42))

        # Categorical features
        self.categorical_features = config.get("cat_features", [])

        # Loss function
        if self.task_type == "regression":
            self.loss_function = config.get("loss_function", "RMSE")
        else:
            self.loss_function = config.get("loss_function", "Logloss")

        self.eval_metric = config.get("eval_metric", self.loss_function)

        # Feature importance
        self.feature_importance: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None

        # Build model
        self._build_model()

        logger.info(
            "CatBoost model initialized",
            task_type=self.task_type,
            iterations=self.iterations,
            device=self.device_type
        )

    def _build_model(self) -> None:
        """Build the CatBoost model."""
        model_params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "task_type": self.device_type,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

        if self.task_type == "regression":
            self.model = CatBoostRegressor(**model_params)
        else:
            self.model = CatBoostClassifier(**model_params)

        logger.debug("CatBoost model built", params=model_params)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the CatBoost model.

        Args:
            features: Input features of shape (n_samples, n_features)
            labels: Target values of shape (n_samples,)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails
        """
        try:
            logger.info("Starting CatBoost training")

            # Validate inputs
            self.validate_input(features, labels)

            # Flatten labels if needed
            if len(labels.shape) > 1:
                labels = labels.ravel()

            # Create Pool for CatBoost
            if self.categorical_features:
                train_pool = Pool(
                    data=features,
                    label=labels,
                    cat_features=self.categorical_features
                )
            else:
                train_pool = Pool(data=features, label=labels)

            # Train model
            self.model.fit(
                train_pool,
                verbose=self.verbose,
            )

            # Extract feature importance
            self.feature_importance = self.model.get_feature_importance()

            # Update training status
            self.is_trained = True
            self.last_trained = datetime.utcnow()

            # Get training metrics
            train_score = self.model.get_best_score()
            final_loss = train_score.get("learn", {}).get(self.eval_metric, 0.0)

            self.training_history.append({
                "timestamp": self.last_trained.isoformat(),
                "iterations": self.model.tree_count_,
                "final_loss": float(final_loss),
                "best_iteration": self.model.get_best_iteration(),
            })

            logger.info(
                "Training completed",
                iterations=self.model.tree_count_,
                best_iteration=self.model.get_best_iteration(),
                final_loss=f"{final_loss:.6f}"
            )

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise RuntimeError(f"Training failed: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the trained model.

        Args:
            features: Input features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model is not trained or prediction fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        try:
            logger.debug("Generating predictions", samples=features.shape[0])

            # Validate inputs
            self.validate_input(features)

            # Create Pool for CatBoost
            if self.categorical_features:
                pred_pool = Pool(
                    data=features,
                    cat_features=self.categorical_features
                )
            else:
                pred_pool = Pool(data=features)

            # Generate predictions
            predictions = self.model.predict(pred_pool)

            logger.debug("Predictions generated", shape=predictions.shape)

            return predictions

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Generate probability predictions for classification.

        Args:
            features: Input features of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)

        Raises:
            ValueError: If model is not a classifier
            RuntimeError: If prediction fails
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        try:
            logger.debug("Generating probability predictions")

            # Validate inputs
            self.validate_input(features)

            # Create Pool for CatBoost
            if self.categorical_features:
                pred_pool = Pool(
                    data=features,
                    cat_features=self.categorical_features
                )
            else:
                pred_pool = Pool(data=features)

            # Generate probability predictions
            probabilities = self.model.predict_proba(pred_pool)

            return probabilities

        except Exception as e:
            logger.error("Probability prediction failed", error=str(e))
            raise RuntimeError(f"Probability prediction failed: {e}")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If evaluation fails
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        try:
            logger.debug("Evaluating model")

            # Generate predictions
            predictions = self.predict(features)

            # Flatten labels if needed
            if len(labels.shape) > 1:
                labels = labels.ravel()

            metrics = {}

            if self.task_type == "regression":
                # Regression metrics
                mse = mean_squared_error(labels, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(labels, predictions)
                r2 = r2_score(labels, predictions)

                # MAPE (avoid division by zero)
                labels_nonzero = labels != 0
                if np.any(labels_nonzero):
                    mape = np.mean(np.abs(
                        (labels[labels_nonzero] - predictions[labels_nonzero]) /
                        labels[labels_nonzero]
                    )) * 100
                else:
                    mape = float('inf')

                metrics = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "mape": float(mape),
                }

            else:
                # Classification metrics
                # Round predictions for classification
                predictions_class = np.round(predictions).astype(int)

                accuracy = accuracy_score(labels, predictions_class)

                # Handle binary and multiclass
                average = 'binary' if len(np.unique(labels)) == 2 else 'weighted'

                try:
                    precision = precision_score(
                        labels,
                        predictions_class,
                        average=average,
                        zero_division=0
                    )
                    recall = recall_score(
                        labels,
                        predictions_class,
                        average=average,
                        zero_division=0
                    )
                    f1 = f1_score(
                        labels,
                        predictions_class,
                        average=average,
                        zero_division=0
                    )
                except Exception:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0

                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                }

            logger.info("Evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise RuntimeError(f"Evaluation failed: {e}")

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.feature_importance is None:
            raise RuntimeError("Model must be trained to get feature importance")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]

        importance_dict = dict(zip(feature_names, self.feature_importance))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        logger.debug("Feature importance computed", top_features=list(importance_dict.keys())[:5])

        return {k: float(v) for k, v in importance_dict.items()}

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path for saving

        Raises:
            IOError: If save fails
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save CatBoost model
            self.model.save_model(str(save_path))

            # Save metadata and feature importance
            self.save_metadata(str(save_path))

            logger.info("Model saved successfully", path=str(save_path))

        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise IOError(f"Failed to save model: {e}")

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path for loading

        Raises:
            IOError: If load fails
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {load_path}")

            # Rebuild model with same parameters
            self._build_model()

            # Load CatBoost model
            self.model.load_model(str(load_path))

            # Load metadata
            self.load_metadata(str(load_path))

            # Extract feature importance
            try:
                self.feature_importance = self.model.get_feature_importance()
            except Exception:
                logger.warning("Could not load feature importance")

            logger.info("Model loaded successfully", path=str(load_path))

        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise IOError(f"Failed to load model: {e}")
