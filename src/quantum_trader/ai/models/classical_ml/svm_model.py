"""Support Vector Machine model for trading signal classification.

This module implements an SVM-based classifier for predicting trading signals
with comprehensive hyperparameter tuning and evaluation capabilities.
"""

from __future__ import annotations

import asyncio
import os
import pickle
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class SVMModel(BaseMLModel):
    """Support Vector Machine classifier for trading signals.

    Implements SVM with RBF kernel for multi-class signal classification,
    including BUY, SELL, and HOLD predictions.

    Attributes:
        config: Model configuration dictionary
        model: Underlying sklearn SVC model
        scaler: Feature scaler for normalization
        is_trained: Whether model has been trained

    Examples:
        >>> config = {
        ...     "kernel": "rbf",
        ...     "C": 1.0,
        ...     "gamma": "scale",
        ...     "class_weight": "balanced"
        ... }
        >>> model = SVMModel(config)
        >>> model.train(features, labels)
        >>> predictions = model.predict(test_features)
    """

    def __init__(self, config: Dict) -> None:
        """Initialize SVM model.

        Args:
            config: Configuration with kernel, C, gamma, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        super().__init__(config)
        self._validate_config()

        # Model hyperparameters from config
        self.kernel: str = config.get("kernel", os.getenv("SVM_KERNEL", "rbf"))
        self.C: float = float(config.get("C", os.getenv("SVM_C", "1.0")))
        self.gamma: str = config.get("gamma", os.getenv("SVM_GAMMA", "scale"))
        self.class_weight: Optional[str] = config.get("class_weight", os.getenv("SVM_CLASS_WEIGHT", "balanced"))
        self.probability: bool = config.get("probability", os.getenv("SVM_PROBABILITY", "true").lower() == "true")
        self.random_state: int = int(config.get("random_state", os.getenv("SVM_RANDOM_STATE", "42")))

        # Grid search parameters
        self.use_grid_search: bool = config.get("use_grid_search", os.getenv("SVM_USE_GRID_SEARCH", "false").lower() == "true")
        self.grid_params: Dict[str, List[Any]] = config.get("grid_params", {})

        # Initialize model and scaler
        self.model: Optional[SVC] = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False

        self._initialize_model()

        logger.info(
            "SVM model initialized",
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        valid_kernels = ["linear", "poly", "rbf", "sigmoid"]
        if "kernel" in self.config:
            if self.config["kernel"] not in valid_kernels:
                raise ValueError(f"kernel must be one of {valid_kernels}")

    def _initialize_model(self) -> None:
        """Initialize the SVC model with configured parameters."""
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=self.probability,
            random_state=self.random_state,
            cache_size=int(os.getenv("SVM_CACHE_SIZE", "200"))
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the SVM model.

        Args:
            features: Training features array (n_samples, n_features)
            labels: Training labels array (n_samples,)

        Raises:
            ValueError: If input arrays invalid
        """
        try:
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Features and labels must have same number of samples: "
                    f"{features.shape[0]} vs {labels.shape[0]}"
                )

            if features.shape[0] == 0:
                raise ValueError("Cannot train on empty dataset")

            logger.info(
                "Starting SVM training",
                n_samples=features.shape[0],
                n_features=features.shape[1],
                n_classes=len(np.unique(labels))
            )

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Train with grid search if enabled
            if self.use_grid_search and self.grid_params:
                self._train_with_grid_search(features_scaled, labels)
            else:
                self.model.fit(features_scaled, labels)

            self.is_trained = True

            logger.info(
                "SVM training completed",
                support_vectors=self.model.n_support_.tolist() if hasattr(self.model, 'n_support_') else None
            )

        except Exception as e:
            logger.error(
                "SVM training failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def _train_with_grid_search(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train model using grid search for hyperparameter tuning.

        Args:
            features: Scaled training features
            labels: Training labels
        """
        logger.info("Starting grid search for hyperparameter tuning")

        cv_folds = int(os.getenv("SVM_CV_FOLDS", "5"))

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.grid_params,
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=int(os.getenv("SVM_N_JOBS", "-1")),
            verbose=0
        )

        grid_search.fit(features, labels)

        self.model = grid_search.best_estimator_

        logger.info(
            "Grid search completed",
            best_params=grid_search.best_params_,
            best_score=float(grid_search.best_score_)
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Features array for prediction (n_samples, n_features)

        Returns:
            Predictions array (n_samples,)

        Raises:
            RuntimeError: If model not trained
            ValueError: If features invalid
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            if features.shape[0] == 0:
                return np.array([])

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Make predictions
            predictions = self.model.predict(features_scaled)

            logger.debug(
                "Predictions generated",
                n_samples=len(predictions)
            )

            return predictions

        except Exception as e:
            logger.error(
                "Prediction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            features: Features array for prediction

        Returns:
            Probability array (n_samples, n_classes)

        Raises:
            RuntimeError: If model not trained or probability not enabled
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            if not self.probability:
                raise RuntimeError("Probability prediction not enabled. Set probability=True in config")

            features_scaled = self.scaler.transform(features)
            probabilities = self.model.predict_proba(features_scaled)

            return probabilities

        except Exception as e:
            logger.error(
                "Probability prediction failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            features: Test features array
            labels: True labels array

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Calculate metrics
            metrics = {
                "accuracy": float(accuracy_score(labels, predictions)),
                "precision_weighted": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
                "recall_weighted": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
                "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0))
            }

            # Add ROC AUC if probability enabled and binary classification
            if self.probability and len(np.unique(labels)) == 2:
                try:
                    probabilities = self.predict_proba(features)
                    metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
                except Exception as e:
                    logger.warning("Could not calculate ROC AUC", error=str(e))

            logger.info(
                "Model evaluation completed",
                **metrics
            )

            return metrics

        except Exception as e:
            logger.error(
                "Evaluation failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model (directory will be created if needed)

        Raises:
            RuntimeError: If model not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")

            # Create directory if needed
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save model and scaler
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "config": self.config,
                "is_trained": self.is_trained,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("Model saved", path=path)

        except Exception as e:
            logger.error(
                "Model save failed",
                error=str(e),
                path=path
            )
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If model file not found
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.config = model_data.get("config", self.config)
            self.is_trained = model_data.get("is_trained", True)

            logger.info(
                "Model loaded",
                path=path,
                timestamp=model_data.get("timestamp")
            )

        except Exception as e:
            logger.error(
                "Model load failed",
                error=str(e),
                path=path
            )
            raise

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (for linear kernel only).

        Returns:
            Feature importance array or None if not available

        Note:
            Only available for linear kernel SVM
        """
        if not self.is_trained:
            logger.warning("Model not trained, cannot get feature importance")
            return None

        if self.kernel != "linear":
            logger.warning(f"Feature importance not available for kernel: {self.kernel}")
            return None

        # For linear SVM, coefficients represent feature importance
        return np.abs(self.model.coef_[0])
