"""
Support Vector Machine Model for Trading

Production-ready SVM implementation for classification and regression trading tasks.
Supports multi-class classification and SVR for price prediction.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class BaseMLModel(ABC):
    """Abstract base for all ML models"""

    def __init__(self, config: Dict) -> None:
        ...

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


class SVMModel(BaseMLModel):
    """
    Support Vector Machine model for trading.

    Implements SVM for both classification (signal direction) and
    regression (price prediction) tasks with production-ready features.

    Attributes:
        config: Configuration dictionary
        model_type: 'classification' or 'regression'
        model: Scikit-learn SVM model
        scaler: Feature scaler
        is_trained: Training status flag

    Example:
        >>> config = {"model": {"svm": {"kernel": "rbf", "C": "1.0"}}}
        >>> model = SVMModel(config, model_type="classification")
        >>> model.train(features, labels)
        >>> predictions = model.predict(new_features)
    """

    def __init__(self, config: Dict, model_type: str = "classification") -> None:
        """
        Initialize SVM model.

        Args:
            config: Configuration dictionary
            model_type: 'classification' or 'regression'

        Raises:
            ValueError: If configuration or model_type is invalid
        """
        super().__init__(config)
        self.config = config
        self.model_type = model_type
        self._validate_config()

        svm_config = self.config.get("model", {}).get("svm", {})

        # Load SVM parameters
        self.kernel: str = svm_config.get("kernel", os.getenv("SVM_KERNEL", "rbf"))
        self.C: float = float(svm_config.get("C", os.getenv("SVM_C", "1.0")))
        self.gamma: str = svm_config.get("gamma", os.getenv("SVM_GAMMA", "scale"))
        self.epsilon: float = float(svm_config.get("epsilon", os.getenv("SVM_EPSILON", "0.1")))

        # Regularization
        self.class_weight: Optional[str] = svm_config.get("class_weight", os.getenv("SVM_CLASS_WEIGHT"))
        if self.class_weight == "None":
            self.class_weight = None

        # Training parameters
        self.max_iter: int = svm_config.get("max_iter", int(os.getenv("SVM_MAX_ITER", "1000")))
        self.tol: float = float(svm_config.get("tol", os.getenv("SVM_TOL", "0.001")))

        # Feature scaling
        self.scale_features: bool = svm_config.get(
            "scale_features",
            os.getenv("SVM_SCALE_FEATURES", "true").lower() == "true"
        )

        # Initialize model and scaler
        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False

        self._initialize_model()

        logger.info(
            "svm_model_initialized",
            model_type=self.model_type,
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma
        )

    def _validate_config(self) -> None:
        """
        Validate configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if self.model_type not in ["classification", "regression"]:
            raise ValueError("model_type must be 'classification' or 'regression'")

        svm_config = self.config.get("model", {}).get("svm", {})

        if svm_config:
            kernel = svm_config.get("kernel", "rbf")
            valid_kernels = ["linear", "poly", "rbf", "sigmoid"]
            if kernel not in valid_kernels:
                raise ValueError(f"kernel must be one of {valid_kernels}")

            C = svm_config.get("C", 1.0)
            if float(C) <= 0:
                raise ValueError("C must be positive")

    def _initialize_model(self) -> None:
        """Initialize SVM model based on type."""
        if self.model_type == "classification":
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                tol=self.tol,
                probability=True,  # Enable probability estimates
                random_state=42
            )
        else:  # regression
            self.model = SVR(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol
            )

        if self.scale_features:
            self.scaler = StandardScaler()

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train SVM model.

        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (n_samples,)

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")
            if not isinstance(labels, np.ndarray):
                raise ValueError("labels must be numpy array")

            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Cannot train on empty data")

            if len(features) != len(labels):
                raise ValueError("features and labels must have same length")

            logger.info(
                "svm_training_started",
                n_samples=len(features),
                n_features=features.shape[1] if len(features.shape) > 1 else 1,
                model_type=self.model_type
            )

            # Scale features if enabled
            if self.scale_features and self.scaler is not None:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features

            # Train model
            self.model.fit(features_scaled, labels)
            self.is_trained = True

            # Calculate training metrics
            train_predictions = self.predict(features)
            train_metrics = self.evaluate(features, labels)

            logger.info(
                "svm_training_completed",
                model_type=self.model_type,
                train_metrics=train_metrics
            )

        except Exception as e:
            logger.error("svm_training_failed", error=str(e))
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Predictions array

        Raises:
            ValueError: If model not trained or inputs invalid
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            if not isinstance(features, np.ndarray):
                raise ValueError("features must be numpy array")

            # Scale features if enabled
            if self.scale_features and self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Generate predictions
            predictions = self.model.predict(features_scaled)

            logger.debug(
                "svm_predictions_generated",
                n_samples=len(predictions),
                model_type=self.model_type
            )

            return predictions

        except Exception as e:
            logger.error("svm_prediction_failed", error=str(e))
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions (classification only).

        Args:
            features: Input features

        Returns:
            Probability estimates for each class

        Raises:
            ValueError: If not a classification model or model not trained
        """
        try:
            if self.model_type != "classification":
                raise ValueError("predict_proba only available for classification")

            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")

            # Scale features if enabled
            if self.scale_features and self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Generate probability predictions
            probabilities = self.model.predict_proba(features_scaled)

            return probabilities

        except Exception as e:
            logger.error("svm_predict_proba_failed", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics

        Raises:
            ValueError: If model not trained or inputs invalid
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")

            predictions = self.predict(features)

            if self.model_type == "classification":
                # Classification metrics
                metrics = {
                    "accuracy": float(accuracy_score(labels, predictions)),
                    "precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
                    "recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
                    "f1_score": float(f1_score(labels, predictions, average="weighted", zero_division=0))
                }
            else:
                # Regression metrics
                mse = mean_squared_error(labels, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(labels, predictions)

                # Calculate mean absolute percentage error
                mape = np.mean(np.abs((labels - predictions) / (labels + 1e-10))) * 100

                metrics = {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2_score": float(r2),
                    "mape": float(mape)
                }

            logger.info("svm_evaluation_completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("svm_evaluation_failed", error=str(e))
            raise

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model

        Raises:
            ValueError: If model not trained
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained model")

            # Create directory if needed
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save model and scaler
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "model_type": self.model_type,
                "config": self.config,
                "is_trained": self.is_trained
            }

            with open(path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info("svm_model_saved", path=path)

        except Exception as e:
            logger.error("svm_save_failed", path=path, error=str(e))
            raise

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            with open(path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.model_type = model_data["model_type"]
            self.is_trained = model_data["is_trained"]

            logger.info("svm_model_loaded", path=path, model_type=self.model_type)

        except Exception as e:
            logger.error("svm_load_failed", path=path, error=str(e))
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model details
        """
        info = {
            "model_type": self.model_type,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "is_trained": self.is_trained,
            "scale_features": self.scale_features
        }

        if self.is_trained:
            info["n_support"] = self.model.n_support_.tolist() if hasattr(self.model, "n_support_") else None

        return info
