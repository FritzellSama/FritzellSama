"""Base ML model implementation for Quantum Trader AI.

This module provides the abstract base class for all machine learning models
in the trading system. All ML models must inherit from BaseMLModel and implement
the required abstract methods.

The base model provides:
- Standard interface for training, prediction, and evaluation
- Model persistence (save/load)
- Configuration management
- Validation and error handling
- Logging and monitoring

Example:
    ```python
    from quantum_trader.ai.models.base_model import BaseMLModel
    import numpy as np
    from typing import Dict

    class MyModel(BaseMLModel):
        def train(self, features: np.ndarray, labels: np.ndarray) -> None:
            # Training implementation
            pass

        def predict(self, features: np.ndarray) -> np.ndarray:
            # Prediction implementation
            pass
    ```
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base class for all ML models.

    This class defines the standard interface that all machine learning models
    must implement. It provides common functionality for model lifecycle management
    including training, prediction, evaluation, and persistence.

    Attributes:
        config: Configuration dictionary for model parameters
        model_id: Unique identifier for the model instance
        model_path: Path where model artifacts are stored
        is_trained: Flag indicating if model has been trained
        metadata: Additional model metadata

    Note:
        All numeric computations should use Decimal for precision.
        All data should use Polars DataFrame format.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the ML model.

        Args:
            config: Configuration dictionary containing model parameters.
                   Required keys depend on specific model implementation.

        Raises:
            ValueError: If config is invalid or missing required parameters
        """
        self.config = config
        self._validate_config()

        # Core attributes
        self.model_id: str = config.get("model_id", self._generate_model_id())
        self.model_path: Path = Path(config.get("model_path", "./models"))
        self.is_trained: bool = False
        self.metadata: Dict[str, Any] = config.get("metadata", {})

        # Training history
        self.training_history: List[Dict[str, Any]] = []
        self.last_trained: Optional[datetime] = None

        # Model-specific initialization
        self._initialize_model()

        logger.info(
            "ML model initialized",
            model_id=self.model_id,
            model_type=self.__class__.__name__
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        # Validate model_path exists or can be created
        model_path = Path(self.config.get("model_path", "./models"))
        if not model_path.exists():
            try:
                model_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Created model directory", path=str(model_path))
            except Exception as e:
                raise ValueError(f"Cannot create model directory: {e}")

        # Validate numeric parameters are Decimal
        for key, value in self.config.items():
            if isinstance(value, float):
                logger.warning(
                    "Float detected in config, converting to Decimal",
                    key=key,
                    value=value
                )
                self.config[key] = Decimal(str(value))

        logger.debug("Config validated successfully")

    def _initialize_model(self) -> None:
        """Initialize model-specific components.

        This method can be overridden by subclasses to perform
        model-specific initialization tasks.
        """
        pass

    def _generate_model_id(self) -> str:
        """Generate unique model identifier.

        Returns:
            Unique model ID string
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_type = self.__class__.__name__
        return f"{model_type}_{timestamp}"

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model on provided data.

        Args:
            features: Input features array of shape (n_samples, n_features)
            labels: Target labels array of shape (n_samples,) or (n_samples, n_targets)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If training fails

        Note:
            Implementations must update self.is_trained and self.last_trained
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            features: Input features array of shape (n_samples, n_features)

        Returns:
            Predictions array of shape (n_samples,) or (n_samples, n_outputs)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model is not trained or prediction fails
        """
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on provided data.

        Args:
            features: Input features array of shape (n_samples, n_features)
            labels: True labels array of shape (n_samples,) or (n_samples, n_targets)

        Returns:
            Dictionary containing evaluation metrics (e.g., accuracy, precision, recall, f1)

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If model is not trained or evaluation fails

        Note:
            All metric values must be standard Python floats, not Decimal
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path where model should be saved

        Raises:
            IOError: If save operation fails

        Note:
            Implementations should save both model parameters and metadata
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path from which to load model

        Raises:
            IOError: If load operation fails
            ValueError: If loaded model is incompatible

        Note:
            Implementations should load both model parameters and metadata
            and set self.is_trained appropriately
        """
        pass

    def validate_input(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> None:
        """Validate input data format and shape.

        Args:
            features: Input features to validate
            labels: Optional labels to validate

        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"Features must be numpy array, got {type(features)}")

        if len(features.shape) != 2:
            raise ValueError(
                f"Features must be 2D array, got shape {features.shape}"
            )

        if features.size == 0:
            raise ValueError("Features array is empty")

        if np.isnan(features).any():
            raise ValueError("Features contain NaN values")

        if np.isinf(features).any():
            raise ValueError("Features contain infinite values")

        if labels is not None:
            if not isinstance(labels, np.ndarray):
                raise ValueError(f"Labels must be numpy array, got {type(labels)}")

            if labels.size == 0:
                raise ValueError("Labels array is empty")

            if len(features) != len(labels):
                raise ValueError(
                    f"Features and labels length mismatch: {len(features)} != {len(labels)}"
                )

            if np.isnan(labels).any():
                raise ValueError("Labels contain NaN values")

            if np.isinf(labels).any():
                raise ValueError("Labels contain infinite values")

        logger.debug(
            "Input validation successful",
            features_shape=features.shape,
            labels_shape=labels.shape if labels is not None else None
        )

    def save_metadata(self, path: str) -> None:
        """Save model metadata to disk.

        Args:
            path: File path for metadata

        Raises:
            IOError: If save fails
        """
        try:
            metadata = {
                "model_id": self.model_id,
                "model_type": self.__class__.__name__,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained.isoformat() if self.last_trained else None,
                "config": self.config,
                "metadata": self.metadata,
                "training_history": self.training_history,
            }

            meta_path = Path(path).with_suffix(".meta")
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info("Metadata saved successfully", path=str(meta_path))

        except Exception as e:
            logger.error("Failed to save metadata", error=str(e))
            raise IOError(f"Failed to save metadata: {e}")

    def load_metadata(self, path: str) -> None:
        """Load model metadata from disk.

        Args:
            path: File path for metadata

        Raises:
            IOError: If load fails
        """
        try:
            meta_path = Path(path).with_suffix(".meta")

            if not meta_path.exists():
                logger.warning("Metadata file not found", path=str(meta_path))
                return

            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            self.model_id = metadata.get("model_id", self.model_id)
            self.is_trained = metadata.get("is_trained", False)

            last_trained = metadata.get("last_trained")
            if last_trained:
                self.last_trained = datetime.fromisoformat(last_trained)

            self.metadata = metadata.get("metadata", {})
            self.training_history = metadata.get("training_history", [])

            logger.info("Metadata loaded successfully", path=str(meta_path))

        except Exception as e:
            logger.error("Failed to load metadata", error=str(e))
            raise IOError(f"Failed to load metadata: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained.isoformat() if self.last_trained else None,
            "config": self.config,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation of model."""
        return (
            f"{self.__class__.__name__}("
            f"model_id={self.model_id}, "
            f"is_trained={self.is_trained})"
        )
