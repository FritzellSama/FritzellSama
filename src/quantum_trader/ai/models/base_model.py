"""Abstract base class for all ML models.

This module provides the base interface that all machine learning models
in the Quantum Trader AI system must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base for all ML models.

    This class defines the interface that all machine learning models
    must implement to be compatible with the Quantum Trader AI system.

    Attributes:
        config: Configuration dictionary for the model
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the ML model.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self._validate_config()
        logger.info("base_ml_model_initialized", model_class=self.__class__.__name__)

    def _validate_config(self) -> None:
        """Validate the model configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model on provided features and labels.

        Args:
            features: Training features as numpy array
            labels: Training labels as numpy array

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for given features.

        Args:
            features: Features to predict on as numpy array

        Returns:
            Predictions as numpy array

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on given data.

        Args:
            features: Evaluation features as numpy array
            labels: True labels as numpy array

        Returns:
            Dictionary of evaluation metrics

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: File path to save the model

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path: File path to load the model from

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass


__all__ = ["BaseMLModel"]
