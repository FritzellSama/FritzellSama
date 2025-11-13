"""Base model abstract class for all ML models.

This module defines the abstract base class that all ML models must inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BaseMLModel(ABC):
    """Abstract base for all ML models.

    All machine learning models in the system must inherit from this base class
    and implement the required methods for training, prediction, evaluation,
    and model persistence.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the base model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config

    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the model on provided data.

        Args:
            features: Training features array
            labels: Training labels array
        """
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on provided features.

        Args:
            features: Features array for prediction

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Evaluation features array
            labels: True labels array

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        pass
