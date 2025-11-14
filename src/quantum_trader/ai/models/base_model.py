"""Base abstract class for all machine learning models.

This module provides the foundation for implementing ML models in the trading system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """Abstract base class for all ML models.

    All machine learning models in the system must inherit from this class
    and implement its abstract methods.

    Attributes:
        config: Model configuration dictionary

    Example:
        >>> class MyModel(BaseMLModel):
        ...     def train(self, features, labels):
        ...         # Training implementation
        ...         pass
        ...     def predict(self, features):
        ...         # Prediction implementation
        ...         return np.array([])
        ...     def evaluate(self, features, labels):
        ...         return {"accuracy": 0.95}
        ...     def save(self, path):
        ...         pass
        ...     def load(self, path):
        ...         pass
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model with configuration.

        Args:
            config: Model configuration dictionary containing hyperparameters
                   and other settings
        """
        self.config = config
        logger.info("model_initialized", model_type=self.__class__.__name__)

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
        """Generate predictions for the given features.

        Args:
            features: Input features as numpy array

        Returns:
            Predictions as numpy array

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            features: Test features as numpy array
            labels: Test labels as numpy array

        Returns:
            Dictionary containing evaluation metrics (e.g., accuracy, loss)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: File path where model should be saved

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path: File path from where model should be loaded

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        pass
