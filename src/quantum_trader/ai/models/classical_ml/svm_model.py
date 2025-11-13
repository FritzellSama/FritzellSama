"""Support Vector Machine model."""
from decimal import Decimal
from typing import Dict
import numpy as np
from sklearn.svm import SVC
from structlog import get_logger

logger = get_logger(__name__)

class SVMModel:
    def __init__(self, config: Dict) -> None:
        self.config = config
        kernel = config.get("kernel", "rbf")
        C = float(config.get("C", 1.0))
        self.model = SVC(kernel=kernel, C=C)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("SVM model trained")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def score(self, X: np.ndarray, y: np.ndarray) -> Decimal:
        accuracy = self.model.score(X, y)
        return Decimal(str(accuracy))
