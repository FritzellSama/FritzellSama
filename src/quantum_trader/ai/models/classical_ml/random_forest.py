"""Random Forest for trading signals."""
from decimal import Decimal
from typing import Dict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from structlog import get_logger

logger = get_logger(__name__)

class RandomForestModel:
    def __init__(self, config: Dict) -> None:
        self.config = config
        n_estimators = config.get("n_estimators", 100)
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        logger.info("Random Forest trained")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
