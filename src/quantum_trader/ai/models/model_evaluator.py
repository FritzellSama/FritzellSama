"""Model evaluation and metrics."""
from decimal import Decimal
from typing import Dict, List
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Decimal]:
        mse = Decimal(str(np.mean((predictions - labels) ** 2)))
        mae = Decimal(str(np.mean(np.abs(predictions - labels))))
        return {"mse": mse, "mae": mae}
