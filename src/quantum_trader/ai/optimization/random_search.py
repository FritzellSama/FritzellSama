"""Random search hyperparameter optimization."""
from decimal import Decimal
from typing import Dict, List
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

class RandomSearchOptimizer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.n_iterations = config.get("n_iterations", 100)
        
    def optimize(self, objective_fn, param_space: Dict) -> Dict:
        best_params = {}
        best_score = Decimal("-inf")
        
        for i in range(self.n_iterations):
            params = {k: np.random.choice(v) for k, v in param_space.items()}
            score = objective_fn(params)
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params
