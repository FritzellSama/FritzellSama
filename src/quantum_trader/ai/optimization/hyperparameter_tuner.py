"""Hyperparameter Tuning."""
import logging
from typing import Dict, Callable

logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, param_space: Dict):
        self.param_space = param_space
    async def tune(self, model_fn: Callable, X, y, n_trials: int = 50):
        import random
        best_score = -float('inf')
        best_params = None
        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in self.param_space.items()}
            model = model_fn(**params)
            model.fit(X, y)
            score = model.score(X, y)
            if score > best_score:
                best_score = score
                best_params = params
        logger.info(f"Best params: {best_params}, score: {best_score}")
        return best_params, best_score
