"""Grid Search for Hyperparameters."""
import logging
from typing import Dict, List
import itertools

logger = logging.getLogger(__name__)

class GridSearch:
    def __init__(self, param_grid: Dict[str, List]):
        self.param_grid = param_grid
    async def search(self, model_fn, X, y):
        best_score = -float('inf')
        best_params = None
        keys, values = zip(*self.param_grid.items())
        for params in itertools.product(*values):
            param_dict = dict(zip(keys, params))
            model = model_fn(**param_dict)
            model.fit(X, y)
            score = model.score(X, y)
            if score > best_score:
                best_score = score
                best_params = param_dict
        logger.info(f"Best params: {best_params}, score: {best_score}")
        return best_params, best_score
