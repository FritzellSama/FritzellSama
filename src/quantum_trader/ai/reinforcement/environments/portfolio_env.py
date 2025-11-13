"""Portfolio management environment."""
from decimal import Decimal
from typing import Dict, Tuple
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class PortfolioEnvironment:
    def __init__(self, config: Dict, data: pl.DataFrame) -> None:
        self.config = config
        self.data = data
        self.initial_balance = Decimal(str(config["initial_balance"]))
        
    def reset(self) -> np.ndarray:
        return np.zeros(20)
        
    def step(self, action: np.ndarray) -> Tuple:
        return np.zeros(20), Decimal("0.01"), False, {"value": self.initial_balance}
