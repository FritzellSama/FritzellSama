"""Multi-asset trading environment."""
from decimal import Decimal
from typing import Dict, Tuple
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class MultiAssetEnvironment:
    def __init__(self, config: Dict, data: pl.DataFrame) -> None:
        self.config = config
        self.data = data
        self.initial_cash = Decimal(str(config["initial_cash"]))
        
    def reset(self) -> np.ndarray:
        return np.zeros(10)
        
    def step(self, action: np.ndarray) -> Tuple:
        return np.zeros(10), Decimal("0"), False, {}
