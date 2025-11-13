"""Comprehensive trading environment."""
from decimal import Decimal
from typing import Dict, Tuple, Any
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class TradingEnvironment:
    def __init__(self, config: Dict, data: pl.DataFrame) -> None:
        self.config = config
        self.data = data
        self.initial_cash = Decimal(str(config["initial_cash"]))
        self.commission_rate = Decimal(str(config.get("commission_rate", 0.001)))
        self.current_step = 0
        self.cash = self.initial_cash
        
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.cash = self.initial_cash
        return self._get_state()
        
    def _get_state(self) -> np.ndarray:
        if self.current_step >= len(self.data):
            return np.zeros(10)
        row = self.data[self.current_step]
        return np.array([float(row["close"][0]), float(row["volume"][0])])
        
    def step(self, action: np.ndarray) -> Tuple:
        reward = Decimal("0.001")
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, done, {"cash": self.cash}
