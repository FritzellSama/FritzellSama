"""Regime switching models."""
from decimal import Decimal
from typing import Dict, List
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class RegimeSwitchingModel:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.n_regimes = config.get("n_regimes", 3)
        self.states = np.zeros(self.n_regimes)
        
    def fit(self, data: pl.DataFrame) -> None:
        logger.info("Fitting regime switching model")
        
    def predict_regime(self, data: pl.DataFrame) -> int:
        return 0
