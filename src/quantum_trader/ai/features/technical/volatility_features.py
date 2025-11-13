"""Volatility-based features."""
from decimal import Decimal
from typing import Dict
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class VolatilityFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.windows = config.get("windows", [10, 20, 50])
        
    def extract(self, data: pl.DataFrame) -> pl.DataFrame:
        result = data
        
        for window in self.windows:
            result = result.with_columns([
                pl.col("close").rolling_std(window).alias(f"volatility_{window}"),
                (pl.col("high") - pl.col("low")).rolling_mean(window).alias(f"atr_{window}")
            ])
            
        return result
