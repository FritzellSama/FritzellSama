"""Rolling window feature engineering."""
from decimal import Decimal
from typing import Dict, List
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class RollingFeatureGenerator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.windows = config.get("windows", [5, 10, 20, 50])
        
    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        result = data
        
        for window in self.windows:
            result = result.with_columns([
                pl.col("close").rolling_mean(window).alias(f"sma_{window}"),
                pl.col("close").rolling_std(window).alias(f"std_{window}"),
                pl.col("volume").rolling_mean(window).alias(f"vol_ma_{window}")
            ])
            
        return result
