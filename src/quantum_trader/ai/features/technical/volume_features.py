"""Volume-based technical features."""
from decimal import Decimal
from typing import Dict
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class VolumeFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.windows = config.get("windows", [10, 20])
        
    def extract(self, data: pl.DataFrame) -> pl.DataFrame:
        result = data
        
        for window in self.windows:
            result = result.with_columns([
                pl.col("volume").rolling_mean(window).alias(f"volume_ma_{window}"),
                (pl.col("volume") / pl.col("volume").rolling_mean(window)).alias(f"volume_ratio_{window}")
            ])
            
        return result
