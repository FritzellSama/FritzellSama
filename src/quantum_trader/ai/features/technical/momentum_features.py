"""Momentum-based technical features."""
from decimal import Decimal
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class MomentumFeatures:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns([
            ((pl.col("close") - pl.col("close").shift(14)) / pl.col("close").shift(14) * 100).alias("rsi"),
            (pl.col("close").rolling_mean(20)).alias("sma_20")
        ])
