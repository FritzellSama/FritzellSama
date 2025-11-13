"""Price-based technical features."""
from decimal import Decimal
from typing import Dict
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class PriceFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def extract(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns([
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("price_range"),
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("price_change")
        ])
