"""Feature Validation."""
import logging
from typing import List
import polars as pl

logger = logging.getLogger(__name__)

class FeatureValidator:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
    async def validate(self, df: pl.DataFrame) -> bool:
        for col in self.feature_cols:
            if col not in df.columns:
                logger.error(f"Missing feature: {col}")
                return False
            if df.select(col).null_count().item() > 0:
                logger.warning(f"Nulls in feature: {col}")
        return True
