"""Feature Extraction Pipeline."""
import logging
from datetime import datetime, timezone
from typing import List
import polars as pl

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
    async def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Extracting features", extra={"timestamp": datetime.now(timezone.utc).isoformat()})
        return df.select(self.feature_cols)
