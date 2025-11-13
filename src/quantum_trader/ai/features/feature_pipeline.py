"""Feature Engineering Pipeline."""
import logging
from datetime import datetime, timezone
from typing import List
import polars as pl

logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(self, steps: List):
        self.steps = steps
    async def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        for step in self.steps:
            df = await step.transform(df)
        logger.info("Pipeline transformation complete", extra={"timestamp": datetime.now(timezone.utc).isoformat()})
        return df
