"""Feature Store for Caching."""
import logging
from datetime import datetime, timezone
from typing import Dict
import polars as pl

logger = logging.getLogger(__name__)

class FeatureStore:
    def __init__(self):
        self.store: Dict[str, pl.DataFrame] = {}
    async def put(self, key: str, df: pl.DataFrame):
        self.store[key] = df
        logger.info(f"Stored features: {key}")
    async def get(self, key: str) -> pl.DataFrame:
        return self.store.get(key)
    def clear(self):
        self.store.clear()
