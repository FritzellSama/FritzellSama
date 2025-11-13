"""Funding Rate Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_funding_features(df: pl.DataFrame, funding_col: str) -> pl.DataFrame:
    funding = df.select(funding_col).to_numpy().mean()
    return pl.DataFrame([{"avg_funding_rate": str(Decimal(str(funding))), "timestamp": datetime.now(timezone.utc).isoformat()}])
