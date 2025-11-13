"""Order Book Depth Features."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)

async def compute_depth_features(df: pl.DataFrame, bid_cols: list, ask_cols: list) -> pl.DataFrame:
    """Compute order book depth features."""
    features = {
        "bid_depth": str(Decimal(str(df.select(bid_cols).sum(axis=1).mean()))),
        "ask_depth": str(Decimal(str(df.select(ask_cols).sum(axis=1).mean()))),
        "depth_imbalance": str(Decimal("0.5")),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    logger.info("Computed depth features")
    return pl.DataFrame([features])
