"""Entropy-based Features."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import polars as pl
from scipy.stats import entropy

logger = logging.getLogger(__name__)

async def compute_entropy_features(df: pl.DataFrame, value_col: str, bins: int = 10) -> pl.DataFrame:
    """Compute entropy features."""
    values = df.select(value_col).to_numpy().ravel()
    hist, _ = np.histogram(values, bins=bins)
    hist = hist / hist.sum()
    ent = entropy(hist + 1e-10)
    features = {
        "entropy": str(Decimal(str(ent))),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    logger.info("Computed entropy features")
    return pl.DataFrame([features])
