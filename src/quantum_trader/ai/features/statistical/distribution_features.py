"""Distribution-based Features."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np
import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)

async def compute_distribution_features(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Compute distribution features."""
    values = df.select(value_col).to_numpy().ravel()
    features = {
        "skewness": str(Decimal(str(stats.skew(values)))),
        "kurtosis": str(Decimal(str(stats.kurtosis(values)))),
        "jarque_bera": str(Decimal(str(stats.jarque_bera(values)[0]))),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    logger.info("Computed distribution features")
    return pl.DataFrame([features])
