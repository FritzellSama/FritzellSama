"""Cointegration Features for Pairs Trading."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import coint, adfuller

logger = logging.getLogger(__name__)

async def compute_cointegration_features(
    df: pl.DataFrame,
    price_cols: List[str]
) -> pl.DataFrame:
    """Compute cointegration features for pairs."""
    features = {}

    for i in range(len(price_cols)):
        for j in range(i + 1, len(price_cols)):
            col1, col2 = price_cols[i], price_cols[j]
            series1 = df.select(col1).to_numpy().ravel()
            series2 = df.select(col2).to_numpy().ravel()

            # Cointegration test
            score, pvalue, _ = coint(series1, series2)
            features[f"coint_{col1}_{col2}_score"] = str(Decimal(str(score)))
            features[f"coint_{col1}_{col2}_pvalue"] = str(Decimal(str(pvalue)))

            # Spread
            spread = series1 - series2
            features[f"spread_{col1}_{col2}_mean"] = str(Decimal(str(np.mean(spread))))
            features[f"spread_{col1}_{col2}_std"] = str(Decimal(str(np.std(spread))))

    logger.info("Computed cointegration features", extra={"timestamp": datetime.now(timezone.utc).isoformat()})
    return pl.DataFrame([features])
