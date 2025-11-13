"""Correlation Features for Market Analysis."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
import numpy as np
import polars as pl
from scipy.stats import spearmanr, kendalltau

logger = logging.getLogger(__name__)

async def compute_correlation_features(
    df: pl.DataFrame,
    price_cols: List[str],
    window: int = 20
) -> pl.DataFrame:
    """Compute rolling correlation features."""
    results = []

    for i in range(window, len(df)):
        window_df = df[i - window:i]
        features = {"index": i}

        for j in range(len(price_cols)):
            for k in range(j + 1, len(price_cols)):
                col1, col2 = price_cols[j], price_cols[k]
                series1 = window_df.select(col1).to_numpy().ravel()
                series2 = window_df.select(col2).to_numpy().ravel()

                # Pearson correlation
                pearson_corr = np.corrcoef(series1, series2)[0, 1]
                features[f"pearson_{col1}_{col2}"] = str(Decimal(str(pearson_corr)))

                # Spearman correlation
                spearman_corr, _ = spearmanr(series1, series2)
                features[f"spearman_{col1}_{col2}"] = str(Decimal(str(spearman_corr)))

        results.append(features)

    logger.info("Computed correlation features", extra={"num_rows": len(results), "timestamp": datetime.now(timezone.utc).isoformat()})
    return pl.DataFrame(results)
