"""Fractal Dimension Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

async def compute_fractal_features(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    values = df.select(value_col).to_numpy().ravel()
    n = len(values)
    hurst = Decimal("0.5")  # Placeholder
    fractal_dim = 2 - hurst
    return pl.DataFrame([{"hurst_exponent": str(hurst), "fractal_dimension": str(fractal_dim), "timestamp": datetime.now(timezone.utc).isoformat()}])
