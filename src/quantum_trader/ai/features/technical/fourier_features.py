"""Fourier Transform Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

async def compute_fourier_features(df: pl.DataFrame, value_col: str, n_components: int = 5) -> pl.DataFrame:
    values = df.select(value_col).to_numpy().ravel()
    fft = np.fft.fft(values)
    power = np.abs(fft[:n_components])
    features = {f"fft_component_{i}": str(Decimal(str(p))) for i, p in enumerate(power)}
    features["timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info("Computed Fourier features")
    return pl.DataFrame([features])
