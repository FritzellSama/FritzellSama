"""Lag Features for Time Series."""
import logging
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_lag_features(df: pl.DataFrame, value_col: str, lags: list = [1, 2, 3]) -> pl.DataFrame:
    for lag in lags:
        df = df.with_columns([df.select(value_col).shift(lag).alias(f"{value_col}_lag_{lag}")])
    logger.info(f"Computed lag features for {value_col}")
    return df
