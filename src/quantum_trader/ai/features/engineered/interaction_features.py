"""Interaction Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_interaction_features(df: pl.DataFrame, col1: str, col2: str) -> pl.DataFrame:
    interaction = df.select(col1).to_numpy() * df.select(col2).to_numpy()
    return df.with_columns([pl.Series(f"{col1}_x_{col2}", interaction.ravel().tolist())])
