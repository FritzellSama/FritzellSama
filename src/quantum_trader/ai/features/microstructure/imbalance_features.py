"""Order Imbalance Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_imbalance_features(df: pl.DataFrame, bid_vol_col: str, ask_vol_col: str) -> pl.DataFrame:
    bid_vol = df.select(bid_vol_col).to_numpy().sum()
    ask_vol = df.select(ask_vol_col).to_numpy().sum()
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
    return pl.DataFrame([{"order_imbalance": str(Decimal(str(imbalance))), "timestamp": datetime.now(timezone.utc).isoformat()}])
