"""Order Flow Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_flow_features(df: pl.DataFrame, volume_col: str, price_col: str) -> pl.DataFrame:
    volume = df.select(volume_col).to_numpy()
    price = df.select(price_col).to_numpy()
    flow = (volume[1:] * price[1:]).sum()
    return pl.DataFrame([{"order_flow": str(Decimal(str(flow))), "timestamp": datetime.now(timezone.utc).isoformat()}])
