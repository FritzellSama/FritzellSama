"""Fear and Greed Index Features."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl

logger = logging.getLogger(__name__)

async def compute_fear_greed_features(df: pl.DataFrame, volatility_col: str) -> pl.DataFrame:
    vol = df.select(volatility_col).to_numpy().mean()
    fear_greed = Decimal("50") - Decimal(str(vol * 100))
    return pl.DataFrame([{"fear_greed_index": str(fear_greed), "timestamp": datetime.now(timezone.utc).isoformat()}])
