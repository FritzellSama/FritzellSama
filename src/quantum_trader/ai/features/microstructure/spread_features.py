"""Bid-ask spread features."""
from decimal import Decimal
from typing import Dict
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class SpreadFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def extract(self, orderbook_data: pl.DataFrame) -> Dict[str, Decimal]:
        if "bid" in orderbook_data.columns and "ask" in orderbook_data.columns:
            bids = orderbook_data["bid"].to_numpy()
            asks = orderbook_data["ask"].to_numpy()
            
            spread = Decimal(str(np.mean(asks - bids)))
            spread_pct = spread / Decimal(str(np.mean(bids))) * Decimal("100") if np.mean(bids) > 0 else Decimal("0")
            
            return {
                "spread": spread,
                "spread_pct": spread_pct
            }
            
        return {"spread": Decimal("0"), "spread_pct": Decimal("0")}
