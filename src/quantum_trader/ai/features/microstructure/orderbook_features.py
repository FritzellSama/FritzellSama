"""Orderbook microstructure features."""
from decimal import Decimal
from typing import Dict
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class OrderbookFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.depth = config.get("depth", 10)
        
    def extract_features(self, orderbook: Dict) -> Dict[str, Decimal]:
        bid_ask_spread = Decimal(str(orderbook.get("ask", 0))) - Decimal(str(orderbook.get("bid", 0)))
        return {"spread": bid_ask_spread, "depth": Decimal(str(self.depth))}
