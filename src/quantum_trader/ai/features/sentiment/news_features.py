"""News sentiment features."""
from decimal import Decimal
from typing import Dict, List
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class NewsFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def extract(self, news_data: List[str]) -> Dict[str, Decimal]:
        return {"sentiment": Decimal("0.5"), "volume": Decimal("100")}
