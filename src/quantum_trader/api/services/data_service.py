"""Data service layer."""
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class DataService:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> pl.DataFrame:
        """Get historical market data."""
        logger.info("Fetching market data", symbol=symbol, timeframe=timeframe)
        
        # Placeholder data
        return pl.DataFrame({
            "timestamp": [datetime.utcnow()],
            "open": [Decimal("50000")],
            "high": [Decimal("51000")],
            "low": [Decimal("49000")],
            "close": [Decimal("50500")],
            "volume": [Decimal("1000")]
        })
        
    async def get_latest_price(self, symbol: str) -> Decimal:
        """Get latest price for symbol."""
        return Decimal("50000")
        
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """Get orderbook for symbol."""
        return {
            "symbol": symbol,
            "bids": [(Decimal("49990"), Decimal("1.5"))],
            "asks": [(Decimal("50010"), Decimal("1.2"))]
        }
