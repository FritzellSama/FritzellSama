"""Market data repository for database operations.

This module provides repository pattern implementation for market data
with support for high-performance queries and polars DataFrame integration.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, List, Any
import polars as pl
from sqlalchemy import select, text, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

logger = get_logger(__name__)


class MarketRepository:
    """Repository for market data operations.

    Provides high-performance queries for market data including OHLCV,
    orderbook snapshots, and tick data with polars DataFrame integration.

    Attributes:
        session: Async database session

    Example:
        >>> repo = MarketRepository(session)
        >>> ohlcv = await repo.get_ohlcv("BTC/USDT", "1h", start_time, end_time)
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize market repository.

        Args:
            session: Async database session

        Example:
            >>> async with db.get_session() as session:
            ...     repo = MarketRepository(session)
        """
        self.session = session
        logger.debug("MarketRepository initialized")

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        exchange: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get OHLCV (candlestick) data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            exchange: Optional exchange filter

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> df = await repo.get_ohlcv(
            ...     "BTC/USDT",
            ...     "1h",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            # Build query
            query = """
                SELECT
                    time_bucket(:timeframe::interval, timestamp) AS bucket,
                    symbol,
                    exchange,
                    (array_agg(price ORDER BY timestamp ASC))[1] as open,
                    MAX(price) as high,
                    MIN(price) as low,
                    (array_agg(price ORDER BY timestamp DESC))[1] as close,
                    SUM(quantity) as volume
                FROM trades
                WHERE symbol = :symbol
                    AND timestamp >= :start_time
                    AND timestamp < :end_time
            """

            if exchange:
                query += " AND exchange = :exchange"

            query += """
                GROUP BY bucket, symbol, exchange
                ORDER BY bucket ASC
            """

            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time,
                "end_time": end_time,
            }

            if exchange:
                params["exchange"] = exchange

            result = await self.session.execute(text(query), params)
            rows = result.fetchall()

            # Convert to polars DataFrame
            if not rows:
                return pl.DataFrame({
                    "timestamp": [],
                    "symbol": [],
                    "exchange": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                })

            data = {
                "timestamp": [row[0] for row in rows],
                "symbol": [row[1] for row in rows],
                "exchange": [row[2] for row in rows],
                "open": [Decimal(str(row[3])) for row in rows],
                "high": [Decimal(str(row[4])) for row in rows],
                "low": [Decimal(str(row[5])) for row in rows],
                "close": [Decimal(str(row[6])) for row in rows],
                "volume": [Decimal(str(row[7])) for row in rows],
            }

            df = pl.DataFrame(data)
            logger.info(
                "OHLCV data retrieved",
                symbol=symbol,
                timeframe=timeframe,
                rows=len(df)
            )
            return df

        except Exception as e:
            logger.error("Failed to get OHLCV data", error=str(e), symbol=symbol)
            raise

    async def get_latest_price(
        self,
        symbol: str,
        exchange: Optional[str] = None
    ) -> Optional[Decimal]:
        """Get latest trade price for a symbol.

        Args:
            symbol: Trading pair
            exchange: Optional exchange filter

        Returns:
            Latest price or None if no trades found

        Example:
            >>> price = await repo.get_latest_price("BTC/USDT", "BINANCE")
        """
        try:
            query = """
                SELECT price
                FROM trades
                WHERE symbol = :symbol
            """

            if exchange:
                query += " AND exchange = :exchange"

            query += """
                ORDER BY timestamp DESC
                LIMIT 1
            """

            params = {"symbol": symbol}
            if exchange:
                params["exchange"] = exchange

            result = await self.session.execute(text(query), params)
            row = result.fetchone()

            if row:
                price = Decimal(str(row[0]))
                logger.debug("Latest price retrieved", symbol=symbol, price=price)
                return price

            return None

        except Exception as e:
            logger.error("Failed to get latest price", error=str(e), symbol=symbol)
            raise

    async def get_volume_stats(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        exchange: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """Get volume statistics for a symbol.

        Args:
            symbol: Trading pair
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            exchange: Optional exchange filter

        Returns:
            Dictionary with volume statistics

        Example:
            >>> stats = await repo.get_volume_stats(
            ...     "BTC/USDT",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            query = """
                SELECT
                    SUM(quantity) as total_volume,
                    AVG(quantity) as avg_volume,
                    MIN(quantity) as min_volume,
                    MAX(quantity) as max_volume,
                    COUNT(*) as trade_count
                FROM trades
                WHERE symbol = :symbol
                    AND timestamp >= :start_time
                    AND timestamp < :end_time
            """

            if exchange:
                query += " AND exchange = :exchange"

            params = {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
            }

            if exchange:
                params["exchange"] = exchange

            result = await self.session.execute(text(query), params)
            row = result.fetchone()

            if row and row[0]:
                stats = {
                    "total_volume": Decimal(str(row[0])),
                    "avg_volume": Decimal(str(row[1])),
                    "min_volume": Decimal(str(row[2])),
                    "max_volume": Decimal(str(row[3])),
                    "trade_count": int(row[4]),
                }
                logger.info("Volume stats retrieved", symbol=symbol, stats=stats)
                return stats

            return {
                "total_volume": Decimal("0"),
                "avg_volume": Decimal("0"),
                "min_volume": Decimal("0"),
                "max_volume": Decimal("0"),
                "trade_count": 0,
            }

        except Exception as e:
            logger.error("Failed to get volume stats", error=str(e), symbol=symbol)
            raise

    async def get_price_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        exchange: Optional[str] = None,
    ) -> Dict[str, Decimal]:
        """Get price range for a symbol.

        Args:
            symbol: Trading pair
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            exchange: Optional exchange filter

        Returns:
            Dictionary with high, low, and price range

        Example:
            >>> range_data = await repo.get_price_range(
            ...     "BTC/USDT",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            query = """
                SELECT
                    MIN(price) as low,
                    MAX(price) as high,
                    AVG(price) as avg_price
                FROM trades
                WHERE symbol = :symbol
                    AND timestamp >= :start_time
                    AND timestamp < :end_time
            """

            if exchange:
                query += " AND exchange = :exchange"

            params = {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
            }

            if exchange:
                params["exchange"] = exchange

            result = await self.session.execute(text(query), params)
            row = result.fetchone()

            if row and row[0]:
                low = Decimal(str(row[0]))
                high = Decimal(str(row[1]))
                avg = Decimal(str(row[2]))

                range_data = {
                    "low": low,
                    "high": high,
                    "avg_price": avg,
                    "range": high - low,
                    "range_percent": ((high - low) / low * Decimal("100")) if low > 0 else Decimal("0"),
                }

                logger.info("Price range retrieved", symbol=symbol, range_data=range_data)
                return range_data

            return {
                "low": Decimal("0"),
                "high": Decimal("0"),
                "avg_price": Decimal("0"),
                "range": Decimal("0"),
                "range_percent": Decimal("0"),
            }

        except Exception as e:
            logger.error("Failed to get price range", error=str(e), symbol=symbol)
            raise

    async def get_trade_count(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        exchange: Optional[str] = None,
    ) -> int:
        """Get trade count for a symbol.

        Args:
            symbol: Trading pair
            start_time: Start timestamp (UTC)
            end_time: End timestamp (UTC)
            exchange: Optional exchange filter

        Returns:
            Number of trades

        Example:
            >>> count = await repo.get_trade_count(
            ...     "BTC/USDT",
            ...     datetime(2024, 1, 1),
            ...     datetime(2024, 1, 2)
            ... )
        """
        try:
            query = """
                SELECT COUNT(*)
                FROM trades
                WHERE symbol = :symbol
                    AND timestamp >= :start_time
                    AND timestamp < :end_time
            """

            if exchange:
                query += " AND exchange = :exchange"

            params = {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
            }

            if exchange:
                params["exchange"] = exchange

            result = await self.session.execute(text(query), params)
            count = result.scalar()

            logger.debug("Trade count retrieved", symbol=symbol, count=count)
            return int(count) if count else 0

        except Exception as e:
            logger.error("Failed to get trade count", error=str(e), symbol=symbol)
            raise
