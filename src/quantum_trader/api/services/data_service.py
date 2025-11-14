"""
Data Service for market data and trading information.

Provides centralized access to market data, order book information,
historical data, and real-time trading data with caching and optimization.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import polars as pl

from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bids and asks."""

    symbol: str
    exchange: str
    bids: List[Tuple[Decimal, Decimal]]  # [(price, quantity)]
    asks: List[Tuple[Decimal, Decimal]]  # [(price, quantity)]
    timestamp: datetime
    sequence: int = 0

    def get_spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return Decimal("0")
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return best_ask - best_bid

    def get_mid_price(self) -> Optional[Decimal]:
        """Calculate mid price."""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return (best_bid + best_ask) / Decimal("2")


@dataclass
class TickerData:
    """Real-time ticker data."""

    symbol: str
    exchange: str
    last_price: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    change_percent_24h: Decimal
    timestamp: datetime


@dataclass
class TradeData:
    """Individual trade data."""

    symbol: str
    exchange: str
    trade_id: str
    price: Decimal
    quantity: Decimal
    side: str  # 'buy' or 'sell'
    timestamp: datetime


class DataService:
    """
    Core data service for market and trading data.

    Provides access to real-time and historical market data with caching,
    rate limiting, and error handling. Aggregates data from multiple sources
    and provides normalized interface.

    Attributes:
        config: Service configuration
        _cache: In-memory data cache
        _cache_ttl: Cache time-to-live settings
        _running: Service running state
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize data service.

        Args:
            config: Configuration with cache settings, data sources, rate limits

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl: Dict[str, int] = {
            "ticker": self.config.get("cache_ttl_ticker", 5),
            "orderbook": self.config.get("cache_ttl_orderbook", 1),
            "ohlcv": self.config.get("cache_ttl_ohlcv", 60),
            "trades": self.config.get("cache_ttl_trades", 10),
        }

        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("data_service_initialized", cache_ttl=self._cache_ttl)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = ["max_cache_size", "cache_cleanup_interval"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config["max_cache_size"] <= 0:
            raise ValueError("max_cache_size must be positive")

        if self.config["cache_cleanup_interval"] <= 0:
            raise ValueError("cache_cleanup_interval must be positive")

    async def start(self) -> None:
        """Start the data service."""
        if self._running:
            logger.warning("data_service_already_running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cache_cleanup_loop())

        logger.info("data_service_started")

    async def stop(self) -> None:
        """Stop the data service gracefully."""
        if not self._running:
            logger.warning("data_service_not_running")
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._cache.clear()
        self._cache_timestamps.clear()

        logger.info("data_service_stopped")

    async def get_ticker(
        self,
        symbol: str,
        exchange: str,
        use_cache: bool = True
    ) -> Optional[TickerData]:
        """
        Get real-time ticker data.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            use_cache: Whether to use cached data

        Returns:
            TickerData if available, None otherwise

        Example:
            >>> ticker = await service.get_ticker("BTC/USDT", "binance")
            >>> print(f"Price: {ticker.last_price}")
        """
        cache_key = f"ticker:{exchange}:{symbol}"

        if use_cache:
            cached = self._get_from_cache(cache_key, "ticker")
            if cached:
                return cached

        try:
            # In production, this would call exchange API
            # For now, we simulate data fetching
            ticker_data = await self._fetch_ticker_from_exchange(symbol, exchange)

            if ticker_data:
                self._put_in_cache(cache_key, ticker_data)

            return ticker_data

        except Exception as e:
            logger.error(
                "get_ticker_error",
                symbol=symbol,
                exchange=exchange,
                error=str(e)
            )
            return None

    async def get_orderbook(
        self,
        symbol: str,
        exchange: str,
        depth: int = 20,
        use_cache: bool = True
    ) -> Optional[OrderBookSnapshot]:
        """
        Get order book snapshot.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            depth: Number of levels to fetch
            use_cache: Whether to use cached data

        Returns:
            OrderBookSnapshot if available, None otherwise

        Example:
            >>> orderbook = await service.get_orderbook("BTC/USDT", "binance", depth=10)
            >>> spread = orderbook.get_spread()
        """
        cache_key = f"orderbook:{exchange}:{symbol}:{depth}"

        if use_cache:
            cached = self._get_from_cache(cache_key, "orderbook")
            if cached:
                return cached

        try:
            orderbook = await self._fetch_orderbook_from_exchange(
                symbol, exchange, depth
            )

            if orderbook:
                self._put_in_cache(cache_key, orderbook)

            return orderbook

        except Exception as e:
            logger.error(
                "get_orderbook_error",
                symbol=symbol,
                exchange=exchange,
                error=str(e)
            )
            return None

    async def get_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> Optional[pl.DataFrame]:
        """
        Get OHLCV (candlestick) data.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            start_time: Start time (UTC)
            end_time: End time (UTC)
            limit: Maximum number of candles

        Returns:
            Polars DataFrame with OHLCV data

        Example:
            >>> df = await service.get_ohlcv("BTC/USDT", "binance", "1h", limit=100)
            >>> print(df.select(["timestamp", "close"]))
        """
        cache_key = f"ohlcv:{exchange}:{symbol}:{timeframe}:{limit}"

        # Check cache
        cached = self._get_from_cache(cache_key, "ohlcv")
        if cached is not None:
            return cached

        try:
            df = await self._fetch_ohlcv_from_exchange(
                symbol, exchange, timeframe, start_time, end_time, limit
            )

            if df is not None and not df.is_empty():
                self._put_in_cache(cache_key, df)

            return df

        except Exception as e:
            logger.error(
                "get_ohlcv_error",
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                error=str(e)
            )
            return None

    async def get_recent_trades(
        self,
        symbol: str,
        exchange: str,
        limit: int = 100
    ) -> List[TradeData]:
        """
        Get recent trades.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name
            limit: Number of trades to fetch

        Returns:
            List of recent trades

        Example:
            >>> trades = await service.get_recent_trades("BTC/USDT", "binance", limit=50)
            >>> for trade in trades[:5]:
            >>>     print(f"{trade.price} @ {trade.quantity}")
        """
        cache_key = f"trades:{exchange}:{symbol}:{limit}"

        cached = self._get_from_cache(cache_key, "trades")
        if cached:
            return cached

        try:
            trades = await self._fetch_trades_from_exchange(symbol, exchange, limit)

            if trades:
                self._put_in_cache(cache_key, trades)

            return trades

        except Exception as e:
            logger.error(
                "get_recent_trades_error",
                symbol=symbol,
                exchange=exchange,
                error=str(e)
            )
            return []

    async def get_market_summary(
        self,
        exchange: str
    ) -> Dict[str, TickerData]:
        """
        Get summary for all markets on exchange.

        Args:
            exchange: Exchange name

        Returns:
            Dictionary mapping symbol to TickerData
        """
        cache_key = f"market_summary:{exchange}"

        cached = self._get_from_cache(cache_key, "ticker")
        if cached:
            return cached

        try:
            summary = await self._fetch_market_summary_from_exchange(exchange)

            if summary:
                self._put_in_cache(cache_key, summary)

            return summary

        except Exception as e:
            logger.error(
                "get_market_summary_error",
                exchange=exchange,
                error=str(e)
            )
            return {}

    # Private methods for data fetching (would integrate with actual exchange APIs)

    async def _fetch_ticker_from_exchange(
        self,
        symbol: str,
        exchange: str
    ) -> Optional[TickerData]:
        """Fetch ticker from exchange API."""
        # In production, this would call the actual exchange API
        # This is a placeholder that would be replaced with real implementation
        logger.debug("fetching_ticker", symbol=symbol, exchange=exchange)
        return None

    async def _fetch_orderbook_from_exchange(
        self,
        symbol: str,
        exchange: str,
        depth: int
    ) -> Optional[OrderBookSnapshot]:
        """Fetch order book from exchange API."""
        logger.debug(
            "fetching_orderbook",
            symbol=symbol,
            exchange=exchange,
            depth=depth
        )
        return None

    async def _fetch_ohlcv_from_exchange(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> Optional[pl.DataFrame]:
        """Fetch OHLCV data from exchange API."""
        logger.debug(
            "fetching_ohlcv",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            limit=limit
        )
        return None

    async def _fetch_trades_from_exchange(
        self,
        symbol: str,
        exchange: str,
        limit: int
    ) -> List[TradeData]:
        """Fetch recent trades from exchange API."""
        logger.debug(
            "fetching_trades",
            symbol=symbol,
            exchange=exchange,
            limit=limit
        )
        return []

    async def _fetch_market_summary_from_exchange(
        self,
        exchange: str
    ) -> Dict[str, TickerData]:
        """Fetch market summary from exchange API."""
        logger.debug("fetching_market_summary", exchange=exchange)
        return {}

    # Cache management

    def _get_from_cache(
        self,
        key: str,
        cache_type: str
    ) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(key)
        if not timestamp:
            return None

        ttl = self._cache_ttl.get(cache_type, 60)
        age = (datetime.utcnow() - timestamp).total_seconds()

        if age > ttl:
            # Expired
            del self._cache[key]
            del self._cache_timestamps[key]
            return None

        return self._cache[key]

    def _put_in_cache(self, key: str, value: Any) -> None:
        """Put item in cache."""
        # Check cache size limit
        if len(self._cache) >= self.config["max_cache_size"]:
            # Remove oldest entry
            oldest_key = min(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]

        self._cache[key] = value
        self._cache_timestamps[key] = datetime.utcnow()

    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up expired cache entries."""
        while self._running:
            try:
                await asyncio.sleep(self.config["cache_cleanup_interval"])
                await self._cleanup_expired_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cache_cleanup_error", error=str(e))

    async def _cleanup_expired_cache(self) -> None:
        """Remove expired entries from cache."""
        now = datetime.utcnow()
        expired_keys = []

        for key, timestamp in self._cache_timestamps.items():
            # Find cache type from key
            cache_type = key.split(":")[0]
            ttl = self._cache_ttl.get(cache_type, 60)

            age = (now - timestamp).total_seconds()
            if age > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            del self._cache_timestamps[key]

        if expired_keys:
            logger.debug(
                "cache_cleanup",
                expired_count=len(expired_keys),
                remaining=len(self._cache)
            )

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        return {
            "total_entries": len(self._cache),
            "max_size": self.config["max_cache_size"],
            "cache_types": {
                cache_type: sum(
                    1 for key in self._cache.keys()
                    if key.startswith(f"{cache_type}:")
                )
                for cache_type in self._cache_ttl.keys()
            }
        }

    def clear_cache(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            cache_type: Optional cache type to clear (clears all if None)

        Returns:
            Number of entries cleared
        """
        if cache_type is None:
            count = len(self._cache)
            self._cache.clear()
            self._cache_timestamps.clear()
            return count

        keys_to_remove = [
            key for key in self._cache.keys()
            if key.startswith(f"{cache_type}:")
        ]

        for key in keys_to_remove:
            del self._cache[key]
            del self._cache_timestamps[key]

        logger.info("cache_cleared", cache_type=cache_type, count=len(keys_to_remove))
        return len(keys_to_remove)
