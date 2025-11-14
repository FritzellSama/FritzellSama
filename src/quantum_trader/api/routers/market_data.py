"""
Market Data API Router.

Provides REST endpoints for accessing market data including ticker information,
order books, OHLCV data, and recent trades.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, validator
import polars as pl
from structlog import get_logger

from quantum_trader.api.dependencies import (
    verify_api_key,
    check_rate_limit,
    get_data_service,
    get_pagination,
    PaginationParams
)
from quantum_trader.api.exceptions import (
    ValidationError,
    InvalidSymbolError,
    InvalidExchangeError,
    ExchangeError,
    ResourceNotFoundError
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/market-data",
    tags=["market-data"],
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)


# Request/Response Models

class TickerResponse(BaseModel):
    """Ticker data response."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    last_price: str = Field(..., description="Last traded price")
    volume_24h: str = Field(..., description="24h trading volume")
    high_24h: str = Field(..., description="24h high price")
    low_24h: str = Field(..., description="24h low price")
    change_24h: str = Field(..., description="24h price change")
    change_percent_24h: str = Field(..., description="24h price change percentage")
    timestamp: str = Field(..., description="Data timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "last_price": "50000.00",
                "volume_24h": "12345.67",
                "high_24h": "51000.00",
                "low_24h": "49000.00",
                "change_24h": "500.00",
                "change_percent_24h": "1.01",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class OrderBookLevel(BaseModel):
    """Order book price level."""

    price: str = Field(..., description="Price level")
    quantity: str = Field(..., description="Quantity at price level")


class OrderBookResponse(BaseModel):
    """Order book snapshot response."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    bids: List[OrderBookLevel] = Field(..., description="Bid levels")
    asks: List[OrderBookLevel] = Field(..., description="Ask levels")
    spread: str = Field(..., description="Bid-ask spread")
    mid_price: Optional[str] = Field(None, description="Mid price")
    timestamp: str = Field(..., description="Snapshot timestamp (ISO 8601)")
    sequence: int = Field(..., description="Sequence number")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "bids": [
                    {"price": "49999.00", "quantity": "1.5"},
                    {"price": "49998.00", "quantity": "2.0"}
                ],
                "asks": [
                    {"price": "50001.00", "quantity": "1.2"},
                    {"price": "50002.00", "quantity": "1.8"}
                ],
                "spread": "2.00",
                "mid_price": "50000.00",
                "timestamp": "2025-01-15T10:30:00Z",
                "sequence": 12345
            }
        }


class OHLCVCandle(BaseModel):
    """OHLCV candlestick data."""

    timestamp: str = Field(..., description="Candle timestamp (ISO 8601)")
    open: str = Field(..., description="Opening price")
    high: str = Field(..., description="Highest price")
    low: str = Field(..., description="Lowest price")
    close: str = Field(..., description="Closing price")
    volume: str = Field(..., description="Trading volume")


class OHLCVResponse(BaseModel):
    """OHLCV data response."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    timeframe: str = Field(..., description="Timeframe (e.g., '1m', '1h')")
    candles: List[OHLCVCandle] = Field(..., description="OHLCV candles")
    count: int = Field(..., description="Number of candles returned")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "exchange": "binance",
                "timeframe": "1h",
                "candles": [
                    {
                        "timestamp": "2025-01-15T10:00:00Z",
                        "open": "49500.00",
                        "high": "50000.00",
                        "low": "49400.00",
                        "close": "49900.00",
                        "volume": "123.45"
                    }
                ],
                "count": 1
            }
        }


class TradeResponse(BaseModel):
    """Individual trade data."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    trade_id: str = Field(..., description="Trade identifier")
    price: str = Field(..., description="Trade price")
    quantity: str = Field(..., description="Trade quantity")
    side: str = Field(..., description="Trade side (buy/sell)")
    timestamp: str = Field(..., description="Trade timestamp (ISO 8601)")


class RecentTradesResponse(BaseModel):
    """Recent trades response."""

    symbol: str = Field(..., description="Trading pair symbol")
    exchange: str = Field(..., description="Exchange name")
    trades: List[TradeResponse] = Field(..., description="Recent trades")
    count: int = Field(..., description="Number of trades returned")


class MarketSummaryItem(BaseModel):
    """Market summary for a single symbol."""

    symbol: str
    last_price: str
    volume_24h: str
    change_percent_24h: str


class MarketSummaryResponse(BaseModel):
    """Market summary response."""

    exchange: str = Field(..., description="Exchange name")
    markets: List[MarketSummaryItem] = Field(..., description="Market summaries")
    count: int = Field(..., description="Number of markets")
    timestamp: str = Field(..., description="Summary timestamp (ISO 8601)")


# Endpoints

@router.get(
    "/ticker/{exchange}/{symbol}",
    response_model=TickerResponse,
    summary="Get ticker data",
    description="Retrieve real-time ticker data for a trading pair"
)
async def get_ticker(
    exchange: str = Query(..., description="Exchange name (e.g., 'binance')"),
    symbol: str = Query(..., description="Trading pair (e.g., 'BTC/USDT')"),
    use_cache: bool = Query(True, description="Use cached data if available"),
    data_service = Depends(get_data_service)
) -> TickerResponse:
    """
    Get real-time ticker data for a trading pair.

    Returns current price, volume, and 24h statistics.

    Example:
        GET /market-data/ticker/binance/BTC/USDT
    """
    try:
        ticker = await data_service.get_ticker(symbol, exchange, use_cache)

        if not ticker:
            raise ResourceNotFoundError(
                resource="Ticker",
                resource_id=f"{exchange}:{symbol}",
                details={"exchange": exchange, "symbol": symbol}
            )

        return TickerResponse(
            symbol=ticker.symbol,
            exchange=ticker.exchange,
            last_price=str(ticker.last_price),
            volume_24h=str(ticker.volume_24h),
            high_24h=str(ticker.high_24h),
            low_24h=str(ticker.low_24h),
            change_24h=str(ticker.change_24h),
            change_percent_24h=str(ticker.change_percent_24h),
            timestamp=ticker.timestamp.isoformat()
        )

    except (InvalidSymbolError, InvalidExchangeError, ResourceNotFoundError):
        raise

    except Exception as e:
        logger.error(
            "get_ticker_error",
            exchange=exchange,
            symbol=symbol,
            error=str(e)
        )
        raise ExchangeError(
            exchange=exchange,
            message=f"Failed to fetch ticker data: {str(e)}"
        )


@router.get(
    "/orderbook/{exchange}/{symbol}",
    response_model=OrderBookResponse,
    summary="Get order book",
    description="Retrieve order book snapshot with bids and asks"
)
async def get_orderbook(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading pair"),
    depth: int = Query(20, ge=1, le=100, description="Order book depth (1-100)"),
    use_cache: bool = Query(True, description="Use cached data if available"),
    data_service = Depends(get_data_service)
) -> OrderBookResponse:
    """
    Get order book snapshot.

    Returns current bids and asks up to specified depth.

    Example:
        GET /market-data/orderbook/binance/BTC/USDT?depth=10
    """
    try:
        orderbook = await data_service.get_orderbook(
            symbol, exchange, depth, use_cache
        )

        if not orderbook:
            raise ResourceNotFoundError(
                resource="OrderBook",
                resource_id=f"{exchange}:{symbol}",
                details={"exchange": exchange, "symbol": symbol}
            )

        return OrderBookResponse(
            symbol=orderbook.symbol,
            exchange=orderbook.exchange,
            bids=[
                OrderBookLevel(price=str(price), quantity=str(qty))
                for price, qty in orderbook.bids
            ],
            asks=[
                OrderBookLevel(price=str(price), quantity=str(qty))
                for price, qty in orderbook.asks
            ],
            spread=str(orderbook.get_spread()),
            mid_price=str(orderbook.get_mid_price()) if orderbook.get_mid_price() else None,
            timestamp=orderbook.timestamp.isoformat(),
            sequence=orderbook.sequence
        )

    except (InvalidSymbolError, InvalidExchangeError, ResourceNotFoundError):
        raise

    except Exception as e:
        logger.error(
            "get_orderbook_error",
            exchange=exchange,
            symbol=symbol,
            error=str(e)
        )
        raise ExchangeError(
            exchange=exchange,
            message=f"Failed to fetch order book: {str(e)}"
        )


@router.get(
    "/ohlcv/{exchange}/{symbol}",
    response_model=OHLCVResponse,
    summary="Get OHLCV data",
    description="Retrieve candlestick (OHLCV) data for technical analysis"
)
async def get_ohlcv(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading pair"),
    timeframe: str = Query(..., description="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of candles (1-1000)"),
    start_time: Optional[str] = Query(None, description="Start time (ISO 8601)"),
    end_time: Optional[str] = Query(None, description="End time (ISO 8601)"),
    data_service = Depends(get_data_service)
) -> OHLCVResponse:
    """
    Get OHLCV (candlestick) data.

    Returns historical price data for specified timeframe.

    Example:
        GET /market-data/ohlcv/binance/BTC/USDT?timeframe=1h&limit=100
    """
    # Validate timeframe
    valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"}
    if timeframe not in valid_timeframes:
        raise ValidationError(
            message=f"Invalid timeframe: {timeframe}",
            field="timeframe",
            value=timeframe,
            details={"valid_timeframes": list(valid_timeframes)}
        )

    # Parse timestamps
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise ValidationError(
                message="Invalid start_time format",
                field="start_time",
                value=start_time
            )

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            raise ValidationError(
                message="Invalid end_time format",
                field="end_time",
                value=end_time
            )

    try:
        df = await data_service.get_ohlcv(
            symbol, exchange, timeframe, start_dt, end_dt, limit
        )

        if df is None or df.is_empty():
            raise ResourceNotFoundError(
                resource="OHLCV",
                resource_id=f"{exchange}:{symbol}:{timeframe}",
                details={
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            )

        # Convert Polars DataFrame to response format
        candles = []
        for row in df.iter_rows(named=True):
            candles.append(OHLCVCandle(
                timestamp=row["timestamp"].isoformat() if isinstance(row["timestamp"], datetime) else row["timestamp"],
                open=str(row["open"]),
                high=str(row["high"]),
                low=str(row["low"]),
                close=str(row["close"]),
                volume=str(row["volume"])
            ))

        return OHLCVResponse(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            candles=candles,
            count=len(candles)
        )

    except (InvalidSymbolError, InvalidExchangeError, ResourceNotFoundError, ValidationError):
        raise

    except Exception as e:
        logger.error(
            "get_ohlcv_error",
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            error=str(e)
        )
        raise ExchangeError(
            exchange=exchange,
            message=f"Failed to fetch OHLCV data: {str(e)}"
        )


@router.get(
    "/trades/{exchange}/{symbol}",
    response_model=RecentTradesResponse,
    summary="Get recent trades",
    description="Retrieve recent trades for a trading pair"
)
async def get_recent_trades(
    exchange: str = Query(..., description="Exchange name"),
    symbol: str = Query(..., description="Trading pair"),
    limit: int = Query(100, ge=1, le=500, description="Number of trades (1-500)"),
    data_service = Depends(get_data_service)
) -> RecentTradesResponse:
    """
    Get recent trades.

    Returns list of recent executed trades.

    Example:
        GET /market-data/trades/binance/BTC/USDT?limit=50
    """
    try:
        trades = await data_service.get_recent_trades(symbol, exchange, limit)

        trade_responses = [
            TradeResponse(
                symbol=trade.symbol,
                exchange=trade.exchange,
                trade_id=trade.trade_id,
                price=str(trade.price),
                quantity=str(trade.quantity),
                side=trade.side,
                timestamp=trade.timestamp.isoformat()
            )
            for trade in trades
        ]

        return RecentTradesResponse(
            symbol=symbol,
            exchange=exchange,
            trades=trade_responses,
            count=len(trade_responses)
        )

    except (InvalidSymbolError, InvalidExchangeError):
        raise

    except Exception as e:
        logger.error(
            "get_recent_trades_error",
            exchange=exchange,
            symbol=symbol,
            error=str(e)
        )
        raise ExchangeError(
            exchange=exchange,
            message=f"Failed to fetch recent trades: {str(e)}"
        )


@router.get(
    "/summary/{exchange}",
    response_model=MarketSummaryResponse,
    summary="Get market summary",
    description="Retrieve summary for all markets on exchange"
)
async def get_market_summary(
    exchange: str = Query(..., description="Exchange name"),
    data_service = Depends(get_data_service)
) -> MarketSummaryResponse:
    """
    Get market summary for all trading pairs on exchange.

    Returns high-level statistics for all markets.

    Example:
        GET /market-data/summary/binance
    """
    try:
        summary = await data_service.get_market_summary(exchange)

        markets = [
            MarketSummaryItem(
                symbol=symbol,
                last_price=str(ticker.last_price),
                volume_24h=str(ticker.volume_24h),
                change_percent_24h=str(ticker.change_percent_24h)
            )
            for symbol, ticker in summary.items()
        ]

        return MarketSummaryResponse(
            exchange=exchange,
            markets=markets,
            count=len(markets),
            timestamp=datetime.utcnow().isoformat()
        )

    except InvalidExchangeError:
        raise

    except Exception as e:
        logger.error(
            "get_market_summary_error",
            exchange=exchange,
            error=str(e)
        )
        raise ExchangeError(
            exchange=exchange,
            message=f"Failed to fetch market summary: {str(e)}"
        )


@router.get(
    "/health",
    summary="Data service health check",
    description="Check health status of market data service"
)
async def health_check(
    data_service = Depends(get_data_service)
) -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns service health and cache statistics.

    Example:
        GET /market-data/health
    """
    try:
        cache_stats = data_service.get_cache_stats()

        return {
            "status": "healthy",
            "service": "market_data",
            "timestamp": datetime.utcnow().isoformat(),
            "cache": cache_stats
        }

    except Exception as e:
        logger.error("health_check_error", error=str(e))
        return {
            "status": "unhealthy",
            "service": "market_data",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
