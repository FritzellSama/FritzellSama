"""
Market data router for Quantum Trader AI API.

Provides endpoints for market data retrieval.
"""

from decimal import Decimal
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from structlog import get_logger

from quantum_trader.api.services.data_service import DataService
from quantum_trader.api.dependencies import (
    get_data_service,
    get_current_user,
    get_optional_user
)
from quantum_trader.api.exceptions import (
    DataNotFoundException,
    DataValidationException
)

logger = get_logger(__name__)

router = APIRouter()


# Response schemas

class TickerResponse(BaseModel):
    """Ticker data response."""
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange name")
    last_price: str = Field(..., description="Last traded price")
    bid: Optional[str] = Field(None, description="Best bid price")
    ask: Optional[str] = Field(None, description="Best ask price")
    volume_24h: str = Field(..., description="24h volume")
    timestamp: str = Field(..., description="Data timestamp")


class OrderBookLevel(BaseModel):
    """Order book level."""
    price: str = Field(..., description="Price level")
    quantity: str = Field(..., description="Quantity at level")


class OrderBookResponse(BaseModel):
    """Order book response."""
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange name")
    bids: List[OrderBookLevel] = Field(..., description="Bid levels")
    asks: List[OrderBookLevel] = Field(..., description="Ask levels")
    timestamp: str = Field(..., description="Data timestamp")


class OHLCVCandle(BaseModel):
    """OHLCV candle."""
    timestamp: str = Field(..., description="Candle timestamp")
    open: str = Field(..., description="Open price")
    high: str = Field(..., description="High price")
    low: str = Field(..., description="Low price")
    close: str = Field(..., description="Close price")
    volume: str = Field(..., description="Volume")


class OHLCVResponse(BaseModel):
    """OHLCV data response."""
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange name")
    timeframe: str = Field(..., description="Timeframe")
    candles: List[OHLCVCandle] = Field(..., description="OHLCV candles")


class TradeResponse(BaseModel):
    """Trade data response."""
    trade_id: str = Field(..., description="Trade ID")
    timestamp: str = Field(..., description="Trade timestamp")
    price: str = Field(..., description="Trade price")
    quantity: str = Field(..., description="Trade quantity")
    side: str = Field(..., description="Trade side (BUY/SELL)")


class SymbolInfo(BaseModel):
    """Trading symbol information."""
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange name")
    base_asset: str = Field(..., description="Base asset")
    quote_asset: str = Field(..., description="Quote asset")
    is_active: bool = Field(..., description="Symbol active status")


# Endpoints

@router.get(
    "/ticker/{exchange}/{symbol}",
    response_model=TickerResponse,
    summary="Get ticker data",
    description="Get current ticker data for a trading pair"
)
async def get_ticker(
    exchange: str,
    symbol: str,
    data_service: DataService = Depends(get_data_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> TickerResponse:
    """Get ticker data.

    Args:
        exchange: Exchange name
        symbol: Trading symbol
        data_service: Data service dependency
        current_user: Current user (optional)

    Returns:
        Ticker data

    Raises:
        HTTPException: If ticker not found
    """
    try:
        logger.info("Get ticker request", exchange=exchange, symbol=symbol)

        ticker_data = await data_service.get_ticker(symbol, exchange)

        return TickerResponse(**ticker_data)

    except DataNotFoundException as e:
        logger.warning("Ticker not found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DataValidationException as e:
        logger.warning("Ticker validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Get ticker failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get ticker"
        )


@router.get(
    "/orderbook/{exchange}/{symbol}",
    response_model=OrderBookResponse,
    summary="Get order book",
    description="Get current order book for a trading pair"
)
async def get_orderbook(
    exchange: str,
    symbol: str,
    depth: int = Query(default=20, ge=1, le=100, description="Order book depth"),
    data_service: DataService = Depends(get_data_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> OrderBookResponse:
    """Get order book.

    Args:
        exchange: Exchange name
        symbol: Trading symbol
        depth: Order book depth
        data_service: Data service dependency
        current_user: Current user (optional)

    Returns:
        Order book data

    Raises:
        HTTPException: If order book not found
    """
    try:
        logger.info("Get orderbook request", exchange=exchange, symbol=symbol, depth=depth)

        orderbook_data = await data_service.get_orderbook(symbol, exchange, depth)

        # Convert to response format
        bids = [OrderBookLevel(price=p, quantity=q) for p, q in orderbook_data["bids"]]
        asks = [OrderBookLevel(price=p, quantity=q) for p, q in orderbook_data["asks"]]

        return OrderBookResponse(
            symbol=orderbook_data["symbol"],
            exchange=orderbook_data["exchange"],
            bids=bids,
            asks=asks,
            timestamp=orderbook_data["timestamp"]
        )

    except DataNotFoundException as e:
        logger.warning("Orderbook not found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DataValidationException as e:
        logger.warning("Orderbook validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Get orderbook failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get orderbook"
        )


@router.get(
    "/ohlcv/{exchange}/{symbol}",
    response_model=OHLCVResponse,
    summary="Get OHLCV data",
    description="Get historical OHLCV candles for a trading pair"
)
async def get_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = Query(..., description="Timeframe (1m, 5m, 1h, 1d, etc.)"),
    start_time: Optional[datetime] = Query(None, description="Start time (UTC)"),
    end_time: Optional[datetime] = Query(None, description="End time (UTC)"),
    limit: Optional[int] = Query(default=100, ge=1, le=1000, description="Max candles"),
    data_service: DataService = Depends(get_data_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> OHLCVResponse:
    """Get OHLCV data.

    Args:
        exchange: Exchange name
        symbol: Trading symbol
        timeframe: Timeframe
        start_time: Start time
        end_time: End time
        limit: Maximum candles
        data_service: Data service dependency
        current_user: Current user (optional)

    Returns:
        OHLCV data

    Raises:
        HTTPException: If data not found
    """
    try:
        logger.info(
            "Get OHLCV request",
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )

        df = await data_service.get_ohlcv(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        # Convert DataFrame to response format
        candles = []
        for row in df.iter_rows(named=True):
            candles.append(OHLCVCandle(
                timestamp=row["timestamp"].isoformat(),
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
            candles=candles
        )

    except DataNotFoundException as e:
        logger.warning("OHLCV not found", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DataValidationException as e:
        logger.warning("OHLCV validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Get OHLCV failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get OHLCV"
        )


@router.get(
    "/trades/{exchange}/{symbol}",
    response_model=List[TradeResponse],
    summary="Get recent trades",
    description="Get recent trades for a trading pair"
)
async def get_trades(
    exchange: str,
    symbol: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Max trades"),
    data_service: DataService = Depends(get_data_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> List[TradeResponse]:
    """Get recent trades.

    Args:
        exchange: Exchange name
        symbol: Trading symbol
        limit: Maximum trades
        data_service: Data service dependency
        current_user: Current user (optional)

    Returns:
        List of recent trades

    Raises:
        HTTPException: If request fails
    """
    try:
        logger.info("Get trades request", exchange=exchange, symbol=symbol, limit=limit)

        df = await data_service.get_recent_trades(symbol, exchange, limit)

        # Convert DataFrame to response format
        trades = []
        for row in df.iter_rows(named=True):
            trades.append(TradeResponse(
                trade_id=row["trade_id"],
                timestamp=row["timestamp"].isoformat(),
                price=str(row["price"]),
                quantity=str(row["quantity"]),
                side=row["side"]
            ))

        return trades

    except DataValidationException as e:
        logger.warning("Trades validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Get trades failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trades"
        )


@router.get(
    "/symbols",
    response_model=List[SymbolInfo],
    summary="Get trading symbols",
    description="Get list of available trading symbols"
)
async def get_symbols(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    data_service: DataService = Depends(get_data_service),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> List[SymbolInfo]:
    """Get trading symbols.

    Args:
        exchange: Optional exchange filter
        data_service: Data service dependency
        current_user: Current user (optional)

    Returns:
        List of trading symbols

    Raises:
        HTTPException: If request fails
    """
    try:
        logger.info("Get symbols request", exchange=exchange)

        symbols = await data_service.get_symbols(exchange)

        return [SymbolInfo(**s) for s in symbols]

    except Exception as e:
        logger.error("Get symbols failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get symbols"
        )
