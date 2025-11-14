"""
Quantum Trader Dashboard Backend API Client

This module provides a comprehensive API client for the quantum trader dashboard backend,
handling communication with the main FastAPI trading system, WebSocket connections,
and data aggregation for dashboard visualization.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import aiohttp
import jwt
from aiohttp import ClientSession, ClientTimeout, WSMsgType
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeFrame(str, Enum):
    """Trading timeframe enumeration"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"


# Pydantic Models for API Requests/Responses

class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    mfa_code: Optional[str] = Field(None, min_length=6, max_length=6)


class TokenResponse(BaseModel):
    """JWT token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours in seconds
    user_id: str
    username: str


class PortfolioSummary(BaseModel):
    """Portfolio summary response schema"""
    total_value: float
    cash_balance: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    total_return_pct: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown_pct: float
    win_rate_pct: float
    num_positions: int
    last_updated: datetime


class Position(BaseModel):
    """Trading position schema"""
    id: str
    symbol: str
    exchange: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float
    cost_basis: float
    opened_at: datetime
    strategy: Optional[str]
    stop_loss: Optional[float]
    take_profit: Optional[float]


class Trade(BaseModel):
    """Trade execution schema"""
    id: str
    order_id: str
    symbol: str
    exchange: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str
    realized_pnl: Optional[float]
    executed_at: datetime
    strategy: Optional[str]


class Order(BaseModel):
    """Order schema"""
    id: str
    symbol: str
    exchange: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    average_price: Optional[float]
    created_at: datetime
    updated_at: datetime
    strategy: Optional[str]


class PerformanceMetric(BaseModel):
    """Performance metrics schema"""
    timestamp: datetime
    total_value: float
    daily_return_pct: float
    cumulative_return_pct: float
    sharpe_ratio: Optional[float]
    volatility: float
    max_drawdown_pct: float


class StrategyStatus(BaseModel):
    """Strategy status schema"""
    id: str
    name: str
    type: str
    status: str  # active, paused, stopped
    positions: int
    pnl: float
    pnl_pct: float
    win_rate_pct: float
    trades_today: int
    avg_holding_period: Optional[float]  # in hours
    parameters: Dict[str, Any]


class RiskMetrics(BaseModel):
    """Risk metrics schema"""
    portfolio_var_95: float  # Value at Risk at 95% confidence
    portfolio_cvar_95: float  # Conditional VaR
    current_leverage: float
    max_leverage: float
    exposure_by_sector: Dict[str, float]
    concentration_risk: float  # Top 5 holdings as % of portfolio
    liquidity_score: float  # 0-100
    risk_score: float  # 0-100, higher is riskier


class MarketDataPoint(BaseModel):
    """Market data OHLCV schema"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BacktestRequest(BaseModel):
    """Backtest request schema"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    symbols: List[str]
    parameters: Optional[Dict[str, Any]]


class BacktestResult(BaseModel):
    """Backtest result schema"""
    id: str
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    completed_at: datetime


class APIClient:
    """
    Asynchronous API client for Quantum Trader dashboard.

    Handles authentication, REST API calls, WebSocket connections,
    and provides high-level methods for dashboard data retrieval.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_url: str = "ws://localhost:8001",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize API client.

        Args:
            base_url: Base URL for REST API
            ws_url: Base URL for WebSocket connections
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.ws_url = ws_url.rstrip('/')
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._session: Optional[ClientSession] = None
        self._ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._user_id: Optional[str] = None
        self._username: Optional[str] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self):
        """Initialize HTTP session"""
        if self._session is None:
            self._session = ClientSession(timeout=self.timeout)

    async def close(self):
        """Close all connections"""
        # Close WebSocket connections
        for ws_name, ws in list(self._ws_connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {ws_name}: {e}")
        self._ws_connections.clear()

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def is_authenticated(self) -> bool:
        """Check if client is authenticated with valid token"""
        if not self._access_token or not self._token_expires_at:
            return False
        return datetime.utcnow() < self._token_expires_at

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for aiohttp request

        Returns:
            Response data as dictionary

        Raises:
            aiohttp.ClientError: On request failure after retries
        """
        if self._session is None:
            await self.connect()

        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers.update(self._get_headers())

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs
                ) as response:
                    response.raise_for_status()

                    # Handle empty responses
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await response.json()
                    else:
                        return {"data": await response.text()}

            except aiohttp.ClientError as e:
                last_exception = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                continue

        raise last_exception

    # Authentication Methods

    async def login(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None
    ) -> TokenResponse:
        """
        Authenticate user and obtain JWT token.

        Args:
            username: Username
            password: Password
            mfa_code: Optional MFA code

        Returns:
            Token response with access token
        """
        request_data = LoginRequest(
            username=username,
            password=password,
            mfa_code=mfa_code
        )

        response = await self._request(
            "POST",
            "/api/v1/auth/login",
            json=request_data.dict(exclude_none=True)
        )

        token_response = TokenResponse(**response)

        # Store token and metadata
        self._access_token = token_response.access_token
        self._user_id = token_response.user_id
        self._username = token_response.username
        self._token_expires_at = datetime.utcnow() + timedelta(
            seconds=token_response.expires_in
        )

        logger.info(f"Successfully authenticated as {username}")
        return token_response

    async def logout(self):
        """Logout and clear authentication state"""
        if self._access_token:
            try:
                await self._request("POST", "/api/v1/auth/logout")
            except Exception as e:
                logger.error(f"Error during logout: {e}")

        self._access_token = None
        self._token_expires_at = None
        self._user_id = None
        self._username = None
        logger.info("Logged out successfully")

    async def refresh_token(self) -> TokenResponse:
        """
        Refresh JWT token before expiration.

        Returns:
            New token response
        """
        response = await self._request("POST", "/api/v1/auth/refresh")
        token_response = TokenResponse(**response)

        self._access_token = token_response.access_token
        self._token_expires_at = datetime.utcnow() + timedelta(
            seconds=token_response.expires_in
        )

        logger.info("Token refreshed successfully")
        return token_response

    # Portfolio Methods

    async def get_portfolio_summary(self) -> PortfolioSummary:
        """Get portfolio summary with key metrics"""
        response = await self._request("GET", "/api/v1/portfolio")
        return PortfolioSummary(**response)

    async def get_positions(self) -> List[Position]:
        """Get all open positions"""
        response = await self._request("GET", "/api/v1/portfolio/positions")
        return [Position(**pos) for pos in response.get("positions", [])]

    async def get_portfolio_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1h"
    ) -> List[PerformanceMetric]:
        """
        Get portfolio performance history.

        Args:
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to now)
            interval: Data interval (1h, 4h, 1d)

        Returns:
            List of performance metrics over time
        """
        params = {"interval": interval}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        response = await self._request(
            "GET",
            "/api/v1/portfolio/history",
            params=params
        )
        return [PerformanceMetric(**metric) for metric in response.get("history", [])]

    # Trading Methods

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trade history.

        Args:
            symbol: Filter by symbol
            start_date: Start date filter
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if start_date:
            params["start_date"] = start_date.isoformat()

        response = await self._request("GET", "/api/v1/trades", params=params)
        return [Trade(**trade) for trade in response.get("trades", [])]

    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get orders.

        Args:
            status: Filter by order status
            symbol: Filter by symbol

        Returns:
            List of orders
        """
        params = {}
        if status:
            params["status"] = status.value
        if symbol:
            params["symbol"] = symbol

        response = await self._request("GET", "/api/v1/orders", params=params)
        return [Order(**order) for order in response.get("orders", [])]

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        exchange: str = "binance"
    ) -> Order:
        """
        Place a new order.

        Args:
            symbol: Trading symbol (e.g., BTC/USDT)
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/etc.)
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            exchange: Exchange name

        Returns:
            Created order
        """
        order_data = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": quantity,
            "exchange": exchange
        }
        if price:
            order_data["price"] = price
        if stop_price:
            order_data["stop_price"] = stop_price

        response = await self._request("POST", "/api/v1/orders", json=order_data)
        return Order(**response)

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        return await self._request("DELETE", f"/api/v1/orders/{order_id}")

    # Strategy Methods

    async def get_strategies(self) -> List[StrategyStatus]:
        """Get all active strategies with their status"""
        response = await self._request("GET", "/api/v1/strategies")
        return [StrategyStatus(**strategy) for strategy in response.get("strategies", [])]

    async def get_strategy(self, strategy_id: str) -> StrategyStatus:
        """Get detailed strategy information"""
        response = await self._request("GET", f"/api/v1/strategies/{strategy_id}")
        return StrategyStatus(**response)

    async def update_strategy_parameters(
        self,
        strategy_id: str,
        parameters: Dict[str, Any]
    ) -> StrategyStatus:
        """Update strategy parameters"""
        response = await self._request(
            "PATCH",
            f"/api/v1/strategies/{strategy_id}",
            json={"parameters": parameters}
        )
        return StrategyStatus(**response)

    async def pause_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Pause a running strategy"""
        return await self._request(
            "POST",
            f"/api/v1/strategies/{strategy_id}/pause"
        )

    async def resume_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Resume a paused strategy"""
        return await self._request(
            "POST",
            f"/api/v1/strategies/{strategy_id}/resume"
        )

    # Risk Methods

    async def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        response = await self._request("GET", "/api/v1/risk/exposure")
        return RiskMetrics(**response)

    async def get_risk_limits(self) -> Dict[str, Any]:
        """Get configured risk limits"""
        return await self._request("GET", "/api/v1/risk/limits")

    async def update_risk_limits(self, limits: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk limits"""
        return await self._request(
            "POST",
            "/api/v1/risk/limits",
            json=limits
        )

    # Market Data Methods

    async def get_market_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500
    ) -> List[MarketDataPoint]:
        """
        Get historical market data (OHLCV).

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum number of candles

        Returns:
            List of OHLCV data points
        """
        params = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "limit": limit
        }
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        response = await self._request("GET", "/api/v1/market/data", params=params)
        return [MarketDataPoint(**point) for point in response.get("data", [])]

    # Backtesting Methods

    async def run_backtest(self, request: BacktestRequest) -> str:
        """
        Start a backtest.

        Args:
            request: Backtest configuration

        Returns:
            Backtest ID
        """
        response = await self._request(
            "POST",
            "/api/v1/backtests",
            json=request.dict()
        )
        return response.get("backtest_id")

    async def get_backtest_result(self, backtest_id: str) -> BacktestResult:
        """Get backtest results"""
        response = await self._request("GET", f"/api/v1/backtests/{backtest_id}")
        return BacktestResult(**response)

    async def list_backtests(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent backtests"""
        response = await self._request(
            "GET",
            "/api/v1/backtests",
            params={"limit": limit}
        )
        return response.get("backtests", [])

    # Analytics Methods

    async def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        return await self._request(
            "GET",
            "/api/v1/analytics/performance",
            params=params
        )

    async def get_drawdown_analysis(self) -> Dict[str, Any]:
        """Get drawdown analysis"""
        return await self._request("GET", "/api/v1/analytics/drawdown")

    # Health & Status Methods

    async def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        return await self._request("GET", "/health")

    async def get_exchange_status(self) -> Dict[str, Any]:
        """Get exchange connectivity status"""
        return await self._request("GET", "/health/exchanges")

    # WebSocket Methods

    async def subscribe_trades(self, callback):
        """
        Subscribe to real-time trade updates.

        Args:
            callback: Async function to call with trade data
        """
        await self._subscribe_websocket("trades", callback)

    async def subscribe_positions(self, callback):
        """Subscribe to real-time position updates"""
        await self._subscribe_websocket("positions", callback)

    async def subscribe_orders(self, callback):
        """Subscribe to real-time order updates"""
        await self._subscribe_websocket("orders", callback)

    async def subscribe_metrics(self, callback):
        """Subscribe to real-time metrics updates"""
        await self._subscribe_websocket("metrics", callback)

    async def _subscribe_websocket(self, channel: str, callback):
        """
        Internal WebSocket subscription handler.

        Args:
            channel: WebSocket channel name
            callback: Async callback function
        """
        ws_url = f"{self.ws_url}/ws/{channel}"

        # Add authentication token to URL
        if self._access_token:
            ws_url += f"?token={self._access_token}"

        try:
            ws = await self._session.ws_connect(ws_url)
            self._ws_connections[channel] = ws

            logger.info(f"Connected to WebSocket channel: {channel}")

            # Start message handler
            asyncio.create_task(self._handle_websocket_messages(channel, ws, callback))

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket {channel}: {e}")
            raise

    async def _handle_websocket_messages(
        self,
        channel: str,
        ws: aiohttp.ClientWebSocketResponse,
        callback
    ):
        """
        Handle incoming WebSocket messages.

        Args:
            channel: Channel name
            ws: WebSocket connection
            callback: Callback function
        """
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = msg.json()
                        await callback(data)
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error on {channel}: {ws.exception()}")
                    break

                elif msg.type == WSMsgType.CLOSED:
                    logger.info(f"WebSocket {channel} closed")
                    break

        except Exception as e:
            logger.error(f"WebSocket handler error for {channel}: {e}")

        finally:
            if channel in self._ws_connections:
                del self._ws_connections[channel]

    async def unsubscribe(self, channel: str):
        """Unsubscribe from a WebSocket channel"""
        if channel in self._ws_connections:
            ws = self._ws_connections[channel]
            await ws.close()
            del self._ws_connections[channel]
            logger.info(f"Unsubscribed from {channel}")


# Convenience functions for quick usage

async def create_client(
    base_url: str = "http://localhost:8000",
    ws_url: str = "ws://localhost:8001"
) -> APIClient:
    """
    Create and connect API client.

    Args:
        base_url: REST API base URL
        ws_url: WebSocket base URL

    Returns:
        Connected API client instance
    """
    client = APIClient(base_url=base_url, ws_url=ws_url)
    await client.connect()
    return client


async def create_authenticated_client(
    username: str,
    password: str,
    base_url: str = "http://localhost:8000",
    ws_url: str = "ws://localhost:8001",
    mfa_code: Optional[str] = None
) -> APIClient:
    """
    Create and authenticate API client.

    Args:
        username: Username
        password: Password
        base_url: REST API base URL
        ws_url: WebSocket base URL
        mfa_code: Optional MFA code

    Returns:
        Authenticated API client instance
    """
    client = await create_client(base_url, ws_url)
    await client.login(username, password, mfa_code)
    return client
