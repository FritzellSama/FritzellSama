"""
Dashboard Backend Server

FastAPI-based backend server for the Quantum Trader AI dashboard.
Provides REST API endpoints and WebSocket connections for real-time data.

This module implements:
- REST API endpoints for trading operations
- WebSocket connections for real-time market data
- Authentication and authorization
- Session management
- CORS and security middleware
"""

import asyncio
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import yaml

from quantum_trader.dashboard.backend.session_manager import SessionManager

logger = structlog.get_logger(__name__)


class ServerConfig:
    """Server configuration loaded from environment and config files."""

    def __init__(self):
        """Initialize configuration from environment variables and config files."""
        # Load from config file
        config_path = os.getenv('DASHBOARD_CONFIG_PATH', 'config/dashboard/server.yaml')
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults", path=config_path)
            config = {}

        # Server settings
        self.host: str = os.getenv('DASHBOARD_HOST', config.get('host', '0.0.0.0'))
        self.port: int = int(os.getenv('DASHBOARD_PORT', config.get('port', 8080)))
        self.reload: bool = os.getenv('DASHBOARD_RELOAD', config.get('reload', False)) == 'true'
        self.workers: int = int(os.getenv('DASHBOARD_WORKERS', config.get('workers', 4)))

        # CORS settings
        cors_origins = os.getenv('DASHBOARD_CORS_ORIGINS', config.get('cors', {}).get('origins', '*'))
        self.cors_origins: List[str] = cors_origins if isinstance(cors_origins, list) else [cors_origins]
        self.cors_credentials: bool = os.getenv('DASHBOARD_CORS_CREDENTIALS',
                                                config.get('cors', {}).get('credentials', True)) == 'true'
        self.cors_methods: List[str] = config.get('cors', {}).get('methods', ['*'])
        self.cors_headers: List[str] = config.get('cors', {}).get('headers', ['*'])

        # Security settings
        self.secret_key: str = os.getenv('DASHBOARD_SECRET_KEY', config.get('security', {}).get('secret_key', ''))
        if not self.secret_key:
            raise ValueError("DASHBOARD_SECRET_KEY must be set")

        self.access_token_expire: int = int(os.getenv('DASHBOARD_TOKEN_EXPIRE',
                                                      config.get('security', {}).get('access_token_expire_minutes', 60)))
        self.algorithm: str = os.getenv('DASHBOARD_ALGORITHM', config.get('security', {}).get('algorithm', 'HS256'))

        # API settings
        self.api_prefix: str = os.getenv('DASHBOARD_API_PREFIX', config.get('api', {}).get('prefix', '/api/v1'))
        self.max_request_size: int = int(os.getenv('DASHBOARD_MAX_REQUEST_SIZE',
                                                   config.get('api', {}).get('max_request_size', 10485760)))  # 10MB

        # WebSocket settings
        self.ws_heartbeat_interval: int = int(os.getenv('DASHBOARD_WS_HEARTBEAT',
                                                        config.get('websocket', {}).get('heartbeat_interval', 30)))
        self.ws_max_message_size: int = int(os.getenv('DASHBOARD_WS_MAX_SIZE',
                                                      config.get('websocket', {}).get('max_message_size', 1048576)))  # 1MB

        # Rate limiting
        self.rate_limit_requests: int = int(os.getenv('DASHBOARD_RATE_LIMIT_REQUESTS',
                                                      config.get('rate_limit', {}).get('requests', 100)))
        self.rate_limit_period: int = int(os.getenv('DASHBOARD_RATE_LIMIT_PERIOD',
                                                    config.get('rate_limit', {}).get('period', 60)))

        logger.info("Server configuration loaded", host=self.host, port=self.port)


# Pydantic models for request/response
class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MarketDataSubscription(BaseModel):
    """Market data subscription request."""
    symbol: str = Field(..., pattern=r'^[A-Z0-9]+/[A-Z0-9]+$')
    exchange: str = Field(..., min_length=1)
    channels: List[str] = Field(..., min_items=1)

    @validator('channels')
    def validate_channels(cls, v):
        """Validate channel names."""
        valid_channels = {'orderbook', 'ticker', 'trades', 'ohlcv'}
        invalid = set(v) - valid_channels
        if invalid:
            raise ValueError(f"Invalid channels: {invalid}")
        return v


class OrderRequest(BaseModel):
    """Order placement request."""
    symbol: str = Field(..., pattern=r'^[A-Z0-9]+/[A-Z0-9]+$')
    side: str = Field(..., pattern=r'^(BUY|SELL)$')
    quantity: str = Field(..., pattern=r'^\d+\.?\d*$')  # Decimal as string
    price: Optional[str] = Field(None, pattern=r'^\d+\.?\d*$')
    order_type: str = Field(..., pattern=r'^(MARKET|LIMIT|STOP_LOSS|TAKE_PROFIT|STOP_LIMIT)$')
    exchange: str = Field(..., min_length=1)
    strategy: str = Field(..., min_length=1)

    @validator('quantity', 'price')
    def validate_decimal(cls, v):
        """Validate decimal values."""
        if v is not None:
            try:
                Decimal(v)
            except Exception:
                raise ValueError("Invalid decimal value")
        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime: float
    connections: Dict[str, int]


# Application state
class AppState:
    """Application state container."""

    def __init__(self, config: ServerConfig):
        """Initialize application state."""
        self.config = config
        self.session_manager: Optional[SessionManager] = None
        self.start_time: datetime = datetime.utcnow()
        self.active_websockets: Dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Quantum Trader Dashboard Server")

    # Initialize components
    config = ServerConfig()
    state = AppState(config)
    state.session_manager = SessionManager(config.secret_key, config.access_token_expire)

    app.state.app_state = state

    logger.info("Server started successfully", host=config.host, port=config.port)

    yield

    # Cleanup
    logger.info("Shutting down server")

    # Close all WebSocket connections
    for ws_id, ws in list(state.active_websockets.items()):
        try:
            await ws.close()
        except Exception as e:
            logger.error("Error closing WebSocket", ws_id=ws_id, error=str(e))

    if state.session_manager:
        await state.session_manager.cleanup()

    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Quantum Trader AI Dashboard API",
    description="Real-time trading dashboard backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add middleware
@app.on_event("startup")
async def configure_middleware():
    """Configure middleware after app creation."""
    config = ServerConfig()

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_credentials,
        allow_methods=config.cors_methods,
        allow_headers=config.cors_headers,
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.warning("HTTP exception", status=exc.status_code, detail=exc.detail, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "timestamp": datetime.utcnow().isoformat()},
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Server health status
    """
    state: AppState = app.state.app_state
    uptime = (datetime.utcnow() - state.start_time).total_seconds()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        uptime=uptime,
        connections={
            "websockets": len(state.active_websockets),
            "sessions": len(state.session_manager.sessions) if state.session_manager else 0,
        },
    )


# Authentication endpoints
@app.post(f"{ServerConfig().api_prefix}/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate user and return access token.

    Args:
        request: Login credentials

    Returns:
        TokenResponse: Access token and metadata

    Raises:
        HTTPException: If authentication fails
    """
    state: AppState = app.state.app_state

    try:
        # Validate credentials (implement actual validation)
        # This is a placeholder - implement proper authentication
        if not request.username or not request.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        # Create session
        token = await state.session_manager.create_session(
            request.username,
            {"username": request.username, "login_time": datetime.utcnow().isoformat()},
        )

        logger.info("User logged in", username=request.username)

        return TokenResponse(
            access_token=token,
            expires_in=state.config.access_token_expire * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )


@app.post(f"{ServerConfig().api_prefix}/auth/logout")
async def logout(token: str = Depends(lambda: None)) -> Dict[str, str]:
    """
    Logout user and invalidate token.

    Args:
        token: Access token

    Returns:
        Success message
    """
    state: AppState = app.state.app_state

    if token:
        await state.session_manager.delete_session(token)

    logger.info("User logged out")

    return {"message": "Logged out successfully"}


@app.get(f"{ServerConfig().api_prefix}/auth/verify")
async def verify_token(token: str = Depends(lambda: None)) -> Dict[str, Any]:
    """
    Verify access token.

    Args:
        token: Access token

    Returns:
        Token validation result

    Raises:
        HTTPException: If token is invalid
    """
    state: AppState = app.state.app_state

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token required",
        )

    session_data = await state.session_manager.get_session(token)
    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return {"valid": True, "user": session_data.get("username")}


# WebSocket endpoint for real-time data
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data and updates.

    Args:
        websocket: WebSocket connection
    """
    state: AppState = app.state.app_state
    ws_id = f"ws_{datetime.utcnow().timestamp()}"

    await websocket.accept()
    state.active_websockets[ws_id] = websocket

    logger.info("WebSocket connected", ws_id=ws_id)

    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Quantum Trader Dashboard",
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Handle messages
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            msg_type = data.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

            elif msg_type == "subscribe":
                # Handle subscription (implement actual subscription logic)
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": data.get("channel"),
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif msg_type == "unsubscribe":
                # Handle unsubscription
                await websocket.send_json({
                    "type": "unsubscribed",
                    "channel": data.get("channel"),
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", ws_id=ws_id)
    except Exception as e:
        logger.error("WebSocket error", ws_id=ws_id, error=str(e), exc_info=True)
    finally:
        if ws_id in state.active_websockets:
            del state.active_websockets[ws_id]


# Run server
if __name__ == "__main__":
    import uvicorn

    config = ServerConfig()

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers if not config.reload else None,
        log_level=os.getenv('LOG_LEVEL', 'info').lower(),
    )
