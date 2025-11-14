"""KuCoin exchange constants and configuration mappings.

This module provides constants, error codes, and mappings for KuCoin exchange integration.
All configurable values are loaded from environment variables or config files.
"""

from decimal import Decimal
from typing import Dict, Final
from enum import Enum
import os


class KuCoinErrorCode(Enum):
    """KuCoin API error codes."""

    # Authentication errors
    INVALID_API_KEY = "400100"
    INVALID_SIGNATURE = "400200"
    INVALID_PASSPHRASE = "400300"
    INVALID_IP = "400400"
    PERMISSION_DENIED = "400500"

    # Order errors
    INSUFFICIENT_BALANCE = "200004"
    INVALID_ORDER_SIZE = "400100"
    INVALID_PRICE = "400200"
    ORDER_NOT_FOUND = "400300"
    INVALID_ORDER_STATUS = "400400"
    BALANCE_INSUFFICIENT = "411100"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "429000"
    TOO_MANY_REQUESTS = "429001"

    # Market errors
    SYMBOL_NOT_FOUND = "400600"
    MARKET_NOT_AVAILABLE = "400700"
    MARKET_SUSPENDED = "400800"

    # System errors
    SYSTEM_ERROR = "500000"
    SERVICE_UNAVAILABLE = "503000"
    SYSTEM_BUSY = "500100"


class KuCoinOrderStatus(Enum):
    """KuCoin order status mappings."""

    ACTIVE = "active"
    DONE = "done"
    CANCELLED = "cancelled"


class KuCoinOrderType(Enum):
    """KuCoin order type mappings."""

    LIMIT = "limit"
    MARKET = "market"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class KuCoinTimeInForce(Enum):
    """KuCoin time in force options."""

    GTC = "GTC"  # Good Till Cancel
    GTT = "GTT"  # Good Till Time
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


# API Configuration - loaded from environment
KUCOIN_API_BASE_URL: str = os.getenv("KUCOIN_API_URL", "https://api.kucoin.com")
KUCOIN_SANDBOX_URL: str = os.getenv("KUCOIN_SANDBOX_URL", "https://openapi-sandbox.kucoin.com")
KUCOIN_WS_BASE_URL: str = os.getenv("KUCOIN_WS_URL", "wss://ws-api.kucoin.com")
KUCOIN_SANDBOX_WS_URL: str = os.getenv("KUCOIN_SANDBOX_WS_URL", "wss://ws-api-sandbox.kucoin.com")

# API Endpoints
API_ENDPOINTS: Final[Dict[str, str]] = {
    "timestamp": "/api/v1/timestamp",
    "symbols": "/api/v1/symbols",
    "currencies": "/api/v1/currencies",
    "order_book": "/api/v1/market/orderbook/level2_100",
    "ticker": "/api/v1/market/orderbook/level1",
    "tickers": "/api/v1/market/allTickers",
    "24h_stats": "/api/v1/market/stats",
    "klines": "/api/v1/market/candles",
    "accounts": "/api/v1/accounts",
    "account_detail": "/api/v1/accounts/{accountId}",
    "orders": "/api/v1/orders",
    "order_detail": "/api/v1/orders/{orderId}",
    "cancel_order": "/api/v1/orders/{orderId}",
    "cancel_all": "/api/v1/orders",
    "fills": "/api/v1/fills",
    "limit_fills": "/api/v1/limit/fills",
    "ws_bullet_public": "/api/v1/bullet-public",
    "ws_bullet_private": "/api/v1/bullet-private",
}

# Rate Limits - loaded from config
RATE_LIMIT_REQUESTS_PER_SECOND: int = int(os.getenv("KUCOIN_RATE_LIMIT_RPS", "100"))
RATE_LIMIT_ORDERS_PER_SECOND: int = int(os.getenv("KUCOIN_RATE_LIMIT_OPS", "45"))
RATE_LIMIT_ORDERS_PER_DAY: int = int(os.getenv("KUCOIN_RATE_LIMIT_OPD", "200000"))

# Timeframe mappings
TIMEFRAME_MAPPING: Final[Dict[str, str]] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "2h": "2hour",
    "4h": "4hour",
    "6h": "6hour",
    "8h": "8hour",
    "12h": "12hour",
    "1d": "1day",
    "1w": "1week",
}

# Order type mappings
ORDER_TYPE_MAPPING: Final[Dict[str, str]] = {
    "MARKET": "market",
    "LIMIT": "limit",
    "STOP_LOSS": "stop_market",
    "STOP_LIMIT": "stop_limit",
}

# Order side mappings
ORDER_SIDE_MAPPING: Final[Dict[str, str]] = {
    "BUY": "buy",
    "SELL": "sell",
}

# Minimum trade amounts - loaded from config
MIN_TRADE_AMOUNT_BTC: Decimal = Decimal(os.getenv("KUCOIN_MIN_BTC", "0.00001"))
MIN_TRADE_AMOUNT_USDT: Decimal = Decimal(os.getenv("KUCOIN_MIN_USDT", "0.1"))

# Precision settings - loaded from config
PRICE_PRECISION: int = int(os.getenv("KUCOIN_PRICE_PRECISION", "8"))
QUANTITY_PRECISION: int = int(os.getenv("KUCOIN_QTY_PRECISION", "8"))

# Retry configuration
MAX_RETRIES: int = int(os.getenv("KUCOIN_MAX_RETRIES", "3"))
RETRY_DELAY_MS: int = int(os.getenv("KUCOIN_RETRY_DELAY_MS", "150"))
BACKOFF_MULTIPLIER: Decimal = Decimal(os.getenv("KUCOIN_BACKOFF_MULTIPLIER", "2"))

# Connection settings
CONNECTION_TIMEOUT_SECONDS: int = int(os.getenv("KUCOIN_CONN_TIMEOUT", "10"))
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("KUCOIN_REQ_TIMEOUT", "30"))
WEBSOCKET_PING_INTERVAL: int = int(os.getenv("KUCOIN_WS_PING_INTERVAL", "20"))
WEBSOCKET_PING_TIMEOUT: int = int(os.getenv("KUCOIN_WS_PING_TIMEOUT", "10"))

# Timestamp tolerance
TIMESTAMP_TOLERANCE_MS: int = int(os.getenv("KUCOIN_TIMESTAMP_TOLERANCE", "5000"))

# Error codes that should trigger retries
RETRYABLE_ERROR_CODES: Final[set] = {
    KuCoinErrorCode.RATE_LIMIT_EXCEEDED.value,
    KuCoinErrorCode.TOO_MANY_REQUESTS.value,
    KuCoinErrorCode.SERVICE_UNAVAILABLE.value,
    KuCoinErrorCode.SYSTEM_ERROR.value,
    KuCoinErrorCode.SYSTEM_BUSY.value,
}

# Error codes that should never be retried
FATAL_ERROR_CODES: Final[set] = {
    KuCoinErrorCode.INVALID_API_KEY.value,
    KuCoinErrorCode.INVALID_SIGNATURE.value,
    KuCoinErrorCode.INVALID_PASSPHRASE.value,
    KuCoinErrorCode.PERMISSION_DENIED.value,
}

# WebSocket stream types
WS_STREAM_TYPES: Final[Dict[str, str]] = {
    "ticker": "/market/ticker",
    "all_tickers": "/market/ticker:all",
    "orderbook": "/market/level2",
    "match": "/market/match",
    "full_orderbook": "/spotMarket/level2Depth5",
    "candles": "/market/candles",
    "account": "/account/balance",
    "order": "/spotMarket/tradeOrders",
}

# Order book depth levels
ORDERBOOK_DEPTH_LEVELS: Final[list] = [20, 100]

# Status code to internal status mapping
STATUS_MAPPING: Final[Dict[str, str]] = {
    "active": "PENDING",
    "done": "FILLED",
    "cancelled": "CANCELLED",
}

# Account types
ACCOUNT_TYPES: Final[Dict[str, str]] = {
    "main": "main",
    "trade": "trade",
    "margin": "margin",
}

# KuCoin specific headers
KC_API_KEY_HEADER: str = "KC-API-KEY"
KC_API_SIGN_HEADER: str = "KC-API-SIGN"
KC_API_TIMESTAMP_HEADER: str = "KC-API-TIMESTAMP"
KC_API_PASSPHRASE_HEADER: str = "KC-API-PASSPHRASE"
KC_API_KEY_VERSION_HEADER: str = "KC-API-KEY-VERSION"

# API version
API_KEY_VERSION: str = os.getenv("KUCOIN_API_VERSION", "2")

# Trading fee tiers - loaded from exchange API
DEFAULT_MAKER_FEE: Decimal = Decimal(os.getenv("KUCOIN_MAKER_FEE", "0.001"))
DEFAULT_TAKER_FEE: Decimal = Decimal(os.getenv("KUCOIN_TAKER_FEE", "0.001"))
