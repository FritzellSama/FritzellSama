"""Bitget exchange constants and configuration mappings.

This module provides constants, error codes, and mappings for Bitget exchange integration.
All configurable values are loaded from environment variables or config files.
"""

from decimal import Decimal
from typing import Dict, Final
from enum import Enum
import os


class BitgetErrorCode(Enum):
    """Bitget API error codes."""

    # Authentication errors
    INVALID_API_KEY = "40001"
    INVALID_SIGNATURE = "40002"
    INVALID_TIMESTAMP = "40003"
    API_KEY_EXPIRED = "40004"
    PERMISSION_DENIED = "40005"

    # Order errors
    INSUFFICIENT_BALANCE = "43004"
    INVALID_ORDER_SIZE = "43005"
    INVALID_PRICE = "43006"
    ORDER_NOT_FOUND = "43007"
    INVALID_ORDER_TYPE = "43008"
    ORDER_PRICE_TOO_HIGH = "43009"
    ORDER_PRICE_TOO_LOW = "43010"

    # Rate limiting
    RATE_LIMIT_EXCEEDED = "40006"
    TOO_MANY_REQUESTS = "40007"

    # Market errors
    SYMBOL_NOT_FOUND = "43011"
    MARKET_CLOSED = "43012"
    TRADING_SUSPENDED = "43013"

    # System errors
    SYSTEM_ERROR = "50000"
    SERVICE_UNAVAILABLE = "50001"
    SYSTEM_BUSY = "50002"
    INTERNAL_ERROR = "50003"


class BitgetOrderStatus(Enum):
    """Bitget order status mappings."""

    INIT = "init"
    NEW = "new"
    PARTIAL_FILL = "partial_fill"
    FULL_FILL = "full_fill"
    CANCELLED = "cancelled"
    FAILED = "failed"


class BitgetOrderType(Enum):
    """Bitget order type mappings."""

    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"


class BitgetOrderSide(Enum):
    """Bitget order side mappings."""

    BUY = "buy"
    SELL = "sell"


# API Configuration - loaded from environment
BITGET_API_BASE_URL: str = os.getenv("BITGET_API_URL", "https://api.bitget.com")
BITGET_TESTNET_URL: str = os.getenv("BITGET_TESTNET_URL", "https://api-sandbox.bitget.com")
BITGET_WS_BASE_URL: str = os.getenv("BITGET_WS_URL", "wss://ws.bitget.com")
BITGET_TESTNET_WS_URL: str = os.getenv("BITGET_TESTNET_WS_URL", "wss://ws-sandbox.bitget.com")

# API Endpoints
API_ENDPOINTS: Final[Dict[str, str]] = {
    "server_time": "/api/spot/v1/public/time",
    "symbols": "/api/spot/v1/public/products",
    "ticker": "/api/spot/v1/market/ticker",
    "tickers": "/api/spot/v1/market/tickers",
    "orderbook": "/api/spot/v1/market/depth",
    "trades": "/api/spot/v1/market/fills",
    "candles": "/api/spot/v1/market/candles",
    "account": "/api/spot/v1/account/assets",
    "bills": "/api/spot/v1/account/bills",
    "place_order": "/api/spot/v1/trade/orders",
    "batch_orders": "/api/spot/v1/trade/batch-orders",
    "cancel_order": "/api/spot/v1/trade/cancel-order",
    "cancel_batch": "/api/spot/v1/trade/cancel-batch-orders",
    "order_info": "/api/spot/v1/trade/orderInfo",
    "open_orders": "/api/spot/v1/trade/open-orders",
    "history_orders": "/api/spot/v1/trade/history",
    "fills": "/api/spot/v1/trade/fills",
}

# Rate Limits - loaded from config
RATE_LIMIT_REQUESTS_PER_SECOND: int = int(os.getenv("BITGET_RATE_LIMIT_RPS", "20"))
RATE_LIMIT_ORDERS_PER_SECOND: int = int(os.getenv("BITGET_RATE_LIMIT_OPS", "10"))
RATE_LIMIT_ORDERS_PER_DAY: int = int(os.getenv("BITGET_RATE_LIMIT_OPD", "100000"))

# Timeframe mappings
TIMEFRAME_MAPPING: Final[Dict[str, str]] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1w",
    "1M": "1M",
}

# Order type mappings
ORDER_TYPE_MAPPING: Final[Dict[str, str]] = {
    "MARKET": "market",
    "LIMIT": "limit",
    "POST_ONLY": "post_only",
    "FOK": "fok",
    "IOC": "ioc",
}

# Order side mappings
ORDER_SIDE_MAPPING: Final[Dict[str, str]] = {
    "BUY": "buy",
    "SELL": "sell",
}

# Minimum trade amounts - loaded from config
MIN_TRADE_AMOUNT_BTC: Decimal = Decimal(os.getenv("BITGET_MIN_BTC", "0.0001"))
MIN_TRADE_AMOUNT_USDT: Decimal = Decimal(os.getenv("BITGET_MIN_USDT", "5"))

# Precision settings - loaded from config
PRICE_PRECISION: int = int(os.getenv("BITGET_PRICE_PRECISION", "8"))
QUANTITY_PRECISION: int = int(os.getenv("BITGET_QTY_PRECISION", "8"))

# Retry configuration
MAX_RETRIES: int = int(os.getenv("BITGET_MAX_RETRIES", "3"))
RETRY_DELAY_MS: int = int(os.getenv("BITGET_RETRY_DELAY_MS", "200"))
BACKOFF_MULTIPLIER: Decimal = Decimal(os.getenv("BITGET_BACKOFF_MULTIPLIER", "2"))

# Connection settings
CONNECTION_TIMEOUT_SECONDS: int = int(os.getenv("BITGET_CONN_TIMEOUT", "10"))
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("BITGET_REQ_TIMEOUT", "30"))
WEBSOCKET_PING_INTERVAL: int = int(os.getenv("BITGET_WS_PING_INTERVAL", "30"))
WEBSOCKET_PING_TIMEOUT: int = int(os.getenv("BITGET_WS_PING_TIMEOUT", "10"))

# Timestamp tolerance
TIMESTAMP_TOLERANCE_MS: int = int(os.getenv("BITGET_TIMESTAMP_TOLERANCE", "5000"))

# Error codes that should trigger retries
RETRYABLE_ERROR_CODES: Final[set] = {
    BitgetErrorCode.RATE_LIMIT_EXCEEDED.value,
    BitgetErrorCode.TOO_MANY_REQUESTS.value,
    BitgetErrorCode.SERVICE_UNAVAILABLE.value,
    BitgetErrorCode.SYSTEM_ERROR.value,
    BitgetErrorCode.SYSTEM_BUSY.value,
    BitgetErrorCode.INTERNAL_ERROR.value,
}

# Error codes that should never be retried
FATAL_ERROR_CODES: Final[set] = {
    BitgetErrorCode.INVALID_API_KEY.value,
    BitgetErrorCode.INVALID_SIGNATURE.value,
    BitgetErrorCode.API_KEY_EXPIRED.value,
    BitgetErrorCode.PERMISSION_DENIED.value,
}

# WebSocket stream types
WS_STREAM_TYPES: Final[Dict[str, str]] = {
    "trade": "trade",
    "ticker": "ticker",
    "depth": "books",
    "depth5": "books5",
    "depth15": "books15",
    "candle": "candle",
    "account": "account",
    "orders": "orders",
}

# Order book depth levels
ORDERBOOK_DEPTH_LEVELS: Final[list] = [5, 15, 50, 100]

# Status code to internal status mapping
STATUS_MAPPING: Final[Dict[str, str]] = {
    "init": "PENDING",
    "new": "PENDING",
    "partial_fill": "PARTIAL",
    "full_fill": "FILLED",
    "cancelled": "CANCELLED",
    "failed": "REJECTED",
}

# Bitget specific headers
BG_ACCESS_KEY_HEADER: str = "ACCESS-KEY"
BG_ACCESS_SIGN_HEADER: str = "ACCESS-SIGN"
BG_ACCESS_TIMESTAMP_HEADER: str = "ACCESS-TIMESTAMP"
BG_ACCESS_PASSPHRASE_HEADER: str = "ACCESS-PASSPHRASE"

# Trading fee tiers - loaded from exchange API
DEFAULT_MAKER_FEE: Decimal = Decimal(os.getenv("BITGET_MAKER_FEE", "0.001"))
DEFAULT_TAKER_FEE: Decimal = Decimal(os.getenv("BITGET_TAKER_FEE", "0.001"))

# Product types
PRODUCT_TYPES: Final[Dict[str, str]] = {
    "spot": "SPOT",
    "margin": "MARGIN",
    "futures": "USDT-FUTURES",
    "coin_futures": "COIN-FUTURES",
}

# Force type for orders
FORCE_TYPE: Final[Dict[str, str]] = {
    "GTC": "normal",
    "IOC": "ioc",
    "FOK": "fok",
    "POST_ONLY": "post_only",
}

# Order source
ORDER_SOURCE: str = os.getenv("BITGET_ORDER_SOURCE", "api")

# Max batch order size
MAX_BATCH_ORDERS: int = int(os.getenv("BITGET_MAX_BATCH_ORDERS", "50"))
