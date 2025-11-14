"""Binance exchange constants and configuration mappings.

This module provides constants, error codes, and mappings for Binance exchange integration.
All configurable values are loaded from environment variables or config files.
"""

from decimal import Decimal
from typing import Dict, Final
from enum import Enum
import os


class BinanceErrorCode(Enum):
    """Binance API error codes."""

    # Authentication errors
    INVALID_API_KEY = -2014
    INVALID_SIGNATURE = -1022
    INVALID_TIMESTAMP = -1021

    # Order errors
    INSUFFICIENT_BALANCE = -2010
    INVALID_ORDER_TYPE = -1116
    INVALID_SIDE = -1104
    INVALID_QUANTITY = -1013
    MIN_NOTIONAL = -1013
    PRICE_QTY_EXCEED_HARD_LIMITS = -1111

    # Rate limiting
    RATE_LIMIT_EXCEEDED = -1003
    WAF_LIMIT_VIOLATED = -1015
    CANCEL_REPLACE_REJECTED = -2021

    # Market errors
    SYMBOL_NOT_FOUND = -1121
    MARKET_NOT_TRADING = -1128

    # System errors
    SYSTEM_ERROR = -1000
    SERVICE_UNAVAILABLE = -1001
    TIMEOUT = -1007
    UNKNOWN_ORDER = -2013
    DUPLICATE_ORDER = -2010


class BinanceOrderStatus(Enum):
    """Binance order status mappings."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class BinanceTimeInForce(Enum):
    """Binance time in force options."""

    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


# API Configuration - loaded from environment
BINANCE_API_BASE_URL: str = os.getenv("BINANCE_API_URL", "https://api.binance.com")
BINANCE_TESTNET_URL: str = os.getenv("BINANCE_TESTNET_URL", "https://testnet.binance.vision")
BINANCE_WS_BASE_URL: str = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443")
BINANCE_TESTNET_WS_URL: str = os.getenv("BINANCE_TESTNET_WS_URL", "wss://testnet.binance.vision")

# API Endpoints
API_ENDPOINTS: Final[Dict[str, str]] = {
    "ping": "/api/v3/ping",
    "time": "/api/v3/time",
    "exchange_info": "/api/v3/exchangeInfo",
    "order_book": "/api/v3/depth",
    "trades": "/api/v3/trades",
    "historical_trades": "/api/v3/historicalTrades",
    "agg_trades": "/api/v3/aggTrades",
    "klines": "/api/v3/klines",
    "ticker_24h": "/api/v3/ticker/24hr",
    "ticker_price": "/api/v3/ticker/price",
    "ticker_book": "/api/v3/ticker/bookTicker",
    "account": "/api/v3/account",
    "order": "/api/v3/order",
    "open_orders": "/api/v3/openOrders",
    "all_orders": "/api/v3/allOrders",
    "my_trades": "/api/v3/myTrades",
    "user_data_stream": "/api/v3/userDataStream",
}

# Rate Limits - loaded from config
RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("BINANCE_RATE_LIMIT_RPM", "1200"))
RATE_LIMIT_ORDERS_PER_SECOND: int = int(os.getenv("BINANCE_RATE_LIMIT_OPS", "10"))
RATE_LIMIT_ORDERS_PER_DAY: int = int(os.getenv("BINANCE_RATE_LIMIT_OPD", "200000"))

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
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M",
}

# Order type mappings
ORDER_TYPE_MAPPING: Final[Dict[str, str]] = {
    "MARKET": "MARKET",
    "LIMIT": "LIMIT",
    "STOP_LOSS": "STOP_LOSS",
    "STOP_LOSS_LIMIT": "STOP_LOSS_LIMIT",
    "TAKE_PROFIT": "TAKE_PROFIT",
    "TAKE_PROFIT_LIMIT": "TAKE_PROFIT_LIMIT",
    "LIMIT_MAKER": "LIMIT_MAKER",
}

# Order side mappings
ORDER_SIDE_MAPPING: Final[Dict[str, str]] = {
    "BUY": "BUY",
    "SELL": "SELL",
}

# Minimum trade amounts - typically loaded from exchange info API
MIN_TRADE_AMOUNT_BTC: Decimal = Decimal(os.getenv("BINANCE_MIN_BTC", "0.00001"))
MIN_TRADE_AMOUNT_USDT: Decimal = Decimal(os.getenv("BINANCE_MIN_USDT", "10"))

# Precision settings - loaded from config
PRICE_PRECISION: int = int(os.getenv("BINANCE_PRICE_PRECISION", "8"))
QUANTITY_PRECISION: int = int(os.getenv("BINANCE_QTY_PRECISION", "8"))

# Retry configuration
MAX_RETRIES: int = int(os.getenv("BINANCE_MAX_RETRIES", "3"))
RETRY_DELAY_MS: int = int(os.getenv("BINANCE_RETRY_DELAY_MS", "100"))
BACKOFF_MULTIPLIER: Decimal = Decimal(os.getenv("BINANCE_BACKOFF_MULTIPLIER", "2"))

# Connection settings
CONNECTION_TIMEOUT_SECONDS: int = int(os.getenv("BINANCE_CONN_TIMEOUT", "10"))
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("BINANCE_REQ_TIMEOUT", "30"))
WEBSOCKET_PING_INTERVAL: int = int(os.getenv("BINANCE_WS_PING_INTERVAL", "20"))
WEBSOCKET_PING_TIMEOUT: int = int(os.getenv("BINANCE_WS_PING_TIMEOUT", "10"))

# Timestamp tolerance
TIMESTAMP_TOLERANCE_MS: int = int(os.getenv("BINANCE_TIMESTAMP_TOLERANCE", "5000"))

# Error codes that should trigger retries
RETRYABLE_ERROR_CODES: Final[set] = {
    BinanceErrorCode.RATE_LIMIT_EXCEEDED.value,
    BinanceErrorCode.WAF_LIMIT_VIOLATED.value,
    BinanceErrorCode.SERVICE_UNAVAILABLE.value,
    BinanceErrorCode.TIMEOUT.value,
    BinanceErrorCode.SYSTEM_ERROR.value,
}

# Error codes that should never be retried
FATAL_ERROR_CODES: Final[set] = {
    BinanceErrorCode.INVALID_API_KEY.value,
    BinanceErrorCode.INVALID_SIGNATURE.value,
    BinanceErrorCode.INVALID_ORDER_TYPE.value,
    BinanceErrorCode.INVALID_SIDE.value,
}

# WebSocket stream types
WS_STREAM_TYPES: Final[Dict[str, str]] = {
    "agg_trade": "@aggTrade",
    "trade": "@trade",
    "kline": "@kline",
    "mini_ticker": "@miniTicker",
    "ticker": "@ticker",
    "book_ticker": "@bookTicker",
    "depth": "@depth",
    "depth_update": "@depth",
}

# Order book depth levels
ORDERBOOK_DEPTH_LEVELS: Final[list] = [5, 10, 20, 50, 100, 500, 1000, 5000]

# Status code to internal status mapping
STATUS_MAPPING: Final[Dict[str, str]] = {
    "NEW": "PENDING",
    "PARTIALLY_FILLED": "PARTIAL",
    "FILLED": "FILLED",
    "CANCELED": "CANCELLED",
    "PENDING_CANCEL": "PENDING",
    "REJECTED": "REJECTED",
    "EXPIRED": "EXPIRED",
}
