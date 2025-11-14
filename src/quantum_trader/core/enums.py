"""
Enums - System-wide enumeration definitions.

This module defines all enumerations used across the trading system
for type safety and consistency.
"""

from enum import Enum, auto


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICE_BERG = "ICE_BERG"
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    POST_ONLY = "POST_ONLY"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class ExecutionStatus(Enum):
    """Execution status enumeration."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Exchange(Enum):
    """Supported exchanges."""

    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    OKX = "OKX"
    KUCOIN = "KUCOIN"
    BITGET = "BITGET"


class TimeFrame(Enum):
    """Trading timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MO1 = "1M"


class SignalAction(Enum):
    """Trading signal actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class PositionSide(Enum):
    """Position side enumeration."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class TradingMode(Enum):
    """Trading mode enumeration."""

    LIVE = "LIVE"
    PAPER = "PAPER"
    BACKTEST = "BACKTEST"


class StrategyState(Enum):
    """Strategy state enumeration."""

    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class RiskLevel(Enum):
    """Risk level enumeration."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MarketCondition(Enum):
    """Market condition enumeration."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class IndicatorType(Enum):
    """Technical indicator types."""

    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"


class EventType(Enum):
    """System event types."""

    ORDER_PLACED = "ORDER_PLACED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_REJECTED = "ORDER_REJECTED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    EXCHANGE_ERROR = "EXCHANGE_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    MARKET_DATA_UPDATE = "MARKET_DATA_UPDATE"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataSource(Enum):
    """Data source types."""

    EXCHANGE = "EXCHANGE"
    DATABASE = "DATABASE"
    CACHE = "CACHE"
    API = "API"
    FILE = "FILE"


class ConnectionState(Enum):
    """Connection state enumeration."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class WebSocketState(Enum):
    """WebSocket connection state."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    SUBSCRIBED = "SUBSCRIBED"
    UNSUBSCRIBED = "UNSUBSCRIBED"
    ERROR = "ERROR"


class BacktestState(Enum):
    """Backtest execution state."""

    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OptimizationMethod(Enum):
    """Parameter optimization methods."""

    GRID_SEARCH = "GRID_SEARCH"
    RANDOM_SEARCH = "RANDOM_SEARCH"
    GENETIC_ALGORITHM = "GENETIC_ALGORITHM"
    BAYESIAN = "BAYESIAN"
    WALK_FORWARD = "WALK_FORWARD"


class MLModelType(Enum):
    """Machine learning model types."""

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"
    REINFORCEMENT_LEARNING = "REINFORCEMENT_LEARNING"
    TIME_SERIES = "TIME_SERIES"
    ENSEMBLE = "ENSEMBLE"


class FeatureType(Enum):
    """Feature engineering types."""

    PRICE = "PRICE"
    VOLUME = "VOLUME"
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    SENTIMENT = "SENTIMENT"
    ALTERNATIVE = "ALTERNATIVE"


class DatabaseOperation(Enum):
    """Database operation types."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SELECT = "SELECT"
    BULK_INSERT = "BULK_INSERT"


class CacheStrategy(Enum):
    """Cache strategies."""

    LRU = "LRU"  # Least Recently Used
    LFU = "LFU"  # Least Frequently Used
    FIFO = "FIFO"  # First In First Out
    TTL = "TTL"  # Time To Live


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationType(Enum):
    """Notification types."""

    EMAIL = "EMAIL"
    SMS = "SMS"
    TELEGRAM = "TELEGRAM"
    WEBHOOK = "WEBHOOK"
    PUSH = "PUSH"


class ReportFormat(Enum):
    """Report output formats."""

    HTML = "HTML"
    PDF = "PDF"
    JSON = "JSON"
    CSV = "CSV"
    EXCEL = "EXCEL"


class PerformanceMetric(Enum):
    """Performance metric types."""

    TOTAL_RETURN = "TOTAL_RETURN"
    SHARPE_RATIO = "SHARPE_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    WIN_RATE = "WIN_RATE"
    PROFIT_FACTOR = "PROFIT_FACTOR"
    CALMAR_RATIO = "CALMAR_RATIO"
    ALPHA = "ALPHA"
    BETA = "BETA"


class SecurityLevel(Enum):
    """Security clearance levels."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


class UserRole(Enum):
    """User role enumeration."""

    ADMIN = "ADMIN"
    TRADER = "TRADER"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"
    API = "API"


class SystemStatus(Enum):
    """Overall system status."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    MAINTENANCE = "MAINTENANCE"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
