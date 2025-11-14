"""
Interfaces - Protocol definitions and abstract interfaces.

This module defines interfaces (protocols) for dependency injection
and loose coupling between components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime
import polars as pl


class IExchange(Protocol):
    """Interface for exchange connectors."""

    async def connect(self) -> None:
        """Establish connection to exchange."""
        ...

    async def disconnect(self) -> None:
        """Close connection to exchange."""
        ...

    async def fetch_balance(self) -> Dict[str, Decimal]:
        """
        Fetch account balance.

        Returns:
            Dictionary mapping currencies to balances
        """
        ...

    async def place_order(self, order: Any) -> str:
        """
        Place trading order.

        Args:
            order: Order object

        Returns:
            Exchange order ID
        """
        ...

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel order.

        Args:
            order_id: Order identifier
            symbol: Trading symbol

        Returns:
            True if cancelled successfully
        """
        ...

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h')
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        ...


class IRiskManager(Protocol):
    """Interface for risk management."""

    def check_order(self, order: Any) -> tuple[bool, Optional[str]]:
        """
        Validate order against risk limits.

        Args:
            order: Order to validate

        Returns:
            Tuple of (approved, rejection_reason)
        """
        ...

    def calculate_position_size(
        self,
        signal: Any,
        balance: Decimal
    ) -> Decimal:
        """
        Calculate position size based on risk.

        Args:
            signal: Trading signal
            balance: Account balance

        Returns:
            Position size
        """
        ...


class IStrategy(Protocol):
    """Interface for trading strategies."""

    def generate_signals(self, market_data: pl.DataFrame) -> List[Any]:
        """
        Generate trading signals from market data.

        Args:
            market_data: Market data DataFrame

        Returns:
            List of trading signals
        """
        ...

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """
        Calculate technical indicators.

        Args:
            data: Price data DataFrame

        Returns:
            Dictionary of indicator values
        """
        ...


class IDataProvider(Protocol):
    """Interface for market data providers."""

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """
        Get OHLCV data for time range.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        ...

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker dictionary
        """
        ...


class IPortfolio(Protocol):
    """Interface for portfolio management."""

    def get_balance(self, currency: str) -> Decimal:
        """
        Get balance for currency.

        Args:
            currency: Currency code

        Returns:
            Balance amount
        """
        ...

    def get_position(self, symbol: str) -> Optional[Any]:
        """
        Get position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object or None
        """
        ...

    def get_all_positions(self) -> List[Any]:
        """
        Get all open positions.

        Returns:
            List of positions
        """
        ...


class ILogger(Protocol):
    """Interface for logging."""

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...


class ICache(Protocol):
    """Interface for caching."""

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        ...

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        ...


class IDatabase(Protocol):
    """Interface for database operations."""

    async def connect(self) -> None:
        """Connect to database."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from database."""
        ...

    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute database query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query result
        """
        ...

    async def fetch_one(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Fetch single row.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Row dictionary or None
        """
        ...

    async def fetch_all(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Fetch all rows.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        ...


class INotifier(Protocol):
    """Interface for notifications."""

    async def send(
        self,
        message: str,
        severity: str,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send notification.

        Args:
            message: Notification message
            severity: Severity level
            recipients: List of recipients

        Returns:
            True if sent successfully
        """
        ...


class IEventBus(Protocol):
    """Interface for event bus."""

    async def publish(self, event: Any) -> None:
        """
        Publish event.

        Args:
            event: Event object
        """
        ...

    async def subscribe(self, event_type: str, handler: Any) -> None:
        """
        Subscribe to event type.

        Args:
            event_type: Type of event
            handler: Event handler function
        """
        ...


class IMetricsCollector(Protocol):
    """Interface for metrics collection."""

    def increment(self, metric: str, value: int = 1, tags: Optional[Dict] = None) -> None:
        """
        Increment counter metric.

        Args:
            metric: Metric name
            value: Increment value
            tags: Metric tags
        """
        ...

    def gauge(self, metric: str, value: float, tags: Optional[Dict] = None) -> None:
        """
        Set gauge metric.

        Args:
            metric: Metric name
            value: Metric value
            tags: Metric tags
        """
        ...

    def histogram(self, metric: str, value: float, tags: Optional[Dict] = None) -> None:
        """
        Record histogram value.

        Args:
            metric: Metric name
            value: Value to record
            tags: Metric tags
        """
        ...
