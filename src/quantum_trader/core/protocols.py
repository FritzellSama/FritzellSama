"""
Protocols - Structural subtyping protocols for type checking.

This module defines protocols for static type checking without inheritance,
using Python's Protocol feature from typing module.
"""

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime
import polars as pl


@runtime_checkable
class Configurable(Protocol):
    """Protocol for objects with configuration."""

    config: Dict[str, Any]

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...


@runtime_checkable
class Initializable(Protocol):
    """Protocol for objects that can be initialized."""

    async def initialize(self) -> None:
        """Initialize object."""
        ...


@runtime_checkable
class Startable(Protocol):
    """Protocol for objects that can be started."""

    async def start(self) -> None:
        """Start object."""
        ...


@runtime_checkable
class Stoppable(Protocol):
    """Protocol for objects that can be stopped."""

    async def stop(self) -> None:
        """Stop object."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for objects with health checks."""

    def is_healthy(self) -> bool:
        """Check if object is healthy."""
        ...


@runtime_checkable
class Connectable(Protocol):
    """Protocol for objects that connect to external services."""

    async def connect(self) -> None:
        """Establish connection."""
        ...

    async def disconnect(self) -> None:
        """Close connection."""
        ...

    def is_connected(self) -> bool:
        """Check if connected."""
        ...


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol for market data providers."""

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pl.DataFrame:
        """Fetch OHLCV data."""
        ...

    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data."""
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Protocol for order execution."""

    async def execute_order(self, order: Any) -> Any:
        """Execute trading order."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for signal generation."""

    def generate_signals(self, data: pl.DataFrame) -> List[Any]:
        """Generate trading signals from data."""
        ...


@runtime_checkable
class RiskChecker(Protocol):
    """Protocol for risk checking."""

    def check_risk(self, order: Any) -> tuple[bool, Optional[str]]:
        """
        Check if order passes risk limits.

        Returns:
            Tuple of (approved, rejection_reason)
        """
        ...


@runtime_checkable
class PositionTracker(Protocol):
    """Protocol for position tracking."""

    def get_position(self, symbol: str) -> Optional[Any]:
        """Get position for symbol."""
        ...

    def get_all_positions(self) -> List[Any]:
        """Get all positions."""
        ...

    def update_position(self, execution: Any) -> Any:
        """Update position with execution result."""
        ...


@runtime_checkable
class BalanceProvider(Protocol):
    """Protocol for balance information."""

    def get_balance(self, currency: str = 'USDT') -> Decimal:
        """Get balance for currency."""
        ...

    def get_all_balances(self) -> Dict[str, Decimal]:
        """Get all balances."""
        ...


@runtime_checkable
class PriceProvider(Protocol):
    """Protocol for price information."""

    def get_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        ...

    async def get_historical_prices(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> pl.DataFrame:
        """Get historical prices."""
        ...


@runtime_checkable
class IndicatorCalculator(Protocol):
    """Protocol for indicator calculation."""

    def calculate(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate indicators from data."""
        ...


@runtime_checkable
class Backtestable(Protocol):
    """Protocol for backtestable strategies."""

    async def backtest(
        self,
        data: pl.DataFrame,
        initial_balance: Decimal
    ) -> Dict[str, Any]:
        """Run backtest on historical data."""
        ...


@runtime_checkable
class Optimizable(Protocol):
    """Protocol for optimizable objects."""

    async def optimize(
        self,
        data: pl.DataFrame,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Optimize parameters."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create from dictionary."""
        ...


@runtime_checkable
class Cacheable(Protocol):
    """Protocol for cacheable objects."""

    def cache_key(self) -> str:
        """Get cache key for this object."""
        ...


@runtime_checkable
class Timestamped(Protocol):
    """Protocol for timestamped objects."""

    created_at: datetime
    updated_at: datetime

    def touch(self) -> None:
        """Update timestamp."""
        ...


@runtime_checkable
class Identifiable(Protocol):
    """Protocol for objects with unique identifiers."""

    id: str

    def get_id(self) -> str:
        """Get unique identifier."""
        ...


@runtime_checkable
class Validated(Protocol):
    """Protocol for validated objects."""

    def validate(self) -> bool:
        """Validate object state."""
        ...

    def is_valid(self) -> bool:
        """Check if object is valid."""
        ...


@runtime_checkable
class Loggable(Protocol):
    """Protocol for objects with logging."""

    def log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        ...

    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Protocol for event emission."""

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event."""
        ...

    def on(self, event_type: str, handler: Any) -> None:
        """Register event handler."""
        ...


@runtime_checkable
class MetricsProvider(Protocol):
    """Protocol for metrics provision."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics dictionary."""
        ...


@runtime_checkable
class StatusProvider(Protocol):
    """Protocol for status information."""

    def get_status(self) -> Dict[str, Any]:
        """Get status dictionary."""
        ...


@runtime_checkable
class PerformanceTracker(Protocol):
    """Protocol for performance tracking."""

    def calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio."""
        ...

    def calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown."""
        ...

    def get_performance_metrics(self) -> Dict[str, Decimal]:
        """Get all performance metrics."""
        ...


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation."""

    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate data quality."""
        ...

    def clean_data(self, data: pl.DataFrame) -> pl.DataFrame:
        """Clean and prepare data."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """Protocol for notifications."""

    async def notify(self, message: str, severity: str) -> bool:
        """Send notification."""
        ...


@runtime_checkable
class Retryable(Protocol):
    """Protocol for retryable operations."""

    max_retries: int

    async def retry(self, operation: Any) -> Any:
        """Retry operation with backoff."""
        ...
