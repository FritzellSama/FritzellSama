"""Database models for strategy performance metrics.

This module provides SQLAlchemy ORM models for tracking strategy performance
metrics over time with TimescaleDB support for efficient time-series queries.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Numeric, Index, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class PerformanceMetricsModel(Base):
    """Database model for strategy performance metrics.

    Stores performance metrics calculated at regular intervals for
    monitoring and analysis of trading strategies.

    Attributes:
        id: Unique metric record identifier (UUID)
        strategy: Strategy name
        exchange: Exchange name
        timestamp: Metric calculation timestamp (UTC)

        # P&L Metrics
        total_pnl: Total realized + unrealized P&L
        realized_pnl: Realized P&L from closed positions
        unrealized_pnl: Unrealized P&L from open positions

        # Performance Metrics
        sharpe_ratio: Sharpe ratio (risk-adjusted returns)
        sortino_ratio: Sortino ratio (downside risk-adjusted)
        max_drawdown: Maximum drawdown percentage
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profit to gross loss

        # Trade Statistics
        total_trades: Total number of trades executed
        winning_trades: Number of profitable trades
        losing_trades: Number of losing trades
        average_win: Average winning trade P&L
        average_loss: Average losing trade P&L

        # Risk Metrics
        value_at_risk: Value at Risk (VaR) estimate
        expected_shortfall: Conditional VaR (CVaR)

        # Position Metrics
        open_positions: Number of currently open positions
        total_exposure: Total market exposure
        leverage_used: Current leverage ratio

        # Additional context
        metadata: Additional metrics and context (JSONB)

    Example:
        >>> metrics = PerformanceMetricsModel(
        ...     strategy="momentum_v1",
        ...     exchange="BINANCE",
        ...     total_pnl=Decimal("15000.50"),
        ...     sharpe_ratio=Decimal("2.5"),
        ...     win_rate=Decimal("65.5")
        ... )
    """

    __tablename__ = "performance_metrics"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Context
    strategy = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    # P&L Metrics
    total_pnl = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("0"))
    realized_pnl = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("0"))
    unrealized_pnl = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("0"))

    # Performance Metrics
    sharpe_ratio = Column(Numeric(precision=28, scale=18), nullable=True)
    sortino_ratio = Column(Numeric(precision=28, scale=18), nullable=True)
    max_drawdown = Column(Numeric(precision=28, scale=18), nullable=True)
    win_rate = Column(Numeric(precision=28, scale=18), nullable=True)
    profit_factor = Column(Numeric(precision=28, scale=18), nullable=True)

    # Trade Statistics
    total_trades = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    winning_trades = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    losing_trades = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    average_win = Column(Numeric(precision=28, scale=18), nullable=True)
    average_loss = Column(Numeric(precision=28, scale=18), nullable=True)

    # Risk Metrics
    value_at_risk = Column(Numeric(precision=28, scale=18), nullable=True)
    expected_shortfall = Column(Numeric(precision=28, scale=18), nullable=True)

    # Position Metrics
    open_positions = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    total_exposure = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("0"))
    leverage_used = Column(Numeric(precision=28, scale=18), nullable=False, default=Decimal("1"))

    # Additional metadata
    metadata = Column(JSONB, nullable=False, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("idx_perf_timestamp_strategy", "timestamp", "strategy"),
        Index("idx_perf_strategy_exchange", "strategy", "exchange"),
    )

    def __repr__(self) -> str:
        """String representation of performance metrics."""
        return (
            f"<PerformanceMetricsModel(strategy={self.strategy}, "
            f"total_pnl={self.total_pnl}, sharpe={self.sharpe_ratio})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of the metrics

        Example:
            >>> metrics.to_dict()
            {
                'strategy': 'momentum_v1',
                'total_pnl': '15000.50',
                'sharpe_ratio': '2.5',
                ...
            }
        """
        return {
            "id": str(self.id),
            "strategy": self.strategy,
            "exchange": self.exchange,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "total_pnl": str(self.total_pnl) if self.total_pnl is not None else None,
            "realized_pnl": str(self.realized_pnl) if self.realized_pnl is not None else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl is not None else None,
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio is not None else None,
            "sortino_ratio": str(self.sortino_ratio) if self.sortino_ratio is not None else None,
            "max_drawdown": str(self.max_drawdown) if self.max_drawdown is not None else None,
            "win_rate": str(self.win_rate) if self.win_rate is not None else None,
            "profit_factor": str(self.profit_factor) if self.profit_factor is not None else None,
            "total_trades": int(self.total_trades) if self.total_trades is not None else 0,
            "winning_trades": int(self.winning_trades) if self.winning_trades is not None else 0,
            "losing_trades": int(self.losing_trades) if self.losing_trades is not None else 0,
            "average_win": str(self.average_win) if self.average_win is not None else None,
            "average_loss": str(self.average_loss) if self.average_loss is not None else None,
            "value_at_risk": str(self.value_at_risk) if self.value_at_risk is not None else None,
            "expected_shortfall": str(self.expected_shortfall) if self.expected_shortfall is not None else None,
            "open_positions": int(self.open_positions) if self.open_positions is not None else 0,
            "total_exposure": str(self.total_exposure) if self.total_exposure is not None else None,
            "leverage_used": str(self.leverage_used) if self.leverage_used is not None else None,
            "metadata": self.metadata,
        }

    @property
    def is_profitable(self) -> bool:
        """Check if strategy is currently profitable."""
        return self.total_pnl > 0

    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Calculate risk-reward ratio.

        Returns:
            Ratio of average win to average loss, or None if not calculable
        """
        if not self.average_loss or self.average_loss == 0:
            return None
        if not self.average_win:
            return None
        return abs(self.average_win / self.average_loss)
