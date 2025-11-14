"""Database models for trading strategies.

This module provides SQLAlchemy ORM models for persisting strategy configurations
and metadata with versioning support.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, DateTime, Numeric, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from structlog import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class StrategyModel(Base):
    """Database model for trading strategies.

    Stores strategy configuration, parameters, and metadata with support
    for versioning and historical tracking.

    Attributes:
        id: Unique strategy identifier (UUID)
        name: Strategy name (unique per version)
        version: Strategy version
        description: Strategy description

        # Configuration
        config: Strategy configuration parameters (JSONB)
        parameters: Strategy parameters (JSONB)

        # Status
        is_active: Whether strategy is currently active
        is_enabled: Whether strategy is enabled for trading

        # Metadata
        created_at: Strategy creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
        activated_at: Last activation timestamp (UTC)
        deactivated_at: Last deactivation timestamp (UTC)

        # Performance tracking
        total_trades: Total trades executed by this strategy
        winning_trades: Number of winning trades
        total_pnl: Cumulative P&L
        max_drawdown: Maximum drawdown observed

        # Risk limits
        max_position_size: Maximum position size allowed
        max_daily_loss: Maximum daily loss limit
        max_open_positions: Maximum concurrent positions

        # Additional context
        metadata: Additional strategy metadata (JSONB)

    Example:
        >>> strategy = StrategyModel(
        ...     name="momentum_v1",
        ...     version="1.0.0",
        ...     config={"lookback": 20, "threshold": 0.02},
        ...     is_active=True
        ... )
    """

    __tablename__ = "strategies"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Strategy identification
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False, default="1.0.0")
    description = Column(Text, nullable=True)

    # Configuration
    config = Column(JSONB, nullable=False, default=dict)
    parameters = Column(JSONB, nullable=False, default=dict)

    # Status
    is_active = Column(Boolean, nullable=False, default=False, index=True)
    is_enabled = Column(Boolean, nullable=False, default=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    deactivated_at = Column(DateTime(timezone=True), nullable=True)

    # Performance tracking
    total_trades = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    winning_trades = Column(Numeric(precision=10, scale=0), nullable=False, default=0)
    total_pnl = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )
    max_drawdown = Column(
        Numeric(precision=28, scale=18),
        nullable=False,
        default=Decimal("0")
    )

    # Risk limits
    max_position_size = Column(Numeric(precision=28, scale=18), nullable=True)
    max_daily_loss = Column(Numeric(precision=28, scale=18), nullable=True)
    max_open_positions = Column(Numeric(precision=10, scale=0), nullable=True)

    # Additional metadata
    metadata = Column(JSONB, nullable=False, default=dict)

    # Indexes for performance
    __table_args__ = (
        Index("idx_strategies_name_version", "name", "version", unique=True),
        Index("idx_strategies_is_active", "is_active"),
    )

    def __repr__(self) -> str:
        """String representation of strategy."""
        return (
            f"<StrategyModel(name={self.name}, version={self.version}, "
            f"active={self.is_active}, pnl={self.total_pnl})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary.

        Returns:
            Dictionary representation of the strategy

        Example:
            >>> strategy.to_dict()
            {
                'name': 'momentum_v1',
                'version': '1.0.0',
                'is_active': True,
                ...
            }
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "parameters": self.parameters,
            "is_active": self.is_active,
            "is_enabled": self.is_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else None,
            "total_trades": int(self.total_trades) if self.total_trades is not None else 0,
            "winning_trades": int(self.winning_trades) if self.winning_trades is not None else 0,
            "total_pnl": str(self.total_pnl) if self.total_pnl is not None else None,
            "max_drawdown": str(self.max_drawdown) if self.max_drawdown is not None else None,
            "max_position_size": str(self.max_position_size) if self.max_position_size is not None else None,
            "max_daily_loss": str(self.max_daily_loss) if self.max_daily_loss is not None else None,
            "max_open_positions": int(self.max_open_positions) if self.max_open_positions is not None else None,
            "metadata": self.metadata,
        }

    @property
    def win_rate(self) -> Decimal:
        """Calculate win rate.

        Returns:
            Win rate as percentage (0-100)
        """
        if not self.total_trades or self.total_trades == 0:
            return Decimal("0")
        return (self.winning_trades / self.total_trades) * Decimal("100")

    @property
    def is_profitable(self) -> bool:
        """Check if strategy is profitable."""
        return self.total_pnl > 0

    @property
    def full_name(self) -> str:
        """Get full strategy name with version.

        Returns:
            Strategy name with version (e.g., 'momentum_v1@1.0.0')
        """
        return f"{self.name}@{self.version}"

    def activate(self) -> None:
        """Activate the strategy.

        Example:
            >>> strategy.activate()
        """
        self.is_active = True
        self.activated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        logger.info("Strategy activated", strategy=self.full_name)

    def deactivate(self) -> None:
        """Deactivate the strategy.

        Example:
            >>> strategy.deactivate()
        """
        self.is_active = False
        self.deactivated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        logger.info("Strategy deactivated", strategy=self.full_name)
