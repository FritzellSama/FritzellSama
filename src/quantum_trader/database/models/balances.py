"""Balance tracking models for multi-exchange portfolio management.

Tracks account balances across multiple exchanges with real-time updates.
All monetary values use Decimal for precision.
"""

from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class BalanceType(Enum):
    """Types of balances in the system."""
    SPOT = "SPOT"
    MARGIN = "MARGIN"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class BalanceStatus(Enum):
    """Status of balance records."""
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"
    LOCKED = "LOCKED"
    INACTIVE = "INACTIVE"


@dataclass
class Balance:
    """Account balance for a specific currency on an exchange.

    Attributes:
        exchange: Exchange name
        currency: Currency code (e.g., 'USDT', 'BTC')
        balance_type: Type of balance (SPOT, MARGIN, etc.)
        total: Total balance (free + locked)
        free: Available balance for trading
        locked: Balance locked in orders
        status: Balance status
        last_updated: UTC timestamp of last update
        metadata: Additional context
        balance_id: Unique identifier (set by database)
    """
    exchange: str
    currency: str
    balance_type: BalanceType
    total: Decimal
    free: Decimal
    locked: Decimal
    status: BalanceStatus
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    balance_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate balance after initialization."""
        if not isinstance(self.last_updated.tzinfo, type(timezone.utc)):
            if self.last_updated.tzinfo is None:
                raise ValueError("last_updated must have UTC timezone")

        # Validate all monetary values are Decimal
        for field_name in ['total', 'free', 'locked']:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise ValueError(f"{field_name} must be Decimal, not {type(value)}")

        # Validate balance consistency
        if self.total != self.free + self.locked:
            raise ValueError(
                f"Balance inconsistency: total ({self.total}) != "
                f"free ({self.free}) + locked ({self.locked})"
            )

        # Validate non-negative balances
        if self.total < Decimal('0'):
            raise ValueError(f"Total balance cannot be negative: {self.total}")
        if self.free < Decimal('0'):
            raise ValueError(f"Free balance cannot be negative: {self.free}")
        if self.locked < Decimal('0'):
            raise ValueError(f"Locked balance cannot be negative: {self.locked}")

    @property
    def locked_percentage(self) -> Decimal:
        """Calculate percentage of balance that is locked.

        Returns:
            Percentage as Decimal (0-100)
        """
        if self.total == Decimal('0'):
            return Decimal('0')

        return (self.locked / self.total * Decimal('100')).quantize(
            Decimal('0.01'), rounding=ROUND_DOWN
        )

    def can_trade(self, amount: Decimal) -> bool:
        """Check if sufficient free balance exists for trade.

        Args:
            amount: Amount to check

        Returns:
            True if sufficient free balance exists
        """
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be Decimal")

        return self.free >= amount and self.status == BalanceStatus.ACTIVE

    def lock_amount(self, amount: Decimal) -> 'Balance':
        """Create new Balance with amount locked.

        Args:
            amount: Amount to lock

        Returns:
            New Balance instance with updated free/locked

        Raises:
            ValueError: If insufficient free balance
        """
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be Decimal")

        if amount > self.free:
            raise ValueError(
                f"Insufficient free balance: requested {amount}, available {self.free}"
            )

        return Balance(
            exchange=self.exchange,
            currency=self.currency,
            balance_type=self.balance_type,
            total=self.total,
            free=self.free - amount,
            locked=self.locked + amount,
            status=self.status,
            last_updated=datetime.now(timezone.utc),
            metadata=self.metadata.copy(),
            balance_id=self.balance_id
        )

    def unlock_amount(self, amount: Decimal) -> 'Balance':
        """Create new Balance with amount unlocked.

        Args:
            amount: Amount to unlock

        Returns:
            New Balance instance with updated free/locked

        Raises:
            ValueError: If insufficient locked balance
        """
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be Decimal")

        if amount > self.locked:
            raise ValueError(
                f"Insufficient locked balance: requested {amount}, locked {self.locked}"
            )

        return Balance(
            exchange=self.exchange,
            currency=self.currency,
            balance_type=self.balance_type,
            total=self.total,
            free=self.free + amount,
            locked=self.locked - amount,
            status=self.status,
            last_updated=datetime.now(timezone.utc),
            metadata=self.metadata.copy(),
            balance_id=self.balance_id
        )

    def update_total(self, new_total: Decimal) -> 'Balance':
        """Create new Balance with updated total (maintains locked amount).

        Args:
            new_total: New total balance

        Returns:
            New Balance instance

        Raises:
            ValueError: If new total less than locked amount
        """
        if not isinstance(new_total, Decimal):
            raise ValueError("New total must be Decimal")

        if new_total < self.locked:
            raise ValueError(
                f"New total ({new_total}) cannot be less than locked ({self.locked})"
            )

        return Balance(
            exchange=self.exchange,
            currency=self.currency,
            balance_type=self.balance_type,
            total=new_total,
            free=new_total - self.locked,
            locked=self.locked,
            status=self.status,
            last_updated=datetime.now(timezone.utc),
            metadata=self.metadata.copy(),
            balance_id=self.balance_id
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert balance to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'balance_id': self.balance_id,
            'exchange': self.exchange,
            'currency': self.currency,
            'balance_type': self.balance_type.value,
            'total': str(self.total),
            'free': str(self.free),
            'locked': str(self.locked),
            'status': self.status.value,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata,
            'locked_percentage': str(self.locked_percentage)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Balance':
        """Create Balance from dictionary.

        Args:
            data: Dictionary with balance data

        Returns:
            Balance instance
        """
        return cls(
            exchange=data['exchange'],
            currency=data['currency'],
            balance_type=BalanceType(data['balance_type']),
            total=Decimal(data['total']),
            free=Decimal(data['free']),
            locked=Decimal(data['locked']),
            status=BalanceStatus(data['status']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            metadata=data.get('metadata', {}),
            balance_id=data.get('balance_id')
        )


@dataclass
class BalanceSnapshot:
    """Portfolio balance snapshot across all exchanges.

    Attributes:
        timestamp: UTC timestamp of snapshot
        balances: List of all balances
        total_value_usd: Total portfolio value in USD
        metadata: Additional context
    """
    timestamp: datetime
    balances: List[Balance]
    total_value_usd: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate snapshot after initialization."""
        if not isinstance(self.timestamp.tzinfo, type(timezone.utc)):
            if self.timestamp.tzinfo is None:
                raise ValueError("timestamp must have UTC timezone")

        if not isinstance(self.total_value_usd, Decimal):
            raise ValueError("total_value_usd must be Decimal")

    def get_balance(
        self,
        exchange: str,
        currency: str,
        balance_type: BalanceType = BalanceType.SPOT
    ) -> Optional[Balance]:
        """Get specific balance from snapshot.

        Args:
            exchange: Exchange name
            currency: Currency code
            balance_type: Type of balance

        Returns:
            Balance if found, None otherwise
        """
        for balance in self.balances:
            if (balance.exchange == exchange and
                balance.currency == currency and
                balance.balance_type == balance_type):
                return balance
        return None

    def get_exchange_balances(self, exchange: str) -> List[Balance]:
        """Get all balances for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            List of balances for the exchange
        """
        return [b for b in self.balances if b.exchange == exchange]

    def get_currency_total(self, currency: str) -> Decimal:
        """Get total balance for a currency across all exchanges.

        Args:
            currency: Currency code

        Returns:
            Total balance as Decimal
        """
        total = Decimal('0')
        for balance in self.balances:
            if balance.currency == currency:
                total += balance.total
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'balances': [b.to_dict() for b in self.balances],
            'total_value_usd': str(self.total_value_usd),
            'metadata': self.metadata,
            'num_balances': len(self.balances),
            'exchanges': list(set(b.exchange for b in self.balances)),
            'currencies': list(set(b.currency for b in self.balances))
        }


class BalanceManager:
    """Manages balance tracking and updates across multiple exchanges.

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        _cache: In-memory balance cache
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize balance manager.

        Args:
            config: Must contain:
                - db_url: PostgreSQL connection string
                - cache_ttl_seconds: Cache time-to-live
        """
        self.config = config
        self._validate_config()

        self.db_pool: Optional[Any] = None
        self._cache: Dict[str, Balance] = {}
        self.cache_ttl = config['cache_ttl_seconds']

        logger.info("BalanceManager initialized")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = ['db_url', 'cache_ttl_seconds']
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

    async def connect(self) -> None:
        """Connect to PostgreSQL database."""
        import asyncpg

        try:
            self.db_pool = await asyncpg.create_pool(
                self.config['db_url'],
                min_size=self.config.get('db_pool_min', 2),
                max_size=self.config.get('db_pool_max', 10)
            )
            logger.info("BalanceManager connected to database")
        except Exception as e:
            logger.error("Failed to connect BalanceManager", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("BalanceManager disconnected")

    async def update_balance(self, balance: Balance) -> None:
        """Update balance in database and cache.

        Args:
            balance: Balance to update
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO balances (
                        exchange, currency, balance_type, total, free, locked,
                        status, last_updated, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (exchange, currency, balance_type)
                    DO UPDATE SET
                        total = $4, free = $5, locked = $6,
                        status = $7, last_updated = $8, metadata = $9
                    """,
                    balance.exchange, balance.currency, balance.balance_type.value,
                    str(balance.total), str(balance.free), str(balance.locked),
                    balance.status.value, balance.last_updated, balance.metadata
                )

            # Update cache
            cache_key = f"{balance.exchange}:{balance.currency}:{balance.balance_type.value}"
            self._cache[cache_key] = balance

            logger.debug(
                "Balance updated",
                exchange=balance.exchange,
                currency=balance.currency,
                total=str(balance.total)
            )

        except Exception as e:
            logger.error("Failed to update balance", error=str(e))
            raise

    async def get_balance(
        self,
        exchange: str,
        currency: str,
        balance_type: BalanceType = BalanceType.SPOT
    ) -> Optional[Balance]:
        """Get balance from cache or database.

        Args:
            exchange: Exchange name
            currency: Currency code
            balance_type: Type of balance

        Returns:
            Balance if found, None otherwise
        """
        cache_key = f"{exchange}:{currency}:{balance_type.value}"

        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query database
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM balances
                    WHERE exchange = $1 AND currency = $2 AND balance_type = $3
                    """,
                    exchange, currency, balance_type.value
                )

            if row:
                balance = Balance.from_dict({
                    'balance_id': str(row['balance_id']),
                    'exchange': row['exchange'],
                    'currency': row['currency'],
                    'balance_type': row['balance_type'],
                    'total': row['total'],
                    'free': row['free'],
                    'locked': row['locked'],
                    'status': row['status'],
                    'last_updated': row['last_updated'].isoformat(),
                    'metadata': row['metadata']
                })

                # Update cache
                self._cache[cache_key] = balance
                return balance

            return None

        except Exception as e:
            logger.error("Failed to get balance", error=str(e))
            raise

    async def create_snapshot(self) -> BalanceSnapshot:
        """Create portfolio balance snapshot.

        Returns:
            BalanceSnapshot with all current balances
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM balances WHERE status = 'ACTIVE' ORDER BY exchange, currency"
                )

            balances = [
                Balance.from_dict({
                    'balance_id': str(row['balance_id']),
                    'exchange': row['exchange'],
                    'currency': row['currency'],
                    'balance_type': row['balance_type'],
                    'total': row['total'],
                    'free': row['free'],
                    'locked': row['locked'],
                    'status': row['status'],
                    'last_updated': row['last_updated'].isoformat(),
                    'metadata': row['metadata']
                })
                for row in rows
            ]

            # Calculate total value (simplified - in production would use real-time prices)
            total_value = Decimal('0')

            snapshot = BalanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                balances=balances,
                total_value_usd=total_value,
                metadata={'num_exchanges': len(set(b.exchange for b in balances))}
            )

            logger.info("Balance snapshot created", num_balances=len(balances))
            return snapshot

        except Exception as e:
            logger.error("Failed to create snapshot", error=str(e))
            raise
