"""TimescaleDB hypertable management for time-series data.

Handles creation, configuration, and maintenance of hypertables for
market data, trades, OHLCV, and other time-series data with optimal
chunking and retention policies.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class ChunkInterval(Enum):
    """Standard chunk time intervals."""
    ONE_HOUR = "1 hour"
    SIX_HOURS = "6 hours"
    ONE_DAY = "1 day"
    ONE_WEEK = "7 days"
    ONE_MONTH = "30 days"


@dataclass
class HypertableConfig:
    """Configuration for creating a hypertable.

    Attributes:
        table_name: Name of the table to convert
        time_column: Column to use as time dimension
        chunk_time_interval: Time interval for chunks
        partitioning_column: Optional space partitioning column
        number_partitions: Number of space partitions
        if_not_exists: Skip if hypertable already exists
        migrate_data: Migrate existing data when creating
    """
    table_name: str
    time_column: str = "timestamp"
    chunk_time_interval: str = "1 day"
    partitioning_column: Optional[str] = None
    number_partitions: Optional[int] = None
    if_not_exists: bool = True
    migrate_data: bool = True

    def __post_init__(self) -> None:
        """Validate hypertable configuration."""
        if not self.table_name:
            raise ValueError("table_name cannot be empty")

        if not self.time_column:
            raise ValueError("time_column cannot be empty")

        if self.partitioning_column and not self.number_partitions:
            raise ValueError("number_partitions required when using partitioning_column")


@dataclass
class RetentionPolicy:
    """Data retention policy for a hypertable.

    Attributes:
        hypertable_name: Name of the hypertable
        drop_after_days: Drop chunks older than this many days
        schedule_interval: How often to run retention job
        if_not_exists: Skip if policy already exists
        policy_id: Database policy ID (set after creation)
    """
    hypertable_name: str
    drop_after_days: int
    schedule_interval: str = "1 day"
    if_not_exists: bool = True
    policy_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate retention policy."""
        if self.drop_after_days < 1:
            raise ValueError("drop_after_days must be >= 1")

        if not self.hypertable_name:
            raise ValueError("hypertable_name cannot be empty")


@dataclass
class HypertableStats:
    """Statistics for a hypertable.

    Attributes:
        hypertable_name: Name of the hypertable
        total_chunks: Total number of chunks
        compressed_chunks: Number of compressed chunks
        total_size_bytes: Total size in bytes
        table_size_bytes: Table size (excluding indexes)
        index_size_bytes: Index size
        num_dimensions: Number of dimensions (time + space)
        chunk_time_interval: Chunk time interval
        last_stats_update: When stats were calculated
    """
    hypertable_name: str
    total_chunks: int
    compressed_chunks: int
    total_size_bytes: int
    table_size_bytes: int
    index_size_bytes: int
    num_dimensions: int
    chunk_time_interval: str
    last_stats_update: datetime

    @property
    def total_size_mb(self) -> Decimal:
        """Get total size in megabytes.

        Returns:
            Size in MB as Decimal
        """
        return (Decimal(self.total_size_bytes) / Decimal(1024 * 1024)).quantize(
            Decimal('0.01')
        )

    @property
    def compression_percentage(self) -> Decimal:
        """Calculate percentage of chunks compressed.

        Returns:
            Percentage as Decimal (0-100)
        """
        if self.total_chunks == 0:
            return Decimal('0')

        return (Decimal(self.compressed_chunks) / Decimal(self.total_chunks) * Decimal('100')).quantize(
            Decimal('0.01')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'hypertable_name': self.hypertable_name,
            'total_chunks': self.total_chunks,
            'compressed_chunks': self.compressed_chunks,
            'total_size_bytes': self.total_size_bytes,
            'total_size_mb': str(self.total_size_mb),
            'table_size_bytes': self.table_size_bytes,
            'index_size_bytes': self.index_size_bytes,
            'num_dimensions': self.num_dimensions,
            'chunk_time_interval': self.chunk_time_interval,
            'compression_percentage': str(self.compression_percentage),
            'last_stats_update': self.last_stats_update.isoformat()
        }


class HypertableManager:
    """Manages TimescaleDB hypertables for time-series data.

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        _hypertables: Registered hypertables
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize hypertable manager.

        Args:
            config: Must contain:
                - db_url: PostgreSQL connection string
                - default_chunk_interval: Default chunk time interval
                - enable_compression: Enable compression by default
                - compression_after_days: Compress chunks after N days

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.db_pool: Optional[Any] = None
        self._hypertables: Dict[str, HypertableConfig] = {}

        self.default_chunk_interval = config['default_chunk_interval']
        self.enable_compression = config.get('enable_compression', True)
        self.compression_after_days = config.get('compression_after_days', 7)

        logger.info(
            "HypertableManager initialized",
            default_chunk_interval=self.default_chunk_interval
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = ['db_url', 'default_chunk_interval']
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

    async def connect(self) -> None:
        """Connect to PostgreSQL/TimescaleDB."""
        import asyncpg

        try:
            self.db_pool = await asyncpg.create_pool(
                self.config['db_url'],
                min_size=self.config.get('db_pool_min', 2),
                max_size=self.config.get('db_pool_max', 5)
            )

            # Verify TimescaleDB extension
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
                )
                if result == 0:
                    raise RuntimeError("TimescaleDB extension not installed")

                # Get TimescaleDB version
                version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                logger.info("Connected to TimescaleDB", version=version)

        except Exception as e:
            logger.error("Failed to connect HypertableManager", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("HypertableManager disconnected")

    async def create_hypertable(
        self,
        config: HypertableConfig
    ) -> None:
        """Create a hypertable from an existing table.

        Args:
            config: Hypertable configuration

        Raises:
            RuntimeError: If hypertable creation fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Build create_hypertable SQL
                sql_parts = [
                    f"SELECT create_hypertable('{config.table_name}', '{config.time_column}'",
                    f"chunk_time_interval => INTERVAL '{config.chunk_time_interval}'"
                ]

                if config.partitioning_column:
                    sql_parts.append(
                        f"partitioning_column => '{config.partitioning_column}', "
                        f"number_partitions => {config.number_partitions}"
                    )

                if config.if_not_exists:
                    sql_parts.append("if_not_exists => TRUE")

                if config.migrate_data:
                    sql_parts.append("migrate_data => TRUE")

                sql = ', '.join(sql_parts) + ");"

                await conn.execute(sql)

            self._hypertables[config.table_name] = config

            logger.info(
                "Hypertable created",
                table=config.table_name,
                chunk_interval=config.chunk_time_interval,
                partitioning=config.partitioning_column
            )

        except Exception as e:
            logger.error("Failed to create hypertable", table=config.table_name, error=str(e))
            raise RuntimeError(f"Create hypertable failed: {e}")

    async def add_retention_policy(
        self,
        policy: RetentionPolicy
    ) -> int:
        """Add data retention policy to a hypertable.

        Args:
            policy: Retention policy configuration

        Returns:
            Policy ID from database

        Raises:
            RuntimeError: If policy creation fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Add retention policy
                sql = f"""
                    SELECT add_retention_policy(
                        '{policy.hypertable_name}',
                        INTERVAL '{policy.drop_after_days} days',
                        if_not_exists => {policy.if_not_exists}
                    )
                """

                policy_id = await conn.fetchval(sql)
                policy.policy_id = policy_id

            logger.info(
                "Retention policy added",
                hypertable=policy.hypertable_name,
                drop_after_days=policy.drop_after_days,
                policy_id=policy_id
            )

            return policy_id

        except Exception as e:
            logger.error(
                "Failed to add retention policy",
                hypertable=policy.hypertable_name,
                error=str(e)
            )
            raise RuntimeError(f"Add retention policy failed: {e}")

    async def remove_retention_policy(
        self,
        hypertable_name: str
    ) -> None:
        """Remove retention policy from a hypertable.

        Args:
            hypertable_name: Name of hypertable

        Raises:
            RuntimeError: If policy removal fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    f"SELECT remove_retention_policy('{hypertable_name}')"
                )

            logger.info("Retention policy removed", hypertable=hypertable_name)

        except Exception as e:
            logger.error("Failed to remove retention policy", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Remove retention policy failed: {e}")

    async def get_hypertable_stats(
        self,
        hypertable_name: str
    ) -> HypertableStats:
        """Get statistics for a hypertable.

        Args:
            hypertable_name: Name of hypertable

        Returns:
            HypertableStats object

        Raises:
            RuntimeError: If query fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get chunk statistics
                chunk_stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = $1
                    """,
                    hypertable_name
                )

                # Get size statistics
                size_stats = await conn.fetchrow(
                    """
                    SELECT
                        hypertable_size($1) as total_size,
                        pg_table_size($1) as table_size,
                        pg_indexes_size($1) as index_size
                    """,
                    hypertable_name
                )

                # Get hypertable metadata
                metadata = await conn.fetchrow(
                    """
                    SELECT
                        num_dimensions,
                        chunk_time_interval::text
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_name = $1
                    """,
                    hypertable_name
                )

            if not chunk_stats or not size_stats or not metadata:
                raise RuntimeError(f"Hypertable not found: {hypertable_name}")

            stats = HypertableStats(
                hypertable_name=hypertable_name,
                total_chunks=chunk_stats['total_chunks'],
                compressed_chunks=chunk_stats['compressed_chunks'],
                total_size_bytes=size_stats['total_size'],
                table_size_bytes=size_stats['table_size'],
                index_size_bytes=size_stats['index_size'],
                num_dimensions=metadata['num_dimensions'],
                chunk_time_interval=metadata['chunk_time_interval'],
                last_stats_update=datetime.now(timezone.utc)
            )

            logger.info(
                "Hypertable stats retrieved",
                hypertable=hypertable_name,
                total_size_mb=str(stats.total_size_mb),
                chunks=stats.total_chunks
            )

            return stats

        except Exception as e:
            logger.error("Failed to get hypertable stats", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Get hypertable stats failed: {e}")

    async def list_hypertables(self) -> List[str]:
        """List all hypertables in the database.

        Returns:
            List of hypertable names

        Raises:
            RuntimeError: If query fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT hypertable_name FROM timescaledb_information.hypertables ORDER BY hypertable_name"
                )

            hypertable_names = [row['hypertable_name'] for row in rows]

            logger.debug("Hypertables listed", count=len(hypertable_names))
            return hypertable_names

        except Exception as e:
            logger.error("Failed to list hypertables", error=str(e))
            raise RuntimeError(f"List hypertables failed: {e}")

    async def set_chunk_time_interval(
        self,
        hypertable_name: str,
        new_interval: str
    ) -> None:
        """Change chunk time interval for a hypertable.

        Args:
            hypertable_name: Name of hypertable
            new_interval: New chunk time interval (e.g., '1 day', '1 hour')

        Raises:
            RuntimeError: If update fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    f"SELECT set_chunk_time_interval('{hypertable_name}', INTERVAL '{new_interval}')"
                )

            logger.info(
                "Chunk time interval updated",
                hypertable=hypertable_name,
                new_interval=new_interval
            )

        except Exception as e:
            logger.error(
                "Failed to set chunk time interval",
                hypertable=hypertable_name,
                error=str(e)
            )
            raise RuntimeError(f"Set chunk time interval failed: {e}")

    async def drop_chunks(
        self,
        hypertable_name: str,
        older_than: datetime,
        verbose: bool = False
    ) -> int:
        """Drop chunks older than specified time.

        Args:
            hypertable_name: Name of hypertable
            older_than: Drop chunks with data before this time
            verbose: Log each dropped chunk

        Returns:
            Number of chunks dropped

        Raises:
            RuntimeError: If drop fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT drop_chunks($1, $2, verbose => $3)",
                    hypertable_name,
                    older_than,
                    verbose
                )

            chunks_dropped = result if result else 0

            logger.info(
                "Chunks dropped",
                hypertable=hypertable_name,
                older_than=older_than.isoformat(),
                chunks_dropped=chunks_dropped
            )

            return chunks_dropped

        except Exception as e:
            logger.error("Failed to drop chunks", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Drop chunks failed: {e}")

    async def create_standard_hypertables(self) -> None:
        """Create standard hypertables for trading system.

        Creates hypertables for:
        - market_data (ticks)
        - trades
        - ohlcv
        - order_book_snapshots
        """
        standard_tables = [
            HypertableConfig(
                table_name='market_data',
                time_column='timestamp',
                chunk_time_interval='1 hour',
                partitioning_column='symbol',
                number_partitions=4
            ),
            HypertableConfig(
                table_name='trades',
                time_column='timestamp',
                chunk_time_interval='1 day',
                partitioning_column='exchange',
                number_partitions=2
            ),
            HypertableConfig(
                table_name='ohlcv',
                time_column='timestamp',
                chunk_time_interval='7 days',
                partitioning_column='symbol',
                number_partitions=4
            ),
            HypertableConfig(
                table_name='order_book_snapshots',
                time_column='timestamp',
                chunk_time_interval='1 hour',
                partitioning_column='symbol',
                number_partitions=4
            )
        ]

        for table_config in standard_tables:
            try:
                await self.create_hypertable(table_config)
            except Exception as e:
                logger.warning(
                    "Failed to create standard hypertable",
                    table=table_config.table_name,
                    error=str(e)
                )

        logger.info("Standard hypertables created")

    async def get_chunk_info(
        self,
        hypertable_name: str
    ) -> List[Dict[str, Any]]:
        """Get detailed information about chunks.

        Args:
            hypertable_name: Name of hypertable

        Returns:
            List of chunk information dictionaries

        Raises:
            RuntimeError: If query fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        chunk_name,
                        range_start,
                        range_end,
                        is_compressed,
                        pg_size_pretty(total_bytes) as size
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = $1
                    ORDER BY range_start DESC
                    """,
                    hypertable_name
                )

            chunks = [
                {
                    'chunk_name': row['chunk_name'],
                    'range_start': row['range_start'].isoformat() if row['range_start'] else None,
                    'range_end': row['range_end'].isoformat() if row['range_end'] else None,
                    'is_compressed': row['is_compressed'],
                    'size': row['size']
                }
                for row in rows
            ]

            logger.debug("Chunk info retrieved", hypertable=hypertable_name, chunks=len(chunks))
            return chunks

        except Exception as e:
            logger.error("Failed to get chunk info", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Get chunk info failed: {e}")
