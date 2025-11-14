"""TimescaleDB compression management for time-series data.

Handles automatic compression policies for market data, trades, and OHLCV data
to optimize storage and query performance for historical data.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from structlog import get_logger

logger = get_logger(__name__)


class CompressionStatus(Enum):
    """Status of compression for a chunk."""
    UNCOMPRESSED = "UNCOMPRESSED"
    COMPRESSED = "COMPRESSED"
    COMPRESSING = "COMPRESSING"
    FAILED = "FAILED"


@dataclass
class CompressionPolicy:
    """Compression policy configuration for a hypertable.

    Attributes:
        hypertable_name: Name of the hypertable
        compress_after_days: Compress chunks older than this many days
        segment_by: Column to segment compressed data by
        order_by: Columns to order compressed data by
        enabled: Whether policy is active
        policy_id: Database policy ID (set after creation)
    """
    hypertable_name: str
    compress_after_days: int
    segment_by: Optional[str] = None
    order_by: str = "timestamp DESC"
    enabled: bool = True
    policy_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate compression policy."""
        if self.compress_after_days < 1:
            raise ValueError("compress_after_days must be >= 1")

        if not self.hypertable_name:
            raise ValueError("hypertable_name cannot be empty")


@dataclass
class CompressionStats:
    """Statistics for compressed chunks.

    Attributes:
        hypertable_name: Name of the hypertable
        total_chunks: Total number of chunks
        compressed_chunks: Number of compressed chunks
        uncompressed_chunks: Number of uncompressed chunks
        total_size_bytes: Total size in bytes
        compressed_size_bytes: Size after compression
        compression_ratio: Compression ratio (original / compressed)
        last_updated: When stats were calculated
    """
    hypertable_name: str
    total_chunks: int
    compressed_chunks: int
    uncompressed_chunks: int
    total_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: Decimal
    last_updated: datetime

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

    @property
    def space_saved_bytes(self) -> int:
        """Calculate space saved by compression.

        Returns:
            Bytes saved
        """
        return self.total_size_bytes - self.compressed_size_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'hypertable_name': self.hypertable_name,
            'total_chunks': self.total_chunks,
            'compressed_chunks': self.compressed_chunks,
            'uncompressed_chunks': self.uncompressed_chunks,
            'total_size_bytes': self.total_size_bytes,
            'compressed_size_bytes': self.compressed_size_bytes,
            'compression_ratio': str(self.compression_ratio),
            'compression_percentage': str(self.compression_percentage),
            'space_saved_bytes': self.space_saved_bytes,
            'last_updated': self.last_updated.isoformat()
        }


class CompressionManager:
    """Manages TimescaleDB compression policies and operations.

    Attributes:
        config: Configuration dictionary
        db_pool: Database connection pool
        _policies: Active compression policies
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize compression manager.

        Args:
            config: Must contain:
                - db_url: PostgreSQL connection string
                - default_compress_after_days: Default compression threshold
                - compression_check_interval_hours: Policy check frequency
                - max_parallel_compressions: Max concurrent compressions

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.db_pool: Optional[Any] = None
        self._policies: Dict[str, CompressionPolicy] = {}
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        self.default_compress_after = config['default_compress_after_days']
        self.check_interval = config['compression_check_interval_hours'] * 3600
        self.max_parallel = config['max_parallel_compressions']

        logger.info(
            "CompressionManager initialized",
            default_compress_after=self.default_compress_after
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = [
            'db_url', 'default_compress_after_days',
            'compression_check_interval_hours', 'max_parallel_compressions'
        ]
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

            logger.info("CompressionManager connected to TimescaleDB")
            self._running = True

        except Exception as e:
            logger.error("Failed to connect CompressionManager", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Disconnect and stop background tasks."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        if self.db_pool:
            await self.db_pool.close()
            logger.info("CompressionManager disconnected")

    async def enable_compression(
        self,
        hypertable_name: str,
        segment_by: Optional[str] = None,
        order_by: str = "timestamp DESC"
    ) -> None:
        """Enable compression on a hypertable.

        Args:
            hypertable_name: Name of hypertable
            segment_by: Optional column to segment by (e.g., 'symbol')
            order_by: Columns to order by in compressed chunks

        Raises:
            RuntimeError: If compression enable fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Enable compression on hypertable
                alter_sql = f"""
                    ALTER TABLE {hypertable_name} SET (
                        timescaledb.compress
                """

                if segment_by:
                    alter_sql += f", timescaledb.compress_segmentby = '{segment_by}'"

                alter_sql += f", timescaledb.compress_orderby = '{order_by}'"
                alter_sql += ");"

                await conn.execute(alter_sql)

            logger.info(
                "Compression enabled",
                hypertable=hypertable_name,
                segment_by=segment_by,
                order_by=order_by
            )

        except Exception as e:
            logger.error("Failed to enable compression", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Enable compression failed: {e}")

    async def add_compression_policy(
        self,
        policy: CompressionPolicy
    ) -> int:
        """Add automatic compression policy to a hypertable.

        Args:
            policy: Compression policy to add

        Returns:
            Policy ID from database

        Raises:
            RuntimeError: If policy creation fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Add compression policy
                policy_id = await conn.fetchval(
                    """
                    SELECT add_compression_policy($1, INTERVAL '%s days')
                    """,
                    policy.hypertable_name,
                    policy.compress_after_days
                )

                policy.policy_id = policy_id
                self._policies[policy.hypertable_name] = policy

            logger.info(
                "Compression policy added",
                hypertable=policy.hypertable_name,
                compress_after_days=policy.compress_after_days,
                policy_id=policy_id
            )

            return policy_id

        except Exception as e:
            logger.error(
                "Failed to add compression policy",
                hypertable=policy.hypertable_name,
                error=str(e)
            )
            raise RuntimeError(f"Add compression policy failed: {e}")

    async def remove_compression_policy(
        self,
        hypertable_name: str
    ) -> None:
        """Remove compression policy from a hypertable.

        Args:
            hypertable_name: Name of hypertable

        Raises:
            RuntimeError: If policy removal fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "SELECT remove_compression_policy($1)",
                    hypertable_name
                )

            self._policies.pop(hypertable_name, None)

            logger.info("Compression policy removed", hypertable=hypertable_name)

        except Exception as e:
            logger.error("Failed to remove compression policy", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Remove compression policy failed: {e}")

    async def compress_chunk(
        self,
        chunk_name: str
    ) -> None:
        """Manually compress a specific chunk.

        Args:
            chunk_name: Name of chunk to compress

        Raises:
            RuntimeError: If compression fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "SELECT compress_chunk($1)",
                    chunk_name
                )

            logger.info("Chunk compressed", chunk=chunk_name)

        except Exception as e:
            logger.error("Failed to compress chunk", chunk=chunk_name, error=str(e))
            raise RuntimeError(f"Compress chunk failed: {e}")

    async def decompress_chunk(
        self,
        chunk_name: str
    ) -> None:
        """Decompress a specific chunk.

        Args:
            chunk_name: Name of chunk to decompress

        Raises:
            RuntimeError: If decompression fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "SELECT decompress_chunk($1)",
                    chunk_name
                )

            logger.info("Chunk decompressed", chunk=chunk_name)

        except Exception as e:
            logger.error("Failed to decompress chunk", chunk=chunk_name, error=str(e))
            raise RuntimeError(f"Decompress chunk failed: {e}")

    async def get_compression_stats(
        self,
        hypertable_name: str
    ) -> CompressionStats:
        """Get compression statistics for a hypertable.

        Args:
            hypertable_name: Name of hypertable

        Returns:
            CompressionStats object

        Raises:
            RuntimeError: If query fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE is_compressed) as compressed_chunks,
                        COUNT(*) FILTER (WHERE NOT is_compressed) as uncompressed_chunks,
                        SUM(total_bytes) as total_size_bytes,
                        COALESCE(SUM(total_bytes) FILTER (WHERE is_compressed), 0) as compressed_size_bytes
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = $1
                    """,
                    hypertable_name
                )

            if not stats:
                raise RuntimeError(f"No stats found for hypertable: {hypertable_name}")

            total_size = stats['total_size_bytes'] or 0
            compressed_size = stats['compressed_size_bytes'] or 0

            # Calculate compression ratio
            if compressed_size > 0:
                compression_ratio = Decimal(total_size) / Decimal(compressed_size)
            else:
                compression_ratio = Decimal('1.0')

            compression_stats = CompressionStats(
                hypertable_name=hypertable_name,
                total_chunks=stats['total_chunks'],
                compressed_chunks=stats['compressed_chunks'],
                uncompressed_chunks=stats['uncompressed_chunks'],
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio.quantize(Decimal('0.01')),
                last_updated=datetime.now(timezone.utc)
            )

            logger.info(
                "Compression stats retrieved",
                hypertable=hypertable_name,
                compression_ratio=str(compression_ratio),
                space_saved_mb=compression_stats.space_saved_bytes / (1024 * 1024)
            )

            return compression_stats

        except Exception as e:
            logger.error("Failed to get compression stats", hypertable=hypertable_name, error=str(e))
            raise RuntimeError(f"Get compression stats failed: {e}")

    async def get_compressible_chunks(
        self,
        hypertable_name: str,
        older_than_days: int
    ) -> List[str]:
        """Get list of chunks eligible for compression.

        Args:
            hypertable_name: Name of hypertable
            older_than_days: Only return chunks older than this

        Returns:
            List of chunk names

        Raises:
            RuntimeError: If query fails
        """
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT chunk_name
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name = $1
                      AND NOT is_compressed
                      AND range_end < NOW() - INTERVAL '%s days'
                    ORDER BY range_end
                    """,
                    hypertable_name,
                    older_than_days
                )

            chunk_names = [row['chunk_name'] for row in rows]

            logger.debug(
                "Compressible chunks found",
                hypertable=hypertable_name,
                count=len(chunk_names)
            )

            return chunk_names

        except Exception as e:
            logger.error("Failed to get compressible chunks", error=str(e))
            raise RuntimeError(f"Get compressible chunks failed: {e}")

    async def compress_old_chunks(
        self,
        hypertable_name: str,
        older_than_days: int,
        batch_size: Optional[int] = None
    ) -> int:
        """Compress chunks older than specified days.

        Args:
            hypertable_name: Name of hypertable
            older_than_days: Compress chunks older than this
            batch_size: Optional limit on number of chunks to compress

        Returns:
            Number of chunks compressed

        Raises:
            RuntimeError: If compression fails
        """
        chunk_names = await self.get_compressible_chunks(hypertable_name, older_than_days)

        if batch_size:
            chunk_names = chunk_names[:batch_size]

        if not chunk_names:
            logger.info("No chunks to compress", hypertable=hypertable_name)
            return 0

        # Compress chunks in parallel (respecting max_parallel limit)
        compressed_count = 0

        for i in range(0, len(chunk_names), self.max_parallel):
            batch = chunk_names[i:i + self.max_parallel]

            tasks = [self.compress_chunk(chunk) for chunk in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for chunk, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("Chunk compression failed", chunk=chunk, error=str(result))
                else:
                    compressed_count += 1

        logger.info(
            "Old chunks compressed",
            hypertable=hypertable_name,
            compressed=compressed_count,
            failed=len(chunk_names) - compressed_count
        )

        return compressed_count
