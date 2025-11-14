"""TimescaleDB setup and hypertable configuration.

This module provides functions to configure TimescaleDB hypertables,
continuous aggregates, and retention policies for high-performance
time-series data storage.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional
from datetime import timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from structlog import get_logger

logger = get_logger(__name__)


class TimescaleSetup:
    """TimescaleDB configuration and setup manager.

    Manages hypertable creation, continuous aggregates, compression,
    and retention policies for optimal time-series performance.

    Example:
        >>> setup = TimescaleSetup(engine, config)
        >>> await setup.initialize()
    """

    def __init__(self, engine: AsyncEngine, config: Dict[str, Any]) -> None:
        """Initialize TimescaleDB setup manager.

        Args:
            engine: SQLAlchemy async engine
            config: TimescaleDB configuration

        Example:
            >>> config = {
            ...     'chunk_time_interval': os.getenv('TIMESCALE_CHUNK_INTERVAL', '1 day'),
            ...     'compression_enabled': os.getenv('TIMESCALE_COMPRESSION', 'true') == 'true',
            ...     'retention_days': int(os.getenv('TIMESCALE_RETENTION_DAYS', '365'))
            ... }
        """
        self.engine = engine
        self.config = config
        logger.info("TimescaleDB setup manager initialized")

    async def initialize(self) -> None:
        """Initialize all TimescaleDB features.

        Sets up hypertables, compression, continuous aggregates, and retention policies.

        Example:
            >>> await setup.initialize()
        """
        try:
            # Create TimescaleDB extension
            await self._create_extension()

            # Create hypertables
            await self._create_hypertables()

            # Setup compression
            if self.config.get("compression_enabled", True):
                await self._setup_compression()

            # Create continuous aggregates
            await self._create_continuous_aggregates()

            # Setup retention policies
            await self._setup_retention_policies()

            logger.info("TimescaleDB initialization complete")

        except Exception as e:
            logger.error("TimescaleDB initialization failed", error=str(e))
            raise

    async def _create_extension(self) -> None:
        """Create TimescaleDB extension if not exists.

        Example:
            >>> await setup._create_extension()
        """
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            logger.info("TimescaleDB extension created")
        except Exception as e:
            logger.error("Failed to create TimescaleDB extension", error=str(e))
            raise

    async def _create_hypertables(self) -> None:
        """Convert tables to hypertables for time-series optimization.

        Converts trades, orders, positions, and performance_metrics tables
        to TimescaleDB hypertables partitioned by timestamp.

        Example:
            >>> await setup._create_hypertables()
        """
        chunk_interval = self.config.get(
            "chunk_time_interval",
            os.getenv("TIMESCALE_CHUNK_INTERVAL", "1 day")
        )

        hypertables = [
            {
                "table": "trades",
                "time_column": "timestamp",
                "chunk_interval": chunk_interval,
            },
            {
                "table": "orders",
                "time_column": "timestamp",
                "chunk_interval": chunk_interval,
            },
            {
                "table": "positions",
                "time_column": "opened_at",
                "chunk_interval": chunk_interval,
            },
            {
                "table": "performance_metrics",
                "time_column": "timestamp",
                "chunk_interval": chunk_interval,
            },
        ]

        try:
            async with self.engine.begin() as conn:
                for ht in hypertables:
                    # Check if already a hypertable
                    check_query = text("""
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.hypertables
                            WHERE hypertable_name = :table_name
                        )
                    """)
                    result = await conn.execute(check_query, {"table_name": ht["table"]})
                    exists = result.scalar()

                    if not exists:
                        # Create hypertable
                        create_query = text(f"""
                            SELECT create_hypertable(
                                '{ht["table"]}',
                                '{ht["time_column"]}',
                                chunk_time_interval => INTERVAL '{ht["chunk_interval"]}',
                                if_not_exists => TRUE
                            )
                        """)
                        await conn.execute(create_query)
                        logger.info("Hypertable created", table=ht["table"])
                    else:
                        logger.info("Hypertable already exists", table=ht["table"])

        except Exception as e:
            logger.error("Failed to create hypertables", error=str(e))
            raise

    async def _setup_compression(self) -> None:
        """Setup compression for hypertables.

        Enables compression on hypertables to reduce storage and improve
        query performance on historical data.

        Example:
            >>> await setup._setup_compression()
        """
        compression_after = self.config.get(
            "compression_after",
            os.getenv("TIMESCALE_COMPRESSION_AFTER", "7 days")
        )

        compression_configs = [
            {
                "table": "trades",
                "segment_by": "symbol, exchange",
                "order_by": "timestamp DESC",
            },
            {
                "table": "orders",
                "segment_by": "symbol, exchange",
                "order_by": "timestamp DESC",
            },
            {
                "table": "positions",
                "segment_by": "symbol, exchange",
                "order_by": "opened_at DESC",
            },
            {
                "table": "performance_metrics",
                "segment_by": "strategy, exchange",
                "order_by": "timestamp DESC",
            },
        ]

        try:
            async with self.engine.begin() as conn:
                for config in compression_configs:
                    # Enable compression
                    compress_query = text(f"""
                        ALTER TABLE {config["table"]} SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = '{config["segment_by"]}',
                            timescaledb.compress_orderby = '{config["order_by"]}'
                        )
                    """)
                    await conn.execute(compress_query)

                    # Add compression policy
                    policy_query = text(f"""
                        SELECT add_compression_policy(
                            '{config["table"]}',
                            INTERVAL '{compression_after}',
                            if_not_exists => TRUE
                        )
                    """)
                    await conn.execute(policy_query)

                    logger.info("Compression enabled", table=config["table"])

        except Exception as e:
            logger.error("Failed to setup compression", error=str(e))
            # Don't raise - compression is optional
            logger.warning("Continuing without compression")

    async def _create_continuous_aggregates(self) -> None:
        """Create continuous aggregates for pre-computed metrics.

        Creates materialized views for common aggregations like hourly/daily
        trading metrics for improved query performance.

        Example:
            >>> await setup._create_continuous_aggregates()
        """
        aggregates = [
            {
                "name": "hourly_trade_metrics",
                "table": "trades",
                "interval": "1 hour",
                "sql": """
                    SELECT
                        time_bucket('1 hour', timestamp) AS bucket,
                        symbol,
                        exchange,
                        strategy,
                        COUNT(*) as trade_count,
                        SUM(quantity) as total_volume,
                        AVG(price) as avg_price,
                        MIN(price) as low_price,
                        MAX(price) as high_price,
                        SUM(fee) as total_fees
                    FROM trades
                    GROUP BY bucket, symbol, exchange, strategy
                """,
            },
            {
                "name": "daily_performance_metrics",
                "table": "performance_metrics",
                "interval": "1 day",
                "sql": """
                    SELECT
                        time_bucket('1 day', timestamp) AS bucket,
                        strategy,
                        exchange,
                        AVG(total_pnl) as avg_pnl,
                        AVG(sharpe_ratio) as avg_sharpe,
                        AVG(win_rate) as avg_win_rate,
                        SUM(total_trades) as total_trades
                    FROM performance_metrics
                    GROUP BY bucket, strategy, exchange
                """,
            },
        ]

        try:
            async with self.engine.begin() as conn:
                for agg in aggregates:
                    # Check if continuous aggregate exists
                    check_query = text("""
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.continuous_aggregates
                            WHERE view_name = :view_name
                        )
                    """)
                    result = await conn.execute(check_query, {"view_name": agg["name"]})
                    exists = result.scalar()

                    if not exists:
                        # Create continuous aggregate
                        create_query = text(f"""
                            CREATE MATERIALIZED VIEW {agg["name"]}
                            WITH (timescaledb.continuous) AS
                            {agg["sql"]}
                            WITH NO DATA
                        """)
                        await conn.execute(create_query)

                        # Add refresh policy
                        refresh_query = text(f"""
                            SELECT add_continuous_aggregate_policy(
                                '{agg["name"]}',
                                start_offset => INTERVAL '3 days',
                                end_offset => INTERVAL '1 hour',
                                schedule_interval => INTERVAL '1 hour',
                                if_not_exists => TRUE
                            )
                        """)
                        await conn.execute(refresh_query)

                        logger.info("Continuous aggregate created", name=agg["name"])
                    else:
                        logger.info("Continuous aggregate already exists", name=agg["name"])

        except Exception as e:
            logger.error("Failed to create continuous aggregates", error=str(e))
            # Don't raise - aggregates are optional
            logger.warning("Continuing without continuous aggregates")

    async def _setup_retention_policies(self) -> None:
        """Setup data retention policies.

        Configures automatic data retention to manage storage by dropping
        old chunks based on configured retention period.

        Example:
            >>> await setup._setup_retention_policies()
        """
        retention_days = self.config.get(
            "retention_days",
            int(os.getenv("TIMESCALE_RETENTION_DAYS", "365"))
        )

        tables = ["trades", "orders", "positions", "performance_metrics"]

        try:
            async with self.engine.begin() as conn:
                for table in tables:
                    # Add retention policy
                    policy_query = text(f"""
                        SELECT add_retention_policy(
                            '{table}',
                            INTERVAL '{retention_days} days',
                            if_not_exists => TRUE
                        )
                    """)
                    await conn.execute(policy_query)
                    logger.info(
                        "Retention policy added",
                        table=table,
                        retention_days=retention_days
                    )

        except Exception as e:
            logger.error("Failed to setup retention policies", error=str(e))
            # Don't raise - retention is optional
            logger.warning("Continuing without retention policies")


async def initialize_timescale(engine: AsyncEngine, config: Dict[str, Any]) -> None:
    """Initialize TimescaleDB with all features.

    Args:
        engine: SQLAlchemy async engine
        config: TimescaleDB configuration

    Example:
        >>> await initialize_timescale(engine, config)
    """
    setup = TimescaleSetup(engine, config)
    await setup.initialize()
