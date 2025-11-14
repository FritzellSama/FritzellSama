"""Database session management for Quantum Trader AI.

This module provides async database session management using SQLAlchemy
and asyncpg for high-performance PostgreSQL/TimescaleDB connections.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from decimal import Decimal

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text
from structlog import get_logger

logger = get_logger(__name__)


class DatabaseSession:
    """Async database session manager.

    Manages database connections, session lifecycle, and connection pooling
    for high-performance async operations.

    Attributes:
        engine: SQLAlchemy async engine
        session_factory: Async session factory
        config: Database configuration

    Example:
        >>> db = DatabaseSession(config)
        >>> await db.initialize()
        >>> async with db.get_session() as session:
        ...     result = await session.execute(query)
        >>> await db.close()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize database session manager.

        Args:
            config: Database configuration dictionary

        Example:
            >>> config = {
            ...     'host': os.getenv('DB_HOST', 'localhost'),
            ...     'port': int(os.getenv('DB_PORT', '5432')),
            ...     'database': os.getenv('DB_NAME', 'quantum_trader'),
            ...     'user': os.getenv('DB_USER', 'trader'),
            ...     'password': os.getenv('DB_PASSWORD'),
            ... }
            >>> db = DatabaseSession(config)
        """
        self.config = config
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._initialized = False

        logger.info("Database session manager created", database=config.get("database"))

    async def initialize(self) -> None:
        """Initialize database engine and session factory.

        Creates async engine with connection pooling and session factory.

        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If database connection fails

        Example:
            >>> await db.initialize()
        """
        try:
            # Validate configuration
            self._validate_config()

            # Build connection URL
            db_url = self._build_connection_url()

            # Get pool configuration from config
            pool_size = self.config.get("pool_size", int(os.getenv("DB_POOL_SIZE", "20")))
            max_overflow = self.config.get("max_overflow", int(os.getenv("DB_MAX_OVERFLOW", "10")))
            pool_timeout = self.config.get("pool_timeout", int(os.getenv("DB_POOL_TIMEOUT", "30")))
            pool_recycle = self.config.get("pool_recycle", int(os.getenv("DB_POOL_RECYCLE", "3600")))

            # Create async engine
            self.engine = create_async_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # Verify connections before using
                echo=self.config.get("echo", False),
                future=True,
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False,
            )

            # Test connection
            await self._test_connection()

            self._initialized = True
            logger.info(
                "Database initialized",
                database=self.config.get("database"),
                pool_size=pool_size,
                max_overflow=max_overflow,
            )

        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise ConnectionError(f"Database initialization failed: {e}") from e

    async def close(self) -> None:
        """Close database engine and cleanup resources.

        Example:
            >>> await db.close()
        """
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database connection closed")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session context manager.

        Yields:
            AsyncSession: Database session

        Raises:
            RuntimeError: If database not initialized

        Example:
            >>> async with db.get_session() as session:
            ...     result = await session.execute(query)
            ...     await session.commit()
        """
        if not self._initialized or not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Session error, rolling back", error=str(e))
            raise
        finally:
            await session.close()

    async def execute_raw(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result

        Example:
            >>> result = await db.execute_raw(
            ...     "SELECT * FROM trades WHERE symbol = :symbol",
            ...     {"symbol": "BTC/USDT"}
            ... )
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result

    def _validate_config(self) -> None:
        """Validate database configuration.

        Raises:
            ValueError: If required configuration is missing
        """
        required_fields = ["host", "port", "database", "user", "password"]
        missing = [f for f in required_fields if not self.config.get(f)]

        if missing:
            raise ValueError(f"Missing required database configuration: {missing}")

    def _build_connection_url(self) -> str:
        """Build async PostgreSQL connection URL.

        Returns:
            Connection URL string

        Example:
            >>> url = db._build_connection_url()
            'postgresql+asyncpg://user:pass@localhost:5432/dbname'
        """
        host = self.config["host"]
        port = self.config["port"]
        database = self.config["database"]
        user = self.config["user"]
        password = self.config["password"]

        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"

    async def _test_connection(self) -> None:
        """Test database connection.

        Raises:
            ConnectionError: If connection test fails
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()
                logger.info("Database connection test successful")
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            raise ConnectionError(f"Database connection test failed: {e}") from e


async def get_database_session(config: Dict[str, Any]) -> DatabaseSession:
    """Factory function to create and initialize database session.

    Args:
        config: Database configuration

    Returns:
        Initialized DatabaseSession instance

    Example:
        >>> config = load_db_config()
        >>> db = await get_database_session(config)
    """
    db = DatabaseSession(config)
    await db.initialize()
    return db


async def create_all_tables(engine: AsyncEngine) -> None:
    """Create all database tables.

    Args:
        engine: SQLAlchemy async engine

    Example:
        >>> await create_all_tables(engine)
    """
    from quantum_trader.database.models.orders import Base as OrderBase
    from quantum_trader.database.models.trades import Base as TradeBase
    from quantum_trader.database.models.positions import Base as PositionBase
    from quantum_trader.database.models.performance import Base as PerformanceBase
    from quantum_trader.database.models.strategies import Base as StrategyBase

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(OrderBase.metadata.create_all)
        await conn.run_sync(TradeBase.metadata.create_all)
        await conn.run_sync(PositionBase.metadata.create_all)
        await conn.run_sync(PerformanceBase.metadata.create_all)
        await conn.run_sync(StrategyBase.metadata.create_all)

    logger.info("All database tables created")
