"""Base database connection and query management for quantum trader.

Provides async PostgreSQL connection pooling with automatic retries,
health checks, and query performance monitoring.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from structlog import get_logger

logger = get_logger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class ConnectionError(DatabaseError):
    """Database connection error."""
    pass


class QueryError(DatabaseError):
    """Database query execution error."""
    pass


class DatabaseConnection:
    """Async PostgreSQL connection manager with connection pooling.

    Attributes:
        config: Configuration dictionary
        pool: AsyncPG connection pool
        _connected: Connection status flag
        _health_check_task: Background health check task
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize database connection manager.

        Args:
            config: Must contain:
                - db_url: PostgreSQL connection string
                - pool_min_size: Minimum pool size
                - pool_max_size: Maximum pool size
                - command_timeout: Query timeout in seconds
                - connection_timeout: Connection timeout in seconds
                - max_retries: Maximum connection retry attempts
                - retry_delay_ms: Initial retry delay
                - health_check_interval_seconds: Health check frequency

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.pool: Optional[Any] = None
        self._connected = False
        self._health_check_task: Optional[asyncio.Task] = None

        self.db_url = config['db_url']
        self.pool_min_size = config['pool_min_size']
        self.pool_max_size = config['pool_max_size']
        self.command_timeout = config['command_timeout']
        self.connection_timeout = config['connection_timeout']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay_ms'] / 1000.0
        self.health_check_interval = config['health_check_interval_seconds']

        logger.info(
            "DatabaseConnection initialized",
            pool_size=f"{self.pool_min_size}-{self.pool_max_size}"
        )

    def _validate_config(self) -> None:
        """Validate required configuration parameters."""
        required = [
            'db_url', 'pool_min_size', 'pool_max_size', 'command_timeout',
            'connection_timeout', 'max_retries', 'retry_delay_ms',
            'health_check_interval_seconds'
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    async def connect(self) -> None:
        """Establish database connection pool with retry logic.

        Raises:
            ConnectionError: If connection fails after all retries
        """
        import asyncpg

        for attempt in range(self.max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    command_timeout=self.command_timeout,
                    timeout=self.connection_timeout
                )

                # Verify connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')

                self._connected = True

                # Start health check task
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                logger.info(
                    "Database connected",
                    attempt=attempt + 1,
                    pool_size=self.pool.get_size()
                )
                return

            except Exception as e:
                logger.error(
                    "Database connection failed",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e)
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {e}")

    async def disconnect(self) -> None:
        """Gracefully close database connections."""
        self._connected = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self.pool:
            await self.pool.close()
            logger.info("Database disconnected")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool.

        Yields:
            Database connection

        Raises:
            ConnectionError: If pool not connected
        """
        if not self._connected or not self.pool:
            raise ConnectionError("Database not connected")

        async with self.pool.acquire() as connection:
            yield connection

    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """Execute a query without returning results.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Optional query timeout override

        Returns:
            Query status string

        Raises:
            QueryError: If query execution fails
        """
        try:
            async with self.acquire() as conn:
                result = await conn.execute(query, *args, timeout=timeout)
                logger.debug("Query executed", query=query[:100], result=result)
                return result

        except Exception as e:
            logger.error("Query execution failed", query=query[:100], error=str(e))
            raise QueryError(f"Execute failed: {e}")

    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Fetch all rows from a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Optional query timeout override

        Returns:
            List of result rows

        Raises:
            QueryError: If query execution fails
        """
        try:
            async with self.acquire() as conn:
                rows = await conn.fetch(query, *args, timeout=timeout)
                logger.debug("Query fetched", query=query[:100], rows=len(rows))
                return rows

        except Exception as e:
            logger.error("Query fetch failed", query=query[:100], error=str(e))
            raise QueryError(f"Fetch failed: {e}")

    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """Fetch a single row from a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Optional query timeout override

        Returns:
            Result row or None

        Raises:
            QueryError: If query execution fails
        """
        try:
            async with self.acquire() as conn:
                row = await conn.fetchrow(query, *args, timeout=timeout)
                logger.debug("Query fetchrow", query=query[:100], found=row is not None)
                return row

        except Exception as e:
            logger.error("Query fetchrow failed", query=query[:100], error=str(e))
            raise QueryError(f"Fetchrow failed: {e}")

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """Fetch a single value from a query.

        Args:
            query: SQL query to execute
            *args: Query parameters
            column: Column index to return
            timeout: Optional query timeout override

        Returns:
            Single value or None

        Raises:
            QueryError: If query execution fails
        """
        try:
            async with self.acquire() as conn:
                value = await conn.fetchval(query, *args, column=column, timeout=timeout)
                logger.debug("Query fetchval", query=query[:100], value=value)
                return value

        except Exception as e:
            logger.error("Query fetchval failed", query=query[:100], error=str(e))
            raise QueryError(f"Fetchval failed: {e}")

    async def executemany(
        self,
        query: str,
        args: List[Tuple],
        timeout: Optional[float] = None
    ) -> None:
        """Execute a query with multiple parameter sets.

        Args:
            query: SQL query to execute
            args: List of parameter tuples
            timeout: Optional query timeout override

        Raises:
            QueryError: If query execution fails
        """
        try:
            async with self.acquire() as conn:
                await conn.executemany(query, args, timeout=timeout)
                logger.debug("Query executemany", query=query[:100], count=len(args))

        except Exception as e:
            logger.error("Query executemany failed", query=query[:100], error=str(e))
            raise QueryError(f"Executemany failed: {e}")

    async def transaction(self):
        """Create a database transaction context.

        Yields:
            Transaction object

        Example:
            async with db.transaction() as tx:
                await tx.execute("INSERT ...")
                await tx.execute("UPDATE ...")
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def check_health(self) -> bool:
        """Check database connection health.

        Returns:
            True if database is healthy, False otherwise
        """
        if not self._connected or not self.pool:
            return False

        try:
            async with self.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._connected:
            try:
                await asyncio.sleep(self.health_check_interval)

                is_healthy = await self.check_health()

                if not is_healthy:
                    logger.warning("Database health check failed, attempting reconnection")
                    await self._reconnect()
                else:
                    logger.debug(
                        "Database healthy",
                        pool_size=self.pool.get_size(),
                        free_connections=self.pool.get_idle_size()
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))

    async def _reconnect(self) -> None:
        """Attempt to reconnect to database."""
        try:
            if self.pool:
                await self.pool.close()

            await self.connect()
            logger.info("Database reconnected successfully")

        except Exception as e:
            logger.error("Reconnection failed", error=str(e))

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.

        Returns:
            Dictionary with pool stats
        """
        if not self.pool:
            return {'connected': False}

        return {
            'connected': self._connected,
            'pool_size': self.pool.get_size(),
            'free_connections': self.pool.get_idle_size(),
            'min_size': self.pool_min_size,
            'max_size': self.pool_max_size
        }


class QueryBuilder:
    """Helper class for building safe parameterized queries."""

    @staticmethod
    def build_insert(
        table: str,
        columns: List[str],
        values_count: int = 1
    ) -> str:
        """Build INSERT query with parameters.

        Args:
            table: Table name
            columns: Column names
            values_count: Number of value sets

        Returns:
            Parameterized INSERT query
        """
        if not columns:
            raise ValueError("Columns cannot be empty")

        cols = ', '.join(columns)
        placeholders = ', '.join(
            f"({', '.join(f'${i * len(columns) + j + 1}' for j in range(len(columns)))})"
            for i in range(values_count)
        )

        return f"INSERT INTO {table} ({cols}) VALUES {placeholders}"

    @staticmethod
    def build_update(
        table: str,
        columns: List[str],
        where_column: str
    ) -> str:
        """Build UPDATE query with parameters.

        Args:
            table: Table name
            columns: Columns to update
            where_column: WHERE clause column

        Returns:
            Parameterized UPDATE query
        """
        if not columns:
            raise ValueError("Columns cannot be empty")

        set_clause = ', '.join(f"{col} = ${i + 1}" for i, col in enumerate(columns))
        where_param = f"${len(columns) + 1}"

        return f"UPDATE {table} SET {set_clause} WHERE {where_column} = {where_param}"

    @staticmethod
    def build_select(
        table: str,
        columns: List[str] = None,
        where_columns: List[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """Build SELECT query with parameters.

        Args:
            table: Table name
            columns: Columns to select (None = all)
            where_columns: WHERE clause columns
            order_by: ORDER BY clause
            limit: LIMIT value

        Returns:
            Parameterized SELECT query
        """
        cols = ', '.join(columns) if columns else '*'
        query = f"SELECT {cols} FROM {table}"

        if where_columns:
            where_clause = ' AND '.join(
                f"{col} = ${i + 1}" for i, col in enumerate(where_columns)
            )
            query += f" WHERE {where_clause}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit:
            query += f" LIMIT {limit}"

        return query
