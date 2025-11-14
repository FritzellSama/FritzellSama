"""Advanced database connection management with pooling and failover.

Provides production-ready connection pooling, health monitoring, automatic
reconnection, and connection lifecycle management for PostgreSQL/TimescaleDB.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from structlog import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """Database connection state."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


@dataclass
class ConnectionStats:
    """Connection pool statistics.

    Attributes:
        state: Current connection state
        pool_size: Current number of connections
        idle_connections: Number of idle connections
        active_connections: Number of active connections
        total_queries: Total queries executed
        failed_queries: Total failed queries
        avg_query_time_ms: Average query execution time
        uptime_seconds: Time since last connection
        last_error: Last connection error if any
    """
    state: ConnectionState
    pool_size: int
    idle_connections: int
    active_connections: int
    total_queries: int
    failed_queries: int
    avg_query_time_ms: Decimal
    uptime_seconds: int
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'state': self.state.value,
            'pool_size': self.pool_size,
            'idle_connections': self.idle_connections,
            'active_connections': self.active_connections,
            'total_queries': self.total_queries,
            'failed_queries': self.failed_queries,
            'avg_query_time_ms': str(self.avg_query_time_ms),
            'uptime_seconds': self.uptime_seconds,
            'last_error': self.last_error
        }


class ConnectionPool:
    """Production-ready async PostgreSQL connection pool.

    Features:
    - Automatic connection pooling with min/max size
    - Health monitoring and automatic reconnection
    - Query timeout and retry logic
    - Connection lifecycle callbacks
    - Comprehensive statistics tracking

    Attributes:
        config: Configuration dictionary
        pool: AsyncPG connection pool
        _state: Current connection state
        _stats_tracker: Query statistics tracker
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize connection pool.

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
                - enable_statement_cache: Enable prepared statement cache
                - max_cached_statements: Max cached statements

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.pool: Optional[Any] = None
        self._state = ConnectionState.DISCONNECTED
        self._health_check_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, List[Callable]] = {
            'on_connect': [],
            'on_disconnect': [],
            'on_error': []
        }

        # Stats tracking
        self._connection_time: Optional[datetime] = None
        self._total_queries = 0
        self._failed_queries = 0
        self._query_times: List[float] = []
        self._last_error: Optional[str] = None

        # Config values
        self.db_url = config['db_url']
        self.pool_min_size = config['pool_min_size']
        self.pool_max_size = config['pool_max_size']
        self.command_timeout = config['command_timeout']
        self.connection_timeout = config['connection_timeout']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay_ms'] / 1000.0
        self.health_check_interval = config['health_check_interval_seconds']
        self.enable_statement_cache = config.get('enable_statement_cache', True)
        self.max_cached_statements = config.get('max_cached_statements', 100)

        logger.info(
            "ConnectionPool initialized",
            pool_size=f"{self.pool_min_size}-{self.pool_max_size}",
            timeout=self.command_timeout
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = [
            'db_url', 'pool_min_size', 'pool_max_size', 'command_timeout',
            'connection_timeout', 'max_retries', 'retry_delay_ms',
            'health_check_interval_seconds'
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(f"Missing required config: {missing}")

        if self.config['pool_min_size'] > self.config['pool_max_size']:
            raise ValueError("pool_min_size cannot exceed pool_max_size")

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for connection events.

        Args:
            event: Event name ('on_connect', 'on_disconnect', 'on_error')
            callback: Async callback function

        Raises:
            ValueError: If invalid event name
        """
        if event not in self._callbacks:
            raise ValueError(f"Invalid event: {event}. Must be one of {list(self._callbacks.keys())}")

        self._callbacks[event].append(callback)
        logger.debug("Callback registered", event=event)

    async def _execute_callbacks(self, event: str, *args, **kwargs) -> None:
        """Execute all callbacks for an event.

        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error("Callback execution failed", event=event, error=str(e))

    async def connect(self) -> None:
        """Establish database connection pool with retry logic.

        Raises:
            RuntimeError: If connection fails after all retries
        """
        import asyncpg

        self._state = ConnectionState.CONNECTING

        for attempt in range(self.max_retries):
            try:
                # Create connection pool
                self.pool = await asyncpg.create_pool(
                    self.db_url,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    command_timeout=self.command_timeout,
                    timeout=self.connection_timeout,
                    max_cached_statement_lifetime=0 if not self.enable_statement_cache else 300,
                    max_cacheable_statement_size=self.max_cached_statements if self.enable_statement_cache else 0
                )

                # Verify connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')

                    # Check if TimescaleDB is available
                    has_timescale = await conn.fetchval(
                        "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
                    )
                    if has_timescale:
                        logger.info("TimescaleDB extension detected")

                self._state = ConnectionState.CONNECTED
                self._connection_time = datetime.now(timezone.utc)
                self._last_error = None

                # Start health check task
                self._health_check_task = asyncio.create_task(self._health_check_loop())

                # Execute on_connect callbacks
                await self._execute_callbacks('on_connect', self)

                logger.info(
                    "Database connected",
                    attempt=attempt + 1,
                    pool_size=self.pool.get_size()
                )
                return

            except Exception as e:
                self._last_error = str(e)
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
                    self._state = ConnectionState.FAILED
                    await self._execute_callbacks('on_error', self, e)
                    raise RuntimeError(f"Connection failed after {self.max_retries} attempts: {e}")

    async def disconnect(self) -> None:
        """Gracefully close all database connections."""
        logger.info("Disconnecting from database")

        self._state = ConnectionState.DISCONNECTED

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close pool
        if self.pool:
            await self.pool.close()
            logger.info("Connection pool closed")

        # Execute on_disconnect callbacks
        await self._execute_callbacks('on_disconnect', self)

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool.

        Yields:
            Database connection

        Raises:
            RuntimeError: If pool not connected
        """
        if self._state != ConnectionState.CONNECTED or not self.pool:
            raise RuntimeError("Connection pool not connected")

        start_time = asyncio.get_event_loop().time()

        try:
            async with self.pool.acquire() as connection:
                yield connection

            # Track successful query
            query_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._total_queries += 1
            self._query_times.append(query_time)

            # Keep only last 1000 query times for averaging
            if len(self._query_times) > 1000:
                self._query_times = self._query_times[-1000:]

        except Exception as e:
            self._failed_queries += 1
            logger.error("Query execution failed", error=str(e))
            raise

    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """Execute a query without returning results.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            Query status string
        """
        async with self.acquire() as conn:
            result = await conn.execute(query, *args, timeout=timeout or self.command_timeout)
            logger.debug("Query executed", query=query[:100])
            return result

    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Fetch all rows from a query.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            List of result rows
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(query, *args, timeout=timeout or self.command_timeout)
            logger.debug("Query fetched", query=query[:100], rows=len(rows))
            return rows

    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """Fetch a single row from a query.

        Args:
            query: SQL query
            *args: Query parameters
            timeout: Optional timeout override

        Returns:
            Result row or None
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow(query, *args, timeout=timeout or self.command_timeout)
            logger.debug("Query fetchrow", query=query[:100], found=row is not None)
            return row

    async def fetchval(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Optional[Any]:
        """Fetch a single value from a query.

        Args:
            query: SQL query
            *args: Query parameters
            column: Column index to return
            timeout: Optional timeout override

        Returns:
            Single value or None
        """
        async with self.acquire() as conn:
            value = await conn.fetchval(
                query, *args,
                column=column,
                timeout=timeout or self.command_timeout
            )
            logger.debug("Query fetchval", query=query[:100])
            return value

    async def check_health(self) -> bool:
        """Check database connection health.

        Returns:
            True if healthy, False otherwise
        """
        if self._state != ConnectionState.CONNECTED or not self.pool:
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
        while self._state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.health_check_interval)

                is_healthy = await self.check_health()

                if not is_healthy:
                    logger.warning("Database unhealthy, attempting reconnection")
                    await self._reconnect()
                else:
                    logger.debug(
                        "Database healthy",
                        pool_size=self.pool.get_size() if self.pool else 0,
                        idle=self.pool.get_idle_size() if self.pool else 0
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))

    async def _reconnect(self) -> None:
        """Attempt to reconnect to database."""
        self._state = ConnectionState.RECONNECTING

        try:
            if self.pool:
                await self.pool.close()

            await self.connect()
            logger.info("Database reconnected successfully")

        except Exception as e:
            self._state = ConnectionState.FAILED
            self._last_error = str(e)
            logger.error("Reconnection failed", error=str(e))
            await self._execute_callbacks('on_error', self, e)

    def get_stats(self) -> ConnectionStats:
        """Get connection pool statistics.

        Returns:
            ConnectionStats object
        """
        if not self.pool:
            return ConnectionStats(
                state=self._state,
                pool_size=0,
                idle_connections=0,
                active_connections=0,
                total_queries=self._total_queries,
                failed_queries=self._failed_queries,
                avg_query_time_ms=Decimal('0'),
                uptime_seconds=0,
                last_error=self._last_error
            )

        # Calculate average query time
        if self._query_times:
            avg_time = sum(self._query_times) / len(self._query_times)
        else:
            avg_time = 0

        # Calculate uptime
        if self._connection_time:
            uptime = (datetime.now(timezone.utc) - self._connection_time).total_seconds()
        else:
            uptime = 0

        return ConnectionStats(
            state=self._state,
            pool_size=self.pool.get_size(),
            idle_connections=self.pool.get_idle_size(),
            active_connections=self.pool.get_size() - self.pool.get_idle_size(),
            total_queries=self._total_queries,
            failed_queries=self._failed_queries,
            avg_query_time_ms=Decimal(str(avg_time)).quantize(Decimal('0.01')),
            uptime_seconds=int(uptime),
            last_error=self._last_error
        )

    @property
    def is_connected(self) -> bool:
        """Check if pool is connected.

        Returns:
            True if connected, False otherwise
        """
        return self._state == ConnectionState.CONNECTED

    @property
    def state(self) -> ConnectionState:
        """Get current connection state.

        Returns:
            ConnectionState enum
        """
        return self._state
