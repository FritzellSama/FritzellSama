"""
Connection pool manager for exchange HTTP connections.

This module provides efficient connection pooling and management for HTTP
requests to cryptocurrency exchanges, including rate limiting, retries,
and connection lifecycle management.
"""

import asyncio
from typing import Optional, Dict, Any
from decimal import Decimal
import os
import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


class ConnectionPool:
    """Manages HTTP connection pooling for exchange APIs.

    Provides efficient connection reuse, automatic retries with exponential
    backoff, rate limiting, and proper resource cleanup.

    Attributes:
        pool_size: Maximum number of connections per host
        timeout: Request timeout in seconds
        rate_limit: Maximum requests per second
        session: aiohttp ClientSession
    """

    def __init__(
        self,
        pool_size: Optional[int] = None,
        timeout: Optional[int] = None,
        rate_limit: Optional[int] = None
    ) -> None:
        """Initialize connection pool.

        Args:
            pool_size: Max connections per host (from config if None)
            timeout: Request timeout in seconds (from config if None)
            rate_limit: Max requests per second (from config if None)
        """
        # Load from config/env vars - NO hardcoded values
        self.pool_size = pool_size or int(os.getenv('EXCHANGE_POOL_SIZE', '100'))
        self.timeout = timeout or int(os.getenv('EXCHANGE_TIMEOUT', '30'))
        self.rate_limit = rate_limit or int(os.getenv('EXCHANGE_RATE_LIMIT', '10'))

        self.session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(self.rate_limit)
        self._lock = asyncio.Lock()

        logger.info(
            "Connection pool initialized",
            pool_size=self.pool_size,
            timeout=self.timeout,
            rate_limit=self.rate_limit
        )

    async def create_session(self) -> None:
        """Create aiohttp session with connection pooling.

        Raises:
            RuntimeError: If session already exists
        """
        async with self._lock:
            if self.session is not None:
                logger.warning("Session already exists")
                return

            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                limit_per_host=int(os.getenv('EXCHANGE_POOL_SIZE_PER_HOST', '20')),
                ttl_dns_cache=int(os.getenv('DNS_CACHE_TTL', '300')),
                enable_cleanup_closed=True
            )

            timeout_config = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=int(os.getenv('EXCHANGE_CONNECT_TIMEOUT', '10')),
                sock_read=int(os.getenv('EXCHANGE_READ_TIMEOUT', '30'))
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={
                    'User-Agent': os.getenv('USER_AGENT', 'QuantumTrader/1.0'),
                    'Content-Type': 'application/json'
                }
            )

            logger.info("HTTP session created successfully")

    async def close_session(self) -> None:
        """Close aiohttp session and cleanup resources."""
        async with self._lock:
            if self.session is None:
                return

            if not self.session.closed:
                await self.session.close()
                # Wait for graceful shutdown
                await asyncio.sleep(float(os.getenv('SESSION_CLOSE_DELAY', '0.25')))

            self.session = None
            logger.info("HTTP session closed")

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            headers: Optional headers
            params: Optional query parameters
            json_data: Optional JSON body
            retries: Number of retry attempts (from config if None)

        Returns:
            Response JSON as dictionary

        Raises:
            aiohttp.ClientError: If request fails after retries
            asyncio.TimeoutError: If request times out
        """
        if self.session is None:
            await self.create_session()

        max_retries = retries or int(os.getenv('MAX_RETRIES', '3'))
        retry_delay = float(os.getenv('INITIAL_RETRY_DELAY', '1.0'))

        for attempt in range(max_retries + 1):
            try:
                # Rate limiting
                async with self._rate_limiter:
                    async with self.session.request(
                        method=method,
                        url=url,
                        headers=headers,
                        params=params,
                        json=json_data
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                        logger.debug(
                            "Request successful",
                            method=method,
                            url=url,
                            status=response.status,
                            attempt=attempt + 1
                        )

                        return data

            except aiohttp.ClientResponseError as e:
                if e.status >= 500 and attempt < max_retries:
                    # Server error - retry with exponential backoff
                    wait_time = retry_delay * (int(os.getenv('BACKOFF_MULTIPLIER', '2')) ** attempt)
                    logger.warning(
                        "Server error, retrying",
                        status=e.status,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
                    continue
                elif e.status == 429:
                    # Rate limited - wait and retry
                    rate_limit_wait = float(os.getenv('RATE_LIMIT_WAIT', '5.0'))
                    logger.warning(
                        "Rate limited, waiting",
                        wait_time=rate_limit_wait,
                        attempt=attempt + 1
                    )
                    await asyncio.sleep(rate_limit_wait)
                    if attempt < max_retries:
                        continue
                else:
                    logger.error(
                        "Request failed with client error",
                        method=method,
                        url=url,
                        status=e.status,
                        error=str(e)
                    )
                    raise

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    wait_time = retry_delay * (int(os.getenv('BACKOFF_MULTIPLIER', '2')) ** attempt)
                    logger.warning(
                        "Request timeout, retrying",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        wait_time=wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "Request timed out after retries",
                        method=method,
                        url=url,
                        attempts=attempt + 1
                    )
                    raise

            except aiohttp.ClientError as e:
                if attempt < max_retries:
                    wait_time = retry_delay * (int(os.getenv('BACKOFF_MULTIPLIER', '2')) ** attempt)
                    logger.warning(
                        "Connection error, retrying",
                        method=method,
                        url=url,
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e)
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "Request failed after retries",
                        method=method,
                        url=url,
                        attempts=attempt + 1,
                        error=str(e)
                    )
                    raise

        raise RuntimeError(f"Request failed after {max_retries} retries")

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GET request.

        Args:
            url: Request URL
            headers: Optional headers
            params: Optional query parameters

        Returns:
            Response JSON
        """
        return await self.request('GET', url, headers=headers, params=params)

    async def post(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make POST request.

        Args:
            url: Request URL
            headers: Optional headers
            json_data: Optional JSON body

        Returns:
            Response JSON
        """
        return await self.request('POST', url, headers=headers, json_data=json_data)

    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make DELETE request.

        Args:
            url: Request URL
            headers: Optional headers
            params: Optional query parameters

        Returns:
            Response JSON
        """
        return await self.request('DELETE', url, headers=headers, params=params)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConnectionPool(pool_size={self.pool_size}, "
            f"timeout={self.timeout}, rate_limit={self.rate_limit})"
        )
