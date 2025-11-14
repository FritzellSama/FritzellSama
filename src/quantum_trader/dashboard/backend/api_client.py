"""Exchange API client with async HTTP, rate limiting, and retry logic.

Production-ready HTTP client for exchange API interactions with:
- Connection pooling and keep-alive
- Token bucket rate limiting
- Exponential backoff retry logic
- HMAC-SHA256 request signing
- Response validation and metrics
"""

import asyncio
import hashlib
import hmac
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from structlog import get_logger

logger = get_logger(__name__)


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception."""
    pass


class APIError(Exception):
    """API error exception."""
    pass


class TokenBucket:
    """Token bucket rate limiter for API requests.

    Attributes:
        capacity: Maximum tokens in bucket
        fill_rate: Tokens added per second
        tokens: Current token count
        last_update: Last token refill timestamp
    """

    def __init__(self, capacity: int, fill_rate: Decimal) -> None:
        """Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity
            fill_rate: Tokens per second refill rate
        """
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = Decimal(str(capacity))
        self.last_update = Decimal(str(time.time()))
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        async with self._lock:
            await self._refill()

            if self.tokens >= Decimal(str(tokens)):
                self.tokens -= Decimal(str(tokens))
                return True
            return False

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = Decimal(str(time.time()))
        elapsed = now - self.last_update
        tokens_to_add = elapsed * self.fill_rate

        self.tokens = min(
            Decimal(str(self.capacity)),
            self.tokens + tokens_to_add
        )
        self.last_update = now


class APIClient:
    """Async HTTP client for exchange APIs.

    Features:
    - Connection pooling with configurable limits
    - Token bucket rate limiting
    - Exponential backoff retry (max 3 attempts)
    - HMAC-SHA256 request signing
    - Response validation
    - Metrics collection

    Attributes:
        config: Client configuration
        session: aiohttp ClientSession
        rate_limiter: Token bucket rate limiter
        metrics: Request metrics
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize API client.

        Args:
            config: Configuration dict with:
                - base_url: API base URL
                - api_key: API key
                - api_secret: API secret
                - rate_limit_capacity: Token bucket capacity
                - rate_limit_fill_rate: Tokens per second
                - max_connections: Max connection pool size
                - timeout: Request timeout in seconds

        Raises:
            ValueError: If required config missing
        """
        self.config = config
        self._validate_config()

        self.base_url = config['base_url']
        self.api_key = config['api_key']
        self.api_secret = config['api_secret']
        self.timeout = config.get('timeout', 30)

        # Rate limiter
        self.rate_limiter = TokenBucket(
            capacity=config['rate_limit_capacity'],
            fill_rate=Decimal(str(config['rate_limit_fill_rate']))
        )

        # Metrics
        self.metrics = {
            'total_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'rate_limited_requests': 0
        }

        # Session initialized on connect
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(
            "api_client_initialized",
            base_url=self.base_url,
            rate_limit_capacity=config['rate_limit_capacity']
        )

    def _validate_config(self) -> None:
        """Validate required configuration.

        Raises:
            ValueError: If required config missing
        """
        required = [
            'base_url', 'api_key', 'api_secret',
            'rate_limit_capacity', 'rate_limit_fill_rate'
        ]

        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config: {key}")

    async def connect(self) -> None:
        """Initialize HTTP session with connection pooling."""
        if self.session is not None:
            return

        connector = aiohttp.TCPConnector(
            limit=self.config.get('max_connections', 100),
            limit_per_host=self.config.get('max_connections_per_host', 10),
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'QuantumTraderAI/1.0'}
        )

        logger.info("api_client_connected")

    async def disconnect(self) -> None:
        """Close HTTP session gracefully."""
        if self.session is not None:
            await self.session.close()
            self.session = None
            logger.info("api_client_disconnected")

    def _sign_request(self, method: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Sign request with HMAC-SHA256.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters

        Returns:
            Headers with signature
        """
        timestamp = str(int(time.time() * 1000))

        # Create signature payload
        query_string = urlencode(sorted(params.items())) if params else ""
        payload = f"{timestamp}{method}{endpoint}{query_string}"

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'X-API-Key': self.api_key,
            'X-Timestamp': timestamp,
            'X-Signature': signature
        }

    async def _wait_for_rate_limit(self, max_wait: int = 5) -> None:
        """Wait for rate limit token availability.

        Args:
            max_wait: Maximum wait time in seconds

        Raises:
            RateLimitExceeded: If max wait time exceeded
        """
        wait_time = Decimal('0')
        sleep_interval = Decimal('0.1')

        while wait_time < Decimal(str(max_wait)):
            if await self.rate_limiter.consume():
                return

            await asyncio.sleep(float(sleep_interval))
            wait_time += sleep_interval
            self.metrics['rate_limited_requests'] += 1

        raise RateLimitExceeded(f"Rate limit exceeded, waited {max_wait}s")

    async def _execute_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """Execute HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            signed: Whether to sign request

        Returns:
            Response data as dict

        Raises:
            APIError: If request fails after retries
        """
        if self.session is None:
            await self.connect()

        params = params or {}
        url = f"{self.base_url}{endpoint}"

        # Wait for rate limit
        await self._wait_for_rate_limit()

        # Sign request if needed
        headers = {}
        if signed:
            headers = self._sign_request(method, endpoint, params)

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = Decimal('1')

        for attempt in range(max_retries):
            try:
                self.metrics['total_requests'] += 1

                async with self.session.request(
                    method,
                    url,
                    params=params,
                    json=data,
                    headers=headers
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        logger.debug(
                            "api_request_success",
                            method=method,
                            endpoint=endpoint,
                            status=response.status
                        )
                        return response_data

                    # Handle specific error codes
                    if response.status == 429:  # Rate limit
                        if attempt < max_retries - 1:
                            delay = base_delay * (Decimal('2') ** attempt)
                            logger.warning(
                                "api_rate_limit_retry",
                                attempt=attempt + 1,
                                delay=float(delay)
                            )
                            await asyncio.sleep(float(delay))
                            self.metrics['retried_requests'] += 1
                            continue

                    # Other errors
                    logger.error(
                        "api_request_failed",
                        method=method,
                        endpoint=endpoint,
                        status=response.status,
                        response=response_data
                    )
                    self.metrics['failed_requests'] += 1
                    raise APIError(f"API error {response.status}: {response_data}")

            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (Decimal('2') ** attempt)
                    logger.warning(
                        "api_request_retry",
                        error=str(e),
                        attempt=attempt + 1,
                        delay=float(delay)
                    )
                    await asyncio.sleep(float(delay))
                    self.metrics['retried_requests'] += 1
                    continue

                logger.error("api_request_error", error=str(e), method=method, endpoint=endpoint)
                self.metrics['failed_requests'] += 1
                raise APIError(f"Request failed after {max_retries} attempts: {e}")

        raise APIError(f"Request failed after {max_retries} attempts")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """Execute GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            signed: Whether to sign request

        Returns:
            Response data
        """
        return await self._execute_request('GET', endpoint, params=params, signed=signed)

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """Execute POST request.

        Args:
            endpoint: API endpoint
            data: Request body data
            signed: Whether to sign request

        Returns:
            Response data
        """
        return await self._execute_request('POST', endpoint, data=data, signed=signed)

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Dict[str, Any]:
        """Execute DELETE request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            signed: Whether to sign request

        Returns:
            Response data
        """
        return await self._execute_request('DELETE', endpoint, params=params, signed=signed)

    def get_metrics(self) -> Dict[str, int]:
        """Get client metrics.

        Returns:
            Metrics dictionary
        """
        return self.metrics.copy()

    async def health_check(self) -> bool:
        """Check API connectivity.

        Returns:
            True if healthy, False otherwise
        """
        try:
            health_endpoint = self.config.get('health_endpoint', '/ping')
            await self.get(health_endpoint, signed=False)
            return True
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
