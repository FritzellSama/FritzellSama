"""
REST Connector - Universal REST API connector for cryptocurrency exchanges.

Provides robust HTTP client with retry logic, rate limiting, authentication,
and comprehensive error handling for exchange REST APIs.
"""

import asyncio
import hashlib
import hmac
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import urllib.parse

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class HTTPMethod(Enum):
    """HTTP methods for API requests."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class AuthType(Enum):
    """Authentication types for exchanges."""
    HMAC_SHA256 = "HMAC_SHA256"
    HMAC_SHA512 = "HMAC_SHA512"
    RSA = "RSA"
    ED25519 = "ED25519"


@dataclass
class RequestConfig:
    """Configuration for individual request."""
    url: str
    method: HTTPMethod
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    auth_required: bool = True
    timeout: Decimal = Decimal("30")
    retries: int = 3


@dataclass
class RetryConfig:
    """Retry configuration for failed requests."""
    max_retries: int
    base_delay: Decimal
    max_delay: Decimal
    exponential_base: Decimal
    retry_on_status: List[int]


class RESTConnector:
    """
    Universal REST API connector for cryptocurrency exchanges.

    Handles authentication, rate limiting, retries, and error handling
    for REST API calls to various cryptocurrency exchanges.

    Attributes:
        config: Configuration dictionary
        api_key: API key for authentication
        api_secret: API secret for signing
        session: aiohttp ClientSession
        base_url: Base URL for API endpoints

    Example:
        >>> config = {
        ...     'base_url': 'https://api.exchange.com',
        ...     'timeout': 30,
        ...     'retry': {'max_retries': 3, 'base_delay': 1.0},
        ...     'auth_type': 'HMAC_SHA256'
        ... }
        >>> connector = RESTConnector(config, api_key='key', api_secret='secret')
        >>> await connector.initialize()
        >>> response = await connector.request('GET', '/api/v1/balance')
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None
    ) -> None:
        """
        Initialize REST connector.

        Args:
            config: Configuration dictionary
            api_key: API key for authentication
            api_secret: API secret for signing
            passphrase: Optional passphrase (for some exchanges)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        self.base_url = config['base_url'].rstrip('/')
        self.testnet = config.get('testnet', False)

        # Auth configuration
        auth_type_str = config.get('auth_type', 'HMAC_SHA256')
        self.auth_type = AuthType[auth_type_str.upper()]

        # Retry configuration
        retry_config = config.get('retry', {})
        self.retry_config = RetryConfig(
            max_retries=int(retry_config.get('max_retries', 3)),
            base_delay=Decimal(str(retry_config.get('base_delay', '1.0'))),
            max_delay=Decimal(str(retry_config.get('max_delay', '60.0'))),
            exponential_base=Decimal(str(retry_config.get('exponential_base', '2.0'))),
            retry_on_status=retry_config.get('retry_on_status', [429, 500, 502, 503, 504])
        )

        # Timeout configuration
        self.default_timeout = Decimal(str(config.get('timeout', 30)))

        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'total_retries': 0,
            'auth_failures': 0,
            'timeout_errors': 0,
            'network_errors': 0
        }

        logger.info(
            "rest_connector_initialized",
            base_url=self.base_url,
            auth_type=self.auth_type.value,
            testnet=self.testnet
        )

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        if 'base_url' not in self.config:
            raise ValueError("Missing 'base_url' in configuration")

        base_url = self.config['base_url']
        if not base_url.startswith('http://') and not base_url.startswith('https://'):
            raise ValueError(f"Invalid base_url: {base_url}")

    async def initialize(self) -> None:
        """Initialize HTTP session and connection pool."""
        async with self._session_lock:
            if self.session is not None:
                return

            timeout = aiohttp.ClientTimeout(
                total=float(self.default_timeout),
                connect=float(self.config.get('connect_timeout', 10)),
                sock_read=float(self.config.get('read_timeout', 30))
            )

            connector = aiohttp.TCPConnector(
                limit=int(self.config.get('connection_limit', 100)),
                limit_per_host=int(self.config.get('connection_limit_per_host', 30)),
                ttl_dns_cache=int(self.config.get('dns_cache_ttl', 300)),
                enable_cleanup_closed=True
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                raise_for_status=False
            )

            logger.info("rest_connector_session_initialized")

    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        async with self._session_lock:
            if self.session:
                await self.session.close()
                self.session = None
                logger.info("rest_connector_session_closed")

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth_required: bool = True,
        timeout: Optional[Decimal] = None,
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to exchange API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL query parameters
            data: Request body data
            headers: Additional headers
            auth_required: Whether to add authentication
            timeout: Request timeout in seconds
            retries: Number of retries (overrides config)

        Returns:
            Response data as dictionary

        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid response
            RuntimeError: On authentication errors
        """
        if self.session is None:
            await self.initialize()

        http_method = HTTPMethod[method.upper()]

        request_config = RequestConfig(
            url=self._build_url(endpoint),
            method=http_method,
            params=params,
            data=data,
            headers=headers or {},
            auth_required=auth_required,
            timeout=timeout or self.default_timeout,
            retries=retries if retries is not None else self.retry_config.max_retries
        )

        return await self._execute_request(request_config)

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL from endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL
        """
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"

    async def _execute_request(
        self,
        request_config: RequestConfig
    ) -> Dict[str, Any]:
        """
        Execute HTTP request with retries.

        Args:
            request_config: Request configuration

        Returns:
            Response data

        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid response
        """
        last_exception: Optional[Exception] = None
        retry_count = 0

        while retry_count <= request_config.retries:
            try:
                self._metrics['total_requests'] += 1

                if retry_count > 0:
                    self._metrics['retried_requests'] += 1
                    self._metrics['total_retries'] += 1

                response = await self._make_request(request_config)

                self._metrics['successful_requests'] += 1
                return response

            except aiohttp.ClientError as e:
                last_exception = e
                self._metrics['network_errors'] += 1

                logger.warning(
                    "request_network_error",
                    url=request_config.url,
                    method=request_config.method.value,
                    error=str(e),
                    retry=retry_count
                )

            except asyncio.TimeoutError as e:
                last_exception = e
                self._metrics['timeout_errors'] += 1

                logger.warning(
                    "request_timeout",
                    url=request_config.url,
                    method=request_config.method.value,
                    timeout=float(request_config.timeout),
                    retry=retry_count
                )

            except Exception as e:
                last_exception = e
                logger.error(
                    "request_unexpected_error",
                    url=request_config.url,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise

            # Calculate backoff delay
            if retry_count < request_config.retries:
                delay = self._calculate_retry_delay(retry_count)
                await asyncio.sleep(float(delay))

            retry_count += 1

        # All retries exhausted
        self._metrics['failed_requests'] += 1

        logger.error(
            "request_failed_after_retries",
            url=request_config.url,
            retries=request_config.retries
        )

        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Request failed after all retries")

    async def _make_request(
        self,
        request_config: RequestConfig
    ) -> Dict[str, Any]:
        """
        Make single HTTP request.

        Args:
            request_config: Request configuration

        Returns:
            Response data

        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid response
            RuntimeError: On authentication errors
        """
        # Prepare headers
        headers = dict(request_config.headers)
        headers['Content-Type'] = 'application/json'
        headers['User-Agent'] = self.config.get(
            'user_agent',
            'QuantumTraderAI/1.0'
        )

        # Add authentication
        if request_config.auth_required:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("API credentials not configured")

            auth_headers = self._generate_auth_headers(
                request_config.method.value,
                request_config.url,
                request_config.params,
                request_config.data
            )
            headers.update(auth_headers)

        # Make request
        async with self.session.request(
            method=request_config.method.value,
            url=request_config.url,
            params=request_config.params,
            json=request_config.data,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=float(request_config.timeout))
        ) as response:
            response_text = await response.text()

            # Check status code
            if response.status in self.retry_config.retry_on_status:
                logger.warning(
                    "request_retriable_status",
                    status=response.status,
                    url=request_config.url
                )
                raise aiohttp.ClientError(
                    f"Retriable status code: {response.status}"
                )

            if response.status == 401 or response.status == 403:
                self._metrics['auth_failures'] += 1
                logger.error(
                    "authentication_failed",
                    status=response.status,
                    response=response_text[:500]
                )
                raise RuntimeError(f"Authentication failed: {response.status}")

            if response.status >= 400:
                logger.error(
                    "request_error_response",
                    status=response.status,
                    response=response_text[:500]
                )
                raise ValueError(
                    f"HTTP {response.status}: {response_text[:500]}"
                )

            # Parse response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(
                    "json_decode_error",
                    response=response_text[:500],
                    error=str(e)
                )
                raise ValueError(f"Invalid JSON response: {str(e)}")

    def _generate_auth_headers(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate authentication headers based on auth type.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            data: Request body

        Returns:
            Dictionary of auth headers
        """
        timestamp = str(int(time.time() * 1000))

        if self.auth_type == AuthType.HMAC_SHA256:
            return self._generate_hmac_sha256_headers(
                method, url, timestamp, params, data
            )
        elif self.auth_type == AuthType.HMAC_SHA512:
            return self._generate_hmac_sha512_headers(
                method, url, timestamp, params, data
            )
        else:
            raise NotImplementedError(
                f"Auth type {self.auth_type} not implemented"
            )

    def _generate_hmac_sha256_headers(
        self,
        method: str,
        url: str,
        timestamp: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate HMAC-SHA256 authentication headers."""
        # Build signature payload
        query_string = urllib.parse.urlencode(params) if params else ''
        body_string = json.dumps(data, separators=(',', ':')) if data else ''

        # Create signature
        message = f"{timestamp}{method}{url.split(self.base_url)[1]}{query_string}{body_string}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'X-API-KEY': self.api_key,
            'X-TIMESTAMP': timestamp,
            'X-SIGNATURE': signature
        }

        if self.passphrase:
            headers['X-PASSPHRASE'] = self.passphrase

        return headers

    def _generate_hmac_sha512_headers(
        self,
        method: str,
        url: str,
        timestamp: str,
        params: Optional[Dict[str, Any]],
        data: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate HMAC-SHA512 authentication headers."""
        query_string = urllib.parse.urlencode(params) if params else ''
        body_string = json.dumps(data, separators=(',', ':')) if data else ''

        message = f"{timestamp}{method.upper()}{query_string}{body_string}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()

        return {
            'X-API-KEY': self.api_key,
            'X-TIMESTAMP': timestamp,
            'X-SIGNATURE': signature
        }

    def _calculate_retry_delay(self, retry_count: int) -> Decimal:
        """
        Calculate exponential backoff delay.

        Args:
            retry_count: Current retry attempt

        Returns:
            Delay in seconds
        """
        delay = self.retry_config.base_delay * (
            self.retry_config.exponential_base ** Decimal(str(retry_count))
        )

        return min(delay, self.retry_config.max_delay)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get connector metrics.

        Returns:
            Dictionary of metrics
        """
        success_rate = Decimal("0")
        if self._metrics['total_requests'] > 0:
            success_rate = (
                Decimal(str(self._metrics['successful_requests'])) /
                Decimal(str(self._metrics['total_requests']))
            ) * Decimal("100")

        return {
            **self._metrics,
            'success_rate': float(success_rate),
            'average_retries': (
                self._metrics['total_retries'] / max(self._metrics['retried_requests'], 1)
            )
        }

    async def __aenter__(self) -> 'RESTConnector':
        """Context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.close()
