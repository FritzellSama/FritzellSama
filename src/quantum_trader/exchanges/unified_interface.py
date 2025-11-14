"""
Unified Exchange Interface - Single interface for all exchange connectors.

Provides unified access to multiple exchanges through a single API,
handling exchange-specific differences and normalization.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import polars as pl
import structlog

from quantum_trader.models import (
    Order, OrderBook, Ticker, Balance,
    OrderSide, OrderType
)
from quantum_trader.exchanges.base import BaseExchange

logger = structlog.get_logger(__name__)


class ExchangeStatus(Enum):
    """Exchange connection status."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    RATE_LIMITED = "RATE_LIMITED"


@dataclass
class ExchangeHealth:
    """Health status for an exchange."""
    exchange: str
    status: ExchangeStatus
    last_successful_request: Optional[datetime]
    last_error: Optional[str]
    error_count: int
    latency_ms: Optional[Decimal]
    rate_limit_remaining: Optional[int]


class UnifiedExchangeInterface:
    """
    Unified interface for interacting with multiple cryptocurrency exchanges.

    Provides single API for multi-exchange operations with automatic failover,
    load balancing, and exchange-specific normalization.

    Attributes:
        config: Configuration dictionary
        exchanges: Dictionary of exchange connectors
        primary_exchange: Default exchange for operations
        health_check_interval: Interval for health checks in seconds

    Example:
        >>> config = {
        ...     'exchanges': {
        ...         'binance': {...},
        ...         'bybit': {...}
        ...     },
        ...     'primary_exchange': 'binance',
        ...     'failover_enabled': True
        ... }
        >>> interface = UnifiedExchangeInterface(config)
        >>> await interface.initialize()
        >>>
        >>> # Single exchange operation
        >>> balance = await interface.fetch_balance('binance')
        >>>
        >>> # Multi-exchange aggregation
        >>> all_balances = await interface.fetch_all_balances()
        >>>
        >>> # Smart routing
        >>> best_price = await interface.get_best_price('BTC/USDT')
    """

    def __init__(
        self,
        config: Dict[str, Any],
        exchange_factories: Dict[str, Any]
    ) -> None:
        """
        Initialize unified exchange interface.

        Args:
            config: Configuration dictionary
            exchange_factories: Dictionary of exchange factory functions

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.exchange_factories = exchange_factories
        self.exchanges: Dict[str, BaseExchange] = {}
        self.exchange_health: Dict[str, ExchangeHealth] = {}

        self.primary_exchange = config.get('primary_exchange', 'binance')
        self.failover_enabled = config.get('failover_enabled', True)
        self.health_check_interval = Decimal(
            str(config.get('health_check_interval', 60))
        )

        # Operation routing
        self.read_preference = config.get('read_preference', 'primary')  # primary, all, fastest
        self.write_preference = config.get('write_preference', 'primary')  # primary, all

        # Monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics: Dict[str, Any] = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'failover_events': 0,
            'exchange_errors': {}
        }

        logger.info(
            "unified_interface_initialized",
            exchanges=list(exchange_factories.keys()),
            primary=self.primary_exchange,
            failover=self.failover_enabled
        )

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        if 'exchanges' not in self.config or not self.config['exchanges']:
            raise ValueError("No exchanges configured")

        primary = self.config.get('primary_exchange')
        if primary and primary not in self.config['exchanges']:
            raise ValueError(f"Primary exchange '{primary}' not in exchanges")

    async def initialize(self) -> None:
        """Initialize all exchange connectors and start health monitoring."""
        logger.info("initializing_exchanges")

        # Initialize exchange connectors
        for exchange_name, factory in self.exchange_factories.items():
            try:
                exchange_config = self.config['exchanges'].get(exchange_name, {})

                if not exchange_config.get('enabled', True):
                    logger.info(
                        "exchange_disabled",
                        exchange=exchange_name
                    )
                    continue

                # Create exchange instance
                exchange = factory(exchange_config)
                self.exchanges[exchange_name] = exchange

                # Initialize health status
                self.exchange_health[exchange_name] = ExchangeHealth(
                    exchange=exchange_name,
                    status=ExchangeStatus.CONNECTING,
                    last_successful_request=None,
                    last_error=None,
                    error_count=0,
                    latency_ms=None,
                    rate_limit_remaining=None
                )

                # Initialize exchange connection
                # Note: Actual connection happens on first use for most exchanges

                self.exchange_health[exchange_name].status = ExchangeStatus.CONNECTED

                logger.info(
                    "exchange_initialized",
                    exchange=exchange_name
                )

            except Exception as e:
                logger.error(
                    "exchange_initialization_failed",
                    exchange=exchange_name,
                    error=str(e)
                )
                self._metrics['exchange_errors'][exchange_name] = str(e)

        if not self.exchanges:
            raise RuntimeError("No exchanges successfully initialized")

        # Start health monitoring
        self._health_check_task = asyncio.create_task(
            self._health_check_loop()
        )

        logger.info(
            "unified_interface_ready",
            active_exchanges=len(self.exchanges)
        )

    async def close(self) -> None:
        """Close all exchange connections and cleanup."""
        logger.info("closing_unified_interface")

        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
                logger.info(
                    "exchange_closed",
                    exchange=exchange_name
                )
            except Exception as e:
                logger.error(
                    "exchange_close_error",
                    exchange=exchange_name,
                    error=str(e)
                )

        self.exchanges.clear()
        logger.info("unified_interface_closed")

    async def fetch_balance(
        self,
        exchange: Optional[str] = None
    ) -> Dict[str, Decimal]:
        """
        Fetch balance from specified exchange.

        Args:
            exchange: Exchange name (uses primary if not specified)

        Returns:
            Balance dictionary {currency: amount}

        Raises:
            RuntimeError: If operation fails
        """
        exchange_name = exchange or self.primary_exchange
        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.fetch_balance(),
            'fetch_balance'
        )

    async def fetch_all_balances(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Fetch balances from all active exchanges.

        Returns:
            Dictionary mapping exchange name to balance
        """
        results = {}
        tasks = []

        for exchange_name in self.exchanges.keys():
            if self._is_exchange_healthy(exchange_name):
                task = self._safe_operation(
                    exchange_name,
                    lambda ex: ex.fetch_balance(),
                    'fetch_balance'
                )
                tasks.append((exchange_name, task))

        for exchange_name, task in tasks:
            try:
                balance = await task
                results[exchange_name] = balance
            except Exception as e:
                logger.warning(
                    "fetch_balance_failed",
                    exchange=exchange_name,
                    error=str(e)
                )

        return results

    async def fetch_ticker(
        self,
        symbol: str,
        exchange: Optional[str] = None
    ) -> Ticker:
        """
        Fetch ticker for symbol from exchange.

        Args:
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            Ticker data

        Raises:
            RuntimeError: If operation fails
        """
        exchange_name = exchange or self.primary_exchange
        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.fetch_ticker(symbol),
            'fetch_ticker'
        )

    async def fetch_all_tickers(
        self,
        symbol: str
    ) -> Dict[str, Ticker]:
        """
        Fetch ticker from all exchanges for arbitrage analysis.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary mapping exchange to ticker
        """
        results = {}
        tasks = []

        for exchange_name in self.exchanges.keys():
            if self._is_exchange_healthy(exchange_name):
                task = self._safe_operation(
                    exchange_name,
                    lambda ex, sym=symbol: ex.fetch_ticker(sym),
                    'fetch_ticker'
                )
                tasks.append((exchange_name, task))

        for exchange_name, task in tasks:
            try:
                ticker = await task
                results[exchange_name] = ticker
            except Exception as e:
                logger.debug(
                    "fetch_ticker_failed",
                    exchange=exchange_name,
                    symbol=symbol,
                    error=str(e)
                )

        return results

    async def fetch_orderbook(
        self,
        symbol: str,
        depth: int = 20,
        exchange: Optional[str] = None
    ) -> OrderBook:
        """
        Fetch order book for symbol.

        Args:
            symbol: Trading pair symbol
            depth: Order book depth
            exchange: Exchange name

        Returns:
            OrderBook data

        Raises:
            RuntimeError: If operation fails
        """
        exchange_name = exchange or self.primary_exchange
        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.fetch_orderbook(symbol, depth),
            'fetch_orderbook'
        )

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        since: Optional[int] = None,
        exchange: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (1m, 5m, 1h, etc.)
            limit: Number of candles
            since: Start timestamp (ms)
            exchange: Exchange name

        Returns:
            Polars DataFrame with OHLCV data

        Raises:
            RuntimeError: If operation fails
        """
        exchange_name = exchange or self.primary_exchange
        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.fetch_ohlcv(symbol, timeframe, limit, since),
            'fetch_ohlcv'
        )

    async def create_order(
        self,
        order: Order,
        exchange: Optional[str] = None
    ) -> str:
        """
        Create order on exchange.

        Args:
            order: Order object
            exchange: Exchange name (overrides order.exchange)

        Returns:
            Exchange order ID

        Raises:
            RuntimeError: If order creation fails
        """
        exchange_name = exchange or order.exchange or self.primary_exchange

        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.create_order(order),
            'create_order',
            critical=True
        )

    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        exchange: Optional[str] = None
    ) -> bool:
        """
        Cancel order on exchange.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            exchange: Exchange name

        Returns:
            True if cancelled successfully

        Raises:
            RuntimeError: If cancellation fails
        """
        exchange_name = exchange or self.primary_exchange

        return await self._execute_operation(
            exchange_name,
            lambda ex: ex.cancel_order(order_id, symbol),
            'cancel_order',
            critical=True
        )

    async def get_best_price(
        self,
        symbol: str,
        side: OrderSide
    ) -> Dict[str, Any]:
        """
        Find best price across all exchanges.

        Args:
            symbol: Trading pair symbol
            side: BUY or SELL

        Returns:
            Dictionary with best price info {exchange, price, volume}
        """
        tickers = await self.fetch_all_tickers(symbol)

        if not tickers:
            raise RuntimeError(f"No ticker data available for {symbol}")

        best_exchange = None
        best_price = None

        for exchange_name, ticker in tickers.items():
            price = ticker.ask if side == OrderSide.BUY else ticker.bid

            if best_price is None or (
                (side == OrderSide.BUY and price < best_price) or
                (side == OrderSide.SELL and price > best_price)
            ):
                best_price = price
                best_exchange = exchange_name

        return {
            'exchange': best_exchange,
            'price': best_price,
            'symbol': symbol,
            'side': side.value,
            'ticker': tickers[best_exchange]
        }

    async def _execute_operation(
        self,
        exchange_name: str,
        operation: callable,
        operation_name: str,
        critical: bool = False
    ) -> Any:
        """
        Execute operation with optional failover.

        Args:
            exchange_name: Target exchange
            operation: Operation to execute
            operation_name: Operation name for logging
            critical: Whether operation is critical (no failover)

        Returns:
            Operation result

        Raises:
            RuntimeError: If operation fails
        """
        self._metrics['total_operations'] += 1

        # Try primary exchange
        if exchange_name in self.exchanges:
            try:
                result = await self._safe_operation(
                    exchange_name,
                    operation,
                    operation_name
                )
                self._metrics['successful_operations'] += 1
                return result

            except Exception as e:
                logger.warning(
                    "operation_failed",
                    exchange=exchange_name,
                    operation=operation_name,
                    error=str(e)
                )

                if critical or not self.failover_enabled:
                    self._metrics['failed_operations'] += 1
                    raise

        # Try failover to other exchanges
        if self.failover_enabled and not critical:
            for fallback_exchange in self.exchanges.keys():
                if fallback_exchange == exchange_name:
                    continue

                if not self._is_exchange_healthy(fallback_exchange):
                    continue

                try:
                    logger.info(
                        "attempting_failover",
                        from_exchange=exchange_name,
                        to_exchange=fallback_exchange,
                        operation=operation_name
                    )

                    result = await self._safe_operation(
                        fallback_exchange,
                        operation,
                        operation_name
                    )

                    self._metrics['successful_operations'] += 1
                    self._metrics['failover_events'] += 1

                    return result

                except Exception as e:
                    logger.warning(
                        "failover_attempt_failed",
                        exchange=fallback_exchange,
                        error=str(e)
                    )

        self._metrics['failed_operations'] += 1
        raise RuntimeError(
            f"Operation '{operation_name}' failed on all available exchanges"
        )

    async def _safe_operation(
        self,
        exchange_name: str,
        operation: callable,
        operation_name: str
    ) -> Any:
        """
        Execute operation with health tracking.

        Args:
            exchange_name: Exchange name
            operation: Operation to execute
            operation_name: Operation name

        Returns:
            Operation result

        Raises:
            Exception: If operation fails
        """
        exchange = self.exchanges[exchange_name]
        health = self.exchange_health[exchange_name]

        start_time = datetime.now(timezone.utc)

        try:
            result = await operation(exchange)

            # Update health
            health.last_successful_request = datetime.now(timezone.utc)
            health.status = ExchangeStatus.CONNECTED
            health.error_count = 0
            health.last_error = None

            # Calculate latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds()
            health.latency_ms = Decimal(str(latency * 1000))

            return result

        except Exception as e:
            # Update health
            health.error_count += 1
            health.last_error = str(e)

            if health.error_count >= int(self.config.get('error_threshold', 5)):
                health.status = ExchangeStatus.ERROR

            logger.error(
                "exchange_operation_error",
                exchange=exchange_name,
                operation=operation_name,
                error=str(e),
                error_count=health.error_count
            )

            raise

    def _is_exchange_healthy(self, exchange_name: str) -> bool:
        """
        Check if exchange is healthy for operations.

        Args:
            exchange_name: Exchange name

        Returns:
            True if healthy
        """
        health = self.exchange_health.get(exchange_name)
        if not health:
            return False

        return health.status in [
            ExchangeStatus.CONNECTED,
            ExchangeStatus.CONNECTING
        ]

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(float(self.health_check_interval))
                await self._perform_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_check_error",
                    error=str(e)
                )

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all exchanges."""
        for exchange_name in self.exchanges.keys():
            try:
                # Simple ping by fetching server time or similar
                await self._safe_operation(
                    exchange_name,
                    lambda ex: ex.fetch_balance() if hasattr(ex, 'fetch_balance') else True,
                    'health_check'
                )

            except Exception as e:
                logger.debug(
                    "health_check_failed",
                    exchange=exchange_name,
                    error=str(e)
                )

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all exchanges.

        Returns:
            Dictionary of health information
        """
        return {
            exchange_name: {
                'status': health.status.value,
                'last_successful_request': (
                    health.last_successful_request.isoformat()
                    if health.last_successful_request else None
                ),
                'error_count': health.error_count,
                'last_error': health.last_error,
                'latency_ms': float(health.latency_ms) if health.latency_ms else None
            }
            for exchange_name, health in self.exchange_health.items()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get unified interface metrics.

        Returns:
            Dictionary of metrics
        """
        success_rate = Decimal("0")
        if self._metrics['total_operations'] > 0:
            success_rate = (
                Decimal(str(self._metrics['successful_operations'])) /
                Decimal(str(self._metrics['total_operations']))
            ) * Decimal("100")

        return {
            **self._metrics,
            'success_rate': float(success_rate),
            'active_exchanges': len([
                name for name, health in self.exchange_health.items()
                if health.status == ExchangeStatus.CONNECTED
            ])
        }
