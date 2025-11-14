"""
Base exchange connector for all exchange implementations.

This module provides the abstract base class that all exchange connectors must implement.
It defines the standard interface for interacting with cryptocurrency exchanges.
"""

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BaseExchange(ABC):
    """Abstract base for all exchange connectors.

    This class defines the standard interface that all exchange implementations
    must follow. It handles connection lifecycle, market data retrieval,
    order management, and account operations.

    Attributes:
        api_key: Exchange API key
        secret: Exchange API secret
        testnet: Whether to use testnet endpoints
        name: Exchange name
        connected: Connection status
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize exchange connector.

        Args:
            api_key: Exchange API key
            secret: Exchange API secret
            testnet: Whether to use testnet environment

        Raises:
            ValueError: If credentials are invalid
        """
        if not api_key or not secret:
            raise ValueError("API key and secret are required")

        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.name = self.__class__.__name__.replace("Exchange", "").upper()
        self.connected = False
        self._session = None

        logger.info(
            "Exchange connector initialized",
            exchange=self.name,
            testnet=testnet
        )

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        since: Optional[int] = None
    ) -> pl.DataFrame:
        """Fetch OHLCV (candlestick) data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to fetch
            since: Timestamp in milliseconds to fetch from

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
            All price/volume columns are Decimal type

        Raises:
            ExchangeError: If fetch fails
            ValidationError: If symbol or timeframe invalid
        """
        pass

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> 'Ticker':
        """Fetch current ticker data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Ticker object with bid, ask, last, volume

        Raises:
            ExchangeError: If fetch fails
            ValidationError: If symbol invalid
        """
        pass

    @abstractmethod
    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> 'OrderBook':
        """Fetch order book snapshot.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            depth: Number of levels to fetch per side

        Returns:
            OrderBook object with bids and asks

        Raises:
            ExchangeError: If fetch fails
            ValidationError: If symbol invalid
        """
        pass

    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Decimal]:
        """Fetch account balances.

        Returns:
            Dictionary mapping currency to available balance
            Example: {'USDT': Decimal('10000.00'), 'BTC': Decimal('0.5')}

        Raises:
            ExchangeError: If fetch fails
            AuthenticationError: If credentials invalid
        """
        pass

    @abstractmethod
    async def fetch_positions(self) -> List[Dict]:
        """Fetch open positions (for derivatives).

        Returns:
            List of position dictionaries containing:
            - symbol: Trading pair
            - size: Position size (Decimal)
            - entry_price: Average entry price (Decimal)
            - unrealized_pnl: Unrealized P&L (Decimal)
            - leverage: Current leverage (Decimal)

        Raises:
            ExchangeError: If fetch fails
            AuthenticationError: If credentials invalid
        """
        pass

    @abstractmethod
    async def create_order(self, order: 'Order') -> str:
        """Create and submit order to exchange.

        Args:
            order: Order object with all required fields

        Returns:
            Exchange order ID (string)

        Raises:
            ExchangeError: If order submission fails
            ValidationError: If order parameters invalid
            InsufficientBalanceError: If insufficient funds
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            True if cancellation successful

        Raises:
            ExchangeError: If cancellation fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    async def fetch_order_status(self, order_id: str, symbol: str) -> Dict:
        """Fetch order status and details.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dictionary containing:
            - order_id: Exchange order ID
            - status: Order status (pending/filled/cancelled/etc)
            - filled_quantity: Amount filled (Decimal)
            - remaining_quantity: Amount remaining (Decimal)
            - average_price: Average fill price (Decimal)
            - fees: Trading fees paid (Decimal)

        Raises:
            ExchangeError: If fetch fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    async def get_trading_fees(self, symbol: str) -> Dict[str, Decimal]:
        """Get trading fee rates for symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            Dictionary with maker and taker fee rates
            Example: {'maker': Decimal('0.0001'), 'taker': Decimal('0.0002')}

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of available trading symbols.

        Returns:
            List of trading pair strings (e.g., ['BTC/USDT', 'ETH/USDT'])

        Raises:
            ExchangeError: If fetch fails
        """
        pass

    async def connect(self) -> None:
        """Establish connection to exchange.

        This method should be called before making any API calls.
        It initializes the HTTP session and validates credentials.

        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If credentials invalid
        """
        try:
            logger.info("Connecting to exchange", exchange=self.name)
            # Subclasses should implement actual connection logic
            self.connected = True
            logger.info("Connected to exchange successfully", exchange=self.name)
        except Exception as e:
            logger.error(
                "Failed to connect to exchange",
                exchange=self.name,
                error=str(e)
            )
            raise

    async def disconnect(self) -> None:
        """Close connection to exchange.

        This method should be called when done with the exchange
        to clean up resources and close HTTP sessions.
        """
        try:
            logger.info("Disconnecting from exchange", exchange=self.name)
            if self._session and not self._session.closed:
                await self._session.close()
            self.connected = False
            logger.info("Disconnected from exchange", exchange=self.name)
        except Exception as e:
            logger.error(
                "Error during disconnect",
                exchange=self.name,
                error=str(e)
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(testnet={self.testnet}, connected={self.connected})"
