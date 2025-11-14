"""
Binance exchange client implementation.

This module provides the concrete implementation of the BaseExchange
interface for Binance cryptocurrency exchange.
"""

import asyncio
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
import polars as pl
from structlog import get_logger

from ..base_exchange import BaseExchange
from ..connectors.connection_pool import ConnectionPool
from .api_mapper import BinanceAPIMapper

logger = get_logger(__name__)


class BinanceClient(BaseExchange):
    """Binance exchange client implementation.

    Provides full implementation of exchange operations for Binance,
    including spot and futures trading.

    Attributes:
        mapper: API data mapper
        pool: HTTP connection pool
        symbols: Available trading symbols
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize Binance client.

        Args:
            api_key: Binance API key
            secret: Binance API secret
            testnet: Whether to use testnet

        Raises:
            ValueError: If credentials are invalid
        """
        super().__init__(api_key, secret, testnet)

        self.mapper = BinanceAPIMapper(api_key, secret, testnet)
        self.pool = ConnectionPool()
        self.symbols: List[str] = []

        logger.info("Binance client initialized", testnet=testnet)

    async def connect(self) -> None:
        """Establish connection to Binance.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info("Connecting to Binance")

            await self.pool.create_session()
            await self._load_symbols()

            self.connected = True
            logger.info("Connected to Binance successfully", symbols_count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to connect to Binance", error=str(e))
            raise ConnectionError(f"Binance connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to Binance."""
        try:
            logger.info("Disconnecting from Binance")
            await self.pool.close_session()
            self.connected = False
            logger.info("Disconnected from Binance")
        except Exception as e:
            logger.error("Error during Binance disconnect", error=str(e))

    async def _load_symbols(self) -> None:
        """Load available trading symbols from exchange."""
        try:
            endpoint = os.getenv('BINANCE_EXCHANGE_INFO_ENDPOINT', '/api/v3/exchangeInfo')
            url = f"{self.mapper.base_url}{endpoint}"

            data = await self.pool.get(url)

            self.symbols = []
            for symbol_info in data.get('symbols', []):
                if symbol_info['status'] == 'TRADING':
                    base = symbol_info['baseAsset']
                    quote = symbol_info['quoteAsset']
                    self.symbols.append(f"{base}/{quote}")

            logger.info("Symbols loaded", count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to load symbols", error=str(e))
            raise

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100,
        since: Optional[int] = None
    ) -> pl.DataFrame:
        """Fetch OHLCV data from Binance.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
            limit: Number of candles to fetch
            since: Timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_binance = symbol.replace('/', '')
            endpoint = os.getenv('BINANCE_KLINES_ENDPOINT', '/api/v3/klines')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_binance,
                'interval': timeframe,
                'limit': min(limit, int(os.getenv('BINANCE_MAX_KLINES_LIMIT', '1000')))
            }

            if since is not None:
                params['startTime'] = since

            data = await self.pool.get(url, params=params)
            df = self.mapper.map_ohlcv_response(data)

            logger.debug(
                "OHLCV data fetched",
                symbol=symbol,
                timeframe=timeframe,
                rows=len(df)
            )

            return df

        except Exception as e:
            logger.error(
                "Failed to fetch OHLCV",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e)
            )
            raise RuntimeError(f"Failed to fetch OHLCV: {e}")

    async def fetch_ticker(self, symbol: str) -> 'Ticker':
        """Fetch ticker data from Binance.

        Args:
            symbol: Trading pair

        Returns:
            Ticker object

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_binance = symbol.replace('/', '')
            endpoint = os.getenv('BINANCE_TICKER_ENDPOINT', '/api/v3/ticker/24hr')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {'symbol': symbol_binance}
            data = await self.pool.get(url, params=params)

            ticker_data = self.mapper.map_ticker_response(data)

            # Import here to avoid circular dependency
            from quantum_trader.models import Ticker

            ticker = Ticker(
                symbol=symbol,
                bid=ticker_data['bid'],
                ask=ticker_data['ask'],
                last=ticker_data['last'],
                volume=ticker_data['volume'],
                timestamp=ticker_data['timestamp']
            )

            logger.debug("Ticker data fetched", symbol=symbol)
            return ticker

        except Exception as e:
            logger.error("Failed to fetch ticker", symbol=symbol, error=str(e))
            raise RuntimeError(f"Failed to fetch ticker: {e}")

    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> 'OrderBook':
        """Fetch orderbook from Binance.

        Args:
            symbol: Trading pair
            depth: Number of levels per side

        Returns:
            OrderBook object

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_binance = symbol.replace('/', '')
            endpoint = os.getenv('BINANCE_DEPTH_ENDPOINT', '/api/v3/depth')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_binance,
                'limit': min(depth, int(os.getenv('BINANCE_MAX_DEPTH', '5000')))
            }

            data = await self.pool.get(url, params=params)
            bids, asks = self.mapper.map_orderbook_response(data)

            # Import here to avoid circular dependency
            from quantum_trader.models import OrderBook

            orderbook = OrderBook(
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc)
            )

            logger.debug("Orderbook fetched", symbol=symbol, depth=depth)
            return orderbook

        except Exception as e:
            logger.error("Failed to fetch orderbook", symbol=symbol, error=str(e))
            raise RuntimeError(f"Failed to fetch orderbook: {e}")

    async def fetch_balance(self) -> Dict[str, Decimal]:
        """Fetch account balance from Binance.

        Returns:
            Dictionary of currency balances

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BINANCE_ACCOUNT_ENDPOINT', '/api/v3/account')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {}
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            data = await self.pool.get(url, headers=headers, params=params)

            balances = self.mapper.map_balance_response(data)

            logger.debug("Balance fetched", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to fetch balance", error=str(e))
            raise RuntimeError(f"Failed to fetch balance: {e}")

    async def fetch_positions(self) -> List[Dict]:
        """Fetch open positions from Binance Futures.

        Returns:
            List of position dictionaries

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            # Futures endpoint
            base_url = os.getenv('BINANCE_FUTURES_URL', 'https://fapi.binance.com')
            endpoint = os.getenv('BINANCE_POSITIONS_ENDPOINT', '/fapi/v2/positionRisk')
            url = f"{base_url}{endpoint}"

            params = {}
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            data = await self.pool.get(url, headers=headers, params=params)

            positions = self.mapper.map_positions_response(data)

            logger.debug("Positions fetched", count=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise RuntimeError(f"Failed to fetch positions: {e}")

    async def create_order(self, order: 'Order') -> str:
        """Create order on Binance.

        Args:
            order: Order object

        Returns:
            Exchange order ID

        Raises:
            RuntimeError: If order creation fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BINANCE_ORDER_ENDPOINT', '/api/v3/order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = self.mapper.map_order_request(order)
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            data = await self.pool.post(url, headers=headers, json_data=params)

            order_id = self.mapper.map_order_response(data)

            logger.info(
                "Order created",
                order_id=order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=str(order.quantity)
            )

            return order_id

        except Exception as e:
            logger.error(
                "Failed to create order",
                symbol=order.symbol,
                error=str(e)
            )
            raise RuntimeError(f"Failed to create order: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            True if cancellation successful

        Raises:
            RuntimeError: If cancellation fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_binance = symbol.replace('/', '')
            endpoint = os.getenv('BINANCE_ORDER_ENDPOINT', '/api/v3/order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_binance,
                'orderId': order_id
            }
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            await self.pool.delete(url, headers=headers, params=params)

            logger.info("Order cancelled", order_id=order_id, symbol=symbol)
            return True

        except Exception as e:
            logger.error(
                "Failed to cancel order",
                order_id=order_id,
                symbol=symbol,
                error=str(e)
            )
            raise RuntimeError(f"Failed to cancel order: {e}")

    async def fetch_order_status(self, order_id: str, symbol: str) -> Dict:
        """Fetch order status from Binance.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair

        Returns:
            Order status dictionary

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_binance = symbol.replace('/', '')
            endpoint = os.getenv('BINANCE_ORDER_ENDPOINT', '/api/v3/order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_binance,
                'orderId': order_id
            }
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            data = await self.pool.get(url, headers=headers, params=params)

            status = self.mapper.map_order_status_response(data)

            logger.debug("Order status fetched", order_id=order_id, status=status['status'])
            return status

        except Exception as e:
            logger.error(
                "Failed to fetch order status",
                order_id=order_id,
                error=str(e)
            )
            raise RuntimeError(f"Failed to fetch order status: {e}")

    async def get_trading_fees(self, symbol: str) -> Dict[str, Decimal]:
        """Get trading fees for symbol.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with maker and taker fees

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BINANCE_TRADE_FEE_ENDPOINT', '/api/v3/tradeFee')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {}
            signature = self.mapper.sign_request(params)
            params['signature'] = signature

            headers = self.mapper.get_headers()
            data = await self.pool.get(url, headers=headers, params=params)

            fees = self.mapper.map_trading_fees_response(data, symbol)

            logger.debug("Trading fees fetched", symbol=symbol, fees=fees)
            return fees

        except Exception as e:
            logger.error("Failed to fetch trading fees", symbol=symbol, error=str(e))
            raise RuntimeError(f"Failed to fetch trading fees: {e}")

    def get_symbols(self) -> List[str]:
        """Get list of available trading symbols.

        Returns:
            List of trading pairs

        Raises:
            RuntimeError: If not connected
        """
        if not self.connected:
            raise RuntimeError("Not connected to exchange")

        return self.symbols.copy()
