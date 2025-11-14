"""
Bitget exchange client implementation.

This module provides the concrete implementation of the BaseExchange
interface for Bitget cryptocurrency exchange.
"""

import asyncio
import os
import json
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
import polars as pl
from structlog import get_logger

from ..base_exchange import BaseExchange
from ..connectors.connection_pool import ConnectionPool
from .api_mapper import BitgetAPIMapper

logger = get_logger(__name__)


class BitgetClient(BaseExchange):
    """Bitget exchange client implementation.

    Provides full implementation of exchange operations for Bitget,
    including spot and derivatives trading.

    Attributes:
        mapper: API data mapper
        pool: HTTP connection pool
        symbols: Available trading symbols
        passphrase: Bitget API passphrase
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize Bitget client.

        Args:
            api_key: Bitget API key
            secret: Bitget API secret
            testnet: Whether to use testnet

        Raises:
            ValueError: If credentials are invalid
        """
        super().__init__(api_key, secret, testnet)

        # Bitget requires a passphrase - load from environment
        self.passphrase = os.getenv('BITGET_API_PASSPHRASE', '')
        if not self.passphrase:
            raise ValueError("BITGET_API_PASSPHRASE environment variable is required")

        self.mapper = BitgetAPIMapper(api_key, secret, self.passphrase, testnet)
        self.pool = ConnectionPool()
        self.symbols: List[str] = []

        logger.info("Bitget client initialized", testnet=testnet)

    async def connect(self) -> None:
        """Establish connection to Bitget.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info("Connecting to Bitget")

            await self.pool.create_session()
            await self._load_symbols()

            self.connected = True
            logger.info("Connected to Bitget successfully", symbols_count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to connect to Bitget", error=str(e))
            raise ConnectionError(f"Bitget connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to Bitget."""
        try:
            logger.info("Disconnecting from Bitget")
            await self.pool.close_session()
            self.connected = False
            logger.info("Disconnected from Bitget")
        except Exception as e:
            logger.error("Error during Bitget disconnect", error=str(e))

    async def _load_symbols(self) -> None:
        """Load available trading symbols from exchange."""
        try:
            endpoint = os.getenv('BITGET_SYMBOLS_ENDPOINT', '/api/spot/v1/public/products')
            url = f"{self.mapper.base_url}{endpoint}"

            data = await self.pool.get(url)

            self.symbols = []
            for symbol_info in data.get('data', []):
                if symbol_info.get('status') == 'online':
                    base = symbol_info['baseCoin']
                    quote = symbol_info['quoteCoin']
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
        """Fetch OHLCV data from Bitget.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
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

            symbol_bitget = symbol.replace('/', '')
            endpoint = os.getenv('BITGET_CANDLES_ENDPOINT', '/api/spot/v1/market/candles')
            url = f"{self.mapper.base_url}{endpoint}"

            # Bitget timeframe mapping
            timeframe_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1h', '4h': '4h', '1d': '1day', '1w': '1week'
            }
            bitget_timeframe = timeframe_map.get(timeframe, '1h')

            params = {
                'symbol': symbol_bitget,
                'period': bitget_timeframe,
                'limit': str(min(limit, int(os.getenv('BITGET_MAX_CANDLES_LIMIT', '1000'))))
            }

            if since is not None:
                params['after'] = str(since)

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
        """Fetch ticker data from Bitget.

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

            symbol_bitget = symbol.replace('/', '')
            endpoint = os.getenv('BITGET_TICKER_ENDPOINT', '/api/spot/v1/market/ticker')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {'symbol': symbol_bitget}
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
        """Fetch orderbook from Bitget.

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

            symbol_bitget = symbol.replace('/', '')
            endpoint = os.getenv('BITGET_DEPTH_ENDPOINT', '/api/spot/v1/market/depth')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_bitget,
                'limit': str(min(depth, int(os.getenv('BITGET_MAX_DEPTH', '150'))))
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
        """Fetch account balance from Bitget.

        Returns:
            Dictionary of currency balances

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BITGET_ACCOUNTS_ENDPOINT', '/api/spot/v1/account/assets')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('GET', endpoint)
            data = await self.pool.get(url, headers=headers)

            balances = self.mapper.map_balance_response(data)

            logger.debug("Balance fetched", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to fetch balance", error=str(e))
            raise RuntimeError(f"Failed to fetch balance: {e}")

    async def fetch_positions(self) -> List[Dict]:
        """Fetch open positions from Bitget Futures.

        Returns:
            List of position dictionaries

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            # Futures endpoint
            endpoint = os.getenv('BITGET_POSITIONS_ENDPOINT', '/api/mix/v1/position/allPosition')
            url = f"{self.mapper.base_url}{endpoint}"

            product_type = os.getenv('BITGET_PRODUCT_TYPE', 'umcbl')  # umcbl for USDT futures
            params = {'productType': product_type}

            headers = self.mapper.sign_request('GET', endpoint + '?' + '&'.join([f"{k}={v}" for k, v in params.items()]))
            data = await self.pool.get(url, headers=headers, params=params)

            positions = self.mapper.map_positions_response(data)

            logger.debug("Positions fetched", count=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise RuntimeError(f"Failed to fetch positions: {e}")

    async def create_order(self, order: 'Order') -> str:
        """Create order on Bitget.

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

            endpoint = os.getenv('BITGET_ORDERS_ENDPOINT', '/api/spot/v1/trade/orders')
            url = f"{self.mapper.base_url}{endpoint}"

            params = self.mapper.map_order_request(order)
            body = json.dumps(params)

            headers = self.mapper.sign_request('POST', endpoint, body)
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
        """Cancel order on Bitget.

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

            symbol_bitget = symbol.replace('/', '')
            endpoint = os.getenv('BITGET_ORDER_CANCEL_ENDPOINT', '/api/spot/v1/trade/cancel-order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_bitget,
                'orderId': order_id
            }
            body = json.dumps(params)

            headers = self.mapper.sign_request('POST', endpoint, body)
            await self.pool.post(url, headers=headers, json_data=params)

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
        """Fetch order status from Bitget.

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

            symbol_bitget = symbol.replace('/', '')
            endpoint = os.getenv('BITGET_ORDER_INFO_ENDPOINT', '/api/spot/v1/trade/orderInfo')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'symbol': symbol_bitget,
                'orderId': order_id
            }

            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            headers = self.mapper.sign_request('GET', endpoint + '?' + query_string)

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

            # Bitget doesn't have a direct fee endpoint, use default from config
            logger.debug("Using default trading fees for Bitget", symbol=symbol)

            fees = {
                'maker': Decimal(os.getenv('BITGET_DEFAULT_MAKER_FEE', '0.002')),
                'taker': Decimal(os.getenv('BITGET_DEFAULT_TAKER_FEE', '0.002'))
            }

            logger.debug("Trading fees retrieved", symbol=symbol, fees=fees)
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
