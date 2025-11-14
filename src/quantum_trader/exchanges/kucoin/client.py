"""
KuCoin exchange client implementation.

This module provides the concrete implementation of the BaseExchange
interface for KuCoin cryptocurrency exchange.
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
from .api_mapper import KuCoinAPIMapper

logger = get_logger(__name__)


class KuCoinClient(BaseExchange):
    """KuCoin exchange client implementation.

    Provides full implementation of exchange operations for KuCoin,
    including spot and futures trading.

    Attributes:
        mapper: API data mapper
        pool: HTTP connection pool
        symbols: Available trading symbols
        passphrase: KuCoin API passphrase
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize KuCoin client.

        Args:
            api_key: KuCoin API key
            secret: KuCoin API secret
            testnet: Whether to use testnet

        Raises:
            ValueError: If credentials are invalid
        """
        super().__init__(api_key, secret, testnet)

        # KuCoin requires a passphrase - load from environment
        self.passphrase = os.getenv('KUCOIN_API_PASSPHRASE', '')
        if not self.passphrase:
            raise ValueError("KUCOIN_API_PASSPHRASE environment variable is required")

        self.mapper = KuCoinAPIMapper(api_key, secret, self.passphrase, testnet)
        self.pool = ConnectionPool()
        self.symbols: List[str] = []

        logger.info("KuCoin client initialized", testnet=testnet)

    async def connect(self) -> None:
        """Establish connection to KuCoin.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info("Connecting to KuCoin")

            await self.pool.create_session()
            await self._load_symbols()

            self.connected = True
            logger.info("Connected to KuCoin successfully", symbols_count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to connect to KuCoin", error=str(e))
            raise ConnectionError(f"KuCoin connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to KuCoin."""
        try:
            logger.info("Disconnecting from KuCoin")
            await self.pool.close_session()
            self.connected = False
            logger.info("Disconnected from KuCoin")
        except Exception as e:
            logger.error("Error during KuCoin disconnect", error=str(e))

    async def _load_symbols(self) -> None:
        """Load available trading symbols from exchange."""
        try:
            endpoint = os.getenv('KUCOIN_SYMBOLS_ENDPOINT', '/api/v1/symbols')
            url = f"{self.mapper.base_url}{endpoint}"

            data = await self.pool.get(url)

            self.symbols = []
            for symbol_info in data.get('data', []):
                if symbol_info.get('enableTrading'):
                    # KuCoin uses BTC-USDT format, convert to BTC/USDT
                    symbol = symbol_info['symbol'].replace('-', '/')
                    self.symbols.append(symbol)

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
        """Fetch OHLCV data from KuCoin.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1min', '5min', '1hour', '1day')
            limit: Number of candles to fetch
            since: Timestamp in seconds

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            symbol_kucoin = symbol.replace('/', '-')
            endpoint = os.getenv('KUCOIN_KLINES_ENDPOINT', '/api/v1/market/candles')
            url = f"{self.mapper.base_url}{endpoint}"

            # KuCoin timeframe mapping
            timeframe_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1hour', '4h': '4hour', '1d': '1day', '1w': '1week'
            }
            kucoin_timeframe = timeframe_map.get(timeframe, '1hour')

            params = {
                'symbol': symbol_kucoin,
                'type': kucoin_timeframe
            }

            if since is not None:
                params['startAt'] = since

            data = await self.pool.get(url, params=params)
            df = self.mapper.map_ohlcv_response(data)

            # Limit results
            if len(df) > limit:
                df = df.head(limit)

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
        """Fetch ticker data from KuCoin.

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

            symbol_kucoin = symbol.replace('/', '-')
            endpoint = os.getenv('KUCOIN_TICKER_ENDPOINT', '/api/v1/market/orderbook/level1')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {'symbol': symbol_kucoin}
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
        """Fetch orderbook from KuCoin.

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

            symbol_kucoin = symbol.replace('/', '-')
            endpoint = os.getenv('KUCOIN_ORDERBOOK_ENDPOINT', '/api/v1/market/orderbook/level2_20')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {'symbol': symbol_kucoin}
            data = await self.pool.get(url, params=params)
            bids, asks = self.mapper.map_orderbook_response(data)

            # Limit depth
            bids = bids[:depth]
            asks = asks[:depth]

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
        """Fetch account balance from KuCoin.

        Returns:
            Dictionary of currency balances

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('KUCOIN_ACCOUNTS_ENDPOINT', '/api/v1/accounts')
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
        """Fetch open positions from KuCoin Futures.

        Returns:
            List of position dictionaries

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            # Futures endpoint
            endpoint = os.getenv('KUCOIN_POSITIONS_ENDPOINT', '/api/v1/positions')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('GET', endpoint)
            data = await self.pool.get(url, headers=headers)

            positions = self.mapper.map_positions_response(data)

            logger.debug("Positions fetched", count=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise RuntimeError(f"Failed to fetch positions: {e}")

    async def create_order(self, order: 'Order') -> str:
        """Create order on KuCoin.

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

            endpoint = os.getenv('KUCOIN_ORDERS_ENDPOINT', '/api/v1/orders')
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
        """Cancel order on KuCoin.

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

            endpoint = f"/api/v1/orders/{order_id}"
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('DELETE', endpoint)
            await self.pool.delete(url, headers=headers)

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
        """Fetch order status from KuCoin.

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

            endpoint = f"/api/v1/orders/{order_id}"
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('GET', endpoint)
            data = await self.pool.get(url, headers=headers)

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

            endpoint = os.getenv('KUCOIN_BASE_FEE_ENDPOINT', '/api/v1/base-fee')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('GET', endpoint)
            data = await self.pool.get(url, headers=headers)

            fees = self.mapper.map_trading_fees_response(data)

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
