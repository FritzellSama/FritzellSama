"""
Bybit exchange client implementation.

This module provides the concrete implementation of the BaseExchange
interface for Bybit cryptocurrency exchange.
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
from .api_mapper import BybitAPIMapper

logger = get_logger(__name__)


class BybitClient(BaseExchange):
    """Bybit exchange client implementation.

    Provides full implementation of exchange operations for Bybit,
    including spot and derivatives trading.

    Attributes:
        mapper: API data mapper
        pool: HTTP connection pool
        symbols: Available trading symbols
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize Bybit client.

        Args:
            api_key: Bybit API key
            secret: Bybit API secret
            testnet: Whether to use testnet

        Raises:
            ValueError: If credentials are invalid
        """
        super().__init__(api_key, secret, testnet)

        self.mapper = BybitAPIMapper(api_key, secret, testnet)
        self.pool = ConnectionPool()
        self.symbols: List[str] = []

        logger.info("Bybit client initialized", testnet=testnet)

    async def connect(self) -> None:
        """Establish connection to Bybit.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info("Connecting to Bybit")

            await self.pool.create_session()
            await self._load_symbols()

            self.connected = True
            logger.info("Connected to Bybit successfully", symbols_count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to connect to Bybit", error=str(e))
            raise ConnectionError(f"Bybit connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to Bybit."""
        try:
            logger.info("Disconnecting from Bybit")
            await self.pool.close_session()
            self.connected = False
            logger.info("Disconnected from Bybit")
        except Exception as e:
            logger.error("Error during Bybit disconnect", error=str(e))

    async def _load_symbols(self) -> None:
        """Load available trading symbols from exchange."""
        try:
            endpoint = os.getenv('BYBIT_INSTRUMENTS_ENDPOINT', '/v5/market/instruments-info')
            url = f"{self.mapper.base_url}{endpoint}"

            category = os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear')
            params = {'category': category}

            data = await self.pool.get(url, params=params)

            self.symbols = []
            result = data.get('result', {})
            for symbol_info in result.get('list', []):
                if symbol_info.get('status') == 'Trading':
                    symbol = symbol_info['symbol']
                    # Try to format as BASE/QUOTE
                    base = symbol_info.get('baseCoin', '')
                    quote = symbol_info.get('quoteCoin', '')
                    if base and quote:
                        self.symbols.append(f"{base}/{quote}")
                    else:
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
        """Fetch OHLCV data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1', '5', '60', 'D')
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

            symbol_bybit = symbol.replace('/', '')
            endpoint = os.getenv('BYBIT_KLINE_ENDPOINT', '/v5/market/kline')
            url = f"{self.mapper.base_url}{endpoint}"

            # Bybit uses different interval format
            interval_map = {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(timeframe, '60')

            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit,
                'interval': bybit_interval,
                'limit': min(limit, int(os.getenv('BYBIT_MAX_KLINES_LIMIT', '1000')))
            }

            if since is not None:
                params['start'] = since

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
        """Fetch ticker data from Bybit.

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

            symbol_bybit = symbol.replace('/', '')
            endpoint = os.getenv('BYBIT_TICKER_ENDPOINT', '/v5/market/tickers')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit
            }

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
        """Fetch orderbook from Bybit.

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

            symbol_bybit = symbol.replace('/', '')
            endpoint = os.getenv('BYBIT_ORDERBOOK_ENDPOINT', '/v5/market/orderbook')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit,
                'limit': min(depth, int(os.getenv('BYBIT_MAX_DEPTH', '500')))
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
        """Fetch account balance from Bybit.

        Returns:
            Dictionary of currency balances

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BYBIT_WALLET_ENDPOINT', '/v5/account/wallet-balance')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.get_headers()
            params = {
                'accountType': os.getenv('BYBIT_ACCOUNT_TYPE', 'UNIFIED')
            }

            data = await self.pool.get(url, headers=headers, params=params)
            balances = self.mapper.map_balance_response(data)

            logger.debug("Balance fetched", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to fetch balance", error=str(e))
            raise RuntimeError(f"Failed to fetch balance: {e}")

    async def fetch_positions(self) -> List[Dict]:
        """Fetch open positions from Bybit.

        Returns:
            List of position dictionaries

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('BYBIT_POSITION_ENDPOINT', '/v5/position/list')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.get_headers()
            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'settleCoin': os.getenv('BYBIT_SETTLE_COIN', 'USDT')
            }

            data = await self.pool.get(url, headers=headers, params=params)
            positions = self.mapper.map_positions_response(data)

            logger.debug("Positions fetched", count=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise RuntimeError(f"Failed to fetch positions: {e}")

    async def create_order(self, order: 'Order') -> str:
        """Create order on Bybit.

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

            endpoint = os.getenv('BYBIT_ORDER_ENDPOINT', '/v5/order/create')
            url = f"{self.mapper.base_url}{endpoint}"

            params = self.mapper.map_order_request(order)
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
        """Cancel order on Bybit.

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

            symbol_bybit = symbol.replace('/', '')
            endpoint = os.getenv('BYBIT_ORDER_CANCEL_ENDPOINT', '/v5/order/cancel')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit,
                'orderId': order_id
            }

            headers = self.mapper.get_headers()
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
        """Fetch order status from Bybit.

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

            symbol_bybit = symbol.replace('/', '')
            endpoint = os.getenv('BYBIT_ORDER_HISTORY_ENDPOINT', '/v5/order/realtime')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit,
                'orderId': order_id
            }

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

            endpoint = os.getenv('BYBIT_FEE_RATE_ENDPOINT', '/v5/account/fee-rate')
            url = f"{self.mapper.base_url}{endpoint}"

            symbol_bybit = symbol.replace('/', '')
            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),
                'symbol': symbol_bybit
            }

            headers = self.mapper.get_headers()
            data = await self.pool.get(url, headers=headers, params=params)

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
