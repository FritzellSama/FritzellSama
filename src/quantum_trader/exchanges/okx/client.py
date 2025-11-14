"""
OKX exchange client implementation.

This module provides the concrete implementation of the BaseExchange
interface for OKX cryptocurrency exchange.
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
from .api_mapper import OKXAPIMapper

logger = get_logger(__name__)


class OKXClient(BaseExchange):
    """OKX exchange client implementation.

    Provides full implementation of exchange operations for OKX,
    including spot and derivatives trading.

    Attributes:
        mapper: API data mapper
        pool: HTTP connection pool
        symbols: Available trading symbols
        passphrase: OKX API passphrase
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize OKX client.

        Args:
            api_key: OKX API key
            secret: OKX API secret
            testnet: Whether to use testnet

        Raises:
            ValueError: If credentials are invalid
        """
        super().__init__(api_key, secret, testnet)

        # OKX requires a passphrase - load from environment
        self.passphrase = os.getenv('OKX_API_PASSPHRASE', '')
        if not self.passphrase:
            raise ValueError("OKX_API_PASSPHRASE environment variable is required")

        self.mapper = OKXAPIMapper(api_key, secret, self.passphrase, testnet)
        self.pool = ConnectionPool()
        self.symbols: List[str] = []

        logger.info("OKX client initialized", testnet=testnet)

    async def connect(self) -> None:
        """Establish connection to OKX.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info("Connecting to OKX")

            await self.pool.create_session()
            await self._load_symbols()

            self.connected = True
            logger.info("Connected to OKX successfully", symbols_count=len(self.symbols))

        except Exception as e:
            logger.error("Failed to connect to OKX", error=str(e))
            raise ConnectionError(f"OKX connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to OKX."""
        try:
            logger.info("Disconnecting from OKX")
            await self.pool.close_session()
            self.connected = False
            logger.info("Disconnected from OKX")
        except Exception as e:
            logger.error("Error during OKX disconnect", error=str(e))

    async def _load_symbols(self) -> None:
        """Load available trading symbols from exchange."""
        try:
            endpoint = os.getenv('OKX_INSTRUMENTS_ENDPOINT', '/api/v5/public/instruments')
            url = f"{self.mapper.base_url}{endpoint}"

            inst_type = os.getenv('OKX_DEFAULT_INST_TYPE', 'SPOT')
            params = {'instType': inst_type}

            data = await self.pool.get(url, params=params)

            self.symbols = []
            for symbol_info in data.get('data', []):
                inst_id = symbol_info['instId']
                # Convert OKX format (BTC-USDT) to standard (BTC/USDT)
                self.symbols.append(inst_id.replace('-', '/'))

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
        """Fetch OHLCV data from OKX.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1H', '1D')
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

            symbol_okx = symbol.replace('/', '-')
            endpoint = os.getenv('OKX_CANDLES_ENDPOINT', '/api/v5/market/candles')
            url = f"{self.mapper.base_url}{endpoint}"

            # OKX uses uppercase for timeframes
            bar_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
            }
            okx_bar = bar_map.get(timeframe, '1H')

            params = {
                'instId': symbol_okx,
                'bar': okx_bar,
                'limit': str(min(limit, int(os.getenv('OKX_MAX_CANDLES_LIMIT', '300'))))
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
        """Fetch ticker data from OKX.

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

            symbol_okx = symbol.replace('/', '-')
            endpoint = os.getenv('OKX_TICKER_ENDPOINT', '/api/v5/market/ticker')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {'instId': symbol_okx}
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
        """Fetch orderbook from OKX.

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

            symbol_okx = symbol.replace('/', '-')
            endpoint = os.getenv('OKX_ORDERBOOK_ENDPOINT', '/api/v5/market/books')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'instId': symbol_okx,
                'sz': str(min(depth, int(os.getenv('OKX_MAX_DEPTH', '400'))))
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
        """Fetch account balance from OKX.

        Returns:
            Dictionary of currency balances

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('OKX_BALANCE_ENDPOINT', '/api/v5/account/balance')
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
        """Fetch open positions from OKX.

        Returns:
            List of position dictionaries

        Raises:
            RuntimeError: If fetch fails
        """
        try:
            if not self.connected:
                raise RuntimeError("Not connected to exchange")

            endpoint = os.getenv('OKX_POSITIONS_ENDPOINT', '/api/v5/account/positions')
            url = f"{self.mapper.base_url}{endpoint}"

            headers = self.mapper.sign_request('GET', endpoint)
            params = {
                'instType': os.getenv('OKX_POSITION_INST_TYPE', 'SWAP')
            }

            data = await self.pool.get(url, headers=headers, params=params)
            positions = self.mapper.map_positions_response(data)

            logger.debug("Positions fetched", count=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to fetch positions", error=str(e))
            raise RuntimeError(f"Failed to fetch positions: {e}")

    async def create_order(self, order: 'Order') -> str:
        """Create order on OKX.

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

            endpoint = os.getenv('OKX_ORDER_ENDPOINT', '/api/v5/trade/order')
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
        """Cancel order on OKX.

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

            symbol_okx = symbol.replace('/', '-')
            endpoint = os.getenv('OKX_ORDER_CANCEL_ENDPOINT', '/api/v5/trade/cancel-order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'instId': symbol_okx,
                'ordId': order_id
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
        """Fetch order status from OKX.

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

            symbol_okx = symbol.replace('/', '-')
            endpoint = os.getenv('OKX_ORDER_STATUS_ENDPOINT', '/api/v5/trade/order')
            url = f"{self.mapper.base_url}{endpoint}"

            params = {
                'instId': symbol_okx,
                'ordId': order_id
            }

            # For GET with params, include params in signature
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signed_endpoint = f"{endpoint}?{query_string}"

            headers = self.mapper.sign_request('GET', signed_endpoint)
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

            endpoint = os.getenv('OKX_TRADE_FEE_ENDPOINT', '/api/v5/account/trade-fee')
            url = f"{self.mapper.base_url}{endpoint}"

            symbol_okx = symbol.replace('/', '-')
            params = {
                'instType': os.getenv('OKX_DEFAULT_INST_TYPE', 'SPOT'),
                'instId': symbol_okx
            }

            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signed_endpoint = f"{endpoint}?{query_string}"

            headers = self.mapper.sign_request('GET', signed_endpoint)
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
