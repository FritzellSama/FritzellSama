"""
Bybit API mapper for data transformation.

This module handles transformation between Bybit API responses and
internal data models, ensuring type safety and proper decimal handling.
"""

import hmac
import hashlib
import time
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BybitAPIMapper:
    """Maps Bybit API data to internal models.

    Handles request signing, data transformation, and validation
    for all Bybit API interactions.

    Attributes:
        api_key: Bybit API key
        secret: Bybit API secret
        testnet: Whether using testnet
        base_url: API base URL
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize API mapper.

        Args:
            api_key: Bybit API key
            secret: Bybit API secret
            testnet: Whether to use testnet endpoints
        """
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet

        # Load URLs from environment
        if testnet:
            self.base_url = os.getenv('BYBIT_TESTNET_URL', 'https://api-testnet.bybit.com')
        else:
            self.base_url = os.getenv('BYBIT_API_URL', 'https://api.bybit.com')

        logger.info(
            "Bybit API mapper initialized",
            testnet=testnet,
            base_url=self.base_url
        )

    def sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request with HMAC-SHA256.

        Args:
            params: Request parameters

        Returns:
            Signature string
        """
        # Add timestamp and recv_window
        timestamp = str(int(time.time() * 1000))
        params['api_key'] = self.api_key
        params['timestamp'] = timestamp
        params['recv_window'] = os.getenv('BYBIT_RECV_WINDOW', '5000')

        # Create query string (sorted alphabetically)
        query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])

        # Generate signature
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        logger.debug("Request signed", params_count=len(params), timestamp=timestamp)
        return signature

    def get_headers(self) -> Dict[str, str]:
        """Get request headers.

        Returns:
            Headers dictionary
        """
        return {
            'Content-Type': 'application/json',
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-TIMESTAMP': str(int(time.time() * 1000)),
            'X-BAPI-RECV-WINDOW': os.getenv('BYBIT_RECV_WINDOW', '5000')
        }

    def map_ohlcv_response(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Transform Bybit OHLCV response to DataFrame.

        Args:
            data: Raw OHLCV data from Bybit

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]

        Raises:
            ValueError: If data format is invalid
        """
        try:
            result = data.get('result', {})
            candles = result.get('list', [])

            if not candles:
                return pl.DataFrame({
                    'timestamp': [],
                    'open': [],
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': []
                })

            # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for candle in candles:
                timestamps.append(datetime.fromtimestamp(int(candle[0]) / 1000, tz=timezone.utc))
                opens.append(Decimal(str(candle[1])))
                highs.append(Decimal(str(candle[2])))
                lows.append(Decimal(str(candle[3])))
                closes.append(Decimal(str(candle[4])))
                volumes.append(Decimal(str(candle[5])))

            df = pl.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })

            logger.debug("OHLCV data mapped", rows=len(df))
            return df

        except Exception as e:
            logger.error("Failed to map OHLCV data", error=str(e))
            raise ValueError(f"Invalid OHLCV data format: {e}")

    def map_ticker_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Bybit ticker response.

        Args:
            data: Raw ticker data from Bybit

        Returns:
            Normalized ticker dictionary
        """
        try:
            result = data.get('result', {})
            ticker = result.get('list', [{}])[0]

            return {
                'symbol': ticker['symbol'],
                'bid': Decimal(str(ticker['bid1Price'])),
                'ask': Decimal(str(ticker['ask1Price'])),
                'last': Decimal(str(ticker['lastPrice'])),
                'volume': Decimal(str(ticker['volume24h'])),
                'timestamp': datetime.now(timezone.utc)
            }
        except Exception as e:
            logger.error("Failed to map ticker data", error=str(e))
            raise ValueError(f"Invalid ticker data format: {e}")

    def map_orderbook_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Transform Bybit orderbook response.

        Args:
            data: Raw orderbook data from Bybit

        Returns:
            Tuple of (bids, asks) where each is list of (price, size) tuples
        """
        try:
            result = data.get('result', {})
            bids = [(Decimal(str(b[0])), Decimal(str(b[1]))) for b in result.get('b', [])]
            asks = [(Decimal(str(a[0])), Decimal(str(a[1]))) for a in result.get('a', [])]

            logger.debug(
                "Orderbook data mapped",
                bids_count=len(bids),
                asks_count=len(asks)
            )

            return bids, asks

        except Exception as e:
            logger.error("Failed to map orderbook data", error=str(e))
            raise ValueError(f"Invalid orderbook data format: {e}")

    def map_balance_response(self, data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Transform Bybit balance response.

        Args:
            data: Raw balance data from Bybit

        Returns:
            Dictionary mapping currency to available balance
        """
        try:
            result = data.get('result', {})
            balances = {}

            # Bybit V5 API structure
            balance_list = result.get('list', [])
            for account in balance_list:
                for coin in account.get('coin', []):
                    available = Decimal(str(coin.get('availableToWithdraw', '0')))
                    if available > Decimal('0'):
                        balances[coin['coin']] = available

            logger.debug("Balance data mapped", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to map balance data", error=str(e))
            raise ValueError(f"Invalid balance data format: {e}")

    def map_positions_response(self, data: Dict[str, Any]) -> List[Dict]:
        """Transform Bybit positions response.

        Args:
            data: Raw positions data from Bybit

        Returns:
            List of normalized position dictionaries
        """
        try:
            result = data.get('result', {})
            positions = []

            for pos in result.get('list', []):
                size = Decimal(str(pos.get('size', '0')))
                if size != Decimal('0'):
                    positions.append({
                        'symbol': pos['symbol'],
                        'size': size,
                        'entry_price': Decimal(str(pos.get('avgPrice', '0'))),
                        'unrealized_pnl': Decimal(str(pos.get('unrealisedPnl', '0'))),
                        'leverage': Decimal(str(pos.get('leverage', '1')))
                    })

            logger.debug("Positions data mapped", positions=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to map positions data", error=str(e))
            raise ValueError(f"Invalid positions data format: {e}")

    def map_order_request(self, order: 'Order') -> Dict[str, Any]:
        """Transform internal order to Bybit order request.

        Args:
            order: Internal order object

        Returns:
            Bybit order request parameters
        """
        try:
            params = {
                'category': os.getenv('BYBIT_DEFAULT_CATEGORY', 'linear'),  # spot, linear, inverse
                'symbol': order.symbol.replace('/', ''),
                'side': order.side.value.capitalize(),  # Buy or Sell
                'orderType': self._map_order_type(order.order_type),
                'qty': str(order.quantity)
            }

            if order.price is not None:
                params['price'] = str(order.price)
                params['timeInForce'] = os.getenv('BYBIT_DEFAULT_TIME_IN_FORCE', 'GTC')

            logger.debug("Order request mapped", order_type=order.order_type)
            return params

        except Exception as e:
            logger.error("Failed to map order request", error=str(e))
            raise ValueError(f"Invalid order format: {e}")

    def _map_order_type(self, order_type: 'OrderType') -> str:
        """Map internal order type to Bybit order type.

        Args:
            order_type: Internal order type

        Returns:
            Bybit order type string
        """
        mapping = {
            'MARKET': 'Market',
            'LIMIT': 'Limit',
            'STOP_LOSS': 'Stop',
            'TAKE_PROFIT': 'Stop',
            'STOP_LIMIT': 'Stop'
        }
        return mapping.get(order_type.value, 'Market')

    def map_order_response(self, data: Dict[str, Any]) -> str:
        """Extract order ID from Bybit order response.

        Args:
            data: Raw order response from Bybit

        Returns:
            Exchange order ID
        """
        try:
            result = data.get('result', {})
            order_id = str(result['orderId'])
            logger.debug("Order response mapped", order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Failed to map order response", error=str(e))
            raise ValueError(f"Invalid order response format: {e}")

    def map_order_status_response(self, data: Dict[str, Any]) -> Dict:
        """Transform Bybit order status response.

        Args:
            data: Raw order status data from Bybit

        Returns:
            Normalized order status dictionary
        """
        try:
            result = data.get('result', {})
            order_list = result.get('list', [])

            if not order_list:
                raise ValueError("Order not found")

            order = order_list[0]

            return {
                'order_id': str(order['orderId']),
                'status': order['orderStatus'].lower(),
                'filled_quantity': Decimal(str(order.get('cumExecQty', '0'))),
                'remaining_quantity': Decimal(str(order.get('qty', '0'))) - Decimal(str(order.get('cumExecQty', '0'))),
                'average_price': Decimal(str(order.get('avgPrice', '0'))),
                'fees': Decimal(str(order.get('cumExecFee', '0')))
            }
        except Exception as e:
            logger.error("Failed to map order status", error=str(e))
            raise ValueError(f"Invalid order status format: {e}")

    def map_trading_fees_response(self, data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Transform Bybit trading fees response.

        Args:
            data: Raw trading fees data from Bybit

        Returns:
            Dictionary with maker and taker fees
        """
        try:
            result = data.get('result', {})
            fee_list = result.get('list', [])

            if fee_list:
                fee = fee_list[0]
                return {
                    'maker': Decimal(str(fee.get('makerFeeRate', '0.001'))),
                    'taker': Decimal(str(fee.get('takerFeeRate', '0.001')))
                }

            # Default fees from config if not found
            return {
                'maker': Decimal(os.getenv('BYBIT_DEFAULT_MAKER_FEE', '0.001')),
                'taker': Decimal(os.getenv('BYBIT_DEFAULT_TAKER_FEE', '0.001'))
            }

        except Exception as e:
            logger.error("Failed to map trading fees", error=str(e))
            raise ValueError(f"Invalid trading fees format: {e}")
