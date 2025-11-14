"""
OKX API mapper for data transformation.

This module handles transformation between OKX API responses and
internal data models, ensuring type safety and proper decimal handling.
"""

import hmac
import hashlib
import base64
import time
import os
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class OKXAPIMapper:
    """Maps OKX API data to internal models.

    Handles request signing, data transformation, and validation
    for all OKX API interactions.

    Attributes:
        api_key: OKX API key
        secret: OKX API secret
        passphrase: OKX API passphrase
        testnet: Whether using testnet
        base_url: API base URL
    """

    def __init__(self, api_key: str, secret: str, passphrase: str, testnet: bool = False) -> None:
        """Initialize API mapper.

        Args:
            api_key: OKX API key
            secret: OKX API secret
            passphrase: OKX API passphrase
            testnet: Whether to use testnet endpoints
        """
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.testnet = testnet

        # Load URLs from environment
        if testnet:
            self.base_url = os.getenv('OKX_TESTNET_URL', 'https://www.okx.com')
        else:
            self.base_url = os.getenv('OKX_API_URL', 'https://www.okx.com')

        logger.info(
            "OKX API mapper initialized",
            testnet=testnet,
            base_url=self.base_url
        )

    def sign_request(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Sign request with HMAC-SHA256 and return headers.

        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body as JSON string

        Returns:
            Headers dictionary with signature
        """
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        # Create prehash string
        prehash = timestamp + method + request_path + body

        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                self.secret.encode('utf-8'),
                prehash.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()

        # Sign passphrase
        passphrase_signature = base64.b64encode(
            hmac.new(
                self.secret.encode('utf-8'),
                self.passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()

        logger.debug("Request signed", timestamp=timestamp)

        return {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': passphrase_signature,
            'Content-Type': 'application/json'
        }

    def map_ohlcv_response(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Transform OKX OHLCV response to DataFrame.

        Args:
            data: Raw OHLCV data from OKX

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]

        Raises:
            ValueError: If data format is invalid
        """
        try:
            candles = data.get('data', [])

            if not candles:
                return pl.DataFrame({
                    'timestamp': [],
                    'open': [],
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': []
                })

            # OKX returns: [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
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
        """Transform OKX ticker response.

        Args:
            data: Raw ticker data from OKX

        Returns:
            Normalized ticker dictionary
        """
        try:
            ticker_list = data.get('data', [])
            if not ticker_list:
                raise ValueError("No ticker data found")

            ticker = ticker_list[0]

            return {
                'symbol': ticker['instId'],
                'bid': Decimal(str(ticker['bidPx'])),
                'ask': Decimal(str(ticker['askPx'])),
                'last': Decimal(str(ticker['last'])),
                'volume': Decimal(str(ticker['vol24h'])),
                'timestamp': datetime.fromtimestamp(int(ticker['ts']) / 1000, tz=timezone.utc)
            }
        except Exception as e:
            logger.error("Failed to map ticker data", error=str(e))
            raise ValueError(f"Invalid ticker data format: {e}")

    def map_orderbook_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Transform OKX orderbook response.

        Args:
            data: Raw orderbook data from OKX

        Returns:
            Tuple of (bids, asks) where each is list of (price, size) tuples
        """
        try:
            orderbook_list = data.get('data', [])
            if not orderbook_list:
                return [], []

            orderbook = orderbook_list[0]
            bids = [(Decimal(str(b[0])), Decimal(str(b[1]))) for b in orderbook.get('bids', [])]
            asks = [(Decimal(str(a[0])), Decimal(str(a[1]))) for a in orderbook.get('asks', [])]

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
        """Transform OKX balance response.

        Args:
            data: Raw balance data from OKX

        Returns:
            Dictionary mapping currency to available balance
        """
        try:
            balances = {}
            account_list = data.get('data', [])

            for account in account_list:
                for detail in account.get('details', []):
                    available = Decimal(str(detail.get('availBal', '0')))
                    if available > Decimal('0'):
                        balances[detail['ccy']] = available

            logger.debug("Balance data mapped", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to map balance data", error=str(e))
            raise ValueError(f"Invalid balance data format: {e}")

    def map_positions_response(self, data: Dict[str, Any]) -> List[Dict]:
        """Transform OKX positions response.

        Args:
            data: Raw positions data from OKX

        Returns:
            List of normalized position dictionaries
        """
        try:
            positions = []
            position_list = data.get('data', [])

            for pos in position_list:
                size = Decimal(str(pos.get('pos', '0')))
                if size != Decimal('0'):
                    positions.append({
                        'symbol': pos['instId'],
                        'size': size,
                        'entry_price': Decimal(str(pos.get('avgPx', '0'))),
                        'unrealized_pnl': Decimal(str(pos.get('upl', '0'))),
                        'leverage': Decimal(str(pos.get('lever', '1')))
                    })

            logger.debug("Positions data mapped", positions=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to map positions data", error=str(e))
            raise ValueError(f"Invalid positions data format: {e}")

    def map_order_request(self, order: 'Order') -> Dict[str, Any]:
        """Transform internal order to OKX order request.

        Args:
            order: Internal order object

        Returns:
            OKX order request parameters
        """
        try:
            params = {
                'instId': order.symbol.replace('/', '-'),
                'tdMode': os.getenv('OKX_DEFAULT_TD_MODE', 'cash'),  # cash, cross, isolated
                'side': order.side.value.lower(),
                'ordType': self._map_order_type(order.order_type),
                'sz': str(order.quantity)
            }

            if order.price is not None:
                params['px'] = str(order.price)

            logger.debug("Order request mapped", order_type=order.order_type)
            return params

        except Exception as e:
            logger.error("Failed to map order request", error=str(e))
            raise ValueError(f"Invalid order format: {e}")

    def _map_order_type(self, order_type: 'OrderType') -> str:
        """Map internal order type to OKX order type.

        Args:
            order_type: Internal order type

        Returns:
            OKX order type string
        """
        mapping = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP_LOSS': 'stop_market',
            'TAKE_PROFIT': 'stop_limit',
            'STOP_LIMIT': 'stop_limit'
        }
        return mapping.get(order_type.value, 'market')

    def map_order_response(self, data: Dict[str, Any]) -> str:
        """Extract order ID from OKX order response.

        Args:
            data: Raw order response from OKX

        Returns:
            Exchange order ID
        """
        try:
            order_list = data.get('data', [])
            if not order_list:
                raise ValueError("No order data in response")

            order_id = str(order_list[0]['ordId'])
            logger.debug("Order response mapped", order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Failed to map order response", error=str(e))
            raise ValueError(f"Invalid order response format: {e}")

    def map_order_status_response(self, data: Dict[str, Any]) -> Dict:
        """Transform OKX order status response.

        Args:
            data: Raw order status data from OKX

        Returns:
            Normalized order status dictionary
        """
        try:
            order_list = data.get('data', [])
            if not order_list:
                raise ValueError("Order not found")

            order = order_list[0]

            return {
                'order_id': str(order['ordId']),
                'status': order['state'].lower(),
                'filled_quantity': Decimal(str(order.get('accFillSz', '0'))),
                'remaining_quantity': Decimal(str(order.get('sz', '0'))) - Decimal(str(order.get('accFillSz', '0'))),
                'average_price': Decimal(str(order.get('avgPx', '0'))),
                'fees': Decimal(str(order.get('fee', '0')))
            }
        except Exception as e:
            logger.error("Failed to map order status", error=str(e))
            raise ValueError(f"Invalid order status format: {e}")

    def map_trading_fees_response(self, data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Transform OKX trading fees response.

        Args:
            data: Raw trading fees data from OKX

        Returns:
            Dictionary with maker and taker fees
        """
        try:
            fee_list = data.get('data', [])

            if fee_list:
                fee = fee_list[0]
                return {
                    'maker': Decimal(str(fee.get('maker', '0.0008'))),
                    'taker': Decimal(str(fee.get('taker', '0.001')))
                }

            # Default fees from config if not found
            return {
                'maker': Decimal(os.getenv('OKX_DEFAULT_MAKER_FEE', '0.0008')),
                'taker': Decimal(os.getenv('OKX_DEFAULT_TAKER_FEE', '0.001'))
            }

        except Exception as e:
            logger.error("Failed to map trading fees", error=str(e))
            raise ValueError(f"Invalid trading fees format: {e}")
