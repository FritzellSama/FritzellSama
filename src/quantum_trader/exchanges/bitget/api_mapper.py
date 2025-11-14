"""
Bitget API mapper for data transformation.

This module handles transformation between Bitget API responses and
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


class BitgetAPIMapper:
    """Maps Bitget API data to internal models.

    Handles request signing, data transformation, and validation
    for all Bitget API interactions.

    Attributes:
        api_key: Bitget API key
        secret: Bitget API secret
        passphrase: Bitget API passphrase
        testnet: Whether using testnet
        base_url: API base URL
    """

    def __init__(self, api_key: str, secret: str, passphrase: str, testnet: bool = False) -> None:
        """Initialize API mapper.

        Args:
            api_key: Bitget API key
            secret: Bitget API secret
            passphrase: Bitget API passphrase
            testnet: Whether to use testnet endpoints
        """
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.testnet = testnet

        # Load URLs from environment
        if testnet:
            self.base_url = os.getenv('BITGET_TESTNET_URL', 'https://api.bitget.com')
        else:
            self.base_url = os.getenv('BITGET_API_URL', 'https://api.bitget.com')

        logger.info(
            "Bitget API mapper initialized",
            testnet=testnet,
            base_url=self.base_url
        )

    def sign_request(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Sign request with HMAC-SHA256 and return headers.

        Args:
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body as string

        Returns:
            Headers dictionary with signature
        """
        timestamp = str(int(time.time() * 1000))

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

        logger.debug("Request signed", timestamp=timestamp)

        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json',
            'locale': os.getenv('BITGET_LOCALE', 'en-US')
        }

    def map_ohlcv_response(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Transform Bitget OHLCV response to DataFrame.

        Args:
            data: Raw OHLCV data from Bitget

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

            # Bitget returns: [timestamp, open, high, low, close, volume, ...]
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
        """Transform Bitget ticker response.

        Args:
            data: Raw ticker data from Bitget

        Returns:
            Normalized ticker dictionary
        """
        try:
            ticker = data.get('data', {})

            return {
                'symbol': ticker['symbol'],
                'bid': Decimal(str(ticker['bestBid'])),
                'ask': Decimal(str(ticker['bestAsk'])),
                'last': Decimal(str(ticker['lastPr'])),
                'volume': Decimal(str(ticker['baseVolume'])),
                'timestamp': datetime.fromtimestamp(int(ticker['ts']) / 1000, tz=timezone.utc)
            }
        except Exception as e:
            logger.error("Failed to map ticker data", error=str(e))
            raise ValueError(f"Invalid ticker data format: {e}")

    def map_orderbook_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Transform Bitget orderbook response.

        Args:
            data: Raw orderbook data from Bitget

        Returns:
            Tuple of (bids, asks) where each is list of (price, size) tuples
        """
        try:
            orderbook = data.get('data', {})
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
        """Transform Bitget balance response.

        Args:
            data: Raw balance data from Bitget

        Returns:
            Dictionary mapping currency to available balance
        """
        try:
            balances = {}
            balance_list = data.get('data', [])

            for balance in balance_list:
                available = Decimal(str(balance.get('available', '0')))
                if available > Decimal('0'):
                    balances[balance['coin']] = available

            logger.debug("Balance data mapped", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to map balance data", error=str(e))
            raise ValueError(f"Invalid balance data format: {e}")

    def map_positions_response(self, data: Dict[str, Any]) -> List[Dict]:
        """Transform Bitget positions response.

        Args:
            data: Raw positions data from Bitget

        Returns:
            List of normalized position dictionaries
        """
        try:
            positions = []
            position_list = data.get('data', [])

            for pos in position_list:
                total = Decimal(str(pos.get('total', '0')))
                if total != Decimal('0'):
                    positions.append({
                        'symbol': pos['symbol'],
                        'size': total,
                        'entry_price': Decimal(str(pos.get('averageOpenPrice', '0'))),
                        'unrealized_pnl': Decimal(str(pos.get('unrealizedPL', '0'))),
                        'leverage': Decimal(str(pos.get('leverage', '1')))
                    })

            logger.debug("Positions data mapped", positions=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to map positions data", error=str(e))
            raise ValueError(f"Invalid positions data format: {e}")

    def map_order_request(self, order: 'Order') -> Dict[str, Any]:
        """Transform internal order to Bitget order request.

        Args:
            order: Internal order object

        Returns:
            Bitget order request parameters
        """
        try:
            params = {
                'symbol': order.symbol.replace('/', ''),
                'side': order.side.value.lower(),
                'orderType': self._map_order_type(order.order_type),
                'size': str(order.quantity),
                'clientOid': f"{order.strategy}_{int(time.time() * 1000)}"
            }

            if order.price is not None:
                params['price'] = str(order.price)
                params['force'] = os.getenv('BITGET_DEFAULT_FORCE', 'normal')  # normal, post_only, fok, ioc

            logger.debug("Order request mapped", order_type=order.order_type)
            return params

        except Exception as e:
            logger.error("Failed to map order request", error=str(e))
            raise ValueError(f"Invalid order format: {e}")

    def _map_order_type(self, order_type: 'OrderType') -> str:
        """Map internal order type to Bitget order type.

        Args:
            order_type: Internal order type

        Returns:
            Bitget order type string
        """
        mapping = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP_LOSS': 'stop',
            'TAKE_PROFIT': 'profit',
            'STOP_LIMIT': 'stop_limit'
        }
        return mapping.get(order_type.value, 'market')

    def map_order_response(self, data: Dict[str, Any]) -> str:
        """Extract order ID from Bitget order response.

        Args:
            data: Raw order response from Bitget

        Returns:
            Exchange order ID
        """
        try:
            order_data = data.get('data', {})
            order_id = str(order_data['orderId'])
            logger.debug("Order response mapped", order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Failed to map order response", error=str(e))
            raise ValueError(f"Invalid order response format: {e}")

    def map_order_status_response(self, data: Dict[str, Any]) -> Dict:
        """Transform Bitget order status response.

        Args:
            data: Raw order status data from Bitget

        Returns:
            Normalized order status dictionary
        """
        try:
            order = data.get('data', {})

            return {
                'order_id': str(order['orderId']),
                'status': order['state'].lower(),
                'filled_quantity': Decimal(str(order.get('accBaseVolume', '0'))),
                'remaining_quantity': Decimal(str(order.get('size', '0'))) - Decimal(str(order.get('accBaseVolume', '0'))),
                'average_price': Decimal(str(order.get('priceAvg', '0'))),
                'fees': Decimal(str(order.get('fee', '0')))
            }
        except Exception as e:
            logger.error("Failed to map order status", error=str(e))
            raise ValueError(f"Invalid order status format: {e}")

    def map_trading_fees_response(self, data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Transform Bitget trading fees response.

        Args:
            data: Raw trading fees data from Bitget

        Returns:
            Dictionary with maker and taker fees
        """
        try:
            fee_data = data.get('data', {})

            return {
                'maker': Decimal(str(fee_data.get('makerFeeRate', '0.002'))),
                'taker': Decimal(str(fee_data.get('takerFeeRate', '0.002')))
            }

        except Exception as e:
            logger.error("Failed to map trading fees", error=str(e))
            # Default fees from config if mapping fails
            return {
                'maker': Decimal(os.getenv('BITGET_DEFAULT_MAKER_FEE', '0.002')),
                'taker': Decimal(os.getenv('BITGET_DEFAULT_TAKER_FEE', '0.002'))
            }
