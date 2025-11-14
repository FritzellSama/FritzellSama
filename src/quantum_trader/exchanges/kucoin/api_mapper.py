"""
KuCoin API mapper for data transformation.

This module handles transformation between KuCoin API responses and
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


class KuCoinAPIMapper:
    """Maps KuCoin API data to internal models.

    Handles request signing, data transformation, and validation
    for all KuCoin API interactions.

    Attributes:
        api_key: KuCoin API key
        secret: KuCoin API secret
        passphrase: KuCoin API passphrase
        testnet: Whether using testnet
        base_url: API base URL
    """

    def __init__(self, api_key: str, secret: str, passphrase: str, testnet: bool = False) -> None:
        """Initialize API mapper.

        Args:
            api_key: KuCoin API key
            secret: KuCoin API secret
            passphrase: KuCoin API passphrase
            testnet: Whether to use testnet endpoints
        """
        self.api_key = api_key
        self.secret = secret
        self.passphrase = passphrase
        self.testnet = testnet

        # Load URLs from environment
        if testnet:
            self.base_url = os.getenv('KUCOIN_TESTNET_URL', 'https://openapi-sandbox.kucoin.com')
        else:
            self.base_url = os.getenv('KUCOIN_API_URL', 'https://api.kucoin.com')

        logger.info(
            "KuCoin API mapper initialized",
            testnet=testnet,
            base_url=self.base_url
        )

    def sign_request(self, method: str, endpoint: str, body: str = '') -> Dict[str, str]:
        """Sign request with HMAC-SHA256 and return headers.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            body: Request body as string

        Returns:
            Headers dictionary with signature
        """
        timestamp = str(int(time.time() * 1000))

        # Create signature string
        str_to_sign = timestamp + method + endpoint + body

        # Generate signature
        signature = base64.b64encode(
            hmac.new(
                self.secret.encode('utf-8'),
                str_to_sign.encode('utf-8'),
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
            'KC-API-KEY': self.api_key,
            'KC-API-SIGN': signature,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-PASSPHRASE': passphrase_signature,
            'KC-API-KEY-VERSION': os.getenv('KUCOIN_API_VERSION', '2'),
            'Content-Type': 'application/json'
        }

    def map_ohlcv_response(self, data: Dict[str, Any]) -> pl.DataFrame:
        """Transform KuCoin OHLCV response to DataFrame.

        Args:
            data: Raw OHLCV data from KuCoin

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

            # KuCoin returns: [timestamp, open, close, high, low, volume, turnover]
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for candle in candles:
                timestamps.append(datetime.fromtimestamp(int(candle[0]), tz=timezone.utc))
                opens.append(Decimal(str(candle[1])))
                highs.append(Decimal(str(candle[3])))
                lows.append(Decimal(str(candle[4])))
                closes.append(Decimal(str(candle[2])))
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
        """Transform KuCoin ticker response.

        Args:
            data: Raw ticker data from KuCoin

        Returns:
            Normalized ticker dictionary
        """
        try:
            ticker = data.get('data', {})

            return {
                'symbol': ticker['symbol'],
                'bid': Decimal(str(ticker['bestBid'])),
                'ask': Decimal(str(ticker['bestAsk'])),
                'last': Decimal(str(ticker['last'])),
                'volume': Decimal(str(ticker['vol'])),
                'timestamp': datetime.fromtimestamp(int(ticker['time']) / 1000, tz=timezone.utc)
            }
        except Exception as e:
            logger.error("Failed to map ticker data", error=str(e))
            raise ValueError(f"Invalid ticker data format: {e}")

    def map_orderbook_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Transform KuCoin orderbook response.

        Args:
            data: Raw orderbook data from KuCoin

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
        """Transform KuCoin balance response.

        Args:
            data: Raw balance data from KuCoin

        Returns:
            Dictionary mapping currency to available balance
        """
        try:
            balances = {}
            account_list = data.get('data', [])

            for account in account_list:
                available = Decimal(str(account.get('available', '0')))
                if available > Decimal('0'):
                    balances[account['currency']] = available

            logger.debug("Balance data mapped", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to map balance data", error=str(e))
            raise ValueError(f"Invalid balance data format: {e}")

    def map_positions_response(self, data: Dict[str, Any]) -> List[Dict]:
        """Transform KuCoin positions response.

        Args:
            data: Raw positions data from KuCoin

        Returns:
            List of normalized position dictionaries
        """
        try:
            positions = []
            position_list = data.get('data', [])

            for pos in position_list:
                current_qty = Decimal(str(pos.get('currentQty', '0')))
                if current_qty != Decimal('0'):
                    positions.append({
                        'symbol': pos['symbol'],
                        'size': current_qty,
                        'entry_price': Decimal(str(pos.get('avgEntryPrice', '0'))),
                        'unrealized_pnl': Decimal(str(pos.get('unrealisedPnl', '0'))),
                        'leverage': Decimal(str(pos.get('realLeverage', '1')))
                    })

            logger.debug("Positions data mapped", positions=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to map positions data", error=str(e))
            raise ValueError(f"Invalid positions data format: {e}")

    def map_order_request(self, order: 'Order') -> Dict[str, Any]:
        """Transform internal order to KuCoin order request.

        Args:
            order: Internal order object

        Returns:
            KuCoin order request parameters
        """
        try:
            client_oid = f"{order.strategy}_{int(time.time() * 1000)}"

            params = {
                'clientOid': client_oid,
                'side': order.side.value.lower(),
                'symbol': order.symbol.replace('/', '-'),
                'type': self._map_order_type(order.order_type)
            }

            if order.order_type.value == 'MARKET':
                # Market orders use size for sell, funds for buy
                if order.side.value == 'BUY':
                    params['funds'] = str(order.quantity)
                else:
                    params['size'] = str(order.quantity)
            else:
                params['size'] = str(order.quantity)
                params['price'] = str(order.price)

            logger.debug("Order request mapped", order_type=order.order_type)
            return params

        except Exception as e:
            logger.error("Failed to map order request", error=str(e))
            raise ValueError(f"Invalid order format: {e}")

    def _map_order_type(self, order_type: 'OrderType') -> str:
        """Map internal order type to KuCoin order type.

        Args:
            order_type: Internal order type

        Returns:
            KuCoin order type string
        """
        mapping = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP_LOSS': 'stop',
            'TAKE_PROFIT': 'stop',
            'STOP_LIMIT': 'stop'
        }
        return mapping.get(order_type.value, 'market')

    def map_order_response(self, data: Dict[str, Any]) -> str:
        """Extract order ID from KuCoin order response.

        Args:
            data: Raw order response from KuCoin

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
        """Transform KuCoin order status response.

        Args:
            data: Raw order status data from KuCoin

        Returns:
            Normalized order status dictionary
        """
        try:
            order = data.get('data', {})

            return {
                'order_id': str(order['id']),
                'status': order['isActive'] and 'active' or 'done',
                'filled_quantity': Decimal(str(order.get('dealSize', '0'))),
                'remaining_quantity': Decimal(str(order.get('size', '0'))) - Decimal(str(order.get('dealSize', '0'))),
                'average_price': Decimal(str(order.get('dealPrice', '0'))),
                'fees': Decimal(str(order.get('fee', '0')))
            }
        except Exception as e:
            logger.error("Failed to map order status", error=str(e))
            raise ValueError(f"Invalid order status format: {e}")

    def map_trading_fees_response(self, data: Dict[str, Any]) -> Dict[str, Decimal]:
        """Transform KuCoin trading fees response.

        Args:
            data: Raw trading fees data from KuCoin

        Returns:
            Dictionary with maker and taker fees
        """
        try:
            fee_data = data.get('data', {})

            return {
                'maker': Decimal(str(fee_data.get('makerFeeRate', '0.001'))),
                'taker': Decimal(str(fee_data.get('takerFeeRate', '0.001')))
            }

        except Exception as e:
            logger.error("Failed to map trading fees", error=str(e))
            # Default fees from config if mapping fails
            return {
                'maker': Decimal(os.getenv('KUCOIN_DEFAULT_MAKER_FEE', '0.001')),
                'taker': Decimal(os.getenv('KUCOIN_DEFAULT_TAKER_FEE', '0.001'))
            }
