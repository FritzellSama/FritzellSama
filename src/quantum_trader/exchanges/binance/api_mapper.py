"""
Binance API mapper for data transformation.

This module handles transformation between Binance API responses and
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


class BinanceAPIMapper:
    """Maps Binance API data to internal models.

    Handles request signing, data transformation, and validation
    for all Binance API interactions.

    Attributes:
        api_key: Binance API key
        secret: Binance API secret
        testnet: Whether using testnet
        base_url: API base URL
    """

    def __init__(self, api_key: str, secret: str, testnet: bool = False) -> None:
        """Initialize API mapper.

        Args:
            api_key: Binance API key
            secret: Binance API secret
            testnet: Whether to use testnet endpoints
        """
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet

        # Load URLs from environment
        if testnet:
            self.base_url = os.getenv('BINANCE_TESTNET_URL', 'https://testnet.binance.vision')
        else:
            self.base_url = os.getenv('BINANCE_API_URL', 'https://api.binance.com')

        logger.info(
            "Binance API mapper initialized",
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
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)

        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])

        # Generate signature
        signature = hmac.new(
            self.secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        logger.debug(
            "Request signed",
            params_count=len(params),
            timestamp=params['timestamp']
        )

        return signature

    def get_headers(self) -> Dict[str, str]:
        """Get request headers with API key.

        Returns:
            Headers dictionary
        """
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }

    def map_ohlcv_response(self, data: List[List]) -> pl.DataFrame:
        """Transform Binance OHLCV response to DataFrame.

        Args:
            data: Raw OHLCV data from Binance

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]

        Raises:
            ValueError: If data format is invalid
        """
        try:
            if not data:
                return pl.DataFrame({
                    'timestamp': [],
                    'open': [],
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': []
                })

            # Binance returns: [timestamp, open, high, low, close, volume, close_time, ...]
            timestamps = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []

            for candle in data:
                timestamps.append(datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc))
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
        """Transform Binance ticker response.

        Args:
            data: Raw ticker data from Binance

        Returns:
            Normalized ticker dictionary
        """
        try:
            return {
                'symbol': data['symbol'],
                'bid': Decimal(str(data['bidPrice'])),
                'ask': Decimal(str(data['askPrice'])),
                'last': Decimal(str(data['lastPrice'])),
                'volume': Decimal(str(data['volume'])),
                'timestamp': datetime.fromtimestamp(
                    data['closeTime'] / 1000,
                    tz=timezone.utc
                )
            }
        except Exception as e:
            logger.error("Failed to map ticker data", error=str(e))
            raise ValueError(f"Invalid ticker data format: {e}")

    def map_orderbook_response(
        self,
        data: Dict[str, Any]
    ) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        """Transform Binance orderbook response.

        Args:
            data: Raw orderbook data from Binance

        Returns:
            Tuple of (bids, asks) where each is list of (price, size) tuples
        """
        try:
            bids = [(Decimal(str(b[0])), Decimal(str(b[1]))) for b in data['bids']]
            asks = [(Decimal(str(a[0])), Decimal(str(a[1]))) for a in data['asks']]

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
        """Transform Binance balance response.

        Args:
            data: Raw balance data from Binance

        Returns:
            Dictionary mapping currency to available balance
        """
        try:
            balances = {}
            for balance in data.get('balances', []):
                free = Decimal(str(balance['free']))
                if free > Decimal('0'):
                    balances[balance['asset']] = free

            logger.debug("Balance data mapped", currencies=len(balances))
            return balances

        except Exception as e:
            logger.error("Failed to map balance data", error=str(e))
            raise ValueError(f"Invalid balance data format: {e}")

    def map_positions_response(self, data: List[Dict[str, Any]]) -> List[Dict]:
        """Transform Binance positions response.

        Args:
            data: Raw positions data from Binance

        Returns:
            List of normalized position dictionaries
        """
        try:
            positions = []
            for pos in data:
                position_amt = Decimal(str(pos['positionAmt']))
                if position_amt != Decimal('0'):
                    positions.append({
                        'symbol': pos['symbol'],
                        'size': position_amt,
                        'entry_price': Decimal(str(pos['entryPrice'])),
                        'unrealized_pnl': Decimal(str(pos['unrealizedProfit'])),
                        'leverage': Decimal(str(pos['leverage']))
                    })

            logger.debug("Positions data mapped", positions=len(positions))
            return positions

        except Exception as e:
            logger.error("Failed to map positions data", error=str(e))
            raise ValueError(f"Invalid positions data format: {e}")

    def map_order_request(self, order: 'Order') -> Dict[str, Any]:
        """Transform internal order to Binance order request.

        Args:
            order: Internal order object

        Returns:
            Binance order request parameters
        """
        try:
            params = {
                'symbol': order.symbol.replace('/', ''),  # Remove slash
                'side': order.side.value,
                'type': self._map_order_type(order.order_type),
                'quantity': str(order.quantity)
            }

            if order.price is not None:
                params['price'] = str(order.price)
                params['timeInForce'] = os.getenv('BINANCE_DEFAULT_TIME_IN_FORCE', 'GTC')

            logger.debug("Order request mapped", order_type=order.order_type)
            return params

        except Exception as e:
            logger.error("Failed to map order request", error=str(e))
            raise ValueError(f"Invalid order format: {e}")

    def _map_order_type(self, order_type: 'OrderType') -> str:
        """Map internal order type to Binance order type.

        Args:
            order_type: Internal order type

        Returns:
            Binance order type string
        """
        mapping = {
            'MARKET': 'MARKET',
            'LIMIT': 'LIMIT',
            'STOP_LOSS': 'STOP_LOSS',
            'TAKE_PROFIT': 'TAKE_PROFIT',
            'STOP_LIMIT': 'STOP_LOSS_LIMIT'
        }
        return mapping.get(order_type.value, 'MARKET')

    def map_order_response(self, data: Dict[str, Any]) -> str:
        """Extract order ID from Binance order response.

        Args:
            data: Raw order response from Binance

        Returns:
            Exchange order ID
        """
        try:
            order_id = str(data['orderId'])
            logger.debug("Order response mapped", order_id=order_id)
            return order_id
        except Exception as e:
            logger.error("Failed to map order response", error=str(e))
            raise ValueError(f"Invalid order response format: {e}")

    def map_order_status_response(self, data: Dict[str, Any]) -> Dict:
        """Transform Binance order status response.

        Args:
            data: Raw order status data from Binance

        Returns:
            Normalized order status dictionary
        """
        try:
            return {
                'order_id': str(data['orderId']),
                'status': data['status'].lower(),
                'filled_quantity': Decimal(str(data['executedQty'])),
                'remaining_quantity': Decimal(str(data['origQty'])) - Decimal(str(data['executedQty'])),
                'average_price': Decimal(str(data.get('avgPrice', '0'))),
                'fees': Decimal(str(data.get('commission', '0')))
            }
        except Exception as e:
            logger.error("Failed to map order status", error=str(e))
            raise ValueError(f"Invalid order status format: {e}")

    def map_trading_fees_response(self, data: List[Dict[str, Any]], symbol: str) -> Dict[str, Decimal]:
        """Transform Binance trading fees response.

        Args:
            data: Raw trading fees data from Binance
            symbol: Trading symbol

        Returns:
            Dictionary with maker and taker fees
        """
        try:
            symbol_no_slash = symbol.replace('/', '')
            for fee_data in data:
                if fee_data['symbol'] == symbol_no_slash:
                    return {
                        'maker': Decimal(str(fee_data['makerCommission'])),
                        'taker': Decimal(str(fee_data['takerCommission']))
                    }

            # Default fees from config if not found
            return {
                'maker': Decimal(os.getenv('BINANCE_DEFAULT_MAKER_FEE', '0.001')),
                'taker': Decimal(os.getenv('BINANCE_DEFAULT_TAKER_FEE', '0.001'))
            }

        except Exception as e:
            logger.error("Failed to map trading fees", error=str(e))
            raise ValueError(f"Invalid trading fees format: {e}")
