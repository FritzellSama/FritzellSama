"""
Order Matching Engine with Price-Time Priority
Production-ready order matching with comprehensive order book management
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
import logging
import asyncio
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import bisect
import polars as pl
import uuid

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OrderBookEntry:
    """Order book entry"""
    order_id: str
    price: Decimal
    quantity: Decimal
    side: str
    timestamp: datetime
    user_id: str

    def __lt__(self, other):
        """Comparison for sorting (price-time priority)"""
        if self.price != other.price:
            return self.price < other.price
        return self.timestamp < other.timestamp


@dataclass
class MatchResult:
    """Match result data structure"""
    buy_order_id: str
    sell_order_id: str
    match_price: Decimal
    match_quantity: Decimal
    timestamp: str
    buyer_id: str
    seller_id: str


@dataclass
class OrderBook:
    """Order book data structure"""
    symbol: str
    bids: List[OrderBookEntry] = field(default_factory=list)  # Buy orders (sorted desc by price)
    asks: List[OrderBookEntry] = field(default_factory=list)  # Sell orders (sorted asc by price)
    last_update: Optional[datetime] = None

    def get_best_bid(self) -> Optional[OrderBookEntry]:
        """Get best bid (highest buy price)"""
        return self.bids[0] if self.bids else None

    def get_best_ask(self) -> Optional[OrderBookEntry]:
        """Get best ask (lowest sell price)"""
        return self.asks[0] if self.asks else None

    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return best_ask.price - best_bid.price

        return None

    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / Decimal('2')

        return None


class MatchingEngine:
    """Order matching engine with price-time priority"""

    def __init__(self) -> None:
        """Initialize matching engine with configuration"""
        self.config = get_config()
        self._load_config()
        self._order_books: Dict[str, OrderBook] = {}
        self._orders: Dict[str, OrderBookEntry] = {}
        self._match_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        logger.info("MatchingEngine initialized")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        self.order_book_depth = self.config.get_int('engine', 'matching_engine.order_book_depth')
        self.tick_size_usd = self.config.get_decimal('engine', 'matching_engine.tick_size_usd')
        self.price_precision = self.config.get_int('engine', 'matching_engine.price_precision')
        self.quantity_precision = self.config.get_int('engine', 'matching_engine.quantity_precision')
        self.matching_algorithm = self.config.get_string('engine', 'matching_engine.matching_algorithm')

        logger.info(
            f"MatchingEngine configured: algorithm={self.matching_algorithm}, "
            f"depth={self.order_book_depth}"
        )

    def _normalize_price(self, price: Decimal) -> Decimal:
        """
        Normalize price to tick size and precision

        Args:
            price: Raw price

        Returns:
            Normalized price
        """
        # Round to tick size
        ticks = int(price / self.tick_size_usd)
        normalized = Decimal(str(ticks)) * self.tick_size_usd

        # Round to precision
        quantize_str = '0.' + '0' * (self.price_precision - 1) + '1'
        normalized = normalized.quantize(Decimal(quantize_str))

        return normalized

    def _normalize_quantity(self, quantity: Decimal) -> Decimal:
        """
        Normalize quantity to precision

        Args:
            quantity: Raw quantity

        Returns:
            Normalized quantity
        """
        quantize_str = '0.' + '0' * (self.quantity_precision - 1) + '1'
        return quantity.quantize(Decimal(quantize_str))

    async def match_order(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
        user_id: str
    ) -> Tuple[List[MatchResult], Optional[str]]:
        """
        Match an incoming order against the order book

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            price: Order price
            quantity: Order quantity
            user_id: User placing the order

        Returns:
            Tuple of (list of match results, remaining order ID if not fully filled)

        Raises:
            ValueError: If order parameters are invalid
        """
        if quantity <= Decimal('0'):
            raise ValueError(f"Invalid quantity: {quantity}")

        if price <= Decimal('0'):
            raise ValueError(f"Invalid price: {price}")

        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}")

        # Normalize price and quantity
        price = self._normalize_price(price)
        quantity = self._normalize_quantity(quantity)

        async with self._lock:
            # Create order entry
            order_id = str(uuid.uuid4())
            order = OrderBookEntry(
                order_id=order_id,
                price=price,
                quantity=quantity,
                side=side,
                timestamp=datetime.utcnow(),
                user_id=user_id
            )

            # Initialize order book if needed
            if symbol not in self._order_books:
                self._order_books[symbol] = OrderBook(symbol=symbol)

            order_book = self._order_books[symbol]

            # Match order
            if self.matching_algorithm == "price_time":
                matches, remaining_order = await self._match_price_time(
                    order, order_book
                )
            else:
                matches, remaining_order = await self._match_price_time(
                    order, order_book
                )

            # Update order book
            order_book.last_update = datetime.utcnow()

            # Record matches
            for match in matches:
                self._record_match(match)

            logger.info(
                f"Order {order_id} matched: {len(matches)} fills, "
                f"remaining={'Yes' if remaining_order else 'No'}"
            )

            return matches, remaining_order

    async def _match_price_time(
        self,
        order: OrderBookEntry,
        order_book: OrderBook
    ) -> Tuple[List[MatchResult], Optional[str]]:
        """
        Match order using price-time priority algorithm

        Args:
            order: Incoming order
            order_book: Order book to match against

        Returns:
            Tuple of (matches, remaining order ID)
        """
        matches: List[MatchResult] = []
        remaining_qty = order.quantity

        if order.side == OrderSide.BUY.value:
            # Match against asks (sell orders)
            while remaining_qty > Decimal('0') and order_book.asks:
                best_ask = order_book.asks[0]

                # Check if prices cross
                if order.price < best_ask.price:
                    break

                # Calculate match quantity
                match_qty = min(remaining_qty, best_ask.quantity)

                # Create match
                match = MatchResult(
                    buy_order_id=order.order_id,
                    sell_order_id=best_ask.order_id,
                    match_price=best_ask.price,  # Taker pays maker's price
                    match_quantity=match_qty,
                    timestamp=datetime.utcnow().isoformat(),
                    buyer_id=order.user_id,
                    seller_id=best_ask.user_id
                )
                matches.append(match)

                # Update quantities
                remaining_qty -= match_qty
                best_ask.quantity -= match_qty

                # Remove filled orders
                if best_ask.quantity <= Decimal('0'):
                    order_book.asks.pop(0)
                    if best_ask.order_id in self._orders:
                        del self._orders[best_ask.order_id]

        else:  # SELL
            # Match against bids (buy orders)
            while remaining_qty > Decimal('0') and order_book.bids:
                best_bid = order_book.bids[0]

                # Check if prices cross
                if order.price > best_bid.price:
                    break

                # Calculate match quantity
                match_qty = min(remaining_qty, best_bid.quantity)

                # Create match
                match = MatchResult(
                    buy_order_id=best_bid.order_id,
                    sell_order_id=order.order_id,
                    match_price=best_bid.price,  # Taker pays maker's price
                    match_quantity=match_qty,
                    timestamp=datetime.utcnow().isoformat(),
                    buyer_id=best_bid.user_id,
                    seller_id=order.user_id
                )
                matches.append(match)

                # Update quantities
                remaining_qty -= match_qty
                best_bid.quantity -= match_qty

                # Remove filled orders
                if best_bid.quantity <= Decimal('0'):
                    order_book.bids.pop(0)
                    if best_bid.order_id in self._orders:
                        del self._orders[best_bid.order_id]

        # Add remaining order to book if not fully filled
        remaining_order_id = None
        if remaining_qty > Decimal('0'):
            order.quantity = remaining_qty
            remaining_order_id = order.order_id

            # Add to appropriate side
            if order.side == OrderSide.BUY.value:
                # Insert in sorted order (highest price first)
                bisect.insort(order_book.bids, order, key=lambda x: (-x.price, x.timestamp))
                # Trim to depth
                order_book.bids = order_book.bids[:self.order_book_depth]
            else:
                # Insert in sorted order (lowest price first)
                bisect.insort(order_book.asks, order, key=lambda x: (x.price, x.timestamp))
                # Trim to depth
                order_book.asks = order_book.asks[:self.order_book_depth]

            # Store order
            self._orders[order.order_id] = order

        return matches, remaining_order_id

    async def get_orderbook(
        self,
        symbol: str,
        depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get order book for a symbol

        Args:
            symbol: Trading symbol
            depth: Number of levels to return (None for all)

        Returns:
            Dictionary with order book data
        """
        async with self._lock:
            if symbol not in self._order_books:
                return {
                    'symbol': symbol,
                    'bids': [],
                    'asks': [],
                    'spread': None,
                    'mid_price': None,
                    'last_update': None
                }

            order_book = self._order_books[symbol]

            # Determine depth
            if depth is None:
                depth = self.order_book_depth

            # Aggregate by price level
            bids_aggregated = self._aggregate_orders(order_book.bids[:depth])
            asks_aggregated = self._aggregate_orders(order_book.asks[:depth])

            return {
                'symbol': symbol,
                'bids': bids_aggregated,
                'asks': asks_aggregated,
                'spread': str(order_book.get_spread()) if order_book.get_spread() else None,
                'mid_price': str(order_book.get_mid_price()) if order_book.get_mid_price() else None,
                'last_update': order_book.last_update.isoformat() if order_book.last_update else None
            }

    def _aggregate_orders(self, orders: List[OrderBookEntry]) -> List[Dict[str, str]]:
        """
        Aggregate orders by price level

        Args:
            orders: List of order book entries

        Returns:
            List of price level dictionaries
        """
        aggregated = {}

        for order in orders:
            price_str = str(order.price)
            if price_str not in aggregated:
                aggregated[price_str] = Decimal('0')
            aggregated[price_str] += order.quantity

        result = [
            {'price': price, 'quantity': str(qty)}
            for price, qty in aggregated.items()
        ]

        return result

    async def update_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]]
    ) -> bool:
        """
        Update order book from external source (e.g., exchange feed)

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks

        Returns:
            bool: True if update successful
        """
        async with self._lock:
            try:
                # Initialize order book if needed
                if symbol not in self._order_books:
                    self._order_books[symbol] = OrderBook(symbol=symbol)

                order_book = self._order_books[symbol]

                # Clear existing orders
                order_book.bids.clear()
                order_book.asks.clear()

                # Add bids (sorted by price descending)
                for price, quantity in sorted(bids, key=lambda x: -x[0]):
                    if len(order_book.bids) >= self.order_book_depth:
                        break

                    order = OrderBookEntry(
                        order_id=str(uuid.uuid4()),
                        price=self._normalize_price(price),
                        quantity=self._normalize_quantity(quantity),
                        side=OrderSide.BUY.value,
                        timestamp=datetime.utcnow(),
                        user_id="EXTERNAL"
                    )
                    order_book.bids.append(order)

                # Add asks (sorted by price ascending)
                for price, quantity in sorted(asks, key=lambda x: x[0]):
                    if len(order_book.asks) >= self.order_book_depth:
                        break

                    order = OrderBookEntry(
                        order_id=str(uuid.uuid4()),
                        price=self._normalize_price(price),
                        quantity=self._normalize_quantity(quantity),
                        side=OrderSide.SELL.value,
                        timestamp=datetime.utcnow(),
                        user_id="EXTERNAL"
                    )
                    order_book.asks.append(order)

                order_book.last_update = datetime.utcnow()

                logger.debug(
                    f"Updated order book for {symbol}: "
                    f"{len(order_book.bids)} bids, {len(order_book.asks)} asks"
                )

                return True

            except Exception as e:
                logger.error(f"Failed to update order book for {symbol}: {e}")
                return False

    def _record_match(self, match: MatchResult) -> None:
        """Record match in history"""
        match_record = {
            'buy_order_id': match.buy_order_id,
            'sell_order_id': match.sell_order_id,
            'match_price': str(match.match_price),
            'match_quantity': str(match.match_quantity),
            'timestamp': match.timestamp,
            'buyer_id': match.buyer_id,
            'seller_id': match.seller_id
        }

        self._match_history.append(match_record)

    def get_match_history_dataframe(self) -> pl.DataFrame:
        """
        Get match history as Polars DataFrame

        Returns:
            Polars DataFrame with match history
        """
        if not self._match_history:
            return pl.DataFrame()

        df = pl.DataFrame(self._match_history)

        logger.info(f"Retrieved {len(df)} match records")
        return df

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order in the book

        Args:
            order_id: Order ID to cancel

        Returns:
            bool: True if cancellation successful
        """
        async with self._lock:
            if order_id not in self._orders:
                logger.warning(f"Order {order_id} not found")
                return False

            order = self._orders[order_id]

            # Find and remove from order book
            for symbol, order_book in self._order_books.items():
                if order.side == OrderSide.BUY.value:
                    order_book.bids = [o for o in order_book.bids if o.order_id != order_id]
                else:
                    order_book.asks = [o for o in order_book.asks if o.order_id != order_id]

                order_book.last_update = datetime.utcnow()

            # Remove from orders dict
            del self._orders[order_id]

            logger.info(f"Order {order_id} cancelled successfully")
            return True
