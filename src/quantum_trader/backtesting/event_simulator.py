"""
Event Simulator - Simulate market events for backtesting.

This module simulates various market events including order fills, market data updates,
and exchange events to create realistic backtest environments.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from enum import Enum
import uuid
import os

from structlog import get_logger
import polars as pl

logger = get_logger(__name__)


class EventType(str, Enum):
    """Market event types."""
    MARKET_DATA = "market_data"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    ORDER_REJECT = "order_reject"
    POSITION_UPDATE = "position_update"
    ACCOUNT_UPDATE = "account_update"
    SYSTEM_EVENT = "system_event"


class EventSimulator:
    """Simulates market and trading events for backtesting.

    Attributes:
        config: Simulator configuration from environment
        event_queue: Queue of pending events
        event_handlers: Registered event handlers
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize event simulator.

        Args:
            config: Optional configuration override

        Raises:
            ValueError: If configuration invalid
        """
        self.config: Dict[str, Any] = config or self._load_config()
        self.event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['event_queue_size']
        )
        self.event_handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        self._running: bool = False

        logger.info("EventSimulator initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Returns:
            Configuration dictionary
        """
        try:
            config = {
                'event_queue_size': int(os.getenv('EVENT_QUEUE_SIZE', '10000')),
                'fill_latency_ms': int(os.getenv('FILL_LATENCY_MS', '10')),
                'partial_fill_probability': Decimal(os.getenv('PARTIAL_FILL_PROB', '0.05')),
                'reject_probability': Decimal(os.getenv('REJECT_PROB', '0.01')),
                'enable_slippage': os.getenv('ENABLE_SLIPPAGE', 'true').lower() == 'true',
                'slippage_bps': Decimal(os.getenv('SLIPPAGE_BPS', '5')),
            }

            logger.debug("Event simulator config loaded", config=config)
            return config

        except Exception as e:
            logger.error("Failed to load config", error=str(e))
            raise ValueError(f"Configuration load failed: {e}")

    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register an event handler.

        Args:
            event_type: Type of event to handle
            handler: Handler function

        Example:
            >>> def handle_fill(event):
            ...     print(f"Order filled: {event['order_id']}")
            >>> simulator.register_handler(EventType.ORDER_FILL, handle_fill)
        """
        try:
            self.event_handlers[event_type].append(handler)

            logger.debug(
                "Event handler registered",
                event_type=event_type.value,
                handler_count=len(self.event_handlers[event_type])
            )

        except Exception as e:
            logger.error("Failed to register handler", error=str(e))
            raise

    async def start(self) -> None:
        """Start event processing loop."""
        if self._running:
            logger.warning("Event simulator already running")
            return

        self._running = True
        logger.info("Event simulator started")

    async def stop(self) -> None:
        """Stop event processing loop."""
        self._running = False
        logger.info("Event simulator stopped")

    async def simulate_market_data_event(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: Decimal,
        high_price: Decimal,
        low_price: Decimal,
        close_price: Decimal,
        volume: Decimal
    ) -> None:
        """Simulate a market data update event.

        Args:
            symbol: Trading symbol
            timestamp: Event timestamp
            open_price: Open price
            high_price: High price
            low_price: Low price
            close_price: Close price
            volume: Trading volume

        Example:
            >>> await simulator.simulate_market_data_event(
            ...     'BTC/USDT',
            ...     datetime.now(timezone.utc),
            ...     Decimal('50000'),
            ...     Decimal('50100'),
            ...     Decimal('49900'),
            ...     Decimal('50050'),
            ...     Decimal('100')
            ... )
        """
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.MARKET_DATA.value,
                'timestamp': timestamp,
                'symbol': symbol,
                'open': str(open_price),
                'high': str(high_price),
                'low': str(low_price),
                'close': str(close_price),
                'volume': str(volume)
            }

            await self._emit_event(EventType.MARKET_DATA, event)

            logger.debug(
                "Market data event simulated",
                symbol=symbol,
                price=str(close_price)
            )

        except Exception as e:
            logger.error("Failed to simulate market data event", error=str(e))
            raise

    async def simulate_order_submission(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Optional[Decimal],
        order_type: str,
        timestamp: datetime
    ) -> None:
        """Simulate order submission and subsequent fill/reject.

        Args:
            order_id: Order identifier
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Order price (None for market orders)
            order_type: Order type
            timestamp: Submission timestamp

        Example:
            >>> await simulator.simulate_order_submission(
            ...     'ord_123',
            ...     'BTC/USDT',
            ...     'BUY',
            ...     Decimal('1.0'),
            ...     Decimal('50000'),
            ...     'LIMIT',
            ...     datetime.now(timezone.utc)
            ... )
        """
        try:
            logger.debug("Simulating order submission", order_id=order_id)

            # Add fill latency
            latency_seconds = self.config['fill_latency_ms'] / Decimal('1000')
            await asyncio.sleep(float(latency_seconds))

            # Determine if order should be rejected
            import random
            if Decimal(str(random.random())) < self.config['reject_probability']:
                await self._simulate_order_reject(
                    order_id,
                    timestamp,
                    "Simulated rejection"
                )
                return

            # Determine if partial fill
            filled_quantity = quantity
            if Decimal(str(random.random())) < self.config['partial_fill_probability']:
                filled_quantity = quantity * Decimal('0.5')

            # Calculate fill price with slippage
            fill_price = price if price else Decimal('50000')  # Would use market price
            if self.config['enable_slippage']:
                slippage = self.config['slippage_bps'] / Decimal('10000')
                if side == 'BUY':
                    fill_price = fill_price * (Decimal('1') + slippage)
                else:
                    fill_price = fill_price * (Decimal('1') - slippage)

            # Simulate fill event
            await self._simulate_order_fill(
                order_id,
                symbol,
                side,
                filled_quantity,
                fill_price,
                timestamp
            )

        except Exception as e:
            logger.error("Failed to simulate order submission", error=str(e))
            raise

    async def simulate_position_update(
        self,
        position_id: str,
        symbol: str,
        quantity: Decimal,
        entry_price: Decimal,
        current_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Simulate position update event.

        Args:
            position_id: Position identifier
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current market price
            timestamp: Update timestamp
        """
        try:
            unrealized_pnl = (current_price - entry_price) * quantity

            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.POSITION_UPDATE.value,
                'timestamp': timestamp,
                'position_id': position_id,
                'symbol': symbol,
                'quantity': str(quantity),
                'entry_price': str(entry_price),
                'current_price': str(current_price),
                'unrealized_pnl': str(unrealized_pnl)
            }

            await self._emit_event(EventType.POSITION_UPDATE, event)

            logger.debug(
                "Position update event simulated",
                position_id=position_id,
                pnl=str(unrealized_pnl)
            )

        except Exception as e:
            logger.error("Failed to simulate position update", error=str(e))
            raise

    async def simulate_account_update(
        self,
        account_id: str,
        balance: Decimal,
        equity: Decimal,
        margin_used: Decimal,
        timestamp: datetime
    ) -> None:
        """Simulate account balance update event.

        Args:
            account_id: Account identifier
            balance: Account balance
            equity: Account equity
            margin_used: Margin used
            timestamp: Update timestamp
        """
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.ACCOUNT_UPDATE.value,
                'timestamp': timestamp,
                'account_id': account_id,
                'balance': str(balance),
                'equity': str(equity),
                'margin_used': str(margin_used),
                'available_margin': str(equity - margin_used)
            }

            await self._emit_event(EventType.ACCOUNT_UPDATE, event)

            logger.debug("Account update event simulated", account_id=account_id)

        except Exception as e:
            logger.error("Failed to simulate account update", error=str(e))
            raise

    async def simulate_system_event(
        self,
        event_name: str,
        event_data: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """Simulate system event.

        Args:
            event_name: Event name
            event_data: Event data
            timestamp: Event timestamp
        """
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.SYSTEM_EVENT.value,
                'timestamp': timestamp,
                'event_name': event_name,
                'data': event_data
            }

            await self._emit_event(EventType.SYSTEM_EVENT, event)

            logger.debug("System event simulated", event_name=event_name)

        except Exception as e:
            logger.error("Failed to simulate system event", error=str(e))
            raise

    async def _simulate_order_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_quantity: Decimal,
        fill_price: Decimal,
        timestamp: datetime
    ) -> None:
        """Simulate order fill event."""
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.ORDER_FILL.value,
                'timestamp': timestamp,
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'filled_quantity': str(filled_quantity),
                'fill_price': str(fill_price),
                'commission': str(filled_quantity * fill_price * Decimal('0.001'))
            }

            await self._emit_event(EventType.ORDER_FILL, event)

            logger.debug(
                "Order fill event simulated",
                order_id=order_id,
                quantity=str(filled_quantity),
                price=str(fill_price)
            )

        except Exception as e:
            logger.error("Failed to simulate order fill", error=str(e))
            raise

    async def _simulate_order_reject(
        self,
        order_id: str,
        timestamp: datetime,
        reason: str
    ) -> None:
        """Simulate order rejection event."""
        try:
            event = {
                'event_id': str(uuid.uuid4()),
                'event_type': EventType.ORDER_REJECT.value,
                'timestamp': timestamp,
                'order_id': order_id,
                'reason': reason
            }

            await self._emit_event(EventType.ORDER_REJECT, event)

            logger.debug("Order reject event simulated", order_id=order_id, reason=reason)

        except Exception as e:
            logger.error("Failed to simulate order reject", error=str(e))
            raise

    async def _emit_event(
        self,
        event_type: EventType,
        event_data: Dict[str, Any]
    ) -> None:
        """Emit event to registered handlers.

        Args:
            event_type: Type of event
            event_data: Event data
        """
        try:
            # Queue event for async processing
            await self.event_queue.put((event_type, event_data))

            # Call synchronous handlers immediately
            handlers = self.event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(
                        "Event handler error",
                        error=str(e),
                        event_type=event_type.value
                    )

        except Exception as e:
            logger.error("Failed to emit event", error=str(e))
            raise

    async def process_events(self) -> None:
        """Process events from queue."""
        try:
            logger.info("Event processor started")

            while self._running:
                try:
                    # Get event from queue with timeout
                    event_type, event_data = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )

                    # Process event
                    logger.debug(
                        "Processing event",
                        type=event_type.value,
                        event_id=event_data.get('event_id')
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Event processing error", error=str(e))

            logger.info("Event processor stopped")

        except Exception as e:
            logger.error("Event processor failed", error=str(e))
            raise

    def get_queue_size(self) -> int:
        """Get current event queue size.

        Returns:
            Number of events in queue
        """
        return self.event_queue.qsize()

    async def clear_queue(self) -> None:
        """Clear all pending events from queue."""
        try:
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            logger.info("Event queue cleared")

        except Exception as e:
            logger.error("Failed to clear queue", error=str(e))
            raise
