"""
Main Trading Engine Orchestrator
Production-ready trading engine coordinating all subsystems
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any
import logging
import asyncio
from datetime import datetime
import polars as pl
from dataclasses import dataclass, asdict
from enum import Enum

from quantum_trader.utils.config_loader import get_config
from quantum_trader.engine.state_machine import StateMachine, State
from quantum_trader.engine.event_bus import EventBus, EventType
from quantum_trader.engine.order_manager import OrderManager
from quantum_trader.engine.execution_engine import ExecutionEngine
from quantum_trader.engine.matching_engine import MatchingEngine
from quantum_trader.engine.reconciliation import Reconciliation
from quantum_trader.engine.settlement import Settlement

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_id: str
    symbol: str
    signal_type: str
    confidence: Decimal
    target_price: Optional[Decimal]
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    position_size: Decimal
    timestamp: str
    strategy_name: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Position:
    """Position data structure"""
    position_id: str
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    opened_at: str
    updated_at: str
    leverage: Decimal = Decimal('1.0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None


class TradingEngine:
    """Main trading engine orchestrator"""

    def __init__(self) -> None:
        """Initialize trading engine with all subsystems"""
        self.config = get_config()
        self._load_config()

        # Initialize subsystems
        self.state_machine = StateMachine()
        self.event_bus = EventBus()
        self.order_manager = OrderManager()
        self.execution_engine = ExecutionEngine()
        self.matching_engine = MatchingEngine()
        self.reconciliation = Reconciliation()
        self.settlement = Settlement()

        # Engine state
        self._positions: Dict[str, Position] = {}
        self._signal_queue: asyncio.Queue = None
        self._order_queue: asyncio.Queue = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Performance tracking
        self._total_pnl = Decimal('0')
        self._trade_count = 0
        self._start_time: Optional[datetime] = None

        logger.info("TradingEngine initialized with all subsystems")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        self.max_concurrent_orders = self.config.get_int('engine', 'trading_engine.max_concurrent_orders')
        self.order_queue_size = self.config.get_int('engine', 'trading_engine.order_queue_size')
        self.position_update_interval_ms = self.config.get_int('engine', 'trading_engine.position_update_interval_ms')
        self.pnl_calculation_interval_ms = self.config.get_int('engine', 'trading_engine.pnl_calculation_interval_ms')
        self.health_check_interval_seconds = self.config.get_int('engine', 'trading_engine.health_check_interval_seconds')

        logger.info(
            f"TradingEngine configured: max_orders={self.max_concurrent_orders}, "
            f"queue_size={self.order_queue_size}"
        )

    async def start(self) -> bool:
        """
        Start the trading engine

        Returns:
            bool: True if start successful

        Raises:
            RuntimeError: If engine fails to start
        """
        if self._running:
            logger.warning("Trading engine already running")
            return False

        logger.info("Starting trading engine...")

        try:
            # Transition to STARTING state
            await self.state_machine.transition_state(
                State.STARTING.value,
                reason="ENGINE_START",
                initiated_by="USER"
            )

            # Initialize queues
            self._signal_queue = asyncio.Queue(maxsize=1000)
            self._order_queue = asyncio.Queue(maxsize=self.order_queue_size)

            # Connect event bus
            await self.event_bus.connect()

            # Subscribe to events
            await self._subscribe_to_events()

            # Start background tasks
            await self._start_background_tasks()

            # Start reconciliation
            if self.reconciliation.enabled:
                await self.reconciliation.start_continuous_reconciliation()

            # Transition to RUNNING state
            await self.state_machine.transition_state(
                State.RUNNING.value,
                reason="START_COMPLETE",
                initiated_by="SYSTEM"
            )

            self._running = True
            self._start_time = datetime.utcnow()

            # Publish start event
            await self.event_bus.publish_event(
                event_type=EventType.STATE_TRANSITION.value,
                data={
                    'state': State.RUNNING.value,
                    'timestamp': self._start_time.isoformat()
                },
                source="TradingEngine",
                priority=1
            )

            logger.info("Trading engine started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")

            # Transition to FAILED state
            try:
                await self.state_machine.transition_state(
                    State.FAILED.value,
                    reason="START_FAILED",
                    initiated_by="SYSTEM",
                    metadata={'error': str(e)}
                )
            except:
                pass

            raise RuntimeError(f"Trading engine start failed: {e}")

    async def stop(self) -> bool:
        """
        Stop the trading engine

        Returns:
            bool: True if stop successful
        """
        if not self._running:
            logger.warning("Trading engine not running")
            return False

        logger.info("Stopping trading engine...")

        try:
            # Transition to STOPPING state
            await self.state_machine.transition_state(
                State.STOPPING.value,
                reason="ENGINE_STOP",
                initiated_by="USER"
            )

            # Stop accepting new signals
            self._running = False

            # Cancel active orders
            active_orders = await self.order_manager.get_active_orders()
            for order in active_orders:
                try:
                    await self.order_manager.cancel_order(
                        order.order_id,
                        reason="Engine shutdown"
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")

            # Stop background tasks
            await self._stop_background_tasks()

            # Stop reconciliation
            await self.reconciliation.stop_continuous_reconciliation()

            # Close event bus
            await self.event_bus.close()

            # Transition to STOPPED state
            await self.state_machine.transition_state(
                State.STOPPED.value,
                reason="STOP_COMPLETE",
                initiated_by="SYSTEM"
            )

            # Publish stop event
            await self.event_bus.publish_event(
                event_type=EventType.STATE_TRANSITION.value,
                data={
                    'state': State.STOPPED.value,
                    'timestamp': datetime.utcnow().isoformat()
                },
                source="TradingEngine",
                priority=1
            )

            logger.info("Trading engine stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping trading engine: {e}")
            return False

    async def process_signal(
        self,
        signal: TradingSignal
    ) -> Optional[str]:
        """
        Process a trading signal

        Args:
            signal: Trading signal to process

        Returns:
            Order ID if order created, None otherwise
        """
        if not self._running:
            logger.warning("Trading engine not running, signal ignored")
            return None

        try:
            logger.info(
                f"Processing signal {signal.signal_id}: "
                f"{signal.signal_type} {signal.symbol} @ {signal.target_price}"
            )

            # Validate signal
            if signal.confidence < Decimal('0.5'):
                logger.warning(f"Signal {signal.signal_id} confidence too low: {signal.confidence}")
                return None

            # Determine order parameters
            if signal.signal_type in [SignalType.BUY.value, SignalType.SELL.value]:
                side = signal.signal_type
                quantity = signal.position_size

                # Create order
                order = await self.order_manager.create_order(
                    symbol=signal.symbol,
                    side=side,
                    order_type="LIMIT",
                    quantity=quantity,
                    price=signal.target_price,
                    time_in_force="GTC",
                    exchange="BINANCE",
                    user_id="STRATEGY",
                    metadata={
                        'signal_id': signal.signal_id,
                        'strategy': signal.strategy_name,
                        'confidence': str(signal.confidence),
                        'stop_loss': str(signal.stop_loss) if signal.stop_loss else None,
                        'take_profit': str(signal.take_profit) if signal.take_profit else None
                    }
                )

                # Validate order
                validation = await self.order_manager.validate_order(order.order_id)

                if not validation.valid:
                    logger.error(f"Order validation failed: {validation.errors}")
                    return None

                # Add to order queue
                await self._order_queue.put(order)

                # Publish order event
                await self.event_bus.publish_event(
                    event_type=EventType.ORDER_CREATED.value,
                    data={
                        'order_id': order.order_id,
                        'symbol': signal.symbol,
                        'side': side,
                        'quantity': str(quantity),
                        'price': str(signal.target_price)
                    },
                    source="TradingEngine",
                    priority=0
                )

                logger.info(f"Order {order.order_id} created from signal {signal.signal_id}")
                return order.order_id

            elif signal.signal_type in [SignalType.CLOSE_LONG.value, SignalType.CLOSE_SHORT.value]:
                # Close existing position
                await self._close_position(signal.symbol, signal.signal_type)
                return None

            else:
                logger.warning(f"Unknown signal type: {signal.signal_type}")
                return None

        except Exception as e:
            logger.error(f"Error processing signal {signal.signal_id}: {e}")
            return None

    async def manage_positions(self) -> Dict[str, Any]:
        """
        Manage open positions

        Returns:
            Dictionary with position management results
        """
        try:
            updated_positions = 0
            closed_positions = 0

            for position_id, position in list(self._positions.items()):
                # Update current price (in production, get from market data)
                # position.current_price = await self._get_current_price(position.symbol)

                # Update unrealized P&L
                if position.side == "BUY":
                    position.unrealized_pnl = (
                        (position.current_price - position.entry_price) * position.quantity
                    )
                else:  # SELL
                    position.unrealized_pnl = (
                        (position.entry_price - position.current_price) * position.quantity
                    )

                position.updated_at = datetime.utcnow().isoformat()
                updated_positions += 1

                # Check stop loss
                if position.stop_loss:
                    if (position.side == "BUY" and position.current_price <= position.stop_loss) or \
                       (position.side == "SELL" and position.current_price >= position.stop_loss):
                        logger.warning(
                            f"Stop loss triggered for position {position_id}: "
                            f"current={position.current_price}, stop={position.stop_loss}"
                        )
                        await self._close_position(position.symbol, "STOP_LOSS")
                        closed_positions += 1

                # Check take profit
                if position.take_profit:
                    if (position.side == "BUY" and position.current_price >= position.take_profit) or \
                       (position.side == "SELL" and position.current_price <= position.take_profit):
                        logger.info(
                            f"Take profit triggered for position {position_id}: "
                            f"current={position.current_price}, target={position.take_profit}"
                        )
                        await self._close_position(position.symbol, "TAKE_PROFIT")
                        closed_positions += 1

            return {
                'total_positions': len(self._positions),
                'updated_positions': updated_positions,
                'closed_positions': closed_positions,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error managing positions: {e}")
            return {'error': str(e)}

    async def _close_position(self, symbol: str, reason: str) -> bool:
        """Close a position"""
        for position_id, position in list(self._positions.items()):
            if position.symbol == symbol:
                # Create closing order
                closing_side = "SELL" if position.side == "BUY" else "BUY"

                order = await self.order_manager.create_order(
                    symbol=symbol,
                    side=closing_side,
                    order_type="MARKET",
                    quantity=position.quantity,
                    exchange="BINANCE",
                    user_id="SYSTEM",
                    metadata={'reason': reason, 'position_id': position_id}
                )

                # Execute order
                result = await self.execution_engine.execute_order(
                    symbol=symbol,
                    side=closing_side,
                    order_type="MARKET",
                    quantity=position.quantity,
                    exchange="BINANCE"
                )

                # Update position
                position.realized_pnl = position.unrealized_pnl
                self._total_pnl += position.realized_pnl

                # Remove position
                del self._positions[position_id]

                # Publish event
                await self.event_bus.publish_event(
                    event_type=EventType.POSITION_CLOSED.value,
                    data={
                        'position_id': position_id,
                        'symbol': symbol,
                        'pnl': str(position.realized_pnl),
                        'reason': reason
                    },
                    source="TradingEngine",
                    priority=1
                )

                logger.info(f"Position {position_id} closed: PnL={position.realized_pnl}")
                return True

        return False

    async def _subscribe_to_events(self) -> None:
        """Subscribe to event bus events"""
        async def handle_order_filled(event):
            """Handle order filled event"""
            logger.info(f"Order filled event received: {event.event_id}")

        async def handle_risk_breach(event):
            """Handle risk breach event"""
            logger.error(f"Risk breach event received: {event.event_id}")
            await self.state_machine.emergency_stop("Risk breach detected")

        # Subscribe to events
        await self.event_bus.subscribe(EventType.ORDER_FILLED.value, handle_order_filled)
        await self.event_bus.subscribe(EventType.RISK_BREACH.value, handle_risk_breach)

    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        # Order processing task
        order_task = asyncio.create_task(self._order_processing_loop())
        self._tasks.append(order_task)

        # Position update task
        position_task = asyncio.create_task(self._position_update_loop())
        self._tasks.append(position_task)

        # P&L calculation task
        pnl_task = asyncio.create_task(self._pnl_calculation_loop())
        self._tasks.append(pnl_task)

        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self._tasks.append(health_task)

        # Event processing task
        event_task = asyncio.create_task(self.event_bus.process_events())
        self._tasks.append(event_task)

        logger.info(f"Started {len(self._tasks)} background tasks")

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        logger.info("Stopping background tasks...")

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Background tasks stopped")

    async def _order_processing_loop(self) -> None:
        """Background task to process orders"""
        while self._running:
            try:
                order = await asyncio.wait_for(self._order_queue.get(), timeout=1.0)

                # Execute order
                result = await self.execution_engine.execute_order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=order.price,
                    exchange=order.exchange
                )

                # Update order tracking
                await self.order_manager.track_order(
                    order.order_id,
                    status=result.status,
                    filled_quantity=result.filled_quantity,
                    average_fill_price=result.average_price
                )

                self._order_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in order processing loop: {e}")

    async def _position_update_loop(self) -> None:
        """Background task to update positions"""
        interval = self.position_update_interval_ms / 1000.0

        while self._running:
            try:
                await self.manage_positions()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in position update loop: {e}")

    async def _pnl_calculation_loop(self) -> None:
        """Background task to calculate P&L"""
        interval = self.pnl_calculation_interval_ms / 1000.0

        while self._running:
            try:
                total_unrealized = sum(
                    pos.unrealized_pnl for pos in self._positions.values()
                )

                logger.debug(
                    f"P&L: Realized={self._total_pnl}, Unrealized={total_unrealized}"
                )

                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in P&L calculation loop: {e}")

    async def _health_check_loop(self) -> None:
        """Background task for health checks"""
        while self._running:
            try:
                health = await self.get_health_status()

                if not health['healthy']:
                    logger.warning(f"Health check failed: {health}")

                await asyncio.sleep(self.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get engine health status

        Returns:
            Dictionary with health information
        """
        subsystem_health = {
            'state_machine': self.state_machine.get_current_state() == State.RUNNING.value,
            'order_queue': self._order_queue.qsize() < self.order_queue_size * 0.9,
            'positions': len(self._positions) < 1000
        }

        healthy = all(subsystem_health.values())

        return {
            'healthy': healthy,
            'state': self.state_machine.get_current_state(),
            'subsystems': subsystem_health,
            'uptime_seconds': int((datetime.utcnow() - self._start_time).total_seconds()) if self._start_time else 0,
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self._positions.values())

        return {
            'total_realized_pnl': str(self._total_pnl),
            'total_unrealized_pnl': str(total_unrealized_pnl),
            'total_pnl': str(self._total_pnl + total_unrealized_pnl),
            'trade_count': self._trade_count,
            'open_positions': len(self._positions),
            'uptime_seconds': int((datetime.utcnow() - self._start_time).total_seconds()) if self._start_time else 0,
            'state': self.state_machine.get_current_state()
        }
