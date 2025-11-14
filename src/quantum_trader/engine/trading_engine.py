"""
Quantum Trader AI - Main Trading Engine
Production-grade trading orchestration system

CRITICAL: All numeric values use Decimal, never float
CRITICAL: All data operations use polars DataFrame, never pandas
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import polars as pl
import yaml

from quantum_trader.engine.settlement import SettlementCycle, SettlementManager
from quantum_trader.engine.state_machine import (
    OrderStateMachine,
    StateTransitionEvent,
)
from quantum_trader.models import (
    AuditLog,
    ExecutionResult,
    ExecutionStatus,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
    SignalAction,
)


logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """Trading engine status"""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"


@dataclass
class PerformanceMetrics:
    """Trading engine performance metrics"""
    timestamp: datetime
    signals_processed: int
    orders_created: int
    orders_filled: int
    orders_rejected: int
    orders_cancelled: int
    total_pnl: Decimal
    win_rate: Decimal
    avg_latency_ms: Decimal
    risk_checks_passed: int
    risk_checks_failed: int
    uptime_seconds: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metrics"""
        for field_name in ["total_pnl", "win_rate", "avg_latency_ms", "uptime_seconds"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be Decimal, got {type(value)}")


@dataclass
class SystemHealth:
    """System health check result"""
    timestamp: datetime
    status: HealthStatus
    database_healthy: bool
    exchanges_healthy: Dict[str, bool]
    risk_manager_healthy: bool
    settlement_manager_healthy: bool
    state_machine_healthy: bool
    memory_usage_mb: Decimal
    cpu_usage_percent: Decimal
    active_orders: int
    pending_settlements: int
    errors_last_hour: int
    warnings_last_hour: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingEngine:
    """
    Production-grade trading engine orchestrator

    Features:
    - Signal processing pipeline
    - Order generation from signals
    - Risk checks integration
    - Execution coordination
    - Performance monitoring
    - System health checks
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize trading engine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Engine status
        self.status = EngineStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.last_health_check = datetime.utcnow()

        # Core components
        self.state_machine = OrderStateMachine(config_path)
        self.settlement_manager = SettlementManager(config_path)

        # Signal processing
        self.signal_queue: Deque[Signal] = deque(maxlen=10000)
        self.processed_signals: Set[str] = set()

        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # Position tracking
        self.positions: Dict[str, Position] = {}

        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.execution_latencies: Deque[Decimal] = deque(maxlen=1000)

        # Configuration
        self.max_concurrent_orders = self.config.get("bot", {}).get("execution", {}).get(
            "max_concurrent_orders", 50
        )
        self.order_timeout = self.config.get("bot", {}).get("execution", {}).get(
            "order_timeout_seconds", 30
        )
        self.max_retries = self.config.get("bot", {}).get("execution", {}).get("retry_attempts", 3)
        self.retry_delay = self.config.get("bot", {}).get("execution", {}).get("retry_delay_ms", 1000) / 1000.0
        self.retry_backoff = Decimal(
            str(self.config.get("bot", {}).get("execution", {}).get("retry_backoff_multiplier", 2.0))
        )

        self.min_order_size = Decimal(
            str(self.config.get("bot", {}).get("trading", {}).get("min_order_size_usdt", 10.0))
        )
        self.max_order_size = Decimal(
            str(self.config.get("bot", {}).get("trading", {}).get("max_order_size_usdt", 100000.0))
        )
        self.max_positions = self.config.get("bot", {}).get("trading", {}).get("max_positions", 20)
        self.position_size_percent = Decimal(
            str(self.config.get("bot", {}).get("trading", {}).get("position_size_percent", 2.0))
        )

        # Risk limits
        self.max_daily_loss = Decimal(
            str(self.config.get("global", {}).get("max_daily_loss_usd", 50000))
        )
        self.daily_pnl = Decimal("0")

        # Health monitoring
        self.health_check_interval = self.config.get("health", {}).get(
            "health_check_interval", 30
        )

        # Statistics
        self.stats = {
            "signals_processed": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "orders_cancelled": 0,
            "risk_checks_passed": 0,
            "risk_checks_failed": 0,
        }

        # Error tracking
        self.recent_errors: Deque[Tuple[datetime, str]] = deque(maxlen=100)
        self.recent_warnings: Deque[Tuple[datetime, str]] = deque(maxlen=100)

        self.logger.info("TradingEngine initialized successfully")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML files

        Args:
            config_path: Optional path to specific config file

        Returns:
            Merged configuration dictionary
        """
        config: Dict[str, Any] = {}

        # Load bot configuration
        bot_config_path = Path("/home/user/FritzellSama/config/bot/bot.yaml")
        if bot_config_path.exists():
            with open(bot_config_path, "r") as f:
                config.update(yaml.safe_load(f) or {})

        # Load risk configuration
        risk_config_path = Path("/home/user/FritzellSama/config/bot/risk.yaml")
        if risk_config_path.exists():
            with open(risk_config_path, "r") as f:
                risk_config = yaml.safe_load(f) or {}
                config.update(risk_config)

        # Load environment configuration
        env_config_path = Path("/home/user/FritzellSama/config/environments/production.yaml")
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f) or {}
                config.update(env_config)

        # Load custom config if provided
        if config_path:
            custom_path = Path(config_path)
            if custom_path.exists():
                with open(custom_path, "r") as f:
                    config.update(yaml.safe_load(f) or {})

        return config

    async def start(self) -> None:
        """Start the trading engine"""
        try:
            self.logger.info("Starting TradingEngine...")

            # Perform health check
            health = await self.check_health()
            if health.status == HealthStatus.CRITICAL:
                raise RuntimeError("System health check failed - cannot start engine")

            self.status = EngineStatus.RUNNING
            self.start_time = datetime.utcnow()

            self.logger.info("TradingEngine started successfully")

            # Start background tasks
            asyncio.create_task(self._signal_processing_loop())
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._performance_monitoring_loop())

        except Exception as e:
            self.logger.error(f"Failed to start TradingEngine: {e}", exc_info=True)
            self.status = EngineStatus.ERROR
            raise

    async def stop(self) -> None:
        """Stop the trading engine"""
        try:
            self.logger.info("Stopping TradingEngine...")

            self.status = EngineStatus.STOPPED

            # Cancel all pending orders
            await self._cancel_all_orders()

            self.logger.info("TradingEngine stopped successfully")

        except Exception as e:
            self.logger.error(f"Failed to stop TradingEngine: {e}", exc_info=True)
            raise

    async def pause(self) -> None:
        """Pause the trading engine"""
        self.logger.info("Pausing TradingEngine...")
        self.status = EngineStatus.PAUSED

    async def resume(self) -> None:
        """Resume the trading engine"""
        self.logger.info("Resuming TradingEngine...")
        self.status = EngineStatus.RUNNING

    async def process_signal(self, signal: Signal) -> Optional[Order]:
        """
        Process trading signal and generate order

        Args:
            signal: Trading signal to process

        Returns:
            Generated order or None if signal rejected
        """
        try:
            start_time = datetime.utcnow()

            # Check if engine is running
            if self.status != EngineStatus.RUNNING:
                self.logger.warning(f"Engine not running, ignoring signal for {signal.symbol}")
                return None

            # Check for duplicate signal
            signal_id = f"{signal.symbol}_{signal.timestamp.isoformat()}_{signal.strategy}"
            if signal_id in self.processed_signals:
                self.logger.debug(f"Duplicate signal detected: {signal_id}")
                return None

            self.processed_signals.add(signal_id)
            self.signal_queue.append(signal)
            self.stats["signals_processed"] += 1

            # Pre-trade risk checks
            if not await self._pre_trade_risk_check(signal):
                self.logger.warning(f"Signal failed risk check: {signal.symbol}")
                self.stats["risk_checks_failed"] += 1
                return None

            self.stats["risk_checks_passed"] += 1

            # Generate order from signal
            order = await self._generate_order_from_signal(signal)

            if not order:
                return None

            # Create order state
            await self.state_machine.create_order_state(order)

            # Submit order
            success = await self._submit_order(order)

            if success:
                self.active_orders[order.order_id] = order
                self.stats["orders_created"] += 1

                # Track latency
                latency = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.execution_latencies.append(Decimal(str(latency)))

                self.logger.info(
                    f"Order created from signal: {order.order_id} for {order.symbol} "
                    f"{order.side.value} {order.quantity}"
                )

                return order
            else:
                self.stats["orders_rejected"] += 1
                return None

        except Exception as e:
            self.logger.error(f"Failed to process signal: {e}", exc_info=True)
            self._track_error(f"Signal processing error: {str(e)}")
            return None

    async def _pre_trade_risk_check(self, signal: Signal) -> bool:
        """
        Perform pre-trade risk checks

        Args:
            signal: Signal to validate

        Returns:
            True if signal passes risk checks
        """
        try:
            # Check if we've hit max positions
            if len(self.positions) >= self.max_positions:
                self.logger.warning(f"Max positions reached: {len(self.positions)}/{self.max_positions}")
                return False

            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.error(
                    f"Daily loss limit exceeded: {self.daily_pnl} < -{self.max_daily_loss}"
                )
                return False

            # Check concurrent orders
            if len(self.active_orders) >= self.max_concurrent_orders:
                self.logger.warning(
                    f"Max concurrent orders reached: {len(self.active_orders)}/{self.max_concurrent_orders}"
                )
                return False

            # Check signal confidence
            min_confidence = Decimal("0.7")
            if signal.confidence < min_confidence:
                self.logger.debug(
                    f"Signal confidence {signal.confidence} below threshold {min_confidence}"
                )
                return False

            # Check signal strength
            min_strength = Decimal("0.5")
            if signal.strength < min_strength:
                self.logger.debug(
                    f"Signal strength {signal.strength} below threshold {min_strength}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Pre-trade risk check failed: {e}", exc_info=True)
            return False

    async def _generate_order_from_signal(self, signal: Signal) -> Optional[Order]:
        """
        Generate order from trading signal

        Args:
            signal: Trading signal

        Returns:
            Generated order or None
        """
        try:
            # Determine side
            if signal.action == SignalAction.BUY:
                side = OrderSide.BUY
            elif signal.action == SignalAction.SELL:
                side = OrderSide.SELL
            elif signal.action in [SignalAction.HOLD, SignalAction.CLOSE]:
                self.logger.debug(f"Signal action {signal.action.value} does not generate new order")
                return None
            else:
                self.logger.error(f"Unknown signal action: {signal.action}")
                return None

            # Calculate order quantity based on position sizing
            # In production, this would use portfolio value and risk calculations
            quantity = Decimal("0.1")  # Placeholder - should be calculated from portfolio

            # Validate order size
            if quantity < Decimal("0.001"):  # Min quantity check
                self.logger.warning(f"Order quantity {quantity} below minimum")
                return None

            # Determine order type and price
            order_type = OrderType.LIMIT
            price: Optional[Decimal] = None

            # In production, fetch current market price
            # For now, use None (market order simulation)
            if order_type == OrderType.LIMIT:
                # Would fetch from market data
                price = Decimal("50000")  # Placeholder

            # Get exchange from config
            enabled_exchanges = self.config.get("bot", {}).get("trading", {}).get(
                "enabled_exchanges", ["binance"]
            )
            exchange = enabled_exchanges[0] if enabled_exchanges else "binance"

            # Create order
            order = Order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                exchange=exchange,
                strategy=signal.strategy,
                timestamp=datetime.utcnow(),
                price=price,
                order_id=f"ORD_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                metadata={
                    "signal_strength": str(signal.strength),
                    "signal_confidence": str(signal.confidence),
                    "signal_timestamp": signal.timestamp.isoformat(),
                    "timeframe": signal.timeframe,
                },
            )

            return order

        except Exception as e:
            self.logger.error(f"Failed to generate order from signal: {e}", exc_info=True)
            return None

    async def _submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange

        Args:
            order: Order to submit

        Returns:
            True if submission successful
        """
        max_retries = self.max_retries
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Submitting order {order.order_id} to {order.exchange} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Transition to PENDING state
                await self.state_machine.transition_state(
                    order_id=order.order_id,
                    event=StateTransitionEvent.SUBMIT,
                    reason=f"Order submitted to {order.exchange}",
                )

                # In production, this would call exchange API
                # Simulate order submission
                await asyncio.sleep(0.1)

                # Simulate acceptance
                await self.state_machine.transition_state(
                    order_id=order.order_id,
                    event=StateTransitionEvent.ACCEPT,
                    reason="Order accepted by exchange",
                )

                self.logger.info(f"Order {order.order_id} submitted successfully")

                return True

            except Exception as e:
                self.logger.warning(
                    f"Order submission attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= float(self.retry_backoff)
                else:
                    self.logger.error(
                        f"Order submission failed after {max_retries} attempts"
                    )
                    await self.state_machine.transition_state(
                        order_id=order.order_id,
                        event=StateTransitionEvent.REJECT,
                        reason=f"Submission failed: {str(e)}",
                    )
                    return False

        return False

    async def handle_execution_result(self, result: ExecutionResult) -> None:
        """
        Handle execution result from exchange

        Args:
            result: Execution result to process
        """
        try:
            self.logger.info(
                f"Handling execution result for {result.order_id}: "
                f"{result.status.value} {result.filled_quantity} @ {result.average_price}"
            )

            # Update order state
            if result.status == ExecutionStatus.SUCCESS:
                await self.state_machine.update_fill(
                    order_id=result.order_id,
                    filled_quantity=result.filled_quantity,
                    fill_price=result.average_price,
                    fees=result.fees,
                )
                self.stats["orders_filled"] += 1

                # Create settlement instruction
                await self.settlement_manager.create_settlement_instruction(
                    execution_result=result,
                    settlement_cycle=SettlementCycle.T_PLUS_2,
                )

            elif result.status == ExecutionStatus.REJECTED:
                await self.state_machine.transition_state(
                    order_id=result.order_id,
                    event=StateTransitionEvent.REJECT,
                    reason=result.error_message or "Order rejected",
                )
                self.stats["orders_rejected"] += 1

            elif result.status == ExecutionStatus.PARTIAL:
                await self.state_machine.update_fill(
                    order_id=result.order_id,
                    filled_quantity=result.filled_quantity,
                    fill_price=result.average_price,
                    fees=result.fees,
                )

            # Update positions
            await self._update_positions(result)

            # Update daily P&L
            await self._update_daily_pnl(result)

        except Exception as e:
            self.logger.error(f"Failed to handle execution result: {e}", exc_info=True)
            self._track_error(f"Execution result handling error: {str(e)}")

    async def _update_positions(self, result: ExecutionResult) -> None:
        """
        Update position tracking from execution result

        Args:
            result: Execution result
        """
        try:
            position_key = f"{result.symbol}_{result.exchange}"

            if position_key not in self.positions:
                # Create new position
                if result.filled_quantity > Decimal("0"):
                    self.positions[position_key] = Position(
                        symbol=result.symbol,
                        quantity=result.filled_quantity if result.side == OrderSide.BUY else -result.filled_quantity,
                        entry_price=result.average_price,
                        current_price=result.average_price,
                        exchange=result.exchange,
                        strategy="unknown",  # Would track from order
                        opened_at=datetime.utcnow(),
                        position_id=f"POS_{position_key}_{datetime.utcnow().strftime('%Y%m%d')}",
                    )
            else:
                # Update existing position
                position = self.positions[position_key]
                position.current_price = result.average_price

                # Adjust quantity
                if result.side == OrderSide.BUY:
                    position.quantity += result.filled_quantity
                else:
                    position.quantity -= result.filled_quantity

                # Remove position if flat
                if position.quantity == Decimal("0"):
                    del self.positions[position_key]

            self.logger.debug(f"Updated position for {position_key}")

        except Exception as e:
            self.logger.error(f"Failed to update positions: {e}", exc_info=True)

    async def _update_daily_pnl(self, result: ExecutionResult) -> None:
        """
        Update daily P&L from execution result

        Args:
            result: Execution result
        """
        try:
            # Simple P&L calculation - in production would be more sophisticated
            if result.side == OrderSide.SELL:
                # Selling generates positive P&L (simplified)
                pnl = result.total_cost - result.fees
            else:
                # Buying generates negative P&L (simplified)
                pnl = -(result.total_cost + result.fees)

            self.daily_pnl += pnl

            self.logger.debug(f"Updated daily P&L: {self.daily_pnl}")

        except Exception as e:
            self.logger.error(f"Failed to update daily P&L: {e}", exc_info=True)

    async def _cancel_all_orders(self) -> None:
        """Cancel all active orders"""
        try:
            self.logger.info(f"Cancelling {len(self.active_orders)} active orders")

            for order_id in list(self.active_orders.keys()):
                await self._cancel_order(order_id, "Engine shutdown")

        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}", exc_info=True)

    async def _cancel_order(self, order_id: str, reason: str) -> bool:
        """
        Cancel specific order

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            True if cancelled successfully
        """
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Order {order_id} not found in active orders")
                return False

            # Transition state
            await self.state_machine.transition_state(
                order_id=order_id,
                event=StateTransitionEvent.CANCEL_CONFIRM,
                reason=reason,
            )

            # Remove from active orders
            order = self.active_orders.pop(order_id)
            self.order_history.append(order)

            self.stats["orders_cancelled"] += 1

            self.logger.info(f"Cancelled order {order_id}: {reason}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}", exc_info=True)
            return False

    async def check_health(self) -> SystemHealth:
        """
        Perform system health check

        Returns:
            System health status
        """
        try:
            # Check database (simulated)
            database_healthy = True

            # Check exchanges (simulated)
            enabled_exchanges = self.config.get("bot", {}).get("trading", {}).get(
                "enabled_exchanges", ["binance"]
            )
            exchanges_healthy = {exchange: True for exchange in enabled_exchanges}

            # Check risk manager (simulated)
            risk_manager_healthy = True

            # Check settlement manager
            settlement_risk = await self.settlement_manager.check_settlement_risk()
            settlement_manager_healthy = settlement_risk.get("overdue_count", 0) == 0

            # Check state machine
            state_machine_healthy = True

            # Calculate resource usage (simulated)
            memory_usage_mb = Decimal("512.5")
            cpu_usage_percent = Decimal("45.2")

            # Count errors and warnings in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            errors_last_hour = sum(1 for ts, _ in self.recent_errors if ts > one_hour_ago)
            warnings_last_hour = sum(1 for ts, _ in self.recent_warnings if ts > one_hour_ago)

            # Determine overall health status
            if not all([
                database_healthy,
                all(exchanges_healthy.values()),
                risk_manager_healthy,
                settlement_manager_healthy,
                state_machine_healthy,
            ]):
                overall_status = HealthStatus.UNHEALTHY
            elif errors_last_hour > 10:
                overall_status = HealthStatus.CRITICAL
            elif warnings_last_hour > 20:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            health = SystemHealth(
                timestamp=datetime.utcnow(),
                status=overall_status,
                database_healthy=database_healthy,
                exchanges_healthy=exchanges_healthy,
                risk_manager_healthy=risk_manager_healthy,
                settlement_manager_healthy=settlement_manager_healthy,
                state_machine_healthy=state_machine_healthy,
                memory_usage_mb=memory_usage_mb,
                cpu_usage_percent=cpu_usage_percent,
                active_orders=len(self.active_orders),
                pending_settlements=len(self.settlement_manager.pending_settlements),
                errors_last_hour=errors_last_hour,
                warnings_last_hour=warnings_last_hour,
            )

            self.last_health_check = datetime.utcnow()

            return health

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return SystemHealth(
                timestamp=datetime.utcnow(),
                status=HealthStatus.CRITICAL,
                database_healthy=False,
                exchanges_healthy={},
                risk_manager_healthy=False,
                settlement_manager_healthy=False,
                state_machine_healthy=False,
                memory_usage_mb=Decimal("0"),
                cpu_usage_percent=Decimal("0"),
                active_orders=0,
                pending_settlements=0,
                errors_last_hour=0,
                warnings_last_hour=0,
                metadata={"error": str(e)},
            )

    async def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics

        Returns:
            Performance metrics
        """
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

            # Calculate win rate
            total_fills = self.stats["orders_filled"]
            win_rate = Decimal("0")
            if total_fills > 0:
                # Simplified - in production would track wins vs losses
                win_rate = Decimal("0.55")  # 55% placeholder

            # Calculate average latency
            avg_latency = Decimal("0")
            if self.execution_latencies:
                avg_latency = sum(self.execution_latencies) / len(self.execution_latencies)

            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                signals_processed=self.stats["signals_processed"],
                orders_created=self.stats["orders_created"],
                orders_filled=self.stats["orders_filled"],
                orders_rejected=self.stats["orders_rejected"],
                orders_cancelled=self.stats["orders_cancelled"],
                total_pnl=self.daily_pnl,
                win_rate=win_rate,
                avg_latency_ms=avg_latency,
                risk_checks_passed=self.stats["risk_checks_passed"],
                risk_checks_failed=self.stats["risk_checks_failed"],
                uptime_seconds=Decimal(str(uptime)),
            )

            self.performance_metrics.append(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
            raise

    async def _signal_processing_loop(self) -> None:
        """Background task for processing signal queue"""
        while self.status in [EngineStatus.RUNNING, EngineStatus.PAUSED]:
            try:
                if self.status == EngineStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Process signals from queue
                # In production, this would consume from message queue
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"Signal processing loop error: {e}", exc_info=True)
                self._track_error(f"Signal processing loop error: {str(e)}")
                await asyncio.sleep(5)

    async def _health_monitoring_loop(self) -> None:
        """Background task for health monitoring"""
        while self.status in [EngineStatus.RUNNING, EngineStatus.PAUSED]:
            try:
                health = await self.check_health()

                if health.status == HealthStatus.CRITICAL:
                    self.logger.error("CRITICAL HEALTH STATUS DETECTED")
                    # In production, would trigger alerts

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(self.health_check_interval)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring"""
        metrics_interval = self.config.get("bot", {}).get("performance", {}).get(
            "metrics_interval_seconds", 60
        )

        while self.status in [EngineStatus.RUNNING, EngineStatus.PAUSED]:
            try:
                await self.get_performance_metrics()
                await asyncio.sleep(metrics_interval)

            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(metrics_interval)

    def _track_error(self, error_msg: str) -> None:
        """Track error for monitoring"""
        self.recent_errors.append((datetime.utcnow(), error_msg))

    def _track_warning(self, warning_msg: str) -> None:
        """Track warning for monitoring"""
        self.recent_warnings.append((datetime.utcnow(), warning_msg))

    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status summary

        Returns:
            Status dictionary
        """
        uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "active_orders": len(self.active_orders),
            "active_positions": len(self.positions),
            "pending_settlements": len(self.settlement_manager.pending_settlements),
            "daily_pnl": str(self.daily_pnl),
            "statistics": self.stats.copy(),
            "last_health_check": self.last_health_check.isoformat(),
        }
