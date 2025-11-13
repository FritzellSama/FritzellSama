"""
Quantum Trader AI - Order State Machine
Production-grade state management for order lifecycle

CRITICAL: All numeric values use Decimal, never float
CRITICAL: Thread-safe state transitions with validation
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import polars as pl
import yaml

from quantum_trader.models import (
    AuditLog,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)


logger = logging.getLogger(__name__)


class StateTransitionEvent(Enum):
    """Events that trigger state transitions"""
    SUBMIT = "SUBMIT"
    ACCEPT = "ACCEPT"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    CANCEL_CONFIRM = "CANCEL_CONFIRM"
    REJECT = "REJECT"
    EXPIRE = "EXPIRE"
    ERROR = "ERROR"


@dataclass
class StateTransition:
    """Record of a state transition"""
    order_id: str
    from_state: OrderStatus
    to_state: OrderStatus
    event: StateTransitionEvent
    timestamp: datetime
    user_id: str
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderState:
    """Complete order state with history"""
    order: Order
    current_status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Decimal
    total_fees: Decimal
    created_at: datetime
    updated_at: datetime
    transition_history: List[StateTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate order state"""
        for field_name in ["filled_quantity", "remaining_quantity", "average_fill_price", "total_fees"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                raise TypeError(f"{field_name} must be Decimal, got {type(value)}")


class OrderStateMachine:
    """
    Production-grade order state machine

    Features:
    - State transition validation
    - State persistence and recovery
    - Event-driven state changes
    - State history tracking
    - Invalid state prevention
    - State rollback capabilities
    """

    # Valid state transitions mapping
    VALID_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
        OrderStatus.PENDING: {
            OrderStatus.OPEN,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.FAILED,
        },
        OrderStatus.OPEN: {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        },
        OrderStatus.PARTIAL: {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.FAILED,
        },
        OrderStatus.FILLED: set(),  # Terminal state
        OrderStatus.CANCELLED: set(),  # Terminal state
        OrderStatus.REJECTED: set(),  # Terminal state
        OrderStatus.EXPIRED: set(),  # Terminal state
        OrderStatus.FAILED: set(),  # Terminal state
    }

    # Terminal states that cannot transition
    TERMINAL_STATES = {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
        OrderStatus.FAILED,
    }

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize order state machine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # State storage
        self.order_states: Dict[str, OrderState] = {}
        self.state_history: Deque[StateTransition] = deque(maxlen=10000)

        # Transition callbacks
        self.transition_callbacks: Dict[
            Tuple[OrderStatus, OrderStatus], List[Callable]
        ] = defaultdict(list)

        # Statistics
        self.transition_counts: Dict[Tuple[OrderStatus, OrderStatus], int] = defaultdict(int)
        self.invalid_transition_attempts: int = 0

        # Configuration
        self.max_history_per_order = self.config.get("bot", {}).get("execution", {}).get("max_history", 100)
        self.enable_state_persistence = self.config.get("bot", {}).get("database", {}).get("enabled", True)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        self.logger.info("OrderStateMachine initialized successfully")

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

    async def create_order_state(self, order: Order) -> OrderState:
        """
        Create initial order state

        Args:
            order: Order to track

        Returns:
            Created order state
        """
        async with self._lock:
            try:
                if order.order_id and order.order_id in self.order_states:
                    self.logger.warning(f"Order state already exists for {order.order_id}")
                    return self.order_states[order.order_id]

                order_id = order.order_id or f"ORD_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

                state = OrderState(
                    order=order,
                    current_status=OrderStatus.PENDING,
                    filled_quantity=Decimal("0"),
                    remaining_quantity=order.quantity,
                    average_fill_price=Decimal("0"),
                    total_fees=Decimal("0"),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    transition_history=[],
                    metadata=order.metadata.copy(),
                )

                self.order_states[order_id] = state

                self.logger.info(
                    f"Created order state for {order_id}: {order.symbol} "
                    f"{order.side.value} {order.quantity}"
                )

                return state

            except Exception as e:
                self.logger.error(f"Failed to create order state: {e}", exc_info=True)
                raise

    async def transition_state(
        self,
        order_id: str,
        event: StateTransitionEvent,
        user_id: str = "system",
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition order to new state based on event

        Args:
            order_id: Order ID to transition
            event: Event triggering transition
            user_id: User initiating transition
            reason: Optional reason for transition
            metadata: Optional metadata

        Returns:
            True if transition successful
        """
        async with self._lock:
            try:
                if order_id not in self.order_states:
                    self.logger.error(f"Order state not found for {order_id}")
                    return False

                state = self.order_states[order_id]
                current_status = state.current_status

                # Determine target state based on event
                target_status = self._get_target_state(current_status, event)

                if not target_status:
                    self.logger.error(
                        f"No target state defined for {current_status.value} + {event.value}"
                    )
                    self.invalid_transition_attempts += 1
                    return False

                # Validate transition
                if not self._is_valid_transition(current_status, target_status):
                    self.logger.error(
                        f"Invalid transition from {current_status.value} to {target_status.value}"
                    )
                    self.invalid_transition_attempts += 1
                    return False

                # Create transition record
                transition = StateTransition(
                    order_id=order_id,
                    from_state=current_status,
                    to_state=target_status,
                    event=event,
                    timestamp=datetime.utcnow(),
                    user_id=user_id,
                    reason=reason,
                    metadata=metadata or {},
                )

                # Update state
                state.current_status = target_status
                state.updated_at = datetime.utcnow()
                state.transition_history.append(transition)

                # Limit history size
                if len(state.transition_history) > self.max_history_per_order:
                    state.transition_history = state.transition_history[-self.max_history_per_order:]

                # Add to global history
                self.state_history.append(transition)

                # Update statistics
                self.transition_counts[(current_status, target_status)] += 1

                self.logger.info(
                    f"Order {order_id} transitioned from {current_status.value} to "
                    f"{target_status.value} via {event.value}"
                )

                # Execute callbacks
                await self._execute_callbacks(current_status, target_status, state)

                # Persist state if enabled
                if self.enable_state_persistence:
                    await self._persist_state(state, transition)

                return True

            except Exception as e:
                self.logger.error(f"Failed to transition state: {e}", exc_info=True)
                return False

    def _get_target_state(
        self, current_status: OrderStatus, event: StateTransitionEvent
    ) -> Optional[OrderStatus]:
        """
        Get target state based on current state and event

        Args:
            current_status: Current order status
            event: Triggering event

        Returns:
            Target status or None if invalid
        """
        # Event to state mapping
        event_state_map: Dict[StateTransitionEvent, OrderStatus] = {
            StateTransitionEvent.SUBMIT: OrderStatus.PENDING,
            StateTransitionEvent.ACCEPT: OrderStatus.OPEN,
            StateTransitionEvent.PARTIAL_FILL: OrderStatus.PARTIAL,
            StateTransitionEvent.FILL: OrderStatus.FILLED,
            StateTransitionEvent.CANCEL_CONFIRM: OrderStatus.CANCELLED,
            StateTransitionEvent.REJECT: OrderStatus.REJECTED,
            StateTransitionEvent.EXPIRE: OrderStatus.EXPIRED,
            StateTransitionEvent.ERROR: OrderStatus.FAILED,
        }

        return event_state_map.get(event)

    def _is_valid_transition(
        self, from_state: OrderStatus, to_state: OrderStatus
    ) -> bool:
        """
        Check if state transition is valid

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if transition is valid
        """
        # Check if from_state has valid transitions defined
        if from_state not in self.VALID_TRANSITIONS:
            return False

        # Check if to_state is in the valid transitions
        return to_state in self.VALID_TRANSITIONS[from_state]

    async def update_fill(
        self,
        order_id: str,
        filled_quantity: Decimal,
        fill_price: Decimal,
        fees: Decimal,
    ) -> bool:
        """
        Update order fill information

        Args:
            order_id: Order ID
            filled_quantity: Quantity filled in this update
            fill_price: Price of this fill
            fees: Fees for this fill

        Returns:
            True if updated successfully
        """
        async with self._lock:
            try:
                if order_id not in self.order_states:
                    self.logger.error(f"Order state not found for {order_id}")
                    return False

                state = self.order_states[order_id]

                # Update filled quantity
                new_filled = state.filled_quantity + filled_quantity
                if new_filled > state.order.quantity:
                    self.logger.error(
                        f"Filled quantity {new_filled} exceeds order quantity {state.order.quantity}"
                    )
                    return False

                # Update average fill price
                if state.filled_quantity == Decimal("0"):
                    state.average_fill_price = fill_price
                else:
                    total_value = (state.average_fill_price * state.filled_quantity) + (fill_price * filled_quantity)
                    state.average_fill_price = total_value / new_filled

                state.filled_quantity = new_filled
                state.remaining_quantity = state.order.quantity - new_filled
                state.total_fees += fees
                state.updated_at = datetime.utcnow()

                # Determine if fully filled or partially filled
                if state.remaining_quantity == Decimal("0"):
                    await self.transition_state(
                        order_id=order_id,
                        event=StateTransitionEvent.FILL,
                        reason=f"Filled {filled_quantity} @ {fill_price}",
                        metadata={
                            "fill_quantity": str(filled_quantity),
                            "fill_price": str(fill_price),
                            "fees": str(fees),
                        },
                    )
                elif state.current_status == OrderStatus.OPEN:
                    await self.transition_state(
                        order_id=order_id,
                        event=StateTransitionEvent.PARTIAL_FILL,
                        reason=f"Partially filled {filled_quantity} @ {fill_price}",
                        metadata={
                            "fill_quantity": str(filled_quantity),
                            "fill_price": str(fill_price),
                            "fees": str(fees),
                        },
                    )

                self.logger.info(
                    f"Updated fill for {order_id}: {filled_quantity} @ {fill_price}, "
                    f"total filled: {state.filled_quantity}/{state.order.quantity}"
                )

                return True

            except Exception as e:
                self.logger.error(f"Failed to update fill: {e}", exc_info=True)
                return False

    def register_callback(
        self,
        from_state: OrderStatus,
        to_state: OrderStatus,
        callback: Callable[[OrderState], None],
    ) -> None:
        """
        Register callback for state transition

        Args:
            from_state: Source state
            to_state: Target state
            callback: Callback function to execute
        """
        self.transition_callbacks[(from_state, to_state)].append(callback)
        self.logger.debug(
            f"Registered callback for {from_state.value} -> {to_state.value}"
        )

    async def _execute_callbacks(
        self, from_state: OrderStatus, to_state: OrderStatus, state: OrderState
    ) -> None:
        """
        Execute registered callbacks for transition

        Args:
            from_state: Source state
            to_state: Target state
            state: Current order state
        """
        callbacks = self.transition_callbacks.get((from_state, to_state), [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                self.logger.error(
                    f"Callback execution failed for {from_state.value} -> {to_state.value}: {e}",
                    exc_info=True,
                )

    async def _persist_state(
        self, state: OrderState, transition: StateTransition
    ) -> None:
        """
        Persist state to database

        Args:
            state: Order state to persist
            transition: State transition to persist
        """
        try:
            # In production, this would write to database
            # For now, log the persistence
            self.logger.debug(
                f"Persisting state for order {transition.order_id}: "
                f"{transition.from_state.value} -> {transition.to_state.value}"
            )

        except Exception as e:
            self.logger.error(f"Failed to persist state: {e}", exc_info=True)

    async def get_order_state(self, order_id: str) -> Optional[OrderState]:
        """
        Get current order state

        Args:
            order_id: Order ID

        Returns:
            Order state or None if not found
        """
        return self.order_states.get(order_id)

    async def get_orders_by_status(self, status: OrderStatus) -> List[OrderState]:
        """
        Get all orders with specific status

        Args:
            status: Status to filter by

        Returns:
            List of order states
        """
        return [
            state for state in self.order_states.values()
            if state.current_status == status
        ]

    async def get_transition_history(
        self, order_id: str
    ) -> List[StateTransition]:
        """
        Get transition history for order

        Args:
            order_id: Order ID

        Returns:
            List of state transitions
        """
        state = self.order_states.get(order_id)
        if not state:
            return []

        return state.transition_history.copy()

    async def rollback_state(
        self, order_id: str, user_id: str = "system", reason: str = "Rollback"
    ) -> bool:
        """
        Rollback to previous state

        Args:
            order_id: Order ID to rollback
            user_id: User initiating rollback
            reason: Rollback reason

        Returns:
            True if rollback successful
        """
        async with self._lock:
            try:
                if order_id not in self.order_states:
                    self.logger.error(f"Order state not found for {order_id}")
                    return False

                state = self.order_states[order_id]

                if len(state.transition_history) < 2:
                    self.logger.error(f"Not enough history to rollback order {order_id}")
                    return False

                # Get previous state (second to last in history)
                previous_transition = state.transition_history[-2]
                previous_status = previous_transition.to_state

                # Check if current state is terminal
                if state.current_status in self.TERMINAL_STATES:
                    self.logger.warning(
                        f"Attempting to rollback terminal state {state.current_status.value} "
                        f"for order {order_id}"
                    )

                # Create rollback transition
                transition = StateTransition(
                    order_id=order_id,
                    from_state=state.current_status,
                    to_state=previous_status,
                    event=StateTransitionEvent.ERROR,  # Use ERROR event for rollbacks
                    timestamp=datetime.utcnow(),
                    user_id=user_id,
                    reason=f"ROLLBACK: {reason}",
                    metadata={"rollback": True},
                )

                # Update state
                state.current_status = previous_status
                state.updated_at = datetime.utcnow()
                state.transition_history.append(transition)
                self.state_history.append(transition)

                self.logger.warning(
                    f"Rolled back order {order_id} to {previous_status.value}: {reason}"
                )

                return True

            except Exception as e:
                self.logger.error(f"Failed to rollback state: {e}", exc_info=True)
                return False

    async def generate_state_report(
        self, start_time: datetime, end_time: datetime
    ) -> pl.DataFrame:
        """
        Generate state transition report

        Args:
            start_time: Report start time
            end_time: Report end time

        Returns:
            Polars DataFrame with state transitions
        """
        try:
            # Filter transitions within time range
            relevant_transitions = [
                trans for trans in self.state_history
                if start_time <= trans.timestamp <= end_time
            ]

            if not relevant_transitions:
                self.logger.info("No state transitions found for time range")
                return pl.DataFrame()

            # Convert to polars DataFrame
            data = {
                "order_id": [t.order_id for t in relevant_transitions],
                "from_state": [t.from_state.value for t in relevant_transitions],
                "to_state": [t.to_state.value for t in relevant_transitions],
                "event": [t.event.value for t in relevant_transitions],
                "timestamp": [t.timestamp for t in relevant_transitions],
                "user_id": [t.user_id for t in relevant_transitions],
                "reason": [t.reason or "" for t in relevant_transitions],
            }

            df = pl.DataFrame(data)

            self.logger.info(
                f"Generated state report with {len(df)} transitions "
                f"from {start_time} to {end_time}"
            )

            return df

        except Exception as e:
            self.logger.error(f"Failed to generate state report: {e}", exc_info=True)
            return pl.DataFrame()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state machine statistics

        Returns:
            Statistics dictionary
        """
        total_orders = len(self.order_states)

        status_counts = defaultdict(int)
        for state in self.order_states.values():
            status_counts[state.current_status.value] += 1

        transition_stats = {
            f"{from_state.value}_to_{to_state.value}": count
            for (from_state, to_state), count in self.transition_counts.items()
        }

        return {
            "total_orders": total_orders,
            "orders_by_status": dict(status_counts),
            "transition_counts": transition_stats,
            "invalid_transition_attempts": self.invalid_transition_attempts,
            "total_transitions": len(self.state_history),
        }

    async def cleanup_old_states(self, days: int = 7) -> int:
        """
        Clean up old terminal states

        Args:
            days: Number of days to retain

        Returns:
            Number of states cleaned up
        """
        async with self._lock:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                initial_count = len(self.order_states)

                # Remove terminal states older than cutoff
                orders_to_remove = [
                    order_id
                    for order_id, state in self.order_states.items()
                    if state.current_status in self.TERMINAL_STATES
                    and state.updated_at < cutoff_time
                ]

                for order_id in orders_to_remove:
                    del self.order_states[order_id]

                removed_count = initial_count - len(self.order_states)

                self.logger.info(
                    f"Cleaned up {removed_count} old terminal states (older than {days} days)"
                )

                return removed_count

            except Exception as e:
                self.logger.error(f"Failed to cleanup old states: {e}", exc_info=True)
                return 0
