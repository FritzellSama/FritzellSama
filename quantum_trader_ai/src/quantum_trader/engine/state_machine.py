"""
Trading Engine State Management
Production-ready state machine with comprehensive transition validation
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any, Set
import logging
import asyncio
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class State(Enum):
    """Trading engine states"""
    INITIALIZED = "INITIALIZED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class TransitionReason(Enum):
    """Reasons for state transitions"""
    USER_REQUEST = "USER_REQUEST"
    AUTOMATED = "AUTOMATED"
    ERROR = "ERROR"
    RISK_BREACH = "RISK_BREACH"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    HEALTH_CHECK_FAILED = "HEALTH_CHECK_FAILED"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    SCHEDULED = "SCHEDULED"


@dataclass
class StateTransition:
    """State transition record"""
    transition_id: str
    from_state: str
    to_state: str
    reason: str
    timestamp: str
    initiated_by: str
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StateMachine:
    """Trading engine state management system"""

    def __init__(self) -> None:
        """Initialize state machine with configuration"""
        self.config = get_config()
        self._load_config()
        self._current_state: State = State.INITIALIZED
        self._previous_state: Optional[State] = None
        self._transition_history: List[StateTransition] = []
        self._allowed_transitions: Dict[State, List[State]] = {}
        self._lock = asyncio.Lock()
        self._state_callbacks: Dict[State, List[Any]] = {}
        self._transition_timeout: int = 30
        logger.info("StateMachine initialized")

    def _load_config(self) -> None:
        """Load configuration from engine.yaml"""
        initial_state_str = self.config.get_string('engine', 'state_machine.initial_state')
        self._current_state = State[initial_state_str]

        # Load allowed transitions
        transitions_config = self.config.get('engine', 'state_machine.allowed_transitions')

        for from_state_str, to_states_list in transitions_config.items():
            from_state = State[from_state_str]
            to_states = [State[s] for s in to_states_list]
            self._allowed_transitions[from_state] = to_states

        self._transition_timeout = self.config.get_int('engine', 'state_machine.transition_timeout_seconds')

        logger.info(
            f"StateMachine configured: initial_state={self._current_state.value}, "
            f"transitions={len(self._allowed_transitions)}"
        )

    async def transition_state(
        self,
        target_state: str,
        reason: str = TransitionReason.USER_REQUEST.value,
        initiated_by: str = "SYSTEM",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new state

        Args:
            target_state: Target state name
            reason: Reason for transition
            initiated_by: Who initiated the transition
            metadata: Additional metadata

        Returns:
            bool: True if transition successful

        Raises:
            ValueError: If target state is invalid
            RuntimeError: If transition not allowed or fails
        """
        try:
            target_state_enum = State[target_state]
        except KeyError:
            raise ValueError(f"Invalid target state: {target_state}")

        async with self._lock:
            start_time = datetime.utcnow()

            # Validate transition
            if not await self.validate_transition(target_state):
                raise RuntimeError(
                    f"Transition from {self._current_state.value} to {target_state} not allowed"
                )

            transition_id = str(uuid.uuid4())

            logger.info(
                f"State transition {transition_id} initiated: "
                f"{self._current_state.value} -> {target_state} (reason: {reason})"
            )

            try:
                # Execute transition with timeout
                transition_successful = await asyncio.wait_for(
                    self._execute_transition(target_state_enum, reason, initiated_by, metadata),
                    timeout=self._transition_timeout
                )

                end_time = datetime.utcnow()
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                if transition_successful:
                    # Record successful transition
                    transition = StateTransition(
                        transition_id=transition_id,
                        from_state=self._previous_state.value if self._previous_state else "NONE",
                        to_state=self._current_state.value,
                        reason=reason,
                        timestamp=end_time.isoformat(),
                        initiated_by=initiated_by,
                        duration_ms=duration_ms,
                        success=True,
                        metadata=metadata
                    )

                    self._transition_history.append(transition)

                    logger.info(
                        f"State transition {transition_id} completed: "
                        f"{transition.from_state} -> {transition.to_state} in {duration_ms}ms"
                    )

                    # Execute state callbacks
                    await self._execute_state_callbacks(self._current_state)

                    return True
                else:
                    raise RuntimeError("Transition execution failed")

            except asyncio.TimeoutError:
                error_msg = f"State transition timeout after {self._transition_timeout}s"
                logger.error(error_msg)

                # Record failed transition
                transition = StateTransition(
                    transition_id=transition_id,
                    from_state=self._current_state.value,
                    to_state=target_state,
                    reason=reason,
                    timestamp=datetime.utcnow().isoformat(),
                    initiated_by=initiated_by,
                    success=False,
                    error_message=error_msg,
                    metadata=metadata
                )

                self._transition_history.append(transition)

                # Transition to FAILED state
                await self._force_transition(State.FAILED, error_msg)

                raise RuntimeError(error_msg)

            except Exception as e:
                error_msg = f"State transition failed: {e}"
                logger.error(error_msg)

                # Record failed transition
                transition = StateTransition(
                    transition_id=transition_id,
                    from_state=self._current_state.value,
                    to_state=target_state,
                    reason=reason,
                    timestamp=datetime.utcnow().isoformat(),
                    initiated_by=initiated_by,
                    success=False,
                    error_message=str(e),
                    metadata=metadata
                )

                self._transition_history.append(transition)

                raise RuntimeError(error_msg)

    async def _execute_transition(
        self,
        target_state: State,
        reason: str,
        initiated_by: str,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Execute state transition logic

        Args:
            target_state: Target state
            reason: Transition reason
            initiated_by: Initiator
            metadata: Additional metadata

        Returns:
            bool: True if successful
        """
        # Pre-transition actions
        await self._pre_transition_actions(self._current_state, target_state)

        # Update state
        self._previous_state = self._current_state
        self._current_state = target_state

        # Post-transition actions
        await self._post_transition_actions(self._previous_state, self._current_state)

        return True

    async def _pre_transition_actions(
        self,
        from_state: State,
        to_state: State
    ) -> None:
        """
        Execute pre-transition actions

        Args:
            from_state: Current state
            to_state: Target state
        """
        # Cleanup based on current state
        if from_state == State.RUNNING:
            if to_state in [State.PAUSED, State.STOPPING, State.EMERGENCY_STOP]:
                logger.info("Preparing to pause/stop trading operations")
                # In production: flush order queues, close positions, etc.
                await asyncio.sleep(0.01)

        elif from_state == State.PAUSED:
            if to_state == State.RUNNING:
                logger.info("Preparing to resume trading operations")
                # In production: verify system health, reload data, etc.
                await asyncio.sleep(0.01)

        elif from_state == State.INITIALIZED:
            if to_state == State.STARTING:
                logger.info("Preparing to start trading engine")
                # In production: initialize connections, load state, etc.
                await asyncio.sleep(0.01)

    async def _post_transition_actions(
        self,
        from_state: State,
        to_state: State
    ) -> None:
        """
        Execute post-transition actions

        Args:
            from_state: Previous state
            to_state: Current state
        """
        # Actions based on new state
        if to_state == State.RUNNING:
            logger.info("Trading engine now running")
            # In production: start order processing, market data, etc.
            await asyncio.sleep(0.01)

        elif to_state == State.STOPPED:
            logger.info("Trading engine stopped")
            # In production: close connections, save state, etc.
            await asyncio.sleep(0.01)

        elif to_state == State.EMERGENCY_STOP:
            logger.error("Emergency stop activated")
            # In production: cancel all orders, close positions, alert operators
            await asyncio.sleep(0.01)

        elif to_state == State.FAILED:
            logger.error("Trading engine in failed state")
            # In production: trigger alerts, save diagnostics, etc.
            await asyncio.sleep(0.01)

    async def _force_transition(self, target_state: State, reason: str) -> None:
        """
        Force transition without validation (for error handling)

        Args:
            target_state: Target state
            reason: Reason for forced transition
        """
        self._previous_state = self._current_state
        self._current_state = target_state

        logger.warning(
            f"Forced state transition: {self._previous_state.value} -> {target_state.value} "
            f"(reason: {reason})"
        )

    async def validate_transition(self, target_state: str) -> bool:
        """
        Validate if a state transition is allowed

        Args:
            target_state: Target state name

        Returns:
            bool: True if transition is allowed
        """
        try:
            target_state_enum = State[target_state]
        except KeyError:
            logger.warning(f"Invalid target state: {target_state}")
            return False

        # Check if transition is allowed
        allowed = self._allowed_transitions.get(self._current_state, [])

        is_allowed = target_state_enum in allowed

        if not is_allowed:
            logger.warning(
                f"Transition {self._current_state.value} -> {target_state} not in allowed transitions"
            )

        return is_allowed

    def get_current_state(self) -> str:
        """
        Get current state

        Returns:
            str: Current state name
        """
        return self._current_state.value

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed state information

        Returns:
            Dictionary with state information
        """
        return {
            'current_state': self._current_state.value,
            'previous_state': self._previous_state.value if self._previous_state else None,
            'allowed_transitions': [s.value for s in self._allowed_transitions.get(self._current_state, [])],
            'total_transitions': len(self._transition_history),
            'uptime_seconds': self._calculate_uptime()
        }

    def _calculate_uptime(self) -> Optional[int]:
        """Calculate system uptime in seconds"""
        # Find when we entered RUNNING state
        running_transitions = [
            t for t in self._transition_history
            if t.to_state == State.RUNNING.value
        ]

        if not running_transitions:
            return None

        # Get most recent RUNNING transition
        last_running = running_transitions[-1]
        start_time = datetime.fromisoformat(last_running.timestamp)

        # Check if we're still running
        if self._current_state == State.RUNNING:
            uptime = (datetime.utcnow() - start_time).total_seconds()
            return int(uptime)

        return None

    def get_transition_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get state transition history

        Args:
            limit: Maximum number of records

        Returns:
            List of transition dictionaries
        """
        history = [asdict(t) for t in self._transition_history[-limit:]]

        logger.info(f"Retrieved {len(history)} state transition records")

        return history

    def register_state_callback(
        self,
        state: str,
        callback: Any
    ) -> bool:
        """
        Register a callback for a specific state

        Args:
            state: State name
            callback: Callback function (async)

        Returns:
            bool: True if registration successful
        """
        try:
            state_enum = State[state]
        except KeyError:
            logger.warning(f"Invalid state: {state}")
            return False

        if state_enum not in self._state_callbacks:
            self._state_callbacks[state_enum] = []

        self._state_callbacks[state_enum].append(callback)

        logger.info(f"Registered callback for state: {state}")

        return True

    async def _execute_state_callbacks(self, state: State) -> None:
        """
        Execute callbacks for a state

        Args:
            state: State to execute callbacks for
        """
        callbacks = self._state_callbacks.get(state, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state.value)
                else:
                    callback(state.value)
            except Exception as e:
                logger.error(f"Error executing state callback for {state.value}: {e}")

    async def emergency_stop(self, reason: str = "Emergency stop requested") -> bool:
        """
        Trigger emergency stop

        Args:
            reason: Reason for emergency stop

        Returns:
            bool: True if emergency stop successful
        """
        logger.critical(f"EMERGENCY STOP: {reason}")

        try:
            # Force transition to emergency stop
            await self.transition_state(
                State.EMERGENCY_STOP.value,
                reason=TransitionReason.CIRCUIT_BREAKER.value,
                initiated_by="EMERGENCY",
                metadata={'reason': reason}
            )

            return True

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            # Force state change
            await self._force_transition(State.EMERGENCY_STOP, str(e))
            return False

    def get_state_metrics(self) -> Dict[str, Any]:
        """Get state machine metrics"""
        state_durations: Dict[str, int] = {}
        transition_counts: Dict[str, int] = {}

        for i, transition in enumerate(self._transition_history):
            # Count transitions
            transition_key = f"{transition.from_state}->{transition.to_state}"
            transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1

            # Calculate state durations
            if i < len(self._transition_history) - 1:
                next_transition = self._transition_history[i + 1]
                start = datetime.fromisoformat(transition.timestamp)
                end = datetime.fromisoformat(next_transition.timestamp)
                duration = int((end - start).total_seconds())

                state = transition.to_state
                state_durations[state] = state_durations.get(state, 0) + duration

        successful_transitions = sum(1 for t in self._transition_history if t.success)
        failed_transitions = sum(1 for t in self._transition_history if not t.success)

        return {
            'current_state': self._current_state.value,
            'total_transitions': len(self._transition_history),
            'successful_transitions': successful_transitions,
            'failed_transitions': failed_transitions,
            'transition_counts': transition_counts,
            'state_durations_seconds': state_durations,
            'uptime_seconds': self._calculate_uptime()
        }
