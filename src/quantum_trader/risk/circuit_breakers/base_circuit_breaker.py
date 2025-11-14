"""Base Circuit Breaker - CRITICAL RISK MANAGEMENT"""
from decimal import Decimal
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import os, logging

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing if safe to resume

class CircuitBreakerTrigger:
    """Represents a circuit breaker trigger event"""

    def __init__(
        self,
        trigger_id: str,
        trigger_type: str,
        severity: str,
        message: str,
        metrics: Dict[str, Any],
        timestamp: datetime
    ):
        self.trigger_id = trigger_id
        self.trigger_type = trigger_type
        self.severity = severity
        self.message = message
        self.metrics = metrics
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trigger_id': self.trigger_id,
            'trigger_type': self.trigger_type,
            'severity': self.severity,
            'message': self.message,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }

class BaseCircuitBreaker(ABC):
    """
    CRITICAL: Base class for all circuit breakers

    Circuit breakers automatically halt trading when risk thresholds are exceeded.
    They protect the system from catastrophic losses.

    States:
    - CLOSED: Normal operation, trading allowed
    - OPEN: Trading halted, risk threshold exceeded
    - HALF_OPEN: Testing recovery, limited trading allowed
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Core parameters
        self.enabled = config.get(
            'enabled',
            os.getenv('CB_ENABLED', 'true').lower() == 'true'
        )
        self.check_interval_seconds = int(config.get(
            'check_interval_seconds',
            os.getenv('CB_CHECK_INTERVAL', '10')
        ))

        # Recovery parameters
        self.recovery_timeout_seconds = int(config.get(
            'recovery_timeout_seconds',
            os.getenv('CB_RECOVERY_TIMEOUT', '300')
        ))
        self.half_open_test_duration = int(config.get(
            'half_open_test_duration',
            os.getenv('CB_HALF_OPEN_DURATION', '60')
        ))

        # State
        self.state = CircuitBreakerState.CLOSED
        self.last_check_time: Optional[datetime] = None
        self.state_change_time = datetime.utcnow()
        self.trip_count = 0

        # Trigger history
        self.triggers: List[CircuitBreakerTrigger] = []
        self.max_trigger_history = 100

        self.logger.info(
            f"{self.__class__.__name__} initialized: "
            f"enabled={self.enabled}, state={self.state.value}"
        )

    @abstractmethod
    def check_conditions(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if circuit breaker should trip

        Args:
            metrics: Current system metrics

        Returns:
            True if conditions met to trip breaker, False otherwise
        """
        pass

    @abstractmethod
    def get_breach_message(self, metrics: Dict[str, Any]) -> str:
        """Get descriptive message about what threshold was breached"""
        pass

    def update(self, metrics: Dict[str, Any]) -> CircuitBreakerState:
        """
        Update circuit breaker state based on current metrics

        Returns:
            Current state after update
        """
        try:
            if not self.enabled:
                return self.state

            current_time = datetime.utcnow()
            self.last_check_time = current_time

            # State machine logic
            if self.state == CircuitBreakerState.CLOSED:
                # Check if should trip
                if self.check_conditions(metrics):
                    self._trip(metrics)

            elif self.state == CircuitBreakerState.OPEN:
                # Check if ready to test recovery
                time_in_state = (current_time - self.state_change_time).total_seconds()
                if time_in_state >= self.recovery_timeout_seconds:
                    self._transition_to_half_open()

            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Test if conditions have improved
                time_in_state = (current_time - self.state_change_time).total_seconds()

                if self.check_conditions(metrics):
                    # Still breaching, back to OPEN
                    self._trip(metrics)
                elif time_in_state >= self.half_open_test_duration:
                    # Conditions good, recover to CLOSED
                    self._recover()

            return self.state

        except Exception as e:
            self.logger.error(f"Update error: {e}", exc_info=True)
            return self.state

    def _trip(self, metrics: Dict[str, Any]) -> None:
        """Trip the circuit breaker (halt trading)"""
        try:
            previous_state = self.state
            self.state = CircuitBreakerState.OPEN
            self.state_change_time = datetime.utcnow()
            self.trip_count += 1

            # Create trigger event
            trigger = CircuitBreakerTrigger(
                trigger_id=f"trigger_{self.trip_count}_{datetime.utcnow().timestamp()}",
                trigger_type=self.__class__.__name__,
                severity='CRITICAL',
                message=self.get_breach_message(metrics),
                metrics=metrics,
                timestamp=datetime.utcnow()
            )

            self.triggers.append(trigger)
            self._trim_trigger_history()

            self.logger.critical(
                f"CIRCUIT BREAKER TRIPPED: {trigger.message} | "
                f"State: {previous_state.value} -> {self.state.value} | "
                f"Trip count: {self.trip_count}"
            )

            # In production: Send alerts, halt all trading, notify operators
            self._send_alert(trigger)

        except Exception as e:
            self.logger.error(f"Trip error: {e}", exc_info=True)

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state to test recovery"""
        try:
            previous_state = self.state
            self.state = CircuitBreakerState.HALF_OPEN
            self.state_change_time = datetime.utcnow()

            self.logger.warning(
                f"Circuit breaker testing recovery: "
                f"{previous_state.value} -> {self.state.value}"
            )

        except Exception as e:
            self.logger.error(f"Transition error: {e}")

    def _recover(self) -> None:
        """Recover circuit breaker to normal operation"""
        try:
            previous_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.state_change_time = datetime.utcnow()

            self.logger.info(
                f"Circuit breaker RECOVERED: "
                f"{previous_state.value} -> {self.state.value}"
            )

            # In production: Send recovery alert, resume trading
            self._send_recovery_notification()

        except Exception as e:
            self.logger.error(f"Recovery error: {e}")

    def _send_alert(self, trigger: CircuitBreakerTrigger) -> None:
        """Send critical alert about circuit breaker trip"""
        try:
            # In production: Send to monitoring system, PagerDuty, email, SMS, etc.
            alert_message = (
                f"CRITICAL ALERT: Circuit Breaker Tripped\n"
                f"Type: {trigger.trigger_type}\n"
                f"Message: {trigger.message}\n"
                f"Time: {trigger.timestamp}\n"
                f"Metrics: {trigger.metrics}"
            )

            self.logger.critical(alert_message)

            # TODO: Integration with alerting system
            # - Send to PagerDuty
            # - Send email to risk team
            # - Send SMS to on-call engineer
            # - Publish to monitoring dashboard

        except Exception as e:
            self.logger.error(f"Alert error: {e}")

    def _send_recovery_notification(self) -> None:
        """Send notification about circuit breaker recovery"""
        try:
            notification = (
                f"Circuit Breaker Recovered: {self.__class__.__name__}\n"
                f"Time: {datetime.utcnow()}\n"
                f"Previous trips: {self.trip_count}"
            )

            self.logger.info(notification)

            # TODO: Send to monitoring system

        except Exception as e:
            self.logger.error(f"Recovery notification error: {e}")

    def _trim_trigger_history(self) -> None:
        """Trim trigger history to max size"""
        if len(self.triggers) > self.max_trigger_history:
            self.triggers = self.triggers[-self.max_trigger_history:]

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed"""
        if not self.enabled:
            return True

        if self.state == CircuitBreakerState.OPEN:
            return False

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited trading during testing
            return True

        return True

    def manual_trip(self, reason: str) -> None:
        """Manually trip the circuit breaker"""
        try:
            metrics = {'manual_trip': True, 'reason': reason}
            self._trip(metrics)

            self.logger.critical(f"Manual circuit breaker trip: {reason}")

        except Exception as e:
            self.logger.error(f"Manual trip error: {e}")

    def manual_reset(self) -> None:
        """Manually reset the circuit breaker"""
        try:
            previous_state = self.state
            self._recover()

            self.logger.warning(
                f"Manual circuit breaker reset: {previous_state.value} -> {self.state.value}"
            )

        except Exception as e:
            self.logger.error(f"Manual reset error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        try:
            time_in_state = None
            if self.state_change_time:
                time_in_state = (datetime.utcnow() - self.state_change_time).total_seconds()

            recent_triggers = [
                t.to_dict() for t in self.triggers[-10:]
            ]

            return {
                'breaker_type': self.__class__.__name__,
                'enabled': self.enabled,
                'state': self.state.value,
                'time_in_state_seconds': time_in_state,
                'trip_count': self.trip_count,
                'last_check_time': self.last_check_time,
                'recent_triggers': recent_triggers,
                'trading_allowed': self.is_trading_allowed(),
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Status error: {e}")
            return {'error': str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        try:
            # Calculate time spent in each state
            total_time = Decimal('0')
            if self.triggers:
                first_trigger = self.triggers[0].timestamp
                total_time = Decimal(str((datetime.utcnow() - first_trigger).total_seconds()))

            return {
                'breaker_type': self.__class__.__name__,
                'total_trips': self.trip_count,
                'total_triggers': len(self.triggers),
                'uptime_seconds': total_time,
                'avg_time_between_trips': (
                    total_time / Decimal(str(self.trip_count))
                    if self.trip_count > 0 else Decimal('0')
                ),
                'current_state': self.state.value,
                'enabled': self.enabled,
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {'error': str(e)}
