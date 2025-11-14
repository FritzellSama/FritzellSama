"""
Drawdown Circuit Breaker
Quantum Trader AI - Production Risk Management

Monitors and responds to portfolio drawdowns:
- Real-time drawdown tracking
- Peak tracking and updates
- Drawdown threshold triggers
- Recovery monitoring

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


class DrawdownLevel(Enum):
    """Drawdown severity level"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class DrawdownConfig:
    """Drawdown circuit breaker configuration"""
    max_drawdown_tolerance: Decimal
    max_daily_loss_percent: Decimal
    max_intraday_loss_percent: Decimal
    assessment_frequency_ms: int

    # Drawdown-specific thresholds
    warning_drawdown_percent: Decimal = Decimal('5.0')
    critical_drawdown_percent: Decimal = Decimal('8.0')
    emergency_drawdown_percent: Decimal = Decimal('10.0')
    recovery_threshold_percent: Decimal = Decimal('1.0')

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'DrawdownConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_drawdown_tolerance=Decimal(str(risk_config['risk_model']['max_drawdown_tolerance'])),
            max_daily_loss_percent=Decimal(str(risk_config['global']['max_daily_loss_percent'])),
            max_intraday_loss_percent=Decimal(str(risk_config['global']['max_intraday_loss_percent'])),
            assessment_frequency_ms=int(risk_config['global']['assessment_frequency_ms'])
        )


@dataclass
class DrawdownState:
    """Current drawdown state"""
    current_value: Decimal
    peak_value: Decimal
    peak_timestamp: datetime
    drawdown_amount: Decimal
    drawdown_percent: Decimal
    in_drawdown: bool
    drawdown_start: Optional[datetime]
    drawdown_duration_seconds: Optional[int]
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class DrawdownAlert:
    """Drawdown circuit breaker alert"""
    level: DrawdownLevel
    alert_type: str
    current_drawdown_percent: Decimal
    threshold_percent: Decimal
    peak_value: Decimal
    current_value: Decimal
    duration_seconds: int
    recommended_action: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class DrawdownCircuitBreaker:
    """
    Circuit breaker for portfolio drawdown monitoring.

    Tracks peak portfolio value and triggers protective actions when
    drawdown exceeds configured thresholds.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize drawdown circuit breaker with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[DrawdownConfig] = None
        self._load_config()

        # State tracking
        self.peak_value: Decimal = Decimal('0')
        self.peak_timestamp: Optional[datetime] = None
        self.drawdown_start: Optional[datetime] = None
        self.in_drawdown: bool = False
        self.value_history: List[Tuple[datetime, Decimal]] = []
        self.triggered_alerts: List[DrawdownAlert] = []
        self.last_assessment: Optional[datetime] = None

    def _load_config(self) -> None:
        """Load configuration from yaml files with retry logic"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with open(self.risk_config_path, 'r') as f:
                    risk_config = yaml.safe_load(f)

                with open(self.prod_config_path, 'r') as f:
                    prod_config = yaml.safe_load(f)

                self.config = DrawdownConfig.from_yaml(risk_config, prod_config)
                logger.info("Drawdown circuit breaker configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load drawdown CB config after {max_retries} attempts") from e

    async def update_and_check_drawdown(
        self,
        current_portfolio_value: Decimal
    ) -> Optional[DrawdownAlert]:
        """
        Update drawdown state and check for threshold breaches.

        Args:
            current_portfolio_value: Current portfolio value

        Returns:
            DrawdownAlert if threshold breached, None otherwise
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()
            self.last_assessment = timestamp

            # Update peak if current value is higher
            if current_portfolio_value > self.peak_value:
                self.peak_value = current_portfolio_value
                self.peak_timestamp = timestamp

                # Exit drawdown if we were in one
                if self.in_drawdown:
                    logger.info(
                        f"Drawdown recovery: Portfolio reached new peak ${self.peak_value}"
                    )
                    self.in_drawdown = False
                    self.drawdown_start = None

            # Calculate current drawdown
            drawdown_state = await self._calculate_drawdown_state(
                current_portfolio_value,
                timestamp
            )

            # Store in history
            self.value_history.append((timestamp, current_portfolio_value))

            # Keep only recent history (last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.value_history = [
                (t, v) for t, v in self.value_history
                if t > cutoff_time
            ]

            # Check if we entered drawdown
            if not self.in_drawdown and drawdown_state.drawdown_percent > Decimal('0.5'):
                self.in_drawdown = True
                self.drawdown_start = timestamp
                logger.info(f"Entered drawdown: {drawdown_state.drawdown_percent}%")

            # Determine drawdown level
            level = await self._determine_drawdown_level(drawdown_state.drawdown_percent)

            if level == DrawdownLevel.NORMAL:
                return None

            # Calculate duration
            if self.drawdown_start:
                duration_seconds = int((timestamp - self.drawdown_start).total_seconds())
            else:
                duration_seconds = 0

            # Determine alert type
            alert_type = await self._determine_alert_type(level, drawdown_state)

            # Determine recommended action
            recommended_action = await self._determine_drawdown_action(
                level,
                drawdown_state.drawdown_percent,
                duration_seconds
            )

            alert = DrawdownAlert(
                level=level,
                alert_type=alert_type,
                current_drawdown_percent=drawdown_state.drawdown_percent,
                threshold_percent=await self._get_threshold_for_level(level),
                peak_value=self.peak_value,
                current_value=current_portfolio_value,
                duration_seconds=duration_seconds,
                recommended_action=recommended_action,
                timestamp=timestamp,
                metadata={
                    'peak_timestamp': self.peak_timestamp.isoformat() if self.peak_timestamp else None,
                    'drawdown_amount': str(drawdown_state.drawdown_amount)
                }
            )

            # Log alert
            logger.warning(
                f"DRAWDOWN ALERT: Level={level.value}, Type={alert_type}, "
                f"Drawdown={drawdown_state.drawdown_percent}%, "
                f"Duration={duration_seconds}s"
            )

            # Store alert
            self.triggered_alerts.append(alert)

            # Keep only recent alerts (last 100)
            self.triggered_alerts = self.triggered_alerts[-100:]

            return alert

        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            raise

    async def get_drawdown_state(
        self,
        current_portfolio_value: Decimal
    ) -> DrawdownState:
        """
        Get current drawdown state without triggering alerts.

        Args:
            current_portfolio_value: Current portfolio value

        Returns:
            DrawdownState with current metrics
        """
        try:
            timestamp = datetime.utcnow()
            return await self._calculate_drawdown_state(current_portfolio_value, timestamp)

        except Exception as e:
            logger.error(f"Failed to get drawdown state: {e}")
            raise

    async def _calculate_drawdown_state(
        self,
        current_value: Decimal,
        timestamp: datetime
    ) -> DrawdownState:
        """Calculate current drawdown state"""
        try:
            if self.peak_value == Decimal('0'):
                return DrawdownState(
                    current_value=current_value,
                    peak_value=current_value,
                    peak_timestamp=timestamp,
                    drawdown_amount=Decimal('0'),
                    drawdown_percent=Decimal('0'),
                    in_drawdown=False,
                    drawdown_start=None,
                    drawdown_duration_seconds=None,
                    timestamp=timestamp
                )

            # Calculate drawdown
            drawdown_amount = self.peak_value - current_value
            drawdown_percent = (drawdown_amount / self.peak_value) * Decimal('100')

            # Calculate duration
            duration_seconds = None
            if self.drawdown_start:
                duration_seconds = int((timestamp - self.drawdown_start).total_seconds())

            return DrawdownState(
                current_value=current_value,
                peak_value=self.peak_value,
                peak_timestamp=self.peak_timestamp or timestamp,
                drawdown_amount=drawdown_amount,
                drawdown_percent=drawdown_percent,
                in_drawdown=self.in_drawdown,
                drawdown_start=self.drawdown_start,
                drawdown_duration_seconds=duration_seconds,
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"Drawdown state calculation failed: {e}")
            raise

    async def _determine_drawdown_level(
        self,
        drawdown_percent: Decimal
    ) -> DrawdownLevel:
        """Determine drawdown severity level"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if drawdown_percent >= self.config.emergency_drawdown_percent:
                return DrawdownLevel.EMERGENCY
            elif drawdown_percent >= self.config.critical_drawdown_percent:
                return DrawdownLevel.CRITICAL
            elif drawdown_percent >= self.config.warning_drawdown_percent:
                return DrawdownLevel.WARNING
            else:
                return DrawdownLevel.NORMAL

        except Exception as e:
            logger.error(f"Failed to determine drawdown level: {e}")
            raise

    async def _determine_alert_type(
        self,
        level: DrawdownLevel,
        state: DrawdownState
    ) -> str:
        """Determine type of drawdown alert"""
        try:
            if level == DrawdownLevel.EMERGENCY:
                return "emergency_drawdown"
            elif level == DrawdownLevel.CRITICAL:
                return "critical_drawdown"
            elif level == DrawdownLevel.WARNING:
                if state.drawdown_duration_seconds and state.drawdown_duration_seconds > 3600:
                    return "prolonged_drawdown"
                else:
                    return "warning_drawdown"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Failed to determine alert type: {e}")
            raise

    async def _determine_drawdown_action(
        self,
        level: DrawdownLevel,
        drawdown_percent: Decimal,
        duration_seconds: int
    ) -> str:
        """Determine recommended action for drawdown"""
        try:
            if level == DrawdownLevel.EMERGENCY:
                return "EMERGENCY: Halt all trading, liquidate risky positions, preserve capital"
            elif level == DrawdownLevel.CRITICAL:
                return "CRITICAL: Reduce all positions by 50%, halt new trades, review strategy"
            elif level == DrawdownLevel.WARNING:
                if duration_seconds > 7200:  # 2 hours
                    return "WARNING: Prolonged drawdown detected, reduce position sizes by 25%, review risk exposure"
                else:
                    return "WARNING: Drawdown threshold breached, monitor closely, consider position reduction"
            else:
                return "No action required"

        except Exception as e:
            logger.error(f"Failed to determine drawdown action: {e}")
            raise

    async def _get_threshold_for_level(
        self,
        level: DrawdownLevel
    ) -> Decimal:
        """Get threshold value for given level"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if level == DrawdownLevel.EMERGENCY:
                return self.config.emergency_drawdown_percent
            elif level == DrawdownLevel.CRITICAL:
                return self.config.critical_drawdown_percent
            elif level == DrawdownLevel.WARNING:
                return self.config.warning_drawdown_percent
            else:
                return Decimal('0')

        except Exception as e:
            logger.error(f"Failed to get threshold for level: {e}")
            raise

    async def should_trigger_circuit_breaker(
        self,
        alert: DrawdownAlert
    ) -> bool:
        """
        Determine if circuit breaker should trigger.

        Args:
            alert: Drawdown alert

        Returns:
            True if circuit breaker should trigger
        """
        try:
            # Trigger on EMERGENCY level
            if alert.level == DrawdownLevel.EMERGENCY:
                logger.warning("Drawdown circuit breaker TRIGGERED - EMERGENCY level")
                return True

            # Trigger on CRITICAL level
            if alert.level == DrawdownLevel.CRITICAL:
                logger.warning("Drawdown circuit breaker TRIGGERED - CRITICAL level")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to determine circuit breaker trigger: {e}")
            raise

    async def calculate_intraday_drawdown(
        self
    ) -> Decimal:
        """
        Calculate intraday drawdown (from today's peak).

        Returns:
            Intraday drawdown percentage
        """
        try:
            if not self.value_history:
                return Decimal('0')

            timestamp = datetime.utcnow()
            today_start = datetime(timestamp.year, timestamp.month, timestamp.day)

            # Get today's values
            today_values = [
                v for t, v in self.value_history
                if t >= today_start
            ]

            if not today_values:
                return Decimal('0')

            intraday_peak = max(today_values)
            current_value = today_values[-1]

            if intraday_peak == Decimal('0'):
                return Decimal('0')

            intraday_drawdown = ((intraday_peak - current_value) / intraday_peak) * Decimal('100')

            return intraday_drawdown

        except Exception as e:
            logger.error(f"Intraday drawdown calculation failed: {e}")
            raise

    async def calculate_max_historical_drawdown(
        self
    ) -> Tuple[Decimal, Optional[datetime], Optional[datetime]]:
        """
        Calculate maximum historical drawdown from available history.

        Returns:
            Tuple of (max_drawdown_percent, peak_date, trough_date)
        """
        try:
            if len(self.value_history) < 2:
                return Decimal('0'), None, None

            max_drawdown = Decimal('0')
            max_drawdown_peak_time = None
            max_drawdown_trough_time = None

            # Track running peak
            running_peak = self.value_history[0][1]
            running_peak_time = self.value_history[0][0]

            for timestamp, value in self.value_history:
                # Update running peak
                if value > running_peak:
                    running_peak = value
                    running_peak_time = timestamp

                # Calculate drawdown from running peak
                if running_peak > Decimal('0'):
                    drawdown = ((running_peak - value) / running_peak) * Decimal('100')

                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        max_drawdown_peak_time = running_peak_time
                        max_drawdown_trough_time = timestamp

            return max_drawdown, max_drawdown_peak_time, max_drawdown_trough_time

        except Exception as e:
            logger.error(f"Max historical drawdown calculation failed: {e}")
            raise

    async def reset_peak(
        self,
        new_peak: Optional[Decimal] = None
    ) -> None:
        """
        Reset peak value (use with caution).

        Args:
            new_peak: Optional new peak value (default: current peak)
        """
        try:
            timestamp = datetime.utcnow()

            if new_peak is not None:
                self.peak_value = new_peak
            # else keep current peak

            self.peak_timestamp = timestamp
            self.in_drawdown = False
            self.drawdown_start = None

            logger.info(f"Drawdown peak reset to {self.peak_value}")

        except Exception as e:
            logger.error(f"Failed to reset peak: {e}")
            raise

    async def get_recovery_progress(
        self,
        current_value: Decimal
    ) -> Decimal:
        """
        Calculate recovery progress toward peak.

        Args:
            current_value: Current portfolio value

        Returns:
            Recovery percentage (0-100)
        """
        try:
            if not self.in_drawdown or self.drawdown_start is None:
                return Decimal('100')

            # Get value at drawdown start
            drawdown_start_value = None
            for t, v in self.value_history:
                if t >= self.drawdown_start:
                    drawdown_start_value = v
                    break

            if drawdown_start_value is None:
                return Decimal('0')

            # Calculate recovery
            drawdown_amount = self.peak_value - drawdown_start_value
            recovery_amount = current_value - drawdown_start_value

            if drawdown_amount == Decimal('0'):
                return Decimal('100')

            recovery_percent = (recovery_amount / drawdown_amount) * Decimal('100')
            recovery_percent = max(Decimal('0'), min(recovery_percent, Decimal('100')))

            return recovery_percent

        except Exception as e:
            logger.error(f"Recovery progress calculation failed: {e}")
            raise


async def main():
    """Example usage of drawdown circuit breaker"""
    try:
        # Initialize circuit breaker
        cb = DrawdownCircuitBreaker(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Simulate portfolio values
        portfolio_values = [
            Decimal('1000000'),
            Decimal('1010000'),
            Decimal('1020000'),  # Peak
            Decimal('1000000'),
            Decimal('980000'),
            Decimal('960000'),  # 6% drawdown
            Decimal('940000'),  # 8% drawdown - should trigger critical
        ]

        for i, value in enumerate(portfolio_values):
            logger.info(f"Step {i+1}: Portfolio value = ${value}")

            # Update and check drawdown
            alert = await cb.update_and_check_drawdown(value)

            if alert:
                logger.warning(f"Alert triggered: {alert.alert_type}, {alert.recommended_action}")

                # Check if circuit breaker should trigger
                should_trigger = await cb.should_trigger_circuit_breaker(alert)
                if should_trigger:
                    logger.critical("CIRCUIT BREAKER TRIGGERED!")

            # Get current state
            state = await cb.get_drawdown_state(value)
            logger.info(
                f"  State: Drawdown={state.drawdown_percent}%, "
                f"In drawdown={state.in_drawdown}"
            )

            await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Drawdown circuit breaker example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
