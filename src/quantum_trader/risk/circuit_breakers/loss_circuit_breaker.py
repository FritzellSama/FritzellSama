"""Loss Circuit Breaker - CRITICAL RISK PROTECTION"""
from decimal import Decimal
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import os, logging
from .base_circuit_breaker import BaseCircuitBreaker

logger = logging.getLogger(__name__)

class LossCircuitBreaker(BaseCircuitBreaker):
    """
    CRITICAL: Halt trading when losses exceed thresholds

    Protection against:
    - Single large loss exceeding limit
    - Cumulative daily losses
    - Consecutive losing trades
    - Drawdown exceeding maximum
    - Loss velocity (rapid losses)

    When triggered: IMMEDIATELY halts all trading to prevent further losses
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Loss thresholds
        self.max_single_loss = Decimal(str(config.get(
            'max_single_loss',
            os.getenv('CB_MAX_SINGLE_LOSS', '50000')
        )))
        self.max_daily_loss = Decimal(str(config.get(
            'max_daily_loss',
            os.getenv('CB_MAX_DAILY_LOSS', '100000')
        )))
        self.max_drawdown_pct = Decimal(str(config.get(
            'max_drawdown_pct',
            os.getenv('CB_MAX_DRAWDOWN', '10.0')
        )))
        self.max_consecutive_losses = int(config.get(
            'max_consecutive_losses',
            os.getenv('CB_MAX_CONSECUTIVE_LOSSES', '5')
        ))

        # Loss velocity thresholds
        self.max_loss_velocity = Decimal(str(config.get(
            'max_loss_velocity',
            os.getenv('CB_MAX_LOSS_VELOCITY', '25000')  # $25k per 5 minutes
        )))
        self.loss_velocity_window_minutes = int(config.get(
            'loss_velocity_window',
            os.getenv('CB_LOSS_VELOCITY_WINDOW', '5')
        ))

        # Tracking
        self.daily_loss = Decimal('0')
        self.consecutive_losses = 0
        self.loss_history: list[Dict[str, Any]] = []
        self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.peak_value: Optional[Decimal] = None

        self.logger.info(
            f"LossCircuitBreaker initialized: "
            f"max_single={self.max_single_loss}, "
            f"max_daily={self.max_daily_loss}, "
            f"max_drawdown={self.max_drawdown_pct}%, "
            f"max_consecutive={self.max_consecutive_losses}"
        )

    def check_conditions(self, metrics: Dict[str, Any]) -> bool:
        """Check if loss thresholds are breached"""
        try:
            # Reset daily tracking if new day
            self._check_daily_reset()

            # Extract metrics
            current_pnl = metrics.get('current_pnl')
            portfolio_value = metrics.get('portfolio_value')
            last_trade_pnl = metrics.get('last_trade_pnl')

            if current_pnl is None or portfolio_value is None:
                return False

            current_pnl = Decimal(str(current_pnl))
            portfolio_value = Decimal(str(portfolio_value))

            # Track peak value for drawdown calculation
            if self.peak_value is None or portfolio_value > self.peak_value:
                self.peak_value = portfolio_value

            # Check 1: Single loss threshold
            if last_trade_pnl is not None:
                last_trade_pnl = Decimal(str(last_trade_pnl))
                if last_trade_pnl < -self.max_single_loss:
                    self.logger.critical(
                        f"Single loss threshold breached: {last_trade_pnl} < -{self.max_single_loss}"
                    )
                    return True

                # Track consecutive losses
                if last_trade_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

            # Check 2: Daily loss threshold
            if current_pnl < 0:
                daily_loss = abs(current_pnl)
                self.daily_loss = daily_loss

                if daily_loss > self.max_daily_loss:
                    self.logger.critical(
                        f"Daily loss threshold breached: {daily_loss} > {self.max_daily_loss}"
                    )
                    return True

            # Check 3: Drawdown threshold
            if self.peak_value and self.peak_value > 0:
                drawdown = ((self.peak_value - portfolio_value) / self.peak_value) * Decimal('100')
                if drawdown > self.max_drawdown_pct:
                    self.logger.critical(
                        f"Drawdown threshold breached: {drawdown}% > {self.max_drawdown_pct}%"
                    )
                    return True

            # Check 4: Consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.logger.critical(
                    f"Consecutive losses threshold breached: "
                    f"{self.consecutive_losses} >= {self.max_consecutive_losses}"
                )
                return True

            # Check 5: Loss velocity
            if self._check_loss_velocity(current_pnl):
                return True

            return False

        except Exception as e:
            self.logger.error(f"Condition check error: {e}", exc_info=True)
            return False

    def _check_loss_velocity(self, current_pnl: Decimal) -> bool:
        """Check if losses are occurring too quickly"""
        try:
            now = datetime.utcnow()

            # Add current loss to history
            if current_pnl < 0:
                self.loss_history.append({
                    'pnl': current_pnl,
                    'timestamp': now
                })

            # Remove old entries outside window
            window_start = now - timedelta(minutes=self.loss_velocity_window_minutes)
            self.loss_history = [
                entry for entry in self.loss_history
                if entry['timestamp'] >= window_start
            ]

            # Calculate loss within window
            window_loss = sum([
                abs(entry['pnl']) for entry in self.loss_history
            ])

            if window_loss > self.max_loss_velocity:
                self.logger.critical(
                    f"Loss velocity threshold breached: "
                    f"{window_loss} in {self.loss_velocity_window_minutes} minutes > "
                    f"{self.max_loss_velocity}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Loss velocity check error: {e}")
            return False

    def _check_daily_reset(self) -> None:
        """Reset daily tracking at start of new day"""
        try:
            now = datetime.utcnow()
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

            if current_day > self.daily_reset_time:
                self.daily_loss = Decimal('0')
                self.consecutive_losses = 0
                self.daily_reset_time = current_day

                self.logger.info("Daily loss tracking reset")

        except Exception as e:
            self.logger.error(f"Daily reset error: {e}")

    def get_breach_message(self, metrics: Dict[str, Any]) -> str:
        """Get descriptive message about loss threshold breach"""
        try:
            messages = []

            # Check which threshold was breached
            current_pnl = Decimal(str(metrics.get('current_pnl', '0')))
            portfolio_value = Decimal(str(metrics.get('portfolio_value', '0')))
            last_trade_pnl = metrics.get('last_trade_pnl')

            if last_trade_pnl is not None:
                last_trade_pnl = Decimal(str(last_trade_pnl))
                if last_trade_pnl < -self.max_single_loss:
                    messages.append(
                        f"Single loss {last_trade_pnl} exceeds limit -{self.max_single_loss}"
                    )

            if abs(current_pnl) > self.max_daily_loss:
                messages.append(
                    f"Daily loss {abs(current_pnl)} exceeds limit {self.max_daily_loss}"
                )

            if self.peak_value and self.peak_value > 0:
                drawdown = ((self.peak_value - portfolio_value) / self.peak_value) * Decimal('100')
                if drawdown > self.max_drawdown_pct:
                    messages.append(
                        f"Drawdown {drawdown}% exceeds limit {self.max_drawdown_pct}%"
                    )

            if self.consecutive_losses >= self.max_consecutive_losses:
                messages.append(
                    f"Consecutive losses {self.consecutive_losses} exceeds limit {self.max_consecutive_losses}"
                )

            # Loss velocity
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=self.loss_velocity_window_minutes)
            recent_losses = [
                entry for entry in self.loss_history
                if entry['timestamp'] >= window_start
            ]
            window_loss = sum([abs(entry['pnl']) for entry in recent_losses])

            if window_loss > self.max_loss_velocity:
                messages.append(
                    f"Loss velocity {window_loss} in {self.loss_velocity_window_minutes}min "
                    f"exceeds limit {self.max_loss_velocity}"
                )

            if messages:
                return "; ".join(messages)
            else:
                return "Loss threshold breached"

        except Exception as e:
            return f"Loss threshold breached (error: {e})"

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get detailed loss statistics"""
        try:
            # Calculate loss velocity
            now = datetime.utcnow()
            window_start = now - timedelta(minutes=self.loss_velocity_window_minutes)
            recent_losses = [
                entry for entry in self.loss_history
                if entry['timestamp'] >= window_start
            ]
            current_velocity = sum([abs(entry['pnl']) for entry in recent_losses])

            # Calculate current drawdown
            current_drawdown_pct = Decimal('0')
            if self.peak_value and self.peak_value > 0:
                # Would need current portfolio value from metrics
                # For now, return peak value info
                pass

            return {
                'daily_loss': self.daily_loss,
                'consecutive_losses': self.consecutive_losses,
                'loss_velocity': current_velocity,
                'loss_velocity_window_minutes': self.loss_velocity_window_minutes,
                'peak_portfolio_value': self.peak_value,
                'thresholds': {
                    'max_single_loss': self.max_single_loss,
                    'max_daily_loss': self.max_daily_loss,
                    'max_drawdown_pct': self.max_drawdown_pct,
                    'max_consecutive_losses': self.max_consecutive_losses,
                    'max_loss_velocity': self.max_loss_velocity
                },
                'utilization': {
                    'daily_loss_pct': (
                        (self.daily_loss / self.max_daily_loss) * Decimal('100')
                        if self.max_daily_loss > 0 else Decimal('0')
                    ),
                    'loss_velocity_pct': (
                        (current_velocity / self.max_loss_velocity) * Decimal('100')
                        if self.max_loss_velocity > 0 else Decimal('0')
                    ),
                    'consecutive_loss_pct': (
                        (Decimal(str(self.consecutive_losses)) /
                         Decimal(str(self.max_consecutive_losses))) * Decimal('100')
                        if self.max_consecutive_losses > 0 else Decimal('0')
                    )
                },
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Loss statistics error: {e}")
            return {'error': str(e)}

    def reset_daily_limits(self) -> None:
        """Manually reset daily limits (use with caution)"""
        try:
            self.daily_loss = Decimal('0')
            self.consecutive_losses = 0
            self.daily_reset_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

            self.logger.warning("Daily loss limits manually reset")

        except Exception as e:
            self.logger.error(f"Manual reset error: {e}")

    def set_peak_value(self, value: Decimal) -> None:
        """Manually set peak portfolio value (for initialization)"""
        try:
            self.peak_value = value
            self.logger.info(f"Peak portfolio value set to: {value}")

        except Exception as e:
            self.logger.error(f"Set peak value error: {e}")
