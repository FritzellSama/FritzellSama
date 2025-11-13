"""
Volatility Circuit Breaker
CRITICAL: Circuit breaker triggered by volatility spikes and anomalies
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class VolatilityCircuitBreaker:
    """Circuit breaker for volatility spikes and market stress"""

    def __init__(self):
        """Initialize volatility circuit breaker with config"""
        self.config = get_config()

        # Load configuration
        self.enabled = self.config.get_bool("risk", "circuit_breakers.enabled")
        self.volatility_multiplier = self.config.get_decimal("risk", "circuit_breakers.volatility_multiplier")

        # State tracking
        self.is_triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None

        # Historical volatility tracking
        self.volatility_history: deque = deque(maxlen=100)

        # Alert tracking
        self.last_alert_time: Optional[datetime] = None
        self.alert_cooldown_seconds = 60  # 1 minute between alerts

        logger.info(
            f"VolatilityCircuitBreaker initialized: enabled={self.enabled}, "
            f"volatility_multiplier={self.volatility_multiplier}x"
        )

    async def check_volatility_spike(
        self,
        current_volatility: Decimal,
        historical_volatility: Optional[Decimal] = None,
        returns: Optional[pl.DataFrame] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check for volatility spike that should trigger circuit breaker

        Args:
            current_volatility: Current volatility measure
            historical_volatility: Historical average volatility (if known)
            returns: Optional recent returns data for calculation

        Returns:
            Tuple of (should_trigger, trigger_info)
        """
        try:
            if not self.enabled:
                logger.debug("Circuit breaker disabled, skipping volatility check")
                return False, {'reason': 'disabled'}

            # Calculate historical volatility if not provided
            if historical_volatility is None and returns is not None:
                historical_volatility = await self._calculate_historical_volatility(returns)

            if historical_volatility is None or historical_volatility <= Decimal("0"):
                logger.warning("Cannot check volatility spike: invalid historical volatility")
                return False, {'reason': 'invalid_historical_volatility'}

            # Calculate volatility ratio
            volatility_ratio = current_volatility / historical_volatility

            # Check if ratio exceeds threshold
            trigger = volatility_ratio >= self.volatility_multiplier

            trigger_info = {
                'current_volatility': float(current_volatility),
                'historical_volatility': float(historical_volatility),
                'volatility_ratio': float(volatility_ratio),
                'threshold': float(self.volatility_multiplier),
                'exceeded': trigger,
                'timestamp': datetime.now().timestamp()
            }

            if trigger:
                logger.error(
                    f"VOLATILITY SPIKE DETECTED: {volatility_ratio:.2f}x historical "
                    f"(threshold: {self.volatility_multiplier:.2f}x)"
                )
                trigger_info['reason'] = 'volatility_spike'
                trigger_info['severity'] = 'critical' if volatility_ratio >= self.volatility_multiplier * Decimal("1.5") else 'warning'
            else:
                logger.debug(
                    f"Volatility check OK: {volatility_ratio:.2f}x historical "
                    f"(threshold: {self.volatility_multiplier:.2f}x)"
                )

            # Track volatility history
            self.volatility_history.append({
                'timestamp': datetime.now().timestamp(),
                'current_vol': float(current_volatility),
                'historical_vol': float(historical_volatility),
                'ratio': float(volatility_ratio)
            })

            return trigger, trigger_info

        except Exception as e:
            logger.error(f"Error checking volatility spike: {e}", exc_info=True)
            return False, {'reason': 'error', 'error': str(e)}

    async def trigger_on_volatility(
        self,
        volatility_data: Dict[str, any],
        reason: str = "volatility_spike"
    ) -> None:
        """
        Trigger circuit breaker due to volatility

        Args:
            volatility_data: Dictionary with volatility information
            reason: Reason for trigger
        """
        try:
            if self.is_triggered:
                logger.warning("Circuit breaker already triggered, ignoring new trigger")
                return

            self.is_triggered = True
            self.trigger_time = datetime.now()
            self.trigger_reason = reason

            logger.critical(
                f"CIRCUIT BREAKER TRIGGERED: {reason} at {self.trigger_time.isoformat()}"
            )
            logger.critical(f"Volatility data: {volatility_data}")

            # Send alert if not in cooldown
            current_time = datetime.now()
            if (self.last_alert_time is None or
                (current_time - self.last_alert_time).total_seconds() >= self.alert_cooldown_seconds):

                await self._send_alert({
                    'type': 'circuit_breaker_triggered',
                    'reason': reason,
                    'trigger_time': self.trigger_time.isoformat(),
                    'volatility_data': volatility_data,
                    'severity': 'critical'
                })

                self.last_alert_time = current_time

        except Exception as e:
            logger.error(f"Error triggering circuit breaker: {e}", exc_info=True)
            raise

    async def monitor_volatility(
        self,
        returns: pl.DataFrame,
        window_days: int = 20
    ) -> Dict[str, any]:
        """
        Monitor volatility and detect anomalies

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            window_days: Window size for volatility calculation

        Returns:
            Dictionary with monitoring results
        """
        try:
            if returns.is_empty():
                logger.warning("Cannot monitor volatility: empty returns")
                return {
                    'status': 'no_data',
                    'message': 'No returns data available'
                }

            # Calculate current volatility
            recent_returns = returns.tail(window_days)
            returns_array = recent_returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for volatility monitoring")
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 2 data points'
                }

            current_vol = float(np.std(returns_array, ddof=1))
            current_vol_annualized = current_vol * np.sqrt(252)

            # Calculate historical volatility (longer window)
            historical_window = min(len(returns), 252)  # Up to 1 year
            historical_returns = returns.tail(historical_window)
            historical_array = historical_returns.select("return").to_numpy().flatten()

            historical_vol = float(np.std(historical_array, ddof=1))
            historical_vol_annualized = historical_vol * np.sqrt(252)

            # Check for spike
            current_vol_decimal = Decimal(str(current_vol_annualized)).quantize(Decimal("0.000001"))
            historical_vol_decimal = Decimal(str(historical_vol_annualized)).quantize(Decimal("0.000001"))

            should_trigger, trigger_info = await self.check_volatility_spike(
                current_vol_decimal,
                historical_vol_decimal
            )

            # Determine status
            if should_trigger:
                status = 'spike_detected'
                if not self.is_triggered:
                    await self.trigger_on_volatility(trigger_info, 'volatility_spike')
            else:
                status = 'normal'

            result = {
                'status': status,
                'current_volatility': current_vol_annualized,
                'historical_volatility': historical_vol_annualized,
                'volatility_ratio': current_vol_annualized / historical_vol_annualized if historical_vol_annualized > 0 else 0,
                'threshold': float(self.volatility_multiplier),
                'is_triggered': self.is_triggered,
                'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
                'trigger_reason': self.trigger_reason,
                'timestamp': datetime.now().timestamp()
            }

            logger.info(
                f"Volatility monitoring: status={status}, "
                f"current_vol={current_vol_annualized:.4%}, "
                f"historical_vol={historical_vol_annualized:.4%}"
            )

            return result

        except Exception as e:
            logger.error(f"Error monitoring volatility: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().timestamp()
            }

    async def reset_circuit_breaker(
        self,
        reason: str = "manual_reset"
    ) -> None:
        """
        Reset circuit breaker (requires appropriate authorization)

        Args:
            reason: Reason for reset
        """
        try:
            if not self.is_triggered:
                logger.info("Circuit breaker not triggered, nothing to reset")
                return

            previous_trigger_time = self.trigger_time
            previous_reason = self.trigger_reason

            self.is_triggered = False
            self.trigger_time = None
            self.trigger_reason = None

            logger.warning(
                f"Circuit breaker RESET: reason={reason}, "
                f"was_triggered_at={previous_trigger_time}, "
                f"previous_reason={previous_reason}"
            )

            # Send alert
            await self._send_alert({
                'type': 'circuit_breaker_reset',
                'reason': reason,
                'previous_trigger_time': previous_trigger_time.isoformat() if previous_trigger_time else None,
                'previous_reason': previous_reason,
                'reset_time': datetime.now().isoformat(),
                'severity': 'warning'
            })

        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {e}", exc_info=True)
            raise

    async def get_volatility_statistics(self) -> Dict[str, any]:
        """
        Get statistics on tracked volatility

        Returns:
            Dictionary with volatility statistics
        """
        try:
            if not self.volatility_history:
                return {
                    'status': 'no_history',
                    'message': 'No volatility history available'
                }

            # Convert to arrays
            ratios = [h['ratio'] for h in self.volatility_history]

            stats = {
                'count': len(ratios),
                'mean_ratio': np.mean(ratios),
                'median_ratio': np.median(ratios),
                'std_ratio': np.std(ratios),
                'min_ratio': np.min(ratios),
                'max_ratio': np.max(ratios),
                'spike_count': sum(1 for r in ratios if r >= float(self.volatility_multiplier)),
                'spike_percentage': sum(1 for r in ratios if r >= float(self.volatility_multiplier)) / len(ratios) * 100,
                'threshold': float(self.volatility_multiplier)
            }

            logger.info(
                f"Volatility statistics: mean_ratio={stats['mean_ratio']:.2f}, "
                f"spike_count={stats['spike_count']}/{stats['count']}"
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting volatility statistics: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    async def _calculate_historical_volatility(
        self,
        returns: pl.DataFrame,
        window_days: int = 252
    ) -> Optional[Decimal]:
        """Calculate historical average volatility"""
        try:
            if returns.is_empty():
                return None

            # Use longer window for historical volatility
            historical_data = returns.tail(min(len(returns), window_days))
            returns_array = historical_data.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                return None

            vol = float(np.std(returns_array, ddof=1))
            annualized_vol = vol * np.sqrt(252)

            return Decimal(str(annualized_vol)).quantize(Decimal("0.000001"))

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}", exc_info=True)
            return None

    async def _send_alert(self, alert_data: Dict[str, any]) -> None:
        """Send alert (placeholder for integration with alert system)"""
        try:
            # Log alert (in production, would integrate with alert system)
            severity = alert_data.get('severity', 'info')

            log_func = {
                'info': logger.info,
                'warning': logger.warning,
                'critical': logger.critical
            }.get(severity, logger.info)

            log_func(f"VOLATILITY ALERT: {alert_data}")

            # In production, would send to:
            # - Email/SMS
            # - Slack/Discord
            # - PagerDuty
            # - Trading dashboard
            # etc.

        except Exception as e:
            logger.error(f"Error sending alert: {e}", exc_info=True)

    def is_circuit_breaker_triggered(self) -> bool:
        """Check if circuit breaker is currently triggered"""
        return self.is_triggered

    def get_trigger_info(self) -> Optional[Dict[str, any]]:
        """Get information about current trigger"""
        if not self.is_triggered:
            return None

        return {
            'is_triggered': True,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'trigger_reason': self.trigger_reason,
            'time_since_trigger': (datetime.now() - self.trigger_time).total_seconds() if self.trigger_time else None
        }
