"""
Drawdown Circuit Breaker
CRITICAL: Monitor and halt trading on excessive drawdown events
"""

import logging
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DrawdownCircuitBreaker:
    """Circuit breaker for drawdown-based risk management"""

    def __init__(self):
        """Initialize drawdown circuit breaker with config"""
        self.config = get_config()
        self.enabled = self.config.get_bool("risk", "circuit_breakers.enabled", True)
        self.drawdown_trigger = self.config.get_decimal("risk", "circuit_breakers.drawdown_trigger_pct")
        self.max_drawdown = self.config.get_decimal("risk", "loss_limits.max_drawdown_pct")
        self.check_interval = self.config.get_int("risk", "monitoring.check_interval_seconds", 1)
        self.alert_cooldown = self.config.get_int("risk", "monitoring.alert_cooldown_seconds", 300)

        self.breaker_triggered = False
        self.last_trigger_time = None
        self.peak_value = Decimal("0")
        self.peak_timestamp = None
        self.drawdown_history: List[Dict] = []

        logger.info(
            f"DrawdownCircuitBreaker initialized: enabled={self.enabled}, "
            f"trigger={self.drawdown_trigger}, max_drawdown={self.max_drawdown}"
        )

    async def check_drawdown(
        self,
        current_portfolio_value: Decimal,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, any]:
        """
        Check current drawdown level and determine if circuit breaker should trigger

        Args:
            current_portfolio_value: Current total portfolio value in USD
            timestamp: Optional timestamp (default: now)

        Returns:
            Dictionary with drawdown analysis:
            {
                'current_drawdown': Decimal (as fraction),
                'current_drawdown_pct': Decimal,
                'peak_value': Decimal,
                'current_value': Decimal,
                'should_trigger': bool,
                'severity': str,
                'time_in_drawdown': int (seconds),
                'recommendation': str
            }
        """
        try:
            if not self.enabled:
                return {
                    "current_drawdown": Decimal("0"),
                    "current_drawdown_pct": Decimal("0"),
                    "peak_value": current_portfolio_value,
                    "current_value": current_portfolio_value,
                    "should_trigger": False,
                    "severity": "none",
                    "time_in_drawdown": 0,
                    "recommendation": "Circuit breaker disabled"
                }

            if timestamp is None:
                timestamp = datetime.now()

            # Update peak if current value is higher
            if current_portfolio_value > self.peak_value:
                self.peak_value = current_portfolio_value
                self.peak_timestamp = timestamp
                logger.debug(f"New peak portfolio value: ${self.peak_value:,.2f}")

            # Calculate drawdown
            if self.peak_value > 0:
                drawdown = (self.peak_value - current_portfolio_value) / self.peak_value
            else:
                drawdown = Decimal("0")

            drawdown_pct = (drawdown * Decimal("100")).quantize(Decimal("0.01"))

            # Calculate time in drawdown
            time_in_drawdown = 0
            if self.peak_timestamp and drawdown > 0:
                time_in_drawdown = int((timestamp - self.peak_timestamp).total_seconds())

            # Determine severity
            severity = self._assess_drawdown_severity(drawdown)

            # Determine if circuit breaker should trigger
            should_trigger = (
                drawdown >= self.drawdown_trigger and
                not self._is_in_cooldown() and
                not self.breaker_triggered
            )

            # Get recommendation
            recommendation = self._get_recommendation(drawdown, severity, should_trigger)

            result = {
                "current_drawdown": drawdown.quantize(Decimal("0.000001")),
                "current_drawdown_pct": drawdown_pct,
                "peak_value": self.peak_value,
                "current_value": current_portfolio_value,
                "should_trigger": should_trigger,
                "severity": severity,
                "time_in_drawdown": time_in_drawdown,
                "peak_timestamp": self.peak_timestamp.isoformat() if self.peak_timestamp else None,
                "current_timestamp": timestamp.isoformat(),
                "trigger_threshold": self.drawdown_trigger,
                "max_threshold": self.max_drawdown,
                "recommendation": recommendation
            }

            # Record drawdown event
            self.drawdown_history.append({
                "timestamp": timestamp,
                "drawdown": drawdown,
                "portfolio_value": current_portfolio_value,
                "peak_value": self.peak_value
            })

            # Trim history (keep last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.drawdown_history = [
                h for h in self.drawdown_history if h["timestamp"] > cutoff_time
            ]

            if should_trigger:
                logger.critical(
                    f"DRAWDOWN CIRCUIT BREAKER SHOULD TRIGGER: {severity} - "
                    f"Drawdown: {drawdown_pct:.2f}%, Value: ${current_portfolio_value:,.2f}, "
                    f"Peak: ${self.peak_value:,.2f}"
                )
            elif drawdown > Decimal("0.01"):  # Log if drawdown > 1%
                logger.warning(
                    f"Drawdown: {drawdown_pct:.2f}%, severity={severity}, "
                    f"time_in_drawdown={time_in_drawdown}s"
                )

            return result

        except Exception as e:
            logger.error(f"Error checking drawdown: {e}", exc_info=True)
            raise

    async def trigger_on_drawdown(
        self,
        drawdown_result: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Trigger circuit breaker due to excessive drawdown

        Args:
            drawdown_result: Result from check_drawdown()

        Returns:
            Dictionary with trigger details and actions
        """
        try:
            self.breaker_triggered = True
            self.last_trigger_time = datetime.now()

            drawdown_pct = drawdown_result.get("current_drawdown_pct", Decimal("0"))
            severity = drawdown_result.get("severity", "medium")
            current_value = drawdown_result.get("current_value", Decimal("0"))

            logger.critical(
                f"DRAWDOWN CIRCUIT BREAKER TRIGGERED: {severity} - "
                f"Drawdown: {drawdown_pct:.2f}%, Portfolio: ${current_value:,.2f}"
            )

            # Determine actions based on severity
            actions = []
            cooldown_minutes = 5

            if severity == "critical":
                actions = [
                    "HALT_ALL_TRADING_IMMEDIATELY",
                    "LIQUIDATE_LOSING_POSITIONS",
                    "ACTIVATE_EMERGENCY_HEDGES",
                    "NOTIFY_RISK_MANAGEMENT",
                    "REQUIRE_MANUAL_OVERRIDE",
                    "RECORD_INCIDENT"
                ]
                cooldown_minutes = 60

            elif severity == "extreme":
                actions = [
                    "HALT_ALL_NEW_POSITIONS",
                    "REDUCE_POSITION_SIZES_50PCT",
                    "TIGHTEN_ALL_STOP_LOSSES",
                    "ACTIVATE_HEDGES",
                    "ALERT_MANAGEMENT"
                ]
                cooldown_minutes = 30

            elif severity == "high":
                actions = [
                    "HALT_NEW_POSITIONS",
                    "REDUCE_POSITION_SIZES",
                    "TIGHTEN_STOP_LOSSES",
                    "REVIEW_RISK_EXPOSURE",
                    "INCREASE_MONITORING"
                ]
                cooldown_minutes = 15

            else:  # moderate
                actions = [
                    "REDUCE_NEW_POSITION_SIZES",
                    "MONITOR_CLOSELY",
                    "PREPARE_HEDGES",
                    "REVIEW_STRATEGY"
                ]
                cooldown_minutes = 5

            result = {
                "triggered": True,
                "trigger_time": self.last_trigger_time.isoformat(),
                "reason": "Excessive drawdown detected",
                "severity": severity,
                "drawdown_pct": drawdown_pct,
                "trigger_threshold": self.drawdown_trigger,
                "max_threshold": self.max_drawdown,
                "portfolio_value": current_value,
                "peak_value": drawdown_result.get("peak_value", Decimal("0")),
                "actions_taken": actions,
                "trading_halted": severity in ["critical", "extreme"],
                "cooldown_minutes": cooldown_minutes,
                "resume_time": (self.last_trigger_time + timedelta(minutes=cooldown_minutes)).isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error triggering drawdown circuit breaker: {e}", exc_info=True)
            raise

    async def track_peak_values(
        self,
        value_history: pl.DataFrame
    ) -> Dict[str, any]:
        """
        Track peak values and calculate running maximum drawdown from historical data

        Args:
            value_history: DataFrame with columns ['timestamp', 'portfolio_value']

        Returns:
            Dictionary with peak tracking results:
            {
                'current_peak': Decimal,
                'all_time_peak': Decimal,
                'max_drawdown': Decimal,
                'max_drawdown_date': str,
                'recovery_time_days': Optional[int],
                'underwater_periods': List[Dict]
            }
        """
        try:
            if value_history.is_empty():
                logger.error("Cannot track peaks: empty value history")
                raise ValueError("Value history DataFrame is empty")

            # Sort by timestamp
            sorted_history = value_history.sort("timestamp")

            # Calculate running maximum (peak)
            values = sorted_history.select("portfolio_value").to_numpy().flatten()
            timestamps = sorted_history.select("timestamp").to_series().to_list()

            running_max = []
            current_max = Decimal("0")

            for val in values:
                val_decimal = Decimal(str(val))
                if val_decimal > current_max:
                    current_max = val_decimal
                running_max.append(current_max)

            # Calculate drawdowns
            drawdowns = []
            for i, val in enumerate(values):
                val_decimal = Decimal(str(val))
                peak = running_max[i]
                if peak > 0:
                    dd = (peak - val_decimal) / peak
                else:
                    dd = Decimal("0")
                drawdowns.append(dd)

            # Find maximum drawdown
            max_drawdown = max(drawdowns)
            max_dd_idx = drawdowns.index(max_drawdown)
            max_dd_date = timestamps[max_dd_idx]

            # Find underwater periods (drawdown > 0)
            underwater_periods = []
            in_drawdown = False
            drawdown_start = None

            for i, dd in enumerate(drawdowns):
                if dd > Decimal("0.01") and not in_drawdown:  # Entered drawdown
                    in_drawdown = True
                    drawdown_start = timestamps[i]
                elif dd <= Decimal("0.01") and in_drawdown:  # Recovered
                    in_drawdown = False
                    underwater_periods.append({
                        "start": drawdown_start.isoformat() if isinstance(drawdown_start, datetime) else str(drawdown_start),
                        "end": timestamps[i].isoformat() if isinstance(timestamps[i], datetime) else str(timestamps[i]),
                        "duration_seconds": int((timestamps[i] - drawdown_start).total_seconds()) if isinstance(timestamps[i], datetime) else 0
                    })

            # If still in drawdown
            if in_drawdown:
                underwater_periods.append({
                    "start": drawdown_start.isoformat() if isinstance(drawdown_start, datetime) else str(drawdown_start),
                    "end": "ongoing",
                    "duration_seconds": int((datetime.now() - drawdown_start).total_seconds()) if isinstance(drawdown_start, datetime) else 0
                })

            # Calculate average recovery time
            recovery_times = [
                p["duration_seconds"] / 86400  # Convert to days
                for p in underwater_periods
                if p["end"] != "ongoing"
            ]
            avg_recovery_days = int(sum(recovery_times) / len(recovery_times)) if recovery_times else None

            result = {
                "current_peak": running_max[-1],
                "all_time_peak": max(running_max),
                "max_drawdown": max_drawdown.quantize(Decimal("0.000001")),
                "max_drawdown_pct": (max_drawdown * Decimal("100")).quantize(Decimal("0.01")),
                "max_drawdown_date": max_dd_date.isoformat() if isinstance(max_dd_date, datetime) else str(max_dd_date),
                "avg_recovery_time_days": avg_recovery_days,
                "num_underwater_periods": len(underwater_periods),
                "underwater_periods": underwater_periods[-5:],  # Last 5 periods
                "currently_underwater": in_drawdown,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Peak tracking: max_drawdown={result['max_drawdown_pct']:.2f}%, "
                f"underwater_periods={result['num_underwater_periods']}, "
                f"avg_recovery={avg_recovery_days} days"
            )

            return result

        except Exception as e:
            logger.error(f"Error tracking peak values: {e}", exc_info=True)
            raise

    def reset_breaker(self) -> None:
        """Reset circuit breaker state"""
        self.breaker_triggered = False
        self.last_trigger_time = None
        logger.info("Drawdown circuit breaker reset")

    def reset_peak(self, new_peak: Optional[Decimal] = None) -> None:
        """Reset peak value (e.g., after capital injection or withdrawal)"""
        old_peak = self.peak_value
        self.peak_value = new_peak if new_peak is not None else Decimal("0")
        self.peak_timestamp = datetime.now()
        logger.info(f"Peak value reset: ${old_peak:,.2f} -> ${self.peak_value:,.2f}")

    def _assess_drawdown_severity(self, drawdown: Decimal) -> str:
        """Assess drawdown severity level"""
        if drawdown >= self.max_drawdown:
            return "critical"
        elif drawdown >= self.max_drawdown * Decimal("0.9"):
            return "extreme"
        elif drawdown >= self.drawdown_trigger:
            return "high"
        elif drawdown >= self.drawdown_trigger * Decimal("0.75"):
            return "moderate"
        elif drawdown >= Decimal("0.02"):  # 2%
            return "low"
        else:
            return "none"

    def _is_in_cooldown(self) -> bool:
        """Check if circuit breaker is in cooldown period"""
        if self.last_trigger_time is None:
            return False

        cooldown_delta = timedelta(seconds=self.alert_cooldown)
        return datetime.now() < (self.last_trigger_time + cooldown_delta)

    def _get_recommendation(self, drawdown: Decimal, severity: str, should_trigger: bool) -> str:
        """Get recommendation based on drawdown state"""
        if severity == "none":
            return "Continue normal operations"

        if should_trigger:
            if severity == "critical":
                return "HALT ALL TRADING - Critical drawdown exceeded"
            elif severity == "extreme":
                return "HALT NEW POSITIONS - Extreme drawdown"
            else:
                return "REDUCE RISK - High drawdown threshold reached"
        else:
            if self._is_in_cooldown():
                return "Monitor closely - Cooldown period active"
            else:
                return f"Monitor drawdown - Current: {drawdown*100:.2f}%"
