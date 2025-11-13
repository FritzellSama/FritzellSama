"""
Maximum Drawdown Calculator
CRITICAL: Calculate and monitor maximum drawdown and underwater periods
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class MaxDrawdownCalculator:
    """Calculate maximum drawdown and track underwater periods"""

    def __init__(self):
        """Initialize max drawdown calculator with config"""
        self.config = get_config()
        self.max_drawdown_pct = self.config.get_decimal("risk", "loss_limits.max_drawdown_pct")
        self.drawdown_trigger_pct = self.config.get_decimal("risk", "circuit_breakers.drawdown_trigger_pct")

        logger.info(
            f"MaxDrawdownCalculator initialized: max_drawdown={self.max_drawdown_pct}, "
            f"trigger={self.drawdown_trigger_pct}"
        )

    async def calculate_max_drawdown(
        self,
        portfolio_values: pl.DataFrame,
        window_days: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate maximum drawdown from portfolio value series

        Args:
            portfolio_values: DataFrame with columns ['timestamp', 'value']
            window_days: Optional rolling window in days (None = all history)

        Returns:
            Dictionary with drawdown metrics:
            {
                'max_drawdown': Decimal (percentage),
                'max_drawdown_value': Decimal (dollar amount),
                'peak_value': Decimal,
                'trough_value': Decimal,
                'peak_timestamp': float,
                'trough_timestamp': float,
                'current_drawdown': Decimal,
                'recovery_time_days': Decimal
            }
        """
        try:
            if portfolio_values.is_empty():
                logger.error("Cannot calculate max drawdown: empty portfolio values")
                raise ValueError("Portfolio values DataFrame cannot be empty")

            # Apply rolling window if specified
            if window_days:
                cutoff_date = datetime.now().timestamp() - (window_days * 86400)
                portfolio_values = portfolio_values.filter(pl.col("timestamp") >= cutoff_date)

            if len(portfolio_values) < 2:
                logger.warning("Insufficient data for drawdown calculation")
                return {
                    "max_drawdown": Decimal("0"),
                    "max_drawdown_value": Decimal("0"),
                    "peak_value": Decimal("0"),
                    "trough_value": Decimal("0"),
                    "peak_timestamp": 0.0,
                    "trough_timestamp": 0.0,
                    "current_drawdown": Decimal("0"),
                    "recovery_time_days": Decimal("0")
                }

            # Sort by timestamp
            portfolio_values = portfolio_values.sort("timestamp")

            # Extract values and timestamps
            values = portfolio_values.select("value").to_numpy().flatten()
            timestamps = portfolio_values.select("timestamp").to_numpy().flatten()

            # Calculate running maximum (peak)
            running_max = np.maximum.accumulate(values)

            # Calculate drawdown at each point
            drawdowns = (values - running_max) / running_max

            # Find maximum drawdown
            max_dd_idx = np.argmin(drawdowns)
            max_drawdown = float(drawdowns[max_dd_idx])

            # Find the peak before maximum drawdown
            peak_idx = np.argmax(running_max[:max_dd_idx + 1] == running_max[max_dd_idx])

            peak_value = float(values[peak_idx])
            trough_value = float(values[max_dd_idx])
            max_dd_value = peak_value - trough_value

            peak_timestamp = float(timestamps[peak_idx])
            trough_timestamp = float(timestamps[max_dd_idx])

            # Calculate current drawdown
            current_value = float(values[-1])
            current_peak = float(running_max[-1])
            current_drawdown = (current_value - current_peak) / current_peak if current_peak > 0 else 0.0

            # Calculate recovery time
            recovery_time_days = Decimal("0")
            if max_dd_idx < len(values) - 1:
                # Find when portfolio recovered to previous peak
                recovery_idx = None
                for i in range(max_dd_idx + 1, len(values)):
                    if values[i] >= running_max[max_dd_idx]:
                        recovery_idx = i
                        break

                if recovery_idx is not None:
                    recovery_time_seconds = timestamps[recovery_idx] - trough_timestamp
                    recovery_time_days = Decimal(str(recovery_time_seconds / 86400.0)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                else:
                    # Still underwater
                    current_time = timestamps[-1]
                    recovery_time_days = Decimal(str((current_time - trough_timestamp) / 86400.0)).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

            result = {
                "max_drawdown": Decimal(str(abs(max_drawdown))).quantize(Decimal("0.000001")),
                "max_drawdown_value": Decimal(str(max_dd_value)).quantize(Decimal("0.01")),
                "peak_value": Decimal(str(peak_value)).quantize(Decimal("0.01")),
                "trough_value": Decimal(str(trough_value)).quantize(Decimal("0.01")),
                "peak_timestamp": peak_timestamp,
                "trough_timestamp": trough_timestamp,
                "current_drawdown": Decimal(str(abs(current_drawdown))).quantize(Decimal("0.000001")),
                "recovery_time_days": recovery_time_days
            }

            logger.info(
                f"Max drawdown: {result['max_drawdown']:.4%} "
                f"(${result['max_drawdown_value']:,.2f}), "
                f"current: {result['current_drawdown']:.4%}, "
                f"recovery_time: {recovery_time_days} days"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}", exc_info=True)
            raise

    async def get_drawdown_duration(
        self,
        portfolio_values: pl.DataFrame,
        threshold_pct: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate average and maximum drawdown duration

        Args:
            portfolio_values: DataFrame with columns ['timestamp', 'value']
            threshold_pct: Minimum drawdown threshold to count (default: 1%)

        Returns:
            Dictionary with duration metrics:
            {
                'avg_duration_days': Decimal,
                'max_duration_days': Decimal,
                'current_duration_days': Decimal,
                'num_drawdowns': int,
                'time_underwater_pct': Decimal
            }
        """
        try:
            if portfolio_values.is_empty():
                logger.error("Cannot calculate drawdown duration: empty portfolio values")
                raise ValueError("Portfolio values DataFrame cannot be empty")

            if threshold_pct is None:
                threshold_pct = Decimal("0.01")  # 1% default threshold

            # Sort by timestamp
            portfolio_values = portfolio_values.sort("timestamp")

            # Extract values and timestamps
            values = portfolio_values.select("value").to_numpy().flatten()
            timestamps = portfolio_values.select("timestamp").to_numpy().flatten()

            if len(values) < 2:
                return {
                    "avg_duration_days": Decimal("0"),
                    "max_duration_days": Decimal("0"),
                    "current_duration_days": Decimal("0"),
                    "num_drawdowns": 0,
                    "time_underwater_pct": Decimal("0")
                }

            # Calculate running maximum and drawdown
            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max

            # Identify underwater periods (drawdown exceeds threshold)
            underwater = drawdowns < -float(threshold_pct)

            # Find all underwater periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0

            for i, is_underwater in enumerate(underwater):
                if is_underwater and not in_drawdown:
                    # Start of new drawdown period
                    start_idx = i
                    in_drawdown = True
                elif not is_underwater and in_drawdown:
                    # End of drawdown period
                    duration_seconds = timestamps[i] - timestamps[start_idx]
                    duration_days = duration_seconds / 86400.0
                    drawdown_periods.append(duration_days)
                    in_drawdown = False

            # Handle case where still in drawdown
            current_duration_days = Decimal("0")
            if in_drawdown:
                duration_seconds = timestamps[-1] - timestamps[start_idx]
                duration_days = duration_seconds / 86400.0
                drawdown_periods.append(duration_days)
                current_duration_days = Decimal(str(duration_days)).quantize(Decimal("0.01"))

            # Calculate statistics
            num_drawdowns = len(drawdown_periods)
            avg_duration_days = Decimal("0")
            max_duration_days = Decimal("0")

            if num_drawdowns > 0:
                avg_duration_days = Decimal(str(np.mean(drawdown_periods))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                max_duration_days = Decimal(str(np.max(drawdown_periods))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            # Calculate percentage of time underwater
            total_underwater_time = np.sum(underwater) / len(underwater) * 100.0
            time_underwater_pct = Decimal(str(total_underwater_time)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            result = {
                "avg_duration_days": avg_duration_days,
                "max_duration_days": max_duration_days,
                "current_duration_days": current_duration_days,
                "num_drawdowns": num_drawdowns,
                "time_underwater_pct": time_underwater_pct
            }

            logger.info(
                f"Drawdown durations: {num_drawdowns} periods, "
                f"avg={avg_duration_days} days, max={max_duration_days} days, "
                f"underwater={time_underwater_pct}%"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating drawdown duration: {e}", exc_info=True)
            raise

    async def track_underwater_periods(
        self,
        portfolio_values: pl.DataFrame,
        threshold_pct: Optional[Decimal] = None
    ) -> List[Dict[str, any]]:
        """
        Track all underwater periods with detailed information

        Args:
            portfolio_values: DataFrame with columns ['timestamp', 'value']
            threshold_pct: Minimum drawdown threshold to count (default: 1%)

        Returns:
            List of dictionaries, each containing:
            {
                'start_timestamp': float,
                'end_timestamp': float,
                'start_date': str,
                'end_date': str,
                'duration_days': Decimal,
                'max_drawdown_pct': Decimal,
                'peak_value': Decimal,
                'trough_value': Decimal,
                'recovered': bool
            }
        """
        try:
            if portfolio_values.is_empty():
                logger.error("Cannot track underwater periods: empty portfolio values")
                raise ValueError("Portfolio values DataFrame cannot be empty")

            if threshold_pct is None:
                threshold_pct = Decimal("0.01")  # 1% default threshold

            # Sort by timestamp
            portfolio_values = portfolio_values.sort("timestamp")

            # Extract values and timestamps
            values = portfolio_values.select("value").to_numpy().flatten()
            timestamps = portfolio_values.select("timestamp").to_numpy().flatten()

            if len(values) < 2:
                return []

            # Calculate running maximum and drawdown
            running_max = np.maximum.accumulate(values)
            drawdowns = (values - running_max) / running_max

            # Identify underwater periods
            underwater = drawdowns < -float(threshold_pct)

            # Track all periods
            periods = []
            in_drawdown = False
            start_idx = 0
            period_peak = 0.0

            for i, is_underwater in enumerate(underwater):
                if is_underwater and not in_drawdown:
                    # Start of new drawdown period
                    start_idx = i
                    period_peak = running_max[i]
                    in_drawdown = True

                elif not is_underwater and in_drawdown:
                    # End of drawdown period
                    end_idx = i

                    # Find minimum value in this period
                    period_values = values[start_idx:end_idx + 1]
                    trough_value = np.min(period_values)
                    max_dd = (trough_value - period_peak) / period_peak

                    duration_seconds = timestamps[end_idx] - timestamps[start_idx]
                    duration_days = duration_seconds / 86400.0

                    periods.append({
                        "start_timestamp": float(timestamps[start_idx]),
                        "end_timestamp": float(timestamps[end_idx]),
                        "start_date": datetime.fromtimestamp(timestamps[start_idx]).isoformat(),
                        "end_date": datetime.fromtimestamp(timestamps[end_idx]).isoformat(),
                        "duration_days": Decimal(str(duration_days)).quantize(Decimal("0.01")),
                        "max_drawdown_pct": Decimal(str(abs(max_dd))).quantize(Decimal("0.000001")),
                        "peak_value": Decimal(str(period_peak)).quantize(Decimal("0.01")),
                        "trough_value": Decimal(str(trough_value)).quantize(Decimal("0.01")),
                        "recovered": True
                    })

                    in_drawdown = False

            # Handle ongoing drawdown
            if in_drawdown:
                period_values = values[start_idx:]
                trough_value = np.min(period_values)
                max_dd = (trough_value - period_peak) / period_peak

                duration_seconds = timestamps[-1] - timestamps[start_idx]
                duration_days = duration_seconds / 86400.0

                periods.append({
                    "start_timestamp": float(timestamps[start_idx]),
                    "end_timestamp": float(timestamps[-1]),
                    "start_date": datetime.fromtimestamp(timestamps[start_idx]).isoformat(),
                    "end_date": datetime.fromtimestamp(timestamps[-1]).isoformat(),
                    "duration_days": Decimal(str(duration_days)).quantize(Decimal("0.01")),
                    "max_drawdown_pct": Decimal(str(abs(max_dd))).quantize(Decimal("0.000001")),
                    "peak_value": Decimal(str(period_peak)).quantize(Decimal("0.01")),
                    "trough_value": Decimal(str(trough_value)).quantize(Decimal("0.01")),
                    "recovered": False
                })

            logger.info(
                f"Tracked {len(periods)} underwater periods, "
                f"{sum(1 for p in periods if not p['recovered'])} ongoing"
            )

            return periods

        except Exception as e:
            logger.error(f"Error tracking underwater periods: {e}", exc_info=True)
            raise

    async def check_drawdown_breach(
        self,
        current_drawdown: Decimal
    ) -> Tuple[bool, str]:
        """
        Check if current drawdown breaches configured limits

        Args:
            current_drawdown: Current drawdown as decimal (e.g., 0.05 for 5%)

        Returns:
            Tuple of (is_breached, severity_level)
            severity_level: 'safe', 'warning', 'critical', 'halt'
        """
        try:
            if current_drawdown <= Decimal("0"):
                return False, "safe"

            # Check against trigger threshold
            if current_drawdown >= self.max_drawdown_pct:
                logger.critical(
                    f"CRITICAL: Drawdown {current_drawdown:.4%} exceeds max limit "
                    f"{self.max_drawdown_pct:.4%}"
                )
                return True, "halt"

            elif current_drawdown >= self.drawdown_trigger_pct:
                logger.warning(
                    f"WARNING: Drawdown {current_drawdown:.4%} exceeds trigger "
                    f"{self.drawdown_trigger_pct:.4%}"
                )
                return True, "critical"

            elif current_drawdown >= self.drawdown_trigger_pct * Decimal("0.75"):
                logger.info(
                    f"CAUTION: Drawdown {current_drawdown:.4%} approaching trigger "
                    f"(75% of threshold)"
                )
                return False, "warning"

            else:
                return False, "safe"

        except Exception as e:
            logger.error(f"Error checking drawdown breach: {e}", exc_info=True)
            raise
