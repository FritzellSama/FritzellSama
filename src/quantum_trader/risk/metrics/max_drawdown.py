"""
Maximum Drawdown Calculations for Quantum Trader AI

Production-grade drawdown tracking and analysis with:
- Real-time drawdown monitoring
- Historical maximum drawdown calculation
- Drawdown duration tracking
- Recovery time analysis
- Ulcer index calculation
- Peak-to-trough analysis

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import logging

import polars as pl
import yaml
import numpy as np

from quantum_trader.models import Position


logger = logging.getLogger(__name__)


@dataclass
class DrawdownConfig:
    """Configuration for drawdown calculations"""
    max_drawdown_tolerance: Decimal
    drawdown_alert_threshold: Decimal
    recovery_target_days: int
    ulcer_index_window: int
    track_underwater_periods: bool


@dataclass
class DrawdownEvent:
    """Represents a drawdown event"""
    peak_value: Decimal
    trough_value: Decimal
    peak_timestamp: datetime
    trough_timestamp: datetime
    recovery_timestamp: Optional[datetime]
    drawdown_percent: Decimal
    duration_days: int
    recovery_days: Optional[int]
    is_active: bool


@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown metrics"""
    current_drawdown: Decimal
    max_drawdown: Decimal
    max_drawdown_duration_days: int
    average_drawdown: Decimal
    current_underwater_days: int
    ulcer_index: Decimal
    recovery_factor: Decimal
    num_drawdown_events: int
    time_underwater_percent: Decimal


class MaxDrawdownCalculator:
    """
    Maximum drawdown calculator and tracker

    Provides comprehensive drawdown analysis including:
    - Real-time current drawdown
    - Historical max drawdown
    - Drawdown duration and recovery time
    - Ulcer index for pain measurement
    - Underwater period tracking
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize maximum drawdown calculator

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.peak_value = Decimal('0')
        self.current_value = Decimal('0')
        self.drawdown_events: List[DrawdownEvent] = []
        self.equity_curve: List[Tuple[datetime, Decimal]] = []

        logger.info("MaxDrawdownCalculator initialized")

    def _load_config(self, config_path: str, env_config_path: str) -> DrawdownConfig:
        """
        Load configuration from YAML files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config

        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            risk_model = risk_config.get('risk_model', {})

            return DrawdownConfig(
                max_drawdown_tolerance=Decimal(str(risk_model.get('max_drawdown_tolerance', 10.0))),
                drawdown_alert_threshold=Decimal(str(risk_config.get('global', {}).get('max_daily_loss_percent', 5.0))),
                recovery_target_days=30,
                ulcer_index_window=14,
                track_underwater_periods=True
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def update(self, current_value: Decimal, timestamp: datetime) -> Decimal:
        """
        Update drawdown calculation with new portfolio value

        Args:
            current_value: Current portfolio value
            timestamp: Timestamp of value

        Returns:
            Current drawdown percentage
        """
        if not isinstance(current_value, Decimal):
            raise TypeError(f"current_value must be Decimal, got {type(current_value)}")

        self.current_value = current_value
        self.equity_curve.append((timestamp, current_value))

        # Update peak value
        if current_value > self.peak_value:
            # New peak - check if we're recovering from drawdown
            if self.drawdown_events and self.drawdown_events[-1].is_active:
                # Mark recovery
                self.drawdown_events[-1].recovery_timestamp = timestamp
                self.drawdown_events[-1].is_active = False
                self.drawdown_events[-1].recovery_days = (
                    timestamp - self.drawdown_events[-1].trough_timestamp
                ).days

                logger.info(
                    "Recovered from drawdown: %s%% in %d days",
                    self.drawdown_events[-1].drawdown_percent,
                    self.drawdown_events[-1].recovery_days
                )

            self.peak_value = current_value

        # Calculate current drawdown
        current_drawdown = self.calculate_current_drawdown()

        # Check if this is a new drawdown event
        if current_drawdown > Decimal('0.01') and (  # > 0.01% threshold
            not self.drawdown_events or not self.drawdown_events[-1].is_active
        ):
            # Start new drawdown event
            event = DrawdownEvent(
                peak_value=self.peak_value,
                trough_value=current_value,
                peak_timestamp=timestamp,
                trough_timestamp=timestamp,
                recovery_timestamp=None,
                drawdown_percent=current_drawdown,
                duration_days=0,
                recovery_days=None,
                is_active=True
            )
            self.drawdown_events.append(event)

            logger.warning("New drawdown event started: %s%%", current_drawdown)

        # Update active drawdown
        elif self.drawdown_events and self.drawdown_events[-1].is_active:
            event = self.drawdown_events[-1]

            # Update if new trough
            if current_value < event.trough_value:
                event.trough_value = current_value
                event.trough_timestamp = timestamp

            event.drawdown_percent = current_drawdown
            event.duration_days = (timestamp - event.peak_timestamp).days

        # Alert if threshold breached
        if current_drawdown > self.config.drawdown_alert_threshold:
            logger.warning(
                "Drawdown alert: %s%% (threshold: %s%%)",
                current_drawdown, self.config.drawdown_alert_threshold
            )

        return current_drawdown

    def calculate_current_drawdown(self) -> Decimal:
        """
        Calculate current drawdown from peak

        Returns:
            Current drawdown as percentage
        """
        if self.peak_value <= Decimal('0'):
            return Decimal('0')

        drawdown = ((self.peak_value - self.current_value) / self.peak_value) * Decimal('100')

        return max(Decimal('0'), drawdown)

    def calculate_max_drawdown(
        self,
        equity_curve_df: Optional[pl.DataFrame] = None
    ) -> Decimal:
        """
        Calculate maximum drawdown from equity curve

        Args:
            equity_curve_df: Optional DataFrame with columns [timestamp, value]
                           If None, uses internal equity curve

        Returns:
            Maximum drawdown percentage
        """
        if equity_curve_df is not None:
            if not isinstance(equity_curve_df, pl.DataFrame):
                raise TypeError(f"equity_curve_df must be polars DataFrame")

            values = [Decimal(str(v)) for v in equity_curve_df['value'].to_list()]
        else:
            values = [v for _, v in self.equity_curve]

        if not values:
            return Decimal('0')

        max_drawdown = Decimal('0')
        peak = values[0]

        for value in values:
            if value > peak:
                peak = value

            if peak > Decimal('0'):
                drawdown = ((peak - value) / peak) * Decimal('100')
                max_drawdown = max(max_drawdown, drawdown)

        logger.info("Maximum drawdown calculated: %s%%", max_drawdown)

        return max_drawdown

    def calculate_max_drawdown_duration(
        self,
        equity_curve_df: Optional[pl.DataFrame] = None
    ) -> int:
        """
        Calculate maximum drawdown duration in days

        Args:
            equity_curve_df: Optional DataFrame with columns [timestamp, value]

        Returns:
            Maximum drawdown duration in days
        """
        if equity_curve_df is not None:
            if not isinstance(equity_curve_df, pl.DataFrame):
                raise TypeError(f"equity_curve_df must be polars DataFrame")

            timestamps = equity_curve_df['timestamp'].to_list()
            values = [Decimal(str(v)) for v in equity_curve_df['value'].to_list()]
        else:
            timestamps = [ts for ts, _ in self.equity_curve]
            values = [v for _, v in self.equity_curve]

        if len(values) < 2:
            return 0

        max_duration = 0
        peak_value = values[0]
        peak_timestamp = timestamps[0]
        in_drawdown = False

        for i, value in enumerate(values):
            if value > peak_value:
                # New peak reached
                if in_drawdown:
                    # Exiting drawdown
                    duration = (timestamps[i] - peak_timestamp).days
                    max_duration = max(max_duration, duration)
                    in_drawdown = False

                peak_value = value
                peak_timestamp = timestamps[i]

            elif value < peak_value:
                # In drawdown
                in_drawdown = True

        # If still in drawdown at end
        if in_drawdown:
            duration = (timestamps[-1] - peak_timestamp).days
            max_duration = max(max_duration, duration)

        logger.info("Maximum drawdown duration: %d days", max_duration)

        return max_duration

    def calculate_ulcer_index(
        self,
        equity_curve_df: Optional[pl.DataFrame] = None,
        window: Optional[int] = None
    ) -> Decimal:
        """
        Calculate Ulcer Index - measure of downside volatility

        Ulcer Index measures the depth and duration of drawdowns

        Args:
            equity_curve_df: Optional DataFrame with columns [timestamp, value]
            window: Lookback window in periods (uses config default if None)

        Returns:
            Ulcer Index value
        """
        if window is None:
            window = self.config.ulcer_index_window

        if equity_curve_df is not None:
            if not isinstance(equity_curve_df, pl.DataFrame):
                raise TypeError(f"equity_curve_df must be polars DataFrame")

            values = [Decimal(str(v)) for v in equity_curve_df['value'].to_list()]
        else:
            values = [v for _, v in self.equity_curve]

        if len(values) < window:
            logger.warning("Insufficient data for Ulcer Index calculation")
            return Decimal('0')

        # Use last N values
        values = values[-window:]

        # Calculate percentage drawdowns from running maximum
        squared_drawdowns = []
        running_max = values[0]

        for value in values:
            running_max = max(running_max, value)

            if running_max > Decimal('0'):
                drawdown_pct = ((running_max - value) / running_max) * Decimal('100')
                squared_drawdowns.append(drawdown_pct ** 2)

        if not squared_drawdowns:
            return Decimal('0')

        # Ulcer Index = sqrt(sum(drawdown^2) / N)
        mean_squared = sum(squared_drawdowns) / Decimal(str(len(squared_drawdowns)))

        # Convert to float for sqrt, then back to Decimal
        ulcer_index = Decimal(str(np.sqrt(float(mean_squared))))

        logger.info("Ulcer Index calculated: %s", ulcer_index)

        return ulcer_index

    def calculate_recovery_factor(
        self,
        total_return: Decimal,
        max_drawdown: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate recovery factor: total return / max drawdown

        Higher is better - shows return per unit of risk

        Args:
            total_return: Total return percentage
            max_drawdown: Max drawdown percentage (calculates if None)

        Returns:
            Recovery factor
        """
        if not isinstance(total_return, Decimal):
            raise TypeError(f"total_return must be Decimal")

        if max_drawdown is None:
            max_drawdown = self.calculate_max_drawdown()

        if max_drawdown <= Decimal('0'):
            return Decimal('0')

        recovery_factor = total_return / max_drawdown

        logger.info(
            "Recovery factor: %s (return=%s%%, max_dd=%s%%)",
            recovery_factor, total_return, max_drawdown
        )

        return recovery_factor

    def get_drawdown_periods(self) -> List[DrawdownEvent]:
        """
        Get all drawdown events

        Returns:
            List of drawdown events
        """
        return self.drawdown_events

    def get_active_drawdown(self) -> Optional[DrawdownEvent]:
        """
        Get current active drawdown event if any

        Returns:
            Active drawdown event or None
        """
        if self.drawdown_events and self.drawdown_events[-1].is_active:
            return self.drawdown_events[-1]
        return None

    def calculate_time_underwater(
        self,
        equity_curve_df: Optional[pl.DataFrame] = None
    ) -> Decimal:
        """
        Calculate percentage of time portfolio is underwater (in drawdown)

        Args:
            equity_curve_df: Optional DataFrame with columns [timestamp, value]

        Returns:
            Percentage of time underwater
        """
        if equity_curve_df is not None:
            if not isinstance(equity_curve_df, pl.DataFrame):
                raise TypeError(f"equity_curve_df must be polars DataFrame")

            values = [Decimal(str(v)) for v in equity_curve_df['value'].to_list()]
        else:
            values = [v for _, v in self.equity_curve]

        if len(values) < 2:
            return Decimal('0')

        underwater_periods = 0
        peak = values[0]

        for value in values:
            if value > peak:
                peak = value
            elif value < peak:
                underwater_periods += 1

        underwater_percent = (Decimal(str(underwater_periods)) / Decimal(str(len(values)))) * Decimal('100')

        logger.info(
            "Time underwater: %s%% (%d of %d periods)",
            underwater_percent, underwater_periods, len(values)
        )

        return underwater_percent

    def calculate_average_drawdown(self) -> Decimal:
        """
        Calculate average drawdown across all drawdown events

        Returns:
            Average drawdown percentage
        """
        if not self.drawdown_events:
            return Decimal('0')

        total_drawdown = sum(event.drawdown_percent for event in self.drawdown_events)
        avg_drawdown = total_drawdown / Decimal(str(len(self.drawdown_events)))

        return avg_drawdown

    def get_comprehensive_metrics(
        self,
        equity_curve_df: Optional[pl.DataFrame] = None
    ) -> DrawdownMetrics:
        """
        Get comprehensive drawdown metrics

        Args:
            equity_curve_df: Optional equity curve DataFrame

        Returns:
            Complete drawdown metrics
        """
        current_drawdown = self.calculate_current_drawdown()
        max_drawdown = self.calculate_max_drawdown(equity_curve_df)
        max_duration = self.calculate_max_drawdown_duration(equity_curve_df)
        avg_drawdown = self.calculate_average_drawdown()
        ulcer_index = self.calculate_ulcer_index(equity_curve_df)
        time_underwater = self.calculate_time_underwater(equity_curve_df)

        # Calculate recovery factor if we have positive returns
        if self.equity_curve:
            initial_value = self.equity_curve[0][1]
            current_value = self.equity_curve[-1][1]

            if initial_value > Decimal('0'):
                total_return = ((current_value - initial_value) / initial_value) * Decimal('100')
                recovery_factor = self.calculate_recovery_factor(total_return, max_drawdown)
            else:
                recovery_factor = Decimal('0')
        else:
            recovery_factor = Decimal('0')

        # Current underwater days
        active_dd = self.get_active_drawdown()
        current_underwater_days = active_dd.duration_days if active_dd else 0

        metrics = DrawdownMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=max_duration,
            average_drawdown=avg_drawdown,
            current_underwater_days=current_underwater_days,
            ulcer_index=ulcer_index,
            recovery_factor=recovery_factor,
            num_drawdown_events=len(self.drawdown_events),
            time_underwater_percent=time_underwater
        )

        return metrics

    def is_drawdown_acceptable(self, drawdown: Optional[Decimal] = None) -> bool:
        """
        Check if drawdown is within acceptable limits

        Args:
            drawdown: Drawdown to check (uses current if None)

        Returns:
            True if acceptable, False otherwise
        """
        if drawdown is None:
            drawdown = self.calculate_current_drawdown()

        acceptable = drawdown <= self.config.max_drawdown_tolerance

        if not acceptable:
            logger.error(
                "Drawdown exceeds tolerance: %s%% > %s%%",
                drawdown, self.config.max_drawdown_tolerance
            )

        return acceptable

    def reset_tracking(self) -> None:
        """Reset all tracking data"""
        self.peak_value = Decimal('0')
        self.current_value = Decimal('0')
        self.drawdown_events = []
        self.equity_curve = []

        logger.info("Drawdown tracking reset")

    def export_drawdown_history(self) -> pl.DataFrame:
        """
        Export drawdown event history as DataFrame

        Returns:
            DataFrame with all drawdown events
        """
        if not self.drawdown_events:
            return pl.DataFrame()

        data = {
            'peak_value': [float(e.peak_value) for e in self.drawdown_events],
            'trough_value': [float(e.trough_value) for e in self.drawdown_events],
            'peak_timestamp': [e.peak_timestamp for e in self.drawdown_events],
            'trough_timestamp': [e.trough_timestamp for e in self.drawdown_events],
            'recovery_timestamp': [e.recovery_timestamp for e in self.drawdown_events],
            'drawdown_percent': [float(e.drawdown_percent) for e in self.drawdown_events],
            'duration_days': [e.duration_days for e in self.drawdown_events],
            'recovery_days': [e.recovery_days for e in self.drawdown_events],
            'is_active': [e.is_active for e in self.drawdown_events]
        }

        return pl.DataFrame(data)
