"""
Correlation Spike Circuit Breaker
CRITICAL: Monitor and halt trading on sudden correlation spikes between assets
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class CorrelationCircuitBreaker:
    """Circuit breaker for correlation spike detection and response"""

    def __init__(self):
        """Initialize correlation circuit breaker with config"""
        self.config = get_config()
        self.enabled = self.config.get_bool("risk", "circuit_breakers.enabled", True)
        self.correlation_threshold = self.config.get_decimal(
            "risk", "circuit_breakers.correlation_spike_threshold"
        )
        self.max_correlation = self.config.get_decimal("risk", "risk_metrics.max_correlation")
        self.check_interval = self.config.get_int("risk", "monitoring.check_interval_seconds", 1)
        self.alert_cooldown = self.config.get_int("risk", "monitoring.alert_cooldown_seconds", 300)

        self.breaker_triggered = False
        self.last_trigger_time = None
        self.correlation_alerts: List[Dict] = []
        self.correlation_baseline: Optional[Decimal] = None

        logger.info(
            f"CorrelationCircuitBreaker initialized: enabled={self.enabled}, "
            f"threshold={self.correlation_threshold}, max_correlation={self.max_correlation}"
        )

    async def check_correlation_spike(
        self,
        correlation_matrix: np.ndarray,
        asset_list: List[str],
        historical_correlation: Optional[Decimal] = None
    ) -> Dict[str, any]:
        """
        Check for correlation spikes that may trigger circuit breaker

        Args:
            correlation_matrix: Current correlation matrix (n x n)
            asset_list: List of asset symbols corresponding to matrix rows/columns
            historical_correlation: Optional baseline average correlation

        Returns:
            Dictionary with spike detection results:
            {
                'spike_detected': bool,
                'should_trigger': bool,
                'current_avg_correlation': Decimal,
                'max_correlation': Decimal,
                'spike_magnitude': Decimal,
                'affected_pairs': List[Tuple[str, str, Decimal]],
                'recommendation': str
            }
        """
        try:
            if not self.enabled:
                return {
                    "spike_detected": False,
                    "should_trigger": False,
                    "current_avg_correlation": Decimal("0"),
                    "max_correlation": Decimal("0"),
                    "spike_magnitude": Decimal("0"),
                    "affected_pairs": [],
                    "recommendation": "Circuit breaker disabled"
                }

            # Calculate current correlation metrics
            current_avg = self._calculate_average_correlation(correlation_matrix)
            max_corr = self._calculate_max_correlation(correlation_matrix)

            # Use provided historical baseline or stored baseline
            if historical_correlation is None:
                if self.correlation_baseline is None:
                    # First check, establish baseline
                    self.correlation_baseline = current_avg
                    historical_correlation = current_avg
                else:
                    historical_correlation = self.correlation_baseline

            # Calculate spike magnitude
            spike_magnitude = current_avg - historical_correlation

            # Detect spike (relative or absolute threshold)
            relative_spike = (current_avg / historical_correlation) if historical_correlation > 0 else Decimal("1")
            spike_detected = (
                current_avg > self.correlation_threshold or
                relative_spike > Decimal("1.3") or  # 30% increase
                spike_magnitude > Decimal("0.15")  # Absolute increase
            )

            # Identify highly correlated pairs
            affected_pairs = self._identify_high_correlation_pairs(
                correlation_matrix, asset_list, threshold=float(self.correlation_threshold)
            )

            # Determine if circuit breaker should trigger
            should_trigger = spike_detected and not self._is_in_cooldown()

            # Get recommendation
            recommendation = self._get_recommendation(
                spike_detected, should_trigger, current_avg, max_corr
            )

            result = {
                "spike_detected": spike_detected,
                "should_trigger": should_trigger,
                "current_avg_correlation": current_avg,
                "max_correlation": max_corr,
                "historical_correlation": historical_correlation,
                "spike_magnitude": spike_magnitude.quantize(Decimal("0.0001")),
                "relative_spike": relative_spike.quantize(Decimal("0.01")),
                "affected_pairs": affected_pairs[:10],  # Top 10 most correlated pairs
                "num_high_correlations": len(affected_pairs),
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat()
            }

            if spike_detected:
                logger.warning(
                    f"Correlation spike detected: current={current_avg:.4f}, "
                    f"historical={historical_correlation:.4f}, "
                    f"spike={spike_magnitude:.4f}, pairs={len(affected_pairs)}"
                )

                # Record alert
                self.correlation_alerts.append({
                    "timestamp": datetime.now(),
                    "current_correlation": current_avg,
                    "spike_magnitude": spike_magnitude,
                    "triggered": should_trigger
                })

            return result

        except Exception as e:
            logger.error(f"Error checking correlation spike: {e}", exc_info=True)
            raise

    async def trigger_on_correlation(
        self,
        correlation_spike_result: Dict[str, any],
        severity: str = "high"
    ) -> Dict[str, any]:
        """
        Trigger circuit breaker due to correlation spike

        Args:
            correlation_spike_result: Result from check_correlation_spike()
            severity: Severity level ('medium', 'high', 'critical')

        Returns:
            Dictionary with trigger details and actions
        """
        try:
            self.breaker_triggered = True
            self.last_trigger_time = datetime.now()

            current_corr = correlation_spike_result.get("current_avg_correlation", Decimal("0"))
            affected_pairs = correlation_spike_result.get("affected_pairs", [])

            logger.critical(
                f"CORRELATION CIRCUIT BREAKER TRIGGERED: {severity} - "
                f"Correlation: {current_corr:.4f}, Affected pairs: {len(affected_pairs)}"
            )

            # Determine actions based on severity
            actions = []

            if severity == "critical" or current_corr > Decimal("0.95"):
                actions = [
                    "HALT_ALL_TRADING_IMMEDIATELY",
                    "LIQUIDATE_CORRELATED_POSITIONS",
                    "ACTIVATE_HEDGES",
                    "NOTIFY_RISK_MANAGEMENT",
                    "WAIT_FOR_MANUAL_REVIEW"
                ]
                cooldown_minutes = 30

            elif severity == "high":
                actions = [
                    "HALT_NEW_POSITIONS",
                    "REDUCE_CORRELATED_EXPOSURE",
                    "TIGHTEN_STOP_LOSSES",
                    "INCREASE_MONITORING_FREQUENCY",
                    "ALERT_RISK_TEAM"
                ]
                cooldown_minutes = 15

            else:  # medium
                actions = [
                    "REDUCE_POSITION_SIZES",
                    "AVOID_CORRELATED_ASSETS",
                    "INCREASE_DIVERSIFICATION",
                    "MONITOR_CLOSELY"
                ]
                cooldown_minutes = 5

            result = {
                "triggered": True,
                "trigger_time": self.last_trigger_time.isoformat(),
                "reason": "Correlation spike detected",
                "severity": severity,
                "current_correlation": current_corr,
                "threshold": self.correlation_threshold,
                "affected_pairs": len(affected_pairs),
                "actions_taken": actions,
                "trading_halted": severity == "critical",
                "cooldown_minutes": cooldown_minutes,
                "resume_time": (self.last_trigger_time + timedelta(minutes=cooldown_minutes)).isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error triggering correlation circuit breaker: {e}", exc_info=True)
            raise

    async def monitor_correlations(
        self,
        correlation_matrices: List[np.ndarray],
        timestamps: List[datetime],
        asset_list: List[str],
        lookback_periods: int = 20
    ) -> Dict[str, any]:
        """
        Monitor correlation trends over time

        Args:
            correlation_matrices: List of historical correlation matrices
            timestamps: Corresponding timestamps for each matrix
            asset_list: List of asset symbols
            lookback_periods: Number of periods to analyze

        Returns:
            Dictionary with correlation monitoring results:
            {
                'trend': str ('increasing', 'decreasing', 'stable'),
                'current_correlation': Decimal,
                'ma_short': Decimal (moving average),
                'ma_long': Decimal,
                'volatility': Decimal,
                'warning_level': str
            }
        """
        try:
            if len(correlation_matrices) < 2:
                logger.warning("Insufficient correlation history for monitoring")
                return {
                    "trend": "unknown",
                    "current_correlation": Decimal("0"),
                    "ma_short": Decimal("0"),
                    "ma_long": Decimal("0"),
                    "volatility": Decimal("0"),
                    "warning_level": "none"
                }

            # Calculate average correlation for each period
            avg_correlations = [
                self._calculate_average_correlation(matrix)
                for matrix in correlation_matrices[-lookback_periods:]
            ]

            # Current correlation
            current_corr = avg_correlations[-1]

            # Moving averages
            ma_short_periods = min(5, len(avg_correlations))
            ma_long_periods = min(10, len(avg_correlations))

            ma_short = Decimal(str(np.mean([float(c) for c in avg_correlations[-ma_short_periods:]]))).quantize(
                Decimal("0.0001")
            )
            ma_long = Decimal(str(np.mean([float(c) for c in avg_correlations[-ma_long_periods:]]))).quantize(
                Decimal("0.0001")
            )

            # Volatility of correlations
            correlation_std = Decimal(str(np.std([float(c) for c in avg_correlations], ddof=1))).quantize(
                Decimal("0.0001")
            )

            # Determine trend
            if ma_short > ma_long * Decimal("1.05"):
                trend = "increasing"
            elif ma_short < ma_long * Decimal("0.95"):
                trend = "decreasing"
            else:
                trend = "stable"

            # Determine warning level
            warning_level = self._assess_warning_level(current_corr, ma_long, correlation_std)

            result = {
                "trend": trend,
                "current_correlation": current_corr,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "volatility": correlation_std,
                "warning_level": warning_level,
                "lookback_periods": len(avg_correlations),
                "correlation_history": [float(c) for c in avg_correlations],
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Correlation monitoring: trend={trend}, current={current_corr:.4f}, "
                f"MA(5)={ma_short:.4f}, MA(10)={ma_long:.4f}, warning={warning_level}"
            )

            return result

        except Exception as e:
            logger.error(f"Error monitoring correlations: {e}", exc_info=True)
            raise

    def reset_breaker(self) -> None:
        """Reset circuit breaker state"""
        self.breaker_triggered = False
        self.last_trigger_time = None
        logger.info("Correlation circuit breaker reset")

    def update_baseline(self, new_baseline: Decimal) -> None:
        """Update correlation baseline"""
        old_baseline = self.correlation_baseline
        self.correlation_baseline = new_baseline
        logger.info(f"Correlation baseline updated: {old_baseline} -> {new_baseline}")

    def _calculate_average_correlation(self, corr_matrix: np.ndarray) -> Decimal:
        """Calculate average absolute correlation (excluding diagonal)"""
        n = corr_matrix.shape[0]
        if n <= 1:
            return Decimal("0")

        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        num_elements = (n * (n - 1)) // 2

        if num_elements == 0:
            return Decimal("0")

        avg_corr = np.sum(upper_triangle) / num_elements

        return Decimal(str(avg_corr)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _calculate_max_correlation(self, corr_matrix: np.ndarray) -> Decimal:
        """Calculate maximum absolute correlation (excluding diagonal)"""
        n = corr_matrix.shape[0]
        if n <= 1:
            return Decimal("0")

        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        max_corr = np.max(upper_triangle)

        return Decimal(str(max_corr)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def _identify_high_correlation_pairs(
        self,
        corr_matrix: np.ndarray,
        asset_list: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, str, Decimal]]:
        """Identify pairs of assets with high correlation"""
        n = len(asset_list)
        high_corr_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                corr_value = abs(corr_matrix[i, j])
                if corr_value >= threshold:
                    high_corr_pairs.append((
                        asset_list[i],
                        asset_list[j],
                        Decimal(str(corr_value)).quantize(Decimal("0.0001"))
                    ))

        # Sort by correlation (descending)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        return high_corr_pairs

    def _is_in_cooldown(self) -> bool:
        """Check if circuit breaker is in cooldown period"""
        if self.last_trigger_time is None:
            return False

        cooldown_delta = timedelta(seconds=self.alert_cooldown)
        return datetime.now() < (self.last_trigger_time + cooldown_delta)

    def _get_recommendation(
        self,
        spike_detected: bool,
        should_trigger: bool,
        current_corr: Decimal,
        max_corr: Decimal
    ) -> str:
        """Get recommendation based on correlation state"""
        if not spike_detected:
            return "Continue normal operations"

        if should_trigger:
            if current_corr > Decimal("0.95"):
                return "HALT ALL TRADING - Critical correlation spike"
            elif current_corr > Decimal("0.85"):
                return "HALT NEW POSITIONS - High correlation spike"
            else:
                return "REDUCE CORRELATED EXPOSURE - Moderate correlation spike"
        else:
            return "Monitor closely - Cooldown period active"

    def _assess_warning_level(
        self,
        current_corr: Decimal,
        ma_long: Decimal,
        volatility: Decimal
    ) -> str:
        """Assess warning level based on correlation metrics"""
        if current_corr > self.correlation_threshold:
            return "critical"
        elif current_corr > self.correlation_threshold * Decimal("0.95"):
            return "high"
        elif current_corr > self.max_correlation:
            return "medium"
        elif current_corr > ma_long * Decimal("1.2") or volatility > Decimal("0.1"):
            return "low"
        else:
            return "none"
