"""
Cascade Failure Protection Circuit Breaker
CRITICAL: Detect and prevent cascade failures due to correlation spikes and systemic events
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class CascadeProtection:
    """Circuit breaker for cascade failure prevention"""

    def __init__(self):
        """Initialize cascade protection with config"""
        self.config = get_config()
        self.enabled = self.config.get_bool("risk", "circuit_breakers.cascade_protection_enabled", True)
        self.correlation_threshold = self.config.get_decimal(
            "risk", "circuit_breakers.correlation_spike_threshold"
        )
        self.max_correlation = self.config.get_decimal("risk", "risk_metrics.max_correlation")
        self.check_interval = self.config.get_int("risk", "monitoring.check_interval_seconds", 1)

        self.cascade_triggered = False
        self.last_check_time = None
        self.correlation_history: List[Dict] = []

        logger.info(
            f"CascadeProtection initialized: enabled={self.enabled}, "
            f"correlation_threshold={self.correlation_threshold}, "
            f"max_correlation={self.max_correlation}"
        )

    async def check_cascade_risk(
        self,
        current_positions: pl.DataFrame,
        market_data: pl.DataFrame,
        correlation_matrix: np.ndarray,
        asset_list: List[str]
    ) -> Dict[str, any]:
        """
        Check for cascade failure risk based on correlation spikes and market stress

        Args:
            current_positions: DataFrame with columns ['asset', 'position_size', 'pnl']
            market_data: DataFrame with columns ['timestamp', 'asset', 'price', 'volume']
            correlation_matrix: Current correlation matrix between assets
            asset_list: List of asset symbols corresponding to correlation matrix

        Returns:
            Dictionary with cascade risk assessment:
            {
                'cascade_risk': bool,
                'risk_level': str ('low', 'medium', 'high', 'critical'),
                'trigger_reason': str,
                'avg_correlation': Decimal,
                'max_correlation': Decimal,
                'affected_positions': List[str],
                'recommended_action': str
            }
        """
        try:
            if not self.enabled:
                return {
                    "cascade_risk": False,
                    "risk_level": "low",
                    "trigger_reason": "Cascade protection disabled",
                    "avg_correlation": Decimal("0"),
                    "max_correlation": Decimal("0"),
                    "affected_positions": [],
                    "recommended_action": "none"
                }

            self.last_check_time = datetime.now()

            # Calculate correlation metrics
            avg_correlation = self._calculate_average_correlation(correlation_matrix)
            max_corr = self._calculate_max_correlation(correlation_matrix)

            # Detect correlation surge
            correlation_surge = await self._detect_correlation_surge(avg_correlation)

            # Identify highly correlated clusters
            clusters = self._identify_correlation_clusters(
                correlation_matrix, asset_list, threshold=float(self.correlation_threshold)
            )

            # Check position concentration in clusters
            position_risk = self._assess_cluster_position_risk(current_positions, clusters)

            # Determine risk level
            risk_level, cascade_risk = self._assess_cascade_risk_level(
                avg_correlation, max_corr, correlation_surge, position_risk
            )

            # Identify affected positions
            affected_positions = []
            if cascade_risk:
                for cluster in clusters:
                    if len(cluster) >= 3:  # Cluster of 3+ highly correlated assets
                        affected_positions.extend(cluster)

            # Determine recommended action
            recommended_action = self._get_recommended_action(risk_level, cascade_risk)

            result = {
                "cascade_risk": cascade_risk,
                "risk_level": risk_level,
                "trigger_reason": self._get_trigger_reason(
                    cascade_risk, avg_correlation, max_corr, correlation_surge
                ),
                "avg_correlation": avg_correlation,
                "max_correlation": max_corr,
                "affected_positions": list(set(affected_positions)),
                "recommended_action": recommended_action,
                "correlation_surge": correlation_surge,
                "num_clusters": len(clusters),
                "cluster_position_risk": position_risk,
                "timestamp": self.last_check_time.isoformat()
            }

            if cascade_risk:
                logger.warning(
                    f"CASCADE RISK DETECTED: {risk_level} - {result['trigger_reason']} - "
                    f"Avg Correlation: {avg_correlation:.4f}, Max: {max_corr:.4f}"
                )
            else:
                logger.debug(f"Cascade check OK: risk_level={risk_level}, avg_corr={avg_correlation:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error checking cascade risk: {e}", exc_info=True)
            raise

    async def trigger_cascade_protection(
        self,
        reason: str,
        affected_positions: List[str],
        severity: str = "high"
    ) -> Dict[str, any]:
        """
        Trigger cascade protection circuit breaker

        Args:
            reason: Reason for triggering protection
            affected_positions: List of affected asset symbols
            severity: Severity level ('medium', 'high', 'critical')

        Returns:
            Dictionary with trigger details and actions taken
        """
        try:
            self.cascade_triggered = True
            trigger_time = datetime.now()

            logger.critical(
                f"CASCADE PROTECTION TRIGGERED: {severity} - {reason} - "
                f"Affected positions: {len(affected_positions)}"
            )

            # Determine actions based on severity
            actions = []

            if severity == "critical":
                actions = [
                    "HALT_ALL_TRADING",
                    "CLOSE_CORRELATED_POSITIONS",
                    "REDUCE_LEVERAGE_TO_MINIMUM",
                    "ALERT_RISK_MANAGEMENT"
                ]
            elif severity == "high":
                actions = [
                    "REDUCE_POSITION_SIZES",
                    "HALT_NEW_POSITIONS",
                    "INCREASE_HEDGE_RATIO",
                    "MONITOR_CONTINUOUSLY"
                ]
            elif severity == "medium":
                actions = [
                    "REDUCE_CORRELATED_EXPOSURE",
                    "INCREASE_MONITORING_FREQUENCY",
                    "PREPARE_HEDGES"
                ]

            result = {
                "triggered": True,
                "trigger_time": trigger_time.isoformat(),
                "reason": reason,
                "severity": severity,
                "affected_positions": affected_positions,
                "actions_taken": actions,
                "trading_halted": severity == "critical",
                "cooldown_minutes": 15 if severity == "critical" else 5
            }

            # Record trigger in history
            self.correlation_history.append({
                "timestamp": trigger_time,
                "event": "cascade_protection_triggered",
                "severity": severity,
                "reason": reason
            })

            return result

        except Exception as e:
            logger.error(f"Error triggering cascade protection: {e}", exc_info=True)
            raise

    async def monitor_correlation_surge(
        self,
        correlation_matrix: np.ndarray,
        lookback_minutes: int = 60
    ) -> Dict[str, any]:
        """
        Monitor for sudden correlation surges (potential cascade indicator)

        Args:
            correlation_matrix: Current correlation matrix
            lookback_minutes: Lookback period for comparison

        Returns:
            Dictionary with surge analysis:
            {
                'surge_detected': bool,
                'current_avg_correlation': Decimal,
                'historical_avg_correlation': Decimal,
                'surge_magnitude': Decimal,
                'surge_pct': Decimal
            }
        """
        try:
            current_avg = self._calculate_average_correlation(correlation_matrix)

            # Get historical average correlation
            cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
            historical_correlations = [
                h["avg_correlation"] for h in self.correlation_history
                if h["timestamp"] > cutoff_time and "avg_correlation" in h
            ]

            if not historical_correlations:
                # No history, record current and return no surge
                self.correlation_history.append({
                    "timestamp": datetime.now(),
                    "avg_correlation": current_avg
                })

                return {
                    "surge_detected": False,
                    "current_avg_correlation": current_avg,
                    "historical_avg_correlation": current_avg,
                    "surge_magnitude": Decimal("0"),
                    "surge_pct": Decimal("0")
                }

            historical_avg = Decimal(str(np.mean([float(c) for c in historical_correlations])))

            # Calculate surge
            surge_magnitude = current_avg - historical_avg
            surge_pct = ((current_avg / historical_avg - Decimal("1")) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if historical_avg > Decimal("0") else Decimal("0")

            # Detect surge (>20% increase or absolute increase >0.15)
            surge_detected = (
                surge_pct > Decimal("20") or
                surge_magnitude > Decimal("0.15")
            )

            # Record current correlation
            self.correlation_history.append({
                "timestamp": datetime.now(),
                "avg_correlation": current_avg
            })

            # Trim history (keep last 24 hours)
            cutoff_24h = datetime.now() - timedelta(hours=24)
            self.correlation_history = [
                h for h in self.correlation_history if h["timestamp"] > cutoff_24h
            ]

            if surge_detected:
                logger.warning(
                    f"Correlation surge detected: current={current_avg:.4f}, "
                    f"historical={historical_avg:.4f}, surge={surge_pct:.2f}%"
                )

            return {
                "surge_detected": surge_detected,
                "current_avg_correlation": current_avg,
                "historical_avg_correlation": historical_avg,
                "surge_magnitude": surge_magnitude.quantize(Decimal("0.0001")),
                "surge_pct": surge_pct
            }

        except Exception as e:
            logger.error(f"Error monitoring correlation surge: {e}", exc_info=True)
            raise

    def reset_cascade_protection(self) -> None:
        """Reset cascade protection state"""
        self.cascade_triggered = False
        logger.info("Cascade protection reset")

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

        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        max_corr = np.max(upper_triangle)

        return Decimal(str(max_corr)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    async def _detect_correlation_surge(self, current_avg: Decimal) -> bool:
        """Detect if current correlation is a surge compared to recent history"""
        if len(self.correlation_history) < 10:
            return False

        recent_correlations = [
            h["avg_correlation"] for h in self.correlation_history[-20:]
            if "avg_correlation" in h
        ]

        if not recent_correlations:
            return False

        recent_avg = Decimal(str(np.mean([float(c) for c in recent_correlations])))
        surge_threshold = recent_avg * Decimal("1.25")  # 25% increase

        return current_avg > surge_threshold

    def _identify_correlation_clusters(
        self,
        corr_matrix: np.ndarray,
        asset_list: List[str],
        threshold: float = 0.8
    ) -> List[List[str]]:
        """Identify clusters of highly correlated assets"""
        n = len(asset_list)
        clusters = []

        visited = set()

        for i in range(n):
            if i in visited:
                continue

            cluster = [asset_list[i]]
            visited.add(i)

            for j in range(i + 1, n):
                if j not in visited and abs(corr_matrix[i, j]) >= threshold:
                    cluster.append(asset_list[j])
                    visited.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _assess_cluster_position_risk(
        self,
        positions: pl.DataFrame,
        clusters: List[List[str]]
    ) -> Decimal:
        """Assess risk from position concentration in correlated clusters"""
        if positions.is_empty() or not clusters:
            return Decimal("0")

        total_exposure = positions.select("position_size").sum().item()

        if total_exposure == 0:
            return Decimal("0")

        max_cluster_exposure = Decimal("0")

        for cluster in clusters:
            cluster_positions = positions.filter(pl.col("asset").is_in(cluster))
            cluster_exposure = cluster_positions.select("position_size").sum().item()
            cluster_pct = Decimal(str(abs(cluster_exposure) / abs(total_exposure)))

            max_cluster_exposure = max(max_cluster_exposure, cluster_pct)

        return max_cluster_exposure.quantize(Decimal("0.0001"))

    def _assess_cascade_risk_level(
        self,
        avg_correlation: Decimal,
        max_correlation: Decimal,
        correlation_surge: bool,
        position_risk: Decimal
    ) -> Tuple[str, bool]:
        """Assess overall cascade risk level"""
        cascade_risk = False
        risk_level = "low"

        # Critical: very high correlation + surge + concentrated positions
        if (
            avg_correlation > self.correlation_threshold and
            correlation_surge and
            position_risk > Decimal("0.3")
        ):
            risk_level = "critical"
            cascade_risk = True

        # High: high correlation threshold exceeded
        elif avg_correlation > self.correlation_threshold:
            risk_level = "high"
            cascade_risk = True

        # Medium: approaching threshold or surge detected
        elif avg_correlation > (self.correlation_threshold * Decimal("0.9")) or correlation_surge:
            risk_level = "medium"

        return risk_level, cascade_risk

    def _get_trigger_reason(
        self,
        cascade_risk: bool,
        avg_correlation: Decimal,
        max_correlation: Decimal,
        correlation_surge: bool
    ) -> str:
        """Get human-readable trigger reason"""
        if not cascade_risk:
            return "No cascade risk detected"

        reasons = []

        if avg_correlation > self.correlation_threshold:
            reasons.append(f"Average correlation {avg_correlation:.4f} exceeds threshold {self.correlation_threshold}")

        if max_correlation > Decimal("0.95"):
            reasons.append(f"Maximum correlation {max_correlation:.4f} indicates near-perfect correlation")

        if correlation_surge:
            reasons.append("Sudden correlation surge detected")

        return " | ".join(reasons) if reasons else "Cascade risk threshold exceeded"

    def _get_recommended_action(self, risk_level: str, cascade_risk: bool) -> str:
        """Get recommended action based on risk level"""
        if risk_level == "critical":
            return "HALT_ALL_TRADING_IMMEDIATELY"
        elif risk_level == "high":
            return "REDUCE_CORRELATED_POSITIONS"
        elif risk_level == "medium":
            return "INCREASE_MONITORING"
        else:
            return "CONTINUE_NORMAL_OPERATIONS"
