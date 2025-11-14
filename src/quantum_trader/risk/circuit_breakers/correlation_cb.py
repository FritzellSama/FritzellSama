"""
Correlation Circuit Breaker
Quantum Trader AI - Production Risk Management

Monitors and responds to high correlation events:
- High correlation detection
- Systemic risk alerts
- Position clustering detection
- Automatic diversification triggers

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


class CorrelationLevel(Enum):
    """Correlation risk level"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CorrelationConfig:
    """Correlation circuit breaker configuration"""
    max_position_correlation: Decimal
    enable_correlation_hedging: bool
    max_sector_exposure_percent: Decimal
    assessment_frequency_ms: int

    # Correlation-specific thresholds
    warning_correlation: Decimal = Decimal('0.7')
    critical_correlation: Decimal = Decimal('0.85')
    cluster_size_threshold: int = 3
    systemic_risk_threshold: Decimal = Decimal('0.8')

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'CorrelationConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_position_correlation=Decimal(str(risk_config['position_limits']['max_position_correlation'])),
            enable_correlation_hedging=risk_config['position_limits']['enable_correlation_hedging'],
            max_sector_exposure_percent=Decimal(str(risk_config['position_limits']['max_sector_exposure_percent'])),
            assessment_frequency_ms=int(risk_config['global']['assessment_frequency_ms'])
        )


@dataclass
class CorrelationAlert:
    """Correlation circuit breaker alert"""
    level: CorrelationLevel
    alert_type: str
    affected_symbols: List[str]
    avg_correlation: Decimal
    max_correlation: Decimal
    cluster_count: int
    recommended_action: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class CorrelationCluster:
    """Group of highly correlated positions"""
    cluster_id: str
    symbols: List[str]
    avg_correlation: Decimal
    total_exposure: Decimal
    exposure_percent: Decimal
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class CorrelationCircuitBreaker:
    """
    Circuit breaker for correlation risk.

    Monitors portfolio for high correlation between positions and
    triggers diversification or hedging when thresholds are breached.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize correlation circuit breaker with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[CorrelationConfig] = None
        self._load_config()

        # State tracking
        self.correlation_history: List[Tuple[datetime, Decimal]] = []
        self.triggered_alerts: Set[str] = set()
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

                self.config = CorrelationConfig.from_yaml(risk_config, prod_config)
                logger.info("Correlation circuit breaker configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load correlation CB config after {max_retries} attempts") from e

    async def check_correlation_risk(
        self,
        positions: List[Position],
        correlation_matrix: Dict[Tuple[str, str], Decimal],
        portfolio_value: Decimal
    ) -> Optional[CorrelationAlert]:
        """
        Check for high correlation risk in portfolio.

        Args:
            positions: Current positions
            correlation_matrix: Correlation matrix for all symbols
            portfolio_value: Total portfolio value

        Returns:
            CorrelationAlert if risk detected, None otherwise
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # Update assessment timestamp
            self.last_assessment = timestamp

            if len(positions) < 2:
                return None

            # Calculate average correlation
            symbols = [p.symbol for p in positions]
            correlations = []

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:
                        corr = correlation_matrix.get((symbol1, symbol2), Decimal('0'))
                        correlations.append(abs(corr))

            if not correlations:
                return None

            avg_correlation = sum(correlations) / Decimal(str(len(correlations)))
            max_correlation = max(correlations)

            # Track correlation history
            self.correlation_history.append((timestamp, avg_correlation))

            # Keep only recent history (last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.correlation_history = [
                (t, c) for t, c in self.correlation_history
                if t > cutoff_time
            ]

            # Detect correlation clusters
            clusters = await self._detect_correlation_clusters(
                positions,
                correlation_matrix,
                portfolio_value
            )

            # Determine risk level
            level = await self._determine_correlation_level(
                avg_correlation,
                max_correlation,
                clusters
            )

            if level == CorrelationLevel.LOW:
                return None

            # Determine alert type
            alert_type = await self._determine_alert_type(
                avg_correlation,
                max_correlation,
                clusters
            )

            # Extract affected symbols
            affected_symbols = []
            for cluster in clusters:
                affected_symbols.extend(cluster.symbols)

            # Remove duplicates
            affected_symbols = list(set(affected_symbols))

            # Determine recommended action
            recommended_action = await self._determine_correlation_action(
                level,
                avg_correlation,
                clusters
            )

            alert = CorrelationAlert(
                level=level,
                alert_type=alert_type,
                affected_symbols=affected_symbols,
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                cluster_count=len(clusters),
                recommended_action=recommended_action,
                timestamp=timestamp,
                metadata={
                    'position_count': len(positions),
                    'portfolio_value': str(portfolio_value),
                    'clusters': [
                        {
                            'symbols': c.symbols,
                            'correlation': str(c.avg_correlation),
                            'exposure': str(c.exposure_percent)
                        }
                        for c in clusters
                    ]
                }
            )

            # Log alert
            logger.warning(
                f"CORRELATION ALERT: Level={level.value}, Type={alert_type}, "
                f"Avg={avg_correlation}, Max={max_correlation}, Clusters={len(clusters)}"
            )

            # Track alert
            alert_id = f"{alert_type}_{timestamp.isoformat()}"
            self.triggered_alerts.add(alert_id)

            return alert

        except Exception as e:
            logger.error(f"Correlation risk check failed: {e}")
            raise

    async def _detect_correlation_clusters(
        self,
        positions: List[Position],
        correlation_matrix: Dict[Tuple[str, str], Decimal],
        portfolio_value: Decimal
    ) -> List[CorrelationCluster]:
        """Detect clusters of highly correlated positions"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            symbols = [p.symbol for p in positions]
            clusters = []
            clustered_symbols = set()

            timestamp = datetime.utcnow()

            # Use simple clustering: group symbols with correlation > threshold
            for i, symbol1 in enumerate(symbols):
                if symbol1 in clustered_symbols:
                    continue

                cluster_symbols = [symbol1]
                cluster_correlations = []

                for j, symbol2 in enumerate(symbols):
                    if i != j and symbol2 not in clustered_symbols:
                        corr = abs(correlation_matrix.get((symbol1, symbol2), Decimal('0')))

                        if corr > self.config.warning_correlation:
                            cluster_symbols.append(symbol2)
                            cluster_correlations.append(corr)

                # Only create cluster if it has multiple symbols
                if len(cluster_symbols) >= self.config.cluster_size_threshold:
                    # Mark as clustered
                    for sym in cluster_symbols:
                        clustered_symbols.add(sym)

                    # Calculate cluster metrics
                    if cluster_correlations:
                        avg_correlation = sum(cluster_correlations) / Decimal(str(len(cluster_correlations)))
                    else:
                        avg_correlation = Decimal('0')

                    # Calculate total exposure
                    total_exposure = Decimal('0')
                    for position in positions:
                        if position.symbol in cluster_symbols:
                            exposure = abs(position.quantity) * position.current_price
                            total_exposure += exposure

                    exposure_percent = (total_exposure / portfolio_value * Decimal('100')) if portfolio_value > Decimal('0') else Decimal('0')

                    cluster = CorrelationCluster(
                        cluster_id=f"cluster_{len(clusters)}_{timestamp.isoformat()}",
                        symbols=cluster_symbols,
                        avg_correlation=avg_correlation,
                        total_exposure=total_exposure,
                        exposure_percent=exposure_percent,
                        timestamp=timestamp
                    )

                    clusters.append(cluster)

            logger.info(f"Detected {len(clusters)} correlation clusters")
            return clusters

        except Exception as e:
            logger.error(f"Cluster detection failed: {e}")
            raise

    async def _determine_correlation_level(
        self,
        avg_correlation: Decimal,
        max_correlation: Decimal,
        clusters: List[CorrelationCluster]
    ) -> CorrelationLevel:
        """Determine correlation risk level"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Critical: High average correlation AND large clusters
            if (
                avg_correlation > self.config.critical_correlation and
                len(clusters) >= 2
            ):
                return CorrelationLevel.CRITICAL

            # High: Average above critical OR max above systemic threshold
            if (
                avg_correlation > self.config.critical_correlation or
                max_correlation > self.config.systemic_risk_threshold
            ):
                return CorrelationLevel.HIGH

            # Moderate: Average above warning OR clusters detected
            if (
                avg_correlation > self.config.warning_correlation or
                len(clusters) >= 1
            ):
                return CorrelationLevel.MODERATE

            return CorrelationLevel.LOW

        except Exception as e:
            logger.error(f"Failed to determine correlation level: {e}")
            raise

    async def _determine_alert_type(
        self,
        avg_correlation: Decimal,
        max_correlation: Decimal,
        clusters: List[CorrelationCluster]
    ) -> str:
        """Determine type of correlation alert"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            if max_correlation > self.config.systemic_risk_threshold:
                return "systemic_risk"
            elif len(clusters) >= 2:
                return "multiple_clusters"
            elif len(clusters) == 1:
                return "position_clustering"
            elif avg_correlation > self.config.critical_correlation:
                return "high_correlation"
            else:
                return "correlation_warning"

        except Exception as e:
            logger.error(f"Failed to determine alert type: {e}")
            raise

    async def _determine_correlation_action(
        self,
        level: CorrelationLevel,
        avg_correlation: Decimal,
        clusters: List[CorrelationCluster]
    ) -> str:
        """Determine recommended action for correlation risk"""
        try:
            if level == CorrelationLevel.CRITICAL:
                return "CRITICAL: Reduce clustered positions by 50%, activate hedging, halt new correlated trades"
            elif level == CorrelationLevel.HIGH:
                return "HIGH: Reduce clustered positions by 25%, increase diversification, monitor closely"
            elif level == CorrelationLevel.MODERATE:
                return "MODERATE: Review position clustering, consider diversification, increase monitoring"
            else:
                return "No action required"

        except Exception as e:
            logger.error(f"Failed to determine correlation action: {e}")
            raise

    async def calculate_diversification_trades(
        self,
        positions: List[Position],
        clusters: List[CorrelationCluster],
        target_correlation: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate trades needed to reduce correlation risk.

        Args:
            positions: Current positions
            clusters: Detected correlation clusters
            target_correlation: Target maximum correlation

        Returns:
            Dictionary of symbol -> position reduction percent
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            reduction_trades = {}

            for cluster in clusters:
                if cluster.avg_correlation <= target_correlation:
                    continue

                # Calculate reduction needed
                correlation_excess = cluster.avg_correlation - target_correlation
                reduction_percent = min(
                    correlation_excess / cluster.avg_correlation * Decimal('100'),
                    Decimal('50')  # Max 50% reduction
                )

                # Apply reduction to cluster symbols
                for symbol in cluster.symbols:
                    reduction_trades[symbol] = reduction_percent

            logger.info(f"Calculated diversification trades for {len(reduction_trades)} symbols")
            return reduction_trades

        except Exception as e:
            logger.error(f"Failed to calculate diversification trades: {e}")
            raise

    async def should_trigger_circuit_breaker(
        self,
        alert: CorrelationAlert
    ) -> bool:
        """
        Determine if circuit breaker should trigger.

        Args:
            alert: Correlation alert

        Returns:
            True if circuit breaker should trigger
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Trigger on CRITICAL level
            if alert.level == CorrelationLevel.CRITICAL:
                logger.warning("Correlation circuit breaker TRIGGERED - CRITICAL level")
                return True

            # Trigger on HIGH level if hedging is enabled
            if alert.level == CorrelationLevel.HIGH and self.config.enable_correlation_hedging:
                logger.warning("Correlation circuit breaker TRIGGERED - HIGH level with hedging enabled")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to determine circuit breaker trigger: {e}")
            raise

    async def get_correlation_trend(self) -> Dict[str, Decimal]:
        """
        Get correlation trend over recent history.

        Returns:
            Dictionary with trend metrics
        """
        try:
            if len(self.correlation_history) < 2:
                return {
                    'current': Decimal('0'),
                    'avg_1h': Decimal('0'),
                    'avg_24h': Decimal('0'),
                    'trend': Decimal('0')
                }

            timestamp = datetime.utcnow()

            # Current (most recent)
            current = self.correlation_history[-1][1]

            # Average over last hour
            hour_ago = timestamp - timedelta(hours=1)
            recent_correlations = [
                c for t, c in self.correlation_history
                if t > hour_ago
            ]
            avg_1h = sum(recent_correlations) / Decimal(str(len(recent_correlations))) if recent_correlations else Decimal('0')

            # Average over all history (up to 24h)
            all_correlations = [c for _, c in self.correlation_history]
            avg_24h = sum(all_correlations) / Decimal(str(len(all_correlations)))

            # Trend: difference between recent and historical
            trend = current - avg_24h

            return {
                'current': current,
                'avg_1h': avg_1h,
                'avg_24h': avg_24h,
                'trend': trend
            }

        except Exception as e:
            logger.error(f"Failed to get correlation trend: {e}")
            raise

    async def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker state"""
        try:
            self.triggered_alerts.clear()
            logger.info("Correlation circuit breaker reset")

        except Exception as e:
            logger.error(f"Failed to reset circuit breaker: {e}")
            raise


async def main():
    """Example usage of correlation circuit breaker"""
    try:
        # Initialize circuit breaker
        cb = CorrelationCircuitBreaker(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample positions
        positions = [
            Position(
                symbol='AAPL',
                quantity=Decimal('100'),
                entry_price=Decimal('150'),
                current_price=Decimal('152'),
                exchange='NASDAQ',
                strategy='test',
                opened_at=datetime.utcnow(),
                position_id='pos1'
            ),
            Position(
                symbol='GOOGL',
                quantity=Decimal('50'),
                entry_price=Decimal('140'),
                current_price=Decimal('142'),
                exchange='NASDAQ',
                strategy='test',
                opened_at=datetime.utcnow(),
                position_id='pos2'
            )
        ]

        # Sample correlation matrix (high correlation)
        correlation_matrix = {
            ('AAPL', 'GOOGL'): Decimal('0.9'),
            ('GOOGL', 'AAPL'): Decimal('0.9')
        }

        portfolio_value = Decimal('1000000')

        # Check correlation risk
        alert = await cb.check_correlation_risk(
            positions,
            correlation_matrix,
            portfolio_value
        )

        if alert:
            logger.info(f"Correlation alert: {alert}")

            # Check if circuit breaker should trigger
            should_trigger = await cb.should_trigger_circuit_breaker(alert)
            logger.info(f"Circuit breaker trigger: {should_trigger}")

    except Exception as e:
        logger.error(f"Correlation circuit breaker example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
