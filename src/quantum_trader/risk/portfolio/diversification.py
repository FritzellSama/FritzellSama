"""
Portfolio Diversification Metrics
Quantum Trader AI - Production Risk Management

Calculate portfolio diversification and concentration metrics:
- Herfindahl index calculation
- Concentration risk metrics
- Effective N calculation
- Diversification ratio

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import polars as pl
import yaml

from quantum_trader.models import Position, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class DiversificationConfig:
    """Diversification metrics configuration"""
    max_single_ticker_percent: Decimal
    max_top_5_percent: Decimal
    max_sector_exposure_percent: Decimal
    rebalance_threshold_percent: Decimal

    # Diversification-specific thresholds
    min_effective_n: int = 5
    max_herfindahl_index: Decimal = Decimal('0.25')

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'DiversificationConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_single_ticker_percent=Decimal(str(risk_config['concentration_limits']['max_single_ticker_percent'])),
            max_top_5_percent=Decimal(str(risk_config['concentration_limits']['max_top_5_percent'])),
            max_sector_exposure_percent=Decimal(str(risk_config['position_limits']['max_sector_exposure_percent'])),
            rebalance_threshold_percent=Decimal(str(risk_config['concentration_limits']['rebalance_threshold_percent']))
        )


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics"""
    herfindahl_index: Decimal
    effective_n: Decimal
    diversification_ratio: Decimal
    concentration_top_5: Decimal
    concentration_top_10: Decimal
    max_position_weight: Decimal
    num_positions: int
    sector_concentration: Dict[str, Decimal]
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class ConcentrationAlert:
    """Concentration risk alert"""
    alert_type: str  # 'single_position', 'top_holdings', 'sector', 'low_diversification'
    severity: str  # 'warning', 'critical'
    metric_value: Decimal
    threshold_value: Decimal
    affected_symbols: List[str]
    recommended_action: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class PortfolioDiversification:
    """
    Calculate portfolio diversification and concentration metrics.

    Provides comprehensive analysis of portfolio concentration risk
    using multiple metrics including Herfindahl index and effective N.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize diversification analyzer with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[DiversificationConfig] = None
        self._load_config()

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

                self.config = DiversificationConfig.from_yaml(risk_config, prod_config)
                logger.info("Diversification configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load diversification config after {max_retries} attempts") from e

    async def calculate_diversification_metrics(
        self,
        positions: List[Position],
        portfolio_value: Decimal,
        sector_map: Optional[Dict[str, str]] = None,
        volatilities: Optional[Dict[str, Decimal]] = None
    ) -> DiversificationMetrics:
        """
        Calculate comprehensive diversification metrics.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            sector_map: Optional mapping of symbol -> sector
            volatilities: Optional volatility for each symbol

        Returns:
            DiversificationMetrics with all calculated metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            if len(positions) == 0:
                return DiversificationMetrics(
                    herfindahl_index=Decimal('0'),
                    effective_n=Decimal('0'),
                    diversification_ratio=Decimal('0'),
                    concentration_top_5=Decimal('0'),
                    concentration_top_10=Decimal('0'),
                    max_position_weight=Decimal('0'),
                    num_positions=0,
                    sector_concentration={},
                    timestamp=timestamp
                )

            # Calculate position weights
            weights = await self._calculate_position_weights(positions, portfolio_value)

            # Herfindahl index (sum of squared weights)
            herfindahl_index = await self._calculate_herfindahl_index(weights)

            # Effective N (1 / Herfindahl index)
            effective_n = await self._calculate_effective_n(herfindahl_index)

            # Diversification ratio
            if volatilities:
                diversification_ratio = await self._calculate_diversification_ratio(
                    weights,
                    volatilities
                )
            else:
                diversification_ratio = Decimal('0')

            # Top holdings concentration
            concentration_top_5 = await self._calculate_top_n_concentration(weights, 5)
            concentration_top_10 = await self._calculate_top_n_concentration(weights, 10)

            # Maximum position weight
            max_position_weight = max(weights.values()) if weights else Decimal('0')

            # Sector concentration
            sector_concentration = {}
            if sector_map:
                sector_concentration = await self._calculate_sector_concentration(
                    positions,
                    portfolio_value,
                    sector_map
                )

            metrics = DiversificationMetrics(
                herfindahl_index=herfindahl_index,
                effective_n=effective_n,
                diversification_ratio=diversification_ratio,
                concentration_top_5=concentration_top_5,
                concentration_top_10=concentration_top_10,
                max_position_weight=max_position_weight,
                num_positions=len(positions),
                sector_concentration=sector_concentration,
                timestamp=timestamp,
                metadata={
                    'portfolio_value': str(portfolio_value)
                }
            )

            logger.info(
                f"Diversification metrics: HHI={herfindahl_index}, "
                f"Effective N={effective_n}, Top5={concentration_top_5}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Diversification metrics calculation failed: {e}")
            raise

    async def check_concentration_alerts(
        self,
        metrics: DiversificationMetrics,
        positions: List[Position],
        portfolio_value: Decimal
    ) -> List[ConcentrationAlert]:
        """
        Check for concentration risk violations.

        Args:
            metrics: Calculated diversification metrics
            positions: Current positions
            portfolio_value: Total portfolio value

        Returns:
            List of concentration alerts
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            alerts = []
            timestamp = datetime.utcnow()

            # Check single position concentration
            if metrics.max_position_weight > self.config.max_single_ticker_percent / Decimal('100'):
                # Find the concentrated position
                weights = await self._calculate_position_weights(positions, portfolio_value)
                max_symbol = max(weights.items(), key=lambda x: x[1])[0]

                alerts.append(ConcentrationAlert(
                    alert_type='single_position',
                    severity='critical' if metrics.max_position_weight > self.config.max_single_ticker_percent / Decimal('50') else 'warning',
                    metric_value=metrics.max_position_weight * Decimal('100'),
                    threshold_value=self.config.max_single_ticker_percent,
                    affected_symbols=[max_symbol],
                    recommended_action=f"Reduce {max_symbol} position to below {self.config.max_single_ticker_percent}% of portfolio",
                    timestamp=timestamp
                ))

            # Check top 5 concentration
            if metrics.concentration_top_5 > self.config.max_top_5_percent / Decimal('100'):
                weights = await self._calculate_position_weights(positions, portfolio_value)
                top_5_symbols = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
                top_5_symbols = [s for s, _ in top_5_symbols]

                alerts.append(ConcentrationAlert(
                    alert_type='top_holdings',
                    severity='warning',
                    metric_value=metrics.concentration_top_5 * Decimal('100'),
                    threshold_value=self.config.max_top_5_percent,
                    affected_symbols=top_5_symbols,
                    recommended_action=f"Diversify top holdings to below {self.config.max_top_5_percent}% combined",
                    timestamp=timestamp
                ))

            # Check sector concentration
            for sector, concentration in metrics.sector_concentration.items():
                if concentration > self.config.max_sector_exposure_percent / Decimal('100'):
                    sector_symbols = [
                        p.symbol for p in positions
                        if getattr(p, 'sector', None) == sector
                    ]

                    alerts.append(ConcentrationAlert(
                        alert_type='sector',
                        severity='warning',
                        metric_value=concentration * Decimal('100'),
                        threshold_value=self.config.max_sector_exposure_percent,
                        affected_symbols=sector_symbols,
                        recommended_action=f"Reduce {sector} sector exposure to below {self.config.max_sector_exposure_percent}%",
                        timestamp=timestamp,
                        metadata={'sector': sector}
                    ))

            # Check low diversification
            if metrics.effective_n < Decimal(str(self.config.min_effective_n)):
                alerts.append(ConcentrationAlert(
                    alert_type='low_diversification',
                    severity='critical',
                    metric_value=metrics.effective_n,
                    threshold_value=Decimal(str(self.config.min_effective_n)),
                    affected_symbols=[],
                    recommended_action=f"Increase diversification to at least {self.config.min_effective_n} effective positions",
                    timestamp=timestamp
                ))

            # Check Herfindahl index
            if metrics.herfindahl_index > self.config.max_herfindahl_index:
                alerts.append(ConcentrationAlert(
                    alert_type='high_herfindahl',
                    severity='warning',
                    metric_value=metrics.herfindahl_index,
                    threshold_value=self.config.max_herfindahl_index,
                    affected_symbols=[],
                    recommended_action=f"Reduce portfolio concentration (HHI should be below {self.config.max_herfindahl_index})",
                    timestamp=timestamp
                ))

            if alerts:
                logger.warning(f"Detected {len(alerts)} concentration alerts")

            return alerts

        except Exception as e:
            logger.error(f"Concentration alert check failed: {e}")
            raise

    async def _calculate_position_weights(
        self,
        positions: List[Position],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate position weights as fraction of portfolio"""
        try:
            if portfolio_value == Decimal('0'):
                return {}

            weights = {}
            for position in positions:
                value = abs(position.quantity) * position.current_price
                weight = value / portfolio_value
                weights[position.symbol] = weight

            return weights

        except Exception as e:
            logger.error(f"Position weight calculation failed: {e}")
            raise

    async def _calculate_herfindahl_index(
        self,
        weights: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate Herfindahl-Hirschman Index (HHI).

        HHI = sum of squared weights
        Range: [1/N, 1] where N is number of positions
        Lower values indicate better diversification
        """
        try:
            if not weights:
                return Decimal('0')

            hhi = sum(w ** 2 for w in weights.values())
            return hhi

        except Exception as e:
            logger.error(f"Herfindahl index calculation failed: {e}")
            raise

    async def _calculate_effective_n(
        self,
        herfindahl_index: Decimal
    ) -> Decimal:
        """
        Calculate effective number of positions.

        Effective N = 1 / HHI
        Represents the number of equally-weighted positions that would
        give the same concentration as the actual portfolio
        """
        try:
            if herfindahl_index == Decimal('0'):
                return Decimal('0')

            effective_n = Decimal('1') / herfindahl_index
            return effective_n

        except Exception as e:
            logger.error(f"Effective N calculation failed: {e}")
            raise

    async def _calculate_diversification_ratio(
        self,
        weights: Dict[str, Decimal],
        volatilities: Dict[str, Decimal]
    ) -> Decimal:
        """
        Calculate diversification ratio.

        DR = (weighted average of individual volatilities) / (portfolio volatility)
        Higher values indicate better diversification
        """
        try:
            if not weights or not volatilities:
                return Decimal('0')

            # Weighted average of individual volatilities
            weighted_vol_sum = Decimal('0')
            for symbol, weight in weights.items():
                vol = volatilities.get(symbol, Decimal('0'))
                weighted_vol_sum += weight * vol

            # Simplified portfolio volatility (assumes independence)
            # Full implementation would use covariance matrix
            portfolio_vol = Decimal('0')
            for symbol, weight in weights.items():
                vol = volatilities.get(symbol, Decimal('0'))
                portfolio_vol += (weight * vol) ** 2

            portfolio_vol = portfolio_vol.sqrt()

            if portfolio_vol == Decimal('0'):
                return Decimal('0')

            diversification_ratio = weighted_vol_sum / portfolio_vol
            return diversification_ratio

        except Exception as e:
            logger.error(f"Diversification ratio calculation failed: {e}")
            raise

    async def _calculate_top_n_concentration(
        self,
        weights: Dict[str, Decimal],
        n: int
    ) -> Decimal:
        """Calculate concentration in top N holdings"""
        try:
            if not weights:
                return Decimal('0')

            # Sort by weight descending
            sorted_weights = sorted(weights.values(), reverse=True)

            # Sum top N
            top_n_sum = sum(sorted_weights[:n])

            return top_n_sum

        except Exception as e:
            logger.error(f"Top N concentration calculation failed: {e}")
            raise

    async def _calculate_sector_concentration(
        self,
        positions: List[Position],
        portfolio_value: Decimal,
        sector_map: Dict[str, str]
    ) -> Dict[str, Decimal]:
        """Calculate concentration by sector"""
        try:
            if portfolio_value == Decimal('0'):
                return {}

            sector_exposure = {}

            for position in positions:
                sector = sector_map.get(position.symbol, 'Unknown')
                value = abs(position.quantity) * position.current_price

                if sector not in sector_exposure:
                    sector_exposure[sector] = Decimal('0')

                sector_exposure[sector] += value

            # Convert to percentages
            sector_concentration = {
                sector: exposure / portfolio_value
                for sector, exposure in sector_exposure.items()
            }

            return sector_concentration

        except Exception as e:
            logger.error(f"Sector concentration calculation failed: {e}")
            raise

    async def calculate_rebalancing_needs(
        self,
        positions: List[Position],
        portfolio_value: Decimal,
        target_weights: Optional[Dict[str, Decimal]] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate rebalancing trades needed to improve diversification.

        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            target_weights: Optional target weights (default: equal weight)

        Returns:
            Dictionary of symbol -> weight adjustment needed
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Calculate current weights
            current_weights = await self._calculate_position_weights(positions, portfolio_value)

            # Default to equal weight if no target specified
            if target_weights is None:
                n = len(positions)
                if n == 0:
                    return {}
                equal_weight = Decimal('1') / Decimal(str(n))
                target_weights = {p.symbol: equal_weight for p in positions}

            # Calculate adjustments
            adjustments = {}
            for symbol in current_weights:
                current = current_weights[symbol]
                target = target_weights.get(symbol, Decimal('0'))
                adjustment = target - current

                # Only include if adjustment is significant
                if abs(adjustment) > self.config.rebalance_threshold_percent / Decimal('100'):
                    adjustments[symbol] = adjustment

            logger.info(f"Calculated rebalancing needs: {len(adjustments)} positions require adjustment")
            return adjustments

        except Exception as e:
            logger.error(f"Rebalancing needs calculation failed: {e}")
            raise


async def main():
    """Example usage of portfolio diversification"""
    try:
        # Initialize analyzer
        analyzer = PortfolioDiversification(
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
            ),
            Position(
                symbol='MSFT',
                quantity=Decimal('80'),
                entry_price=Decimal('350'),
                current_price=Decimal('355'),
                exchange='NASDAQ',
                strategy='test',
                opened_at=datetime.utcnow(),
                position_id='pos3'
            )
        ]

        portfolio_value = Decimal('50000')

        # Calculate diversification metrics
        metrics = await analyzer.calculate_diversification_metrics(
            positions,
            portfolio_value
        )

        logger.info(f"Diversification metrics: {metrics}")

        # Check for alerts
        alerts = await analyzer.check_concentration_alerts(
            metrics,
            positions,
            portfolio_value
        )

        if alerts:
            logger.warning(f"Concentration alerts: {len(alerts)}")
            for alert in alerts:
                logger.warning(f"  - {alert.alert_type}: {alert.recommended_action}")

    except Exception as e:
        logger.error(f"Portfolio diversification example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
