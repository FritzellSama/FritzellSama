"""
Cascade Failure Protection Circuit Breaker
Quantum Trader AI - Production Risk Management

Detects and prevents cascade failures in the portfolio:
- Market cascade detection
- Correlation breakdown monitoring
- Liquidity evaporation detection
- Automatic position reduction
- Emergency hedging activation

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


class CascadeLevel(Enum):
    """Cascade failure severity level"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class CascadeConfig:
    """Cascade protection configuration"""
    max_daily_loss_percent: Decimal
    max_intraday_loss_percent: Decimal
    max_position_correlation: Decimal
    min_daily_volume_shares: int
    max_acceptable_spread_percent: Decimal
    volatility_vix_threshold: Decimal
    stress_indicator_threshold: Decimal
    assessment_frequency_ms: int

    # Cascade-specific thresholds
    cascade_correlation_threshold: Decimal = Decimal('0.85')
    cascade_loss_velocity_threshold: Decimal = Decimal('0.02')  # 2% loss in 5 minutes
    liquidity_drop_threshold: Decimal = Decimal('0.5')  # 50% volume drop
    emergency_hedge_threshold: Decimal = Decimal('0.05')  # 5% portfolio loss

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'CascadeConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_daily_loss_percent=Decimal(str(risk_config['global']['max_daily_loss_percent'])),
            max_intraday_loss_percent=Decimal(str(risk_config['global']['max_intraday_loss_percent'])),
            max_position_correlation=Decimal(str(risk_config['position_limits']['max_position_correlation'])),
            min_daily_volume_shares=int(risk_config['liquidity']['min_daily_volume_shares']),
            max_acceptable_spread_percent=Decimal(str(risk_config['liquidity']['max_acceptable_spread_percent'])),
            volatility_vix_threshold=Decimal(str(risk_config['market_conditions']['volatility_vix_threshold'])),
            stress_indicator_threshold=Decimal(str(risk_config['market_conditions']['stress_indicator_threshold'])),
            assessment_frequency_ms=int(risk_config['global']['assessment_frequency_ms'])
        )


@dataclass
class CascadeEvent:
    """Cascade failure event detection"""
    level: CascadeLevel
    event_type: str
    affected_symbols: List[str]
    correlation_spike: Decimal
    liquidity_drop: Decimal
    loss_velocity: Decimal
    recommended_action: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


@dataclass
class ProtectionAction:
    """Protection action to take"""
    action_type: str  # 'reduce_positions', 'emergency_hedge', 'halt_trading'
    symbols: List[str]
    reduction_percent: Decimal
    hedge_required: bool
    priority: str  # 'high', 'critical', 'emergency'
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class CascadeProtection:
    """
    Cascade failure protection system.

    Monitors for systemic risk events where correlated positions
    experience simultaneous losses due to market structure breakdown.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize cascade protection with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[CascadeConfig] = None
        self._load_config()

        # State tracking
        self.recent_losses: List[Tuple[datetime, Decimal]] = []
        self.liquidity_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.correlation_history: List[Tuple[datetime, Decimal]] = []
        self.active_cascade_events: Set[str] = set()

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

                self.config = CascadeConfig.from_yaml(risk_config, prod_config)
                logger.info("Cascade protection configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load cascade config after {max_retries} attempts") from e

    async def detect_cascade(
        self,
        positions: List[Position],
        correlation_matrix: Dict[Tuple[str, str], Decimal],
        portfolio_value: Decimal,
        market_data: pl.DataFrame
    ) -> Optional[CascadeEvent]:
        """
        Detect cascade failure conditions in the portfolio.

        Args:
            positions: Current positions
            correlation_matrix: Symbol correlation matrix
            portfolio_value: Total portfolio value
            market_data: Recent market data [symbol, timestamp, volume, spread]

        Returns:
            CascadeEvent if cascade detected, None otherwise
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # 1. Check for correlation spikes
            correlation_spike = await self._detect_correlation_spike(
                positions,
                correlation_matrix
            )

            # 2. Check for liquidity evaporation
            liquidity_drop = await self._detect_liquidity_evaporation(
                positions,
                market_data
            )

            # 3. Check for loss velocity (rapid losses)
            loss_velocity = await self._calculate_loss_velocity(
                positions,
                portfolio_value
            )

            # 4. Determine cascade level
            level, event_type = await self._determine_cascade_level(
                correlation_spike,
                liquidity_drop,
                loss_velocity
            )

            if level == CascadeLevel.NORMAL:
                return None

            # Extract affected symbols
            affected_symbols = [p.symbol for p in positions]

            # Determine recommended action
            recommended_action = await self._determine_cascade_action(
                level,
                correlation_spike,
                liquidity_drop,
                loss_velocity
            )

            cascade_event = CascadeEvent(
                level=level,
                event_type=event_type,
                affected_symbols=affected_symbols,
                correlation_spike=correlation_spike,
                liquidity_drop=liquidity_drop,
                loss_velocity=loss_velocity,
                recommended_action=recommended_action,
                timestamp=timestamp,
                metadata={
                    'position_count': len(positions),
                    'portfolio_value': str(portfolio_value)
                }
            )

            # Log event
            logger.warning(
                f"CASCADE DETECTED: Level={level.value}, Type={event_type}, "
                f"Correlation={correlation_spike}, Liquidity={liquidity_drop}, "
                f"Velocity={loss_velocity}"
            )

            # Track active event
            event_id = f"{event_type}_{timestamp.isoformat()}"
            self.active_cascade_events.add(event_id)

            return cascade_event

        except Exception as e:
            logger.error(f"Cascade detection failed: {e}")
            raise

    async def calculate_protection_actions(
        self,
        cascade_event: CascadeEvent,
        positions: List[Position],
        portfolio_value: Decimal
    ) -> List[ProtectionAction]:
        """
        Calculate protection actions based on cascade event.

        Args:
            cascade_event: Detected cascade event
            positions: Current positions
            portfolio_value: Total portfolio value

        Returns:
            List of protection actions to execute
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            actions = []
            timestamp = datetime.utcnow()

            if cascade_event.level == CascadeLevel.WARNING:
                # Reduce high-correlation positions
                high_corr_symbols = await self._identify_high_correlation_positions(
                    positions,
                    cascade_event.affected_symbols
                )

                if high_corr_symbols:
                    actions.append(ProtectionAction(
                        action_type='reduce_positions',
                        symbols=high_corr_symbols,
                        reduction_percent=Decimal('25'),
                        hedge_required=False,
                        priority='high',
                        timestamp=timestamp,
                        metadata={'reason': 'correlation_warning'}
                    ))

            elif cascade_event.level == CascadeLevel.CRITICAL:
                # Reduce all positions by 50% and consider hedging
                actions.append(ProtectionAction(
                    action_type='reduce_positions',
                    symbols=cascade_event.affected_symbols,
                    reduction_percent=Decimal('50'),
                    hedge_required=True,
                    priority='critical',
                    timestamp=timestamp,
                    metadata={'reason': 'cascade_critical'}
                ))

                # Activate emergency hedging
                actions.append(ProtectionAction(
                    action_type='emergency_hedge',
                    symbols=['SPY', 'QQQ'],  # Use index ETFs for hedging
                    reduction_percent=Decimal('0'),
                    hedge_required=True,
                    priority='critical',
                    timestamp=timestamp,
                    metadata={
                        'hedge_ratio': str(Decimal('0.5')),
                        'reason': 'cascade_hedge'
                    }
                ))

            elif cascade_event.level == CascadeLevel.EMERGENCY:
                # Halt all new trading and liquidate positions
                actions.append(ProtectionAction(
                    action_type='halt_trading',
                    symbols=[],
                    reduction_percent=Decimal('100'),
                    hedge_required=False,
                    priority='emergency',
                    timestamp=timestamp,
                    metadata={'reason': 'cascade_emergency', 'halt_duration_minutes': 30}
                ))

                # Emergency position liquidation
                actions.append(ProtectionAction(
                    action_type='reduce_positions',
                    symbols=cascade_event.affected_symbols,
                    reduction_percent=Decimal('75'),
                    hedge_required=False,
                    priority='emergency',
                    timestamp=timestamp,
                    metadata={'reason': 'emergency_liquidation'}
                ))

            logger.info(f"Generated {len(actions)} protection actions for cascade level {cascade_event.level.value}")
            return actions

        except Exception as e:
            logger.error(f"Failed to calculate protection actions: {e}")
            raise

    async def _detect_correlation_spike(
        self,
        positions: List[Position],
        correlation_matrix: Dict[Tuple[str, str], Decimal]
    ) -> Decimal:
        """Detect abnormal correlation spikes between positions"""
        try:
            if len(positions) < 2:
                return Decimal('0')

            symbols = [p.symbol for p in positions]
            correlations = []

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:
                        corr = correlation_matrix.get((symbol1, symbol2), Decimal('0'))
                        correlations.append(corr)

            if not correlations:
                return Decimal('0')

            # Calculate average correlation
            avg_correlation = sum(correlations) / Decimal(str(len(correlations)))

            # Track correlation history
            timestamp = datetime.utcnow()
            self.correlation_history.append((timestamp, avg_correlation))

            # Keep only recent history (last hour)
            cutoff_time = timestamp - timedelta(hours=1)
            self.correlation_history = [
                (t, c) for t, c in self.correlation_history
                if t > cutoff_time
            ]

            # Check if correlation is spiking
            if len(self.correlation_history) > 10:
                recent_avg = sum(c for _, c in self.correlation_history[-10:]) / Decimal('10')
                historical_avg = sum(c for _, c in self.correlation_history[:-10]) / Decimal(str(len(self.correlation_history) - 10))

                if historical_avg > Decimal('0'):
                    correlation_spike = (recent_avg - historical_avg) / historical_avg
                else:
                    correlation_spike = Decimal('0')
            else:
                correlation_spike = Decimal('0')

            return max(avg_correlation, correlation_spike)

        except Exception as e:
            logger.error(f"Correlation spike detection failed: {e}")
            raise

    async def _detect_liquidity_evaporation(
        self,
        positions: List[Position],
        market_data: pl.DataFrame
    ) -> Decimal:
        """Detect sudden liquidity drops across positions"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()
            liquidity_drops = []

            for position in positions:
                symbol = position.symbol

                # Get recent volume data
                symbol_data = market_data.filter(pl.col('symbol') == symbol)

                if len(symbol_data) == 0:
                    continue

                current_volume = Decimal(str(symbol_data['volume'].tail(1)[0]))

                # Track volume history
                if symbol not in self.liquidity_history:
                    self.liquidity_history[symbol] = []

                self.liquidity_history[symbol].append((timestamp, current_volume))

                # Keep only recent history (last hour)
                cutoff_time = timestamp - timedelta(hours=1)
                self.liquidity_history[symbol] = [
                    (t, v) for t, v in self.liquidity_history[symbol]
                    if t > cutoff_time
                ]

                # Calculate liquidity drop
                if len(self.liquidity_history[symbol]) > 10:
                    recent_volumes = [v for _, v in self.liquidity_history[symbol][-5:]]
                    historical_volumes = [v for _, v in self.liquidity_history[symbol][:-5]]

                    recent_avg = sum(recent_volumes) / Decimal('5')
                    historical_avg = sum(historical_volumes) / Decimal(str(len(historical_volumes)))

                    if historical_avg > Decimal('0'):
                        liquidity_drop = (historical_avg - recent_avg) / historical_avg
                        liquidity_drops.append(max(liquidity_drop, Decimal('0')))

            if not liquidity_drops:
                return Decimal('0')

            # Return maximum liquidity drop
            max_drop = max(liquidity_drops)
            return max_drop

        except Exception as e:
            logger.error(f"Liquidity evaporation detection failed: {e}")
            raise

    async def _calculate_loss_velocity(
        self,
        positions: List[Position],
        portfolio_value: Decimal
    ) -> Decimal:
        """Calculate the rate of portfolio losses"""
        try:
            timestamp = datetime.utcnow()

            # Calculate current portfolio P&L
            total_pnl = sum(p.pnl for p in positions)

            if portfolio_value == Decimal('0'):
                return Decimal('0')

            pnl_percent = (total_pnl / portfolio_value) * Decimal('100')

            # Track losses
            self.recent_losses.append((timestamp, pnl_percent))

            # Keep only recent losses (last 5 minutes)
            cutoff_time = timestamp - timedelta(minutes=5)
            self.recent_losses = [
                (t, pnl) for t, pnl in self.recent_losses
                if t > cutoff_time
            ]

            # Calculate velocity (rate of change)
            if len(self.recent_losses) < 2:
                return Decimal('0')

            oldest_time, oldest_pnl = self.recent_losses[0]
            newest_time, newest_pnl = self.recent_losses[-1]

            time_diff_seconds = (newest_time - oldest_time).total_seconds()

            if time_diff_seconds == 0:
                return Decimal('0')

            pnl_change = newest_pnl - oldest_pnl

            # Velocity in percent per minute
            velocity = (pnl_change / Decimal(str(time_diff_seconds))) * Decimal('60')

            # Only consider negative velocity (losses)
            return abs(min(velocity, Decimal('0')))

        except Exception as e:
            logger.error(f"Loss velocity calculation failed: {e}")
            raise

    async def _determine_cascade_level(
        self,
        correlation_spike: Decimal,
        liquidity_drop: Decimal,
        loss_velocity: Decimal
    ) -> Tuple[CascadeLevel, str]:
        """Determine cascade severity level"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Emergency: Multiple severe conditions
            if (
                correlation_spike > self.config.cascade_correlation_threshold and
                liquidity_drop > self.config.liquidity_drop_threshold and
                loss_velocity > self.config.cascade_loss_velocity_threshold * Decimal('2')
            ):
                return CascadeLevel.EMERGENCY, "full_cascade"

            # Critical: Two severe conditions or one extreme condition
            if (
                (correlation_spike > self.config.cascade_correlation_threshold and
                 liquidity_drop > self.config.liquidity_drop_threshold) or
                loss_velocity > self.config.cascade_loss_velocity_threshold * Decimal('1.5')
            ):
                return CascadeLevel.CRITICAL, "critical_cascade"

            # Warning: One severe condition
            if (
                correlation_spike > self.config.cascade_correlation_threshold * Decimal('0.9') or
                liquidity_drop > self.config.liquidity_drop_threshold * Decimal('0.7') or
                loss_velocity > self.config.cascade_loss_velocity_threshold
            ):
                return CascadeLevel.WARNING, "warning_cascade"

            return CascadeLevel.NORMAL, "no_cascade"

        except Exception as e:
            logger.error(f"Failed to determine cascade level: {e}")
            raise

    async def _determine_cascade_action(
        self,
        level: CascadeLevel,
        correlation_spike: Decimal,
        liquidity_drop: Decimal,
        loss_velocity: Decimal
    ) -> str:
        """Determine recommended action for cascade event"""
        try:
            if level == CascadeLevel.EMERGENCY:
                return "EMERGENCY: Halt trading, liquidate 75% positions, activate emergency protocols"
            elif level == CascadeLevel.CRITICAL:
                return "CRITICAL: Reduce all positions 50%, activate emergency hedging"
            elif level == CascadeLevel.WARNING:
                return "WARNING: Reduce high-correlation positions 25%, monitor closely"
            else:
                return "No action required"

        except Exception as e:
            logger.error(f"Failed to determine cascade action: {e}")
            raise

    async def _identify_high_correlation_positions(
        self,
        positions: List[Position],
        affected_symbols: List[str]
    ) -> List[str]:
        """Identify positions with high correlation to affected symbols"""
        try:
            # Simplified: return all affected symbols
            # Full implementation would calculate correlation matrix
            return affected_symbols[:int(len(affected_symbols) * 0.5)]

        except Exception as e:
            logger.error(f"Failed to identify high correlation positions: {e}")
            raise

    async def clear_cascade_event(self, event_id: str) -> None:
        """Clear a resolved cascade event"""
        try:
            if event_id in self.active_cascade_events:
                self.active_cascade_events.remove(event_id)
                logger.info(f"Cleared cascade event: {event_id}")

        except Exception as e:
            logger.error(f"Failed to clear cascade event: {e}")
            raise


async def main():
    """Example usage of cascade protection"""
    try:
        # Initialize protection
        protection = CascadeProtection(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample positions
        positions = [
            Position(
                symbol='AAPL',
                quantity=Decimal('100'),
                entry_price=Decimal('150'),
                current_price=Decimal('148'),
                exchange='NASDAQ',
                strategy='test',
                opened_at=datetime.utcnow(),
                position_id='pos1'
            )
        ]

        # Sample correlation matrix
        correlation_matrix = {
            ('AAPL', 'GOOGL'): Decimal('0.9'),
            ('GOOGL', 'AAPL'): Decimal('0.9')
        }

        # Sample market data
        market_data = pl.DataFrame({
            'symbol': ['AAPL'],
            'timestamp': [datetime.utcnow().isoformat()],
            'volume': [1000000.0],
            'spread': [0.01]
        })

        # Detect cascade
        cascade = await protection.detect_cascade(
            positions,
            correlation_matrix,
            Decimal('1000000'),
            market_data
        )

        if cascade:
            logger.info(f"Cascade detected: {cascade}")

    except Exception as e:
        logger.error(f"Cascade protection example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
