"""
Dynamic Position Sizing
Quantum Trader AI - Production Risk Management

Calculate adaptive position sizes based on market conditions:
- Volatility-adjusted sizing
- Kelly criterion integration
- Drawdown-adjusted sizing
- Confidence-based sizing

CRITICAL: All numeric values use Decimal, never float
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import polars as pl
import yaml

from quantum_trader.models import Position, Signal, AuditLog

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Position sizing configuration"""
    max_portfolio_risk_percent: Decimal
    max_position_size_percent: Decimal
    min_position_size_usd: Decimal
    max_leverage: Decimal
    volatility_position_multiplier: Decimal

    # Kelly criterion settings
    kelly_fraction: Decimal = Decimal('0.25')  # Use 25% of full Kelly
    min_win_rate: Decimal = Decimal('0.52')  # Minimum 52% win rate

    @classmethod
    def from_yaml(cls, risk_config: Dict, prod_config: Dict) -> 'PositionSizingConfig':
        """Load configuration from yaml dictionaries"""
        return cls(
            max_portfolio_risk_percent=Decimal(str(risk_config['global']['max_portfolio_risk_percent'])),
            max_position_size_percent=Decimal(str(risk_config['position_limits']['max_position_size_percent'])),
            min_position_size_usd=Decimal(str(risk_config['position_limits']['min_position_size_usd'])),
            max_leverage=Decimal(str(risk_config['position_limits']['max_leverage'])),
            volatility_position_multiplier=Decimal(str(risk_config['market_conditions']['volatility_position_multiplier']))
        )


@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    symbol: str
    recommended_quantity: Decimal
    recommended_value: Decimal
    position_weight: Decimal
    sizing_method: str
    risk_amount: Decimal
    stop_loss_distance: Decimal
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class DynamicPositionSizer:
    """
    Calculate dynamic position sizes based on multiple factors.

    Adapts position sizing to market conditions, volatility, confidence,
    and portfolio state using multiple sizing methodologies.
    """

    def __init__(self, risk_config_path: str, prod_config_path: str):
        """Initialize dynamic position sizer with configuration"""
        self.risk_config_path = risk_config_path
        self.prod_config_path = prod_config_path
        self.config: Optional[PositionSizingConfig] = None
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

                self.config = PositionSizingConfig.from_yaml(risk_config, prod_config)
                logger.info("Position sizing configuration loaded successfully")
                return

            except Exception as e:
                logger.error(f"Failed to load config (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise RuntimeError(f"Failed to load position sizing config after {max_retries} attempts") from e

    async def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: Decimal,
        current_price: Decimal,
        stop_loss_percent: Optional[Decimal] = None,
        volatility: Optional[Decimal] = None,
        win_rate: Optional[Decimal] = None,
        current_drawdown_percent: Optional[Decimal] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using multiple factors.

        Args:
            signal: Trading signal with confidence and strength
            portfolio_value: Total portfolio value
            current_price: Current asset price
            stop_loss_percent: Stop loss distance as percentage
            volatility: Asset volatility (annualized)
            win_rate: Historical win rate for this strategy
            current_drawdown_percent: Current portfolio drawdown

        Returns:
            PositionSizeResult with recommended size and metrics
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # Start with base position size (fixed risk)
            base_size = await self._calculate_fixed_risk_size(
                portfolio_value,
                current_price,
                stop_loss_percent or Decimal('2.5')
            )

            # Adjust for confidence
            confidence_adjusted_size = await self._adjust_for_confidence(
                base_size,
                signal.confidence
            )

            # Adjust for volatility
            if volatility is not None:
                volatility_adjusted_size = await self._adjust_for_volatility(
                    confidence_adjusted_size,
                    volatility
                )
            else:
                volatility_adjusted_size = confidence_adjusted_size

            # Adjust for drawdown
            if current_drawdown_percent is not None:
                drawdown_adjusted_size = await self._adjust_for_drawdown(
                    volatility_adjusted_size,
                    current_drawdown_percent
                )
            else:
                drawdown_adjusted_size = volatility_adjusted_size

            # Calculate Kelly-based size if win rate available
            if win_rate is not None:
                kelly_size = await self._calculate_kelly_size(
                    portfolio_value,
                    current_price,
                    win_rate,
                    signal.strength
                )
                # Use average of adjusted size and Kelly size
                final_size = (drawdown_adjusted_size + kelly_size) / Decimal('2')
                sizing_method = 'hybrid_kelly'
            else:
                final_size = drawdown_adjusted_size
                sizing_method = 'dynamic_adjusted'

            # Apply position size limits
            final_size = await self._apply_position_limits(
                final_size,
                portfolio_value,
                current_price
            )

            # Calculate metrics
            position_value = final_size * current_price
            position_weight = position_value / portfolio_value if portfolio_value > Decimal('0') else Decimal('0')
            risk_amount = position_value * (stop_loss_percent or Decimal('2.5')) / Decimal('100')

            result = PositionSizeResult(
                symbol=signal.symbol,
                recommended_quantity=final_size,
                recommended_value=position_value,
                position_weight=position_weight,
                sizing_method=sizing_method,
                risk_amount=risk_amount,
                stop_loss_distance=stop_loss_percent or Decimal('2.5'),
                timestamp=timestamp,
                metadata={
                    'signal_confidence': str(signal.confidence),
                    'signal_strength': str(signal.strength),
                    'base_size': str(base_size),
                    'final_size': str(final_size),
                    'volatility': str(volatility) if volatility else None,
                    'win_rate': str(win_rate) if win_rate else None
                }
            )

            logger.info(
                f"Position size calculated for {signal.symbol}: "
                f"qty={final_size}, value=${position_value}, weight={position_weight*Decimal('100')}%"
            )

            return result

        except Exception as e:
            logger.error(f"Position size calculation failed for {signal.symbol}: {e}")
            raise

    async def calculate_volatility_adjusted_size(
        self,
        symbol: str,
        base_size: Decimal,
        volatility: Decimal,
        target_volatility: Decimal = Decimal('0.15')
    ) -> PositionSizeResult:
        """
        Calculate position size adjusted for volatility targeting.

        Args:
            symbol: Asset symbol
            base_size: Base position size
            volatility: Asset volatility (annualized)
            target_volatility: Target portfolio volatility contribution

        Returns:
            PositionSizeResult with volatility-adjusted size
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # Volatility scaling: size = target_vol / asset_vol * base_size
            if volatility > Decimal('0'):
                vol_adjusted_size = (target_volatility / volatility) * base_size
            else:
                vol_adjusted_size = base_size

            # Apply multiplier from config for high volatility environments
            if volatility > Decimal('0.3'):  # 30% annualized volatility threshold
                vol_adjusted_size *= self.config.volatility_position_multiplier

            result = PositionSizeResult(
                symbol=symbol,
                recommended_quantity=vol_adjusted_size,
                recommended_value=Decimal('0'),
                position_weight=Decimal('0'),
                sizing_method='volatility_adjusted',
                risk_amount=Decimal('0'),
                stop_loss_distance=Decimal('0'),
                timestamp=timestamp,
                metadata={
                    'volatility': str(volatility),
                    'target_volatility': str(target_volatility),
                    'base_size': str(base_size)
                }
            )

            return result

        except Exception as e:
            logger.error(f"Volatility-adjusted size calculation failed: {e}")
            raise

    async def calculate_kelly_criterion_size(
        self,
        symbol: str,
        portfolio_value: Decimal,
        current_price: Decimal,
        win_rate: Decimal,
        avg_win_loss_ratio: Decimal
    ) -> PositionSizeResult:
        """
        Calculate position size using Kelly criterion.

        Kelly % = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio

        Args:
            symbol: Asset symbol
            portfolio_value: Total portfolio value
            current_price: Current asset price
            win_rate: Historical win rate (0 to 1)
            avg_win_loss_ratio: Average win / average loss ratio

        Returns:
            PositionSizeResult with Kelly-based size
        """
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            timestamp = datetime.utcnow()

            # Validate inputs
            if win_rate < self.config.min_win_rate:
                logger.warning(
                    f"Win rate {win_rate} below minimum {self.config.min_win_rate}, "
                    "using minimum position size"
                )
                kelly_percent = Decimal('0.01')  # 1% minimum
            else:
                # Calculate Kelly percentage
                if avg_win_loss_ratio > Decimal('0'):
                    kelly_percent = (
                        (win_rate * avg_win_loss_ratio - (Decimal('1') - win_rate)) /
                        avg_win_loss_ratio
                    )
                else:
                    kelly_percent = Decimal('0')

                # Apply fractional Kelly for safety
                kelly_percent *= self.config.kelly_fraction

                # Clamp to reasonable range
                kelly_percent = max(Decimal('0.01'), min(kelly_percent, Decimal('0.10')))  # 1-10%

            # Calculate position size
            position_value = portfolio_value * kelly_percent
            position_size = position_value / current_price if current_price > Decimal('0') else Decimal('0')

            result = PositionSizeResult(
                symbol=symbol,
                recommended_quantity=position_size,
                recommended_value=position_value,
                position_weight=kelly_percent,
                sizing_method='kelly_criterion',
                risk_amount=position_value * Decimal('0.5'),  # Estimate 50% risk
                stop_loss_distance=Decimal('0'),
                timestamp=timestamp,
                metadata={
                    'win_rate': str(win_rate),
                    'avg_win_loss_ratio': str(avg_win_loss_ratio),
                    'kelly_percent': str(kelly_percent),
                    'kelly_fraction': str(self.config.kelly_fraction)
                }
            )

            logger.info(
                f"Kelly size for {symbol}: {kelly_percent*Decimal('100')}% of portfolio, "
                f"qty={position_size}"
            )

            return result

        except Exception as e:
            logger.error(f"Kelly criterion calculation failed: {e}")
            raise

    async def _calculate_fixed_risk_size(
        self,
        portfolio_value: Decimal,
        current_price: Decimal,
        stop_loss_percent: Decimal
    ) -> Decimal:
        """Calculate position size for fixed risk amount"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Risk amount = portfolio * max_risk_percent
            risk_amount = portfolio_value * (self.config.max_portfolio_risk_percent / Decimal('100'))

            # Position size = risk_amount / (price * stop_loss_percent / 100)
            if current_price > Decimal('0') and stop_loss_percent > Decimal('0'):
                stop_loss_amount = current_price * (stop_loss_percent / Decimal('100'))
                position_size = risk_amount / stop_loss_amount
            else:
                position_size = Decimal('0')

            return position_size

        except Exception as e:
            logger.error(f"Fixed risk size calculation failed: {e}")
            raise

    async def _adjust_for_confidence(
        self,
        base_size: Decimal,
        confidence: Decimal
    ) -> Decimal:
        """Adjust position size based on signal confidence"""
        try:
            # Scale linearly with confidence
            # confidence of 1.0 = full size, 0.5 = half size, etc.
            adjusted_size = base_size * confidence

            # Minimum 25% of base size even at low confidence
            adjusted_size = max(adjusted_size, base_size * Decimal('0.25'))

            return adjusted_size

        except Exception as e:
            logger.error(f"Confidence adjustment failed: {e}")
            raise

    async def _adjust_for_volatility(
        self,
        base_size: Decimal,
        volatility: Decimal
    ) -> Decimal:
        """Adjust position size based on volatility"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Normal volatility threshold: 15% annualized
            normal_volatility = Decimal('0.15')

            if volatility > normal_volatility:
                # Reduce size in high volatility
                vol_ratio = normal_volatility / volatility
                adjusted_size = base_size * vol_ratio

                # Apply additional multiplier from config
                adjusted_size *= self.config.volatility_position_multiplier
            else:
                # No adjustment for normal/low volatility
                adjusted_size = base_size

            return adjusted_size

        except Exception as e:
            logger.error(f"Volatility adjustment failed: {e}")
            raise

    async def _adjust_for_drawdown(
        self,
        base_size: Decimal,
        drawdown_percent: Decimal
    ) -> Decimal:
        """Adjust position size based on current drawdown"""
        try:
            # Reduce size during drawdowns
            if drawdown_percent < Decimal('2'):
                # No adjustment for small drawdowns
                return base_size
            elif drawdown_percent < Decimal('5'):
                # 5% reduction for 2-5% drawdown
                return base_size * Decimal('0.95')
            elif drawdown_percent < Decimal('10'):
                # 25% reduction for 5-10% drawdown
                return base_size * Decimal('0.75')
            else:
                # 50% reduction for >10% drawdown
                return base_size * Decimal('0.5')

        except Exception as e:
            logger.error(f"Drawdown adjustment failed: {e}")
            raise

    async def _calculate_kelly_size(
        self,
        portfolio_value: Decimal,
        current_price: Decimal,
        win_rate: Decimal,
        expected_return: Decimal
    ) -> Decimal:
        """Calculate position size using simplified Kelly criterion"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            # Simplified Kelly: f = (p * b - q) / b
            # where p = win_rate, q = 1 - win_rate, b = odds (using expected_return as proxy)
            if win_rate < self.config.min_win_rate:
                return Decimal('0')

            q = Decimal('1') - win_rate
            b = max(expected_return, Decimal('0.5'))  # Minimum 0.5 odds

            kelly_fraction = (win_rate * b - q) / b

            # Apply fractional Kelly
            kelly_fraction *= self.config.kelly_fraction

            # Clamp
            kelly_fraction = max(Decimal('0'), min(kelly_fraction, Decimal('0.1')))

            # Calculate size
            position_value = portfolio_value * kelly_fraction
            position_size = position_value / current_price if current_price > Decimal('0') else Decimal('0')

            return position_size

        except Exception as e:
            logger.error(f"Kelly size calculation failed: {e}")
            raise

    async def _apply_position_limits(
        self,
        position_size: Decimal,
        portfolio_value: Decimal,
        current_price: Decimal
    ) -> Decimal:
        """Apply maximum and minimum position size limits"""
        try:
            if self.config is None:
                raise RuntimeError("Configuration not loaded")

            position_value = position_size * current_price

            # Apply maximum position size
            max_position_value = portfolio_value * (self.config.max_position_size_percent / Decimal('100'))
            if position_value > max_position_value:
                position_size = max_position_value / current_price if current_price > Decimal('0') else Decimal('0')
                logger.info(f"Position size capped at max {self.config.max_position_size_percent}%")

            # Apply minimum position size
            if position_value < self.config.min_position_size_usd:
                if position_value > Decimal('0'):
                    logger.warning(
                        f"Position value ${position_value} below minimum ${self.config.min_position_size_usd}, "
                        "setting to zero"
                    )
                position_size = Decimal('0')

            # Round to reasonable precision (e.g., 8 decimals for crypto, 0 for stocks)
            # For now, using 8 decimals
            position_size = position_size.quantize(Decimal('0.00000001'))

            return position_size

        except Exception as e:
            logger.error(f"Failed to apply position limits: {e}")
            raise


async def main():
    """Example usage of dynamic position sizer"""
    try:
        # Initialize sizer
        sizer = DynamicPositionSizer(
            risk_config_path='/home/user/FritzellSama/config/bot/risk.yaml',
            prod_config_path='/home/user/FritzellSama/config/environments/production.yaml'
        )

        # Create sample signal
        from quantum_trader.models import SignalAction
        signal = Signal(
            symbol='AAPL',
            action=SignalAction.BUY,
            strength=Decimal('0.8'),
            confidence=Decimal('0.85'),
            timestamp=datetime.utcnow(),
            strategy='test_strategy',
            timeframe='1h'
        )

        portfolio_value = Decimal('1000000')
        current_price = Decimal('150')
        stop_loss_percent = Decimal('2.5')
        volatility = Decimal('0.20')  # 20% annualized
        win_rate = Decimal('0.60')  # 60% win rate
        current_drawdown = Decimal('3.0')  # 3% drawdown

        # Calculate position size
        result = await sizer.calculate_position_size(
            signal=signal,
            portfolio_value=portfolio_value,
            current_price=current_price,
            stop_loss_percent=stop_loss_percent,
            volatility=volatility,
            win_rate=win_rate,
            current_drawdown_percent=current_drawdown
        )

        logger.info(f"Position sizing result: {result}")
        logger.info(
            f"Recommended: {result.recommended_quantity} shares "
            f"(${result.recommended_value}), "
            f"weight={result.position_weight*Decimal('100')}%"
        )

    except Exception as e:
        logger.error(f"Dynamic position sizer example failed: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
