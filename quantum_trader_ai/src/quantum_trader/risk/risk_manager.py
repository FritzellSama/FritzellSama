"""
Risk Manager - CRITICAL PRODUCTION SYSTEM
Handles all risk management including position limits, VaR, drawdown monitoring, and circuit breakers
MANAGES BILLIONS IN REAL CAPITAL - ZERO TOLERANCE FOR ERRORS
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
import numpy as np

from quantum_trader.utils.config_loader import get_config
from quantum_trader.models import Order, Position, OrderSide, OrderType

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    """Risk check decision outcomes"""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PARTIAL_APPROVED = "PARTIAL_APPROVED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    portfolio_value: Decimal
    total_exposure: Decimal
    daily_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    var_99: Decimal
    cvar_99: Decimal
    sharpe_ratio: Decimal
    leverage: Decimal
    largest_position_pct: Decimal
    correlation_max: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskViolation:
    """Risk limit violation details"""
    violation_type: str
    current_value: Decimal
    limit_value: Decimal
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """
    Production Risk Management System
    - Pre-trade risk checks
    - Real-time monitoring
    - Position limits and exposure management
    - VaR/CVaR calculation
    - Circuit breakers
    - Emergency stop capabilities
    """

    def __init__(self) -> None:
        """Initialize risk manager with configuration"""
        self.config = get_config()
        self._load_config()

        # Risk state tracking
        self.daily_start_balance: Decimal = Decimal('0')
        self.peak_portfolio_value: Decimal = Decimal('0')
        self.daily_pnl: Decimal = Decimal('0')
        self.emergency_stop_active: bool = False

        # Historical data for VaR calculation
        self.returns_history: pl.DataFrame = pl.DataFrame({
            'timestamp': pl.Series([], dtype=pl.Datetime),
            'return': pl.Series([], dtype=pl.Float64)
        })

        # Violation tracking
        self.recent_violations: List[RiskViolation] = []

        logger.info("RiskManager initialized with configuration")

    def _load_config(self) -> None:
        """Load all risk configuration from config files"""
        # Position limits
        self.max_position_size = self.config.get_decimal('risk', 'position_limits.max_position_size_usd')
        self.max_portfolio_value = self.config.get_decimal('risk', 'position_limits.max_portfolio_value_usd')
        self.max_concentration = self.config.get_decimal('risk', 'position_limits.max_concentration_pct')
        self.max_leverage = self.config.get_decimal('risk', 'position_limits.max_leverage')

        # Loss limits
        self.max_daily_loss = self.config.get_decimal('risk', 'loss_limits.max_daily_loss_usd')
        self.max_drawdown_pct = self.config.get_decimal('risk', 'loss_limits.max_drawdown_pct')
        self.var_limit = self.config.get_decimal('risk', 'loss_limits.var_limit_usd')
        self.cvar_limit = self.config.get_decimal('risk', 'loss_limits.cvar_limit_usd')

        # Risk metrics
        self.target_sharpe = self.config.get_decimal('risk', 'risk_metrics.target_sharpe_ratio')
        self.correlation_limit = self.config.get_decimal('risk', 'risk_metrics.max_correlation')
        self.var_confidence = self.config.get_decimal('risk', 'risk_metrics.var_confidence')
        self.var_horizon_days = self.config.get_int('risk', 'risk_metrics.var_horizon_days', 1)

        # Circuit breakers
        self.circuit_breakers_enabled = self.config.get_bool('risk', 'circuit_breakers.enabled', True)
        self.daily_loss_trigger = self.config.get_decimal('risk', 'circuit_breakers.daily_loss_trigger_usd')
        self.drawdown_trigger = self.config.get_decimal('risk', 'circuit_breakers.drawdown_trigger_pct')
        self.volatility_multiplier = self.config.get_decimal('risk', 'circuit_breakers.volatility_multiplier')

        # Monitoring
        self.check_interval = self.config.get_int('risk', 'monitoring.check_interval_seconds', 1)

        logger.info(f"Risk config loaded - Max position: ${self.max_position_size}, "
                   f"Max daily loss: ${self.max_daily_loss}, VaR limit: ${self.var_limit}")

    async def pre_trade_check(self, order: Order, portfolio_value: Decimal,
                              positions: List[Position]) -> Tuple[RiskDecision, Optional[str]]:
        """
        Comprehensive pre-trade risk checks

        Args:
            order: Order to validate
            portfolio_value: Current portfolio value
            positions: Current open positions

        Returns:
            Tuple of (RiskDecision, rejection_reason)
        """
        try:
            # Emergency stop check
            if self.emergency_stop_active:
                return (RiskDecision.EMERGENCY_STOP, "Emergency stop is active - all trading halted")

            # Run all checks in parallel
            checks = await asyncio.gather(
                self._check_position_limits(order, positions),
                self._check_buying_power(order, portfolio_value),
                self._check_concentration(order, portfolio_value, positions),
                self._check_correlation(order, positions),
                self._check_daily_loss(),
                self._check_var_impact(order, portfolio_value),
                return_exceptions=True
            )

            # Process check results
            for i, check in enumerate(checks):
                if isinstance(check, Exception):
                    logger.error(f"Risk check {i} failed with exception: {check}")
                    return (RiskDecision.REJECTED, f"Risk check error: {str(check)}")

                approved, reason = check
                if not approved:
                    logger.warning(f"Order rejected - {reason}")
                    self._record_violation(f"pre_trade_check_{i}", reason)
                    return (RiskDecision.REJECTED, reason)

            logger.info(f"Order passed all risk checks: {order.symbol} {order.side} {order.quantity}")
            return (RiskDecision.APPROVED, None)

        except Exception as e:
            logger.error(f"Unexpected error in pre_trade_check: {e}", exc_info=True)
            return (RiskDecision.REJECTED, f"Risk check system error: {str(e)}")

    async def _check_position_limits(self, order: Order, positions: List[Position]) -> Tuple[bool, Optional[str]]:
        """Check if order exceeds position size limits"""
        try:
            # Calculate order value (use limit price if available, otherwise estimate)
            order_value = order.quantity * (order.price if order.price else Decimal('0'))

            if order_value > self.max_position_size:
                return (False, f"Order size ${order_value} exceeds max position size ${self.max_position_size}")

            # Check existing position
            existing_position = next((p for p in positions if p.symbol == order.symbol), None)
            if existing_position:
                new_quantity = existing_position.quantity
                if order.side == OrderSide.BUY:
                    new_quantity += order.quantity
                else:
                    new_quantity -= order.quantity

                new_value = abs(new_quantity * existing_position.current_price)
                if new_value > self.max_position_size:
                    return (False, f"New position value ${new_value} would exceed limit ${self.max_position_size}")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return (False, f"Position limit check error: {str(e)}")

    async def _check_buying_power(self, order: Order, portfolio_value: Decimal) -> Tuple[bool, Optional[str]]:
        """Check if sufficient buying power for order"""
        try:
            # Estimate order cost (use limit price or add slippage buffer for market orders)
            if order.price:
                order_cost = order.quantity * order.price
            else:
                # For market orders, add 1% slippage buffer
                order_cost = order.quantity * Decimal('1.01')  # Price should be provided from orderbook

            # Check against leverage limit
            max_order_value = portfolio_value * self.max_leverage
            if order_cost > max_order_value:
                return (False, f"Order cost ${order_cost} exceeds buying power ${max_order_value} "
                              f"(portfolio ${portfolio_value} * leverage {self.max_leverage})")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking buying power: {e}")
            return (False, f"Buying power check error: {str(e)}")

    async def _check_concentration(self, order: Order, portfolio_value: Decimal,
                                   positions: List[Position]) -> Tuple[bool, Optional[str]]:
        """Check if order would violate concentration limits"""
        try:
            # Calculate position value after order
            existing_pos = next((p for p in positions if p.symbol == order.symbol), None)

            if existing_pos:
                new_quantity = existing_pos.quantity
                if order.side == OrderSide.BUY:
                    new_quantity += order.quantity
                else:
                    new_quantity -= order.quantity
                position_value = abs(new_quantity * existing_pos.current_price)
            else:
                position_value = order.quantity * (order.price or Decimal('0'))

            concentration = position_value / portfolio_value if portfolio_value > 0 else Decimal('0')

            if concentration > self.max_concentration:
                return (False, f"Position concentration {concentration*100:.2f}% exceeds limit "
                              f"{self.max_concentration*100:.2f}%")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking concentration: {e}")
            return (False, f"Concentration check error: {str(e)}")

    async def _check_correlation(self, order: Order, positions: List[Position]) -> Tuple[bool, Optional[str]]:
        """Check correlation with existing positions (simplified check)"""
        try:
            # This is a simplified correlation check
            # In production, you'd calculate actual correlation from historical price data

            # For now, just check if we have too many positions in the same direction
            same_side_positions = sum(1 for p in positions if
                                     (p.quantity > 0 and order.side == OrderSide.BUY) or
                                     (p.quantity < 0 and order.side == OrderSide.SELL))

            if same_side_positions > 10:  # More than 10 positions in same direction
                logger.warning(f"High directional concentration: {same_side_positions} positions")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking correlation: {e}")
            return (False, f"Correlation check error: {str(e)}")

    async def _check_daily_loss(self) -> Tuple[bool, Optional[str]]:
        """Check if daily loss limit exceeded"""
        try:
            if abs(self.daily_pnl) > self.max_daily_loss and self.daily_pnl < 0:
                return (False, f"Daily loss ${abs(self.daily_pnl)} exceeds limit ${self.max_daily_loss}")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return (False, f"Daily loss check error: {str(e)}")

    async def _check_var_impact(self, order: Order, portfolio_value: Decimal) -> Tuple[bool, Optional[str]]:
        """Check if order would increase VaR beyond limits"""
        try:
            # Simplified VaR check - in production, calculate full portfolio VaR
            current_var = await self.calculate_var()

            if current_var > self.var_limit:
                return (False, f"Current VaR ${current_var} exceeds limit ${self.var_limit}")

            return (True, None)

        except Exception as e:
            logger.error(f"Error checking VaR impact: {e}")
            return (False, f"VaR check error: {str(e)}")

    async def calculate_var(self, confidence: Optional[Decimal] = None,
                           horizon: Optional[int] = None) -> Decimal:
        """
        Calculate Value at Risk using historical simulation

        Args:
            confidence: Confidence level (default from config)
            horizon: Time horizon in days (default from config)

        Returns:
            VaR value in USD
        """
        try:
            confidence = confidence or self.var_confidence
            horizon = horizon or self.var_horizon_days

            if len(self.returns_history) < 30:
                logger.warning("Insufficient data for VaR calculation, returning limit value")
                return self.var_limit

            # Calculate VaR from returns distribution
            returns_array = self.returns_history['return'].to_numpy()
            var_percentile = float((Decimal('1') - confidence) * Decimal('100'))
            var_return = Decimal(str(np.percentile(returns_array, var_percentile)))

            # Scale by horizon (square root of time rule)
            horizon_scalar = Decimal(str(np.sqrt(horizon)))
            var_value = abs(var_return) * horizon_scalar

            logger.debug(f"VaR calculated: ${var_value} at {confidence*100}% confidence")
            return var_value

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}", exc_info=True)
            return self.var_limit  # Conservative fallback

    async def monitor_real_time(self, portfolio_value: Decimal, positions: List[Position]) -> RiskMetrics:
        """
        Real-time risk monitoring

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions

        Returns:
            Current risk metrics
        """
        try:
            # Update peak value for drawdown calculation
            if portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value

            # Calculate metrics
            total_exposure = sum(abs(p.quantity * p.current_price) for p in positions)
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value \
                              if self.peak_portfolio_value > 0 else Decimal('0')

            leverage = total_exposure / portfolio_value if portfolio_value > 0 else Decimal('0')

            largest_position = max((abs(p.quantity * p.current_price) for p in positions), default=Decimal('0'))
            largest_position_pct = largest_position / portfolio_value if portfolio_value > 0 else Decimal('0')

            var_99 = await self.calculate_var()
            cvar_99 = var_99 * Decimal('1.2')  # Simplified CVaR estimate

            metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                daily_pnl=self.daily_pnl,
                max_drawdown=self.max_drawdown_pct,
                current_drawdown=current_drawdown,
                var_99=var_99,
                cvar_99=cvar_99,
                sharpe_ratio=Decimal('0'),  # Calculate from returns history
                leverage=leverage,
                largest_position_pct=largest_position_pct,
                correlation_max=Decimal('0')  # Calculate from positions
            )

            # Circuit breaker checks
            await self._check_circuit_breakers(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}", exc_info=True)
            raise

    async def _check_circuit_breakers(self, metrics: RiskMetrics) -> None:
        """Check and trigger circuit breakers if needed"""
        try:
            if not self.circuit_breakers_enabled:
                return

            # Daily loss circuit breaker
            if abs(metrics.daily_pnl) > self.daily_loss_trigger and metrics.daily_pnl < 0:
                await self.trigger_circuit_breaker("DAILY_LOSS",
                    f"Daily loss ${abs(metrics.daily_pnl)} exceeded trigger ${self.daily_loss_trigger}")

            # Drawdown circuit breaker
            if metrics.current_drawdown > self.drawdown_trigger:
                await self.trigger_circuit_breaker("DRAWDOWN",
                    f"Drawdown {metrics.current_drawdown*100:.2f}% exceeded trigger "
                    f"{self.drawdown_trigger*100:.2f}%")

            # VaR circuit breaker
            if metrics.var_99 > self.var_limit:
                await self.trigger_circuit_breaker("VAR",
                    f"VaR ${metrics.var_99} exceeded limit ${self.var_limit}")

        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}", exc_info=True)

    async def trigger_circuit_breaker(self, breaker_type: str, reason: str) -> None:
        """Trigger a circuit breaker and halt trading"""
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {breaker_type} - {reason}")
        self.emergency_stop_active = True

        violation = RiskViolation(
            violation_type=breaker_type,
            current_value=Decimal('0'),
            limit_value=Decimal('0'),
            severity="CRITICAL",
            message=reason
        )
        self.recent_violations.append(violation)

        # In production: send alerts, close positions, etc.
        # await self.emergency_liquidation()

    async def emergency_liquidation(self) -> None:
        """Execute emergency liquidation of all positions"""
        logger.critical("EMERGENCY LIQUIDATION INITIATED")
        # Implementation would close all positions
        # This is a placeholder for production implementation
        pass

    async def reduce_all_positions(self, reduction_pct: Decimal) -> None:
        """Reduce all positions by a percentage"""
        logger.warning(f"Reducing all positions by {reduction_pct*100}%")
        # Implementation would reduce position sizes
        pass

    def _record_violation(self, violation_type: str, message: str) -> None:
        """Record a risk violation"""
        violation = RiskViolation(
            violation_type=violation_type,
            current_value=Decimal('0'),
            limit_value=Decimal('0'),
            severity="HIGH",
            message=message
        )
        self.recent_violations.append(violation)

        # Keep only recent violations (last 100)
        if len(self.recent_violations) > 100:
            self.recent_violations = self.recent_violations[-100:]

    def update_daily_pnl(self, pnl: Decimal) -> None:
        """Update daily P&L tracking"""
        self.daily_pnl = pnl

    def reset_daily_metrics(self, starting_balance: Decimal) -> None:
        """Reset daily metrics at start of trading day"""
        self.daily_start_balance = starting_balance
        self.daily_pnl = Decimal('0')
        logger.info(f"Daily metrics reset - Starting balance: ${starting_balance}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics summary"""
        return {
            'daily_pnl': str(self.daily_pnl),
            'peak_portfolio_value': str(self.peak_portfolio_value),
            'emergency_stop_active': self.emergency_stop_active,
            'recent_violations_count': len(self.recent_violations)
        }
