"""
Risk Limits Manager
CRITICAL: Risk limit definitions, validation, and enforcement
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VALUE = "portfolio_value"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    VAR = "var"
    CVAR = "cvar"
    CORRELATION = "correlation"


class LimitSeverity(Enum):
    """Severity levels for limit breaches"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    HALT = "halt"


class RiskLimitsManager:
    """Manage and enforce risk limits"""

    def __init__(self):
        """Initialize risk limits manager with config"""
        self.config = get_config()

        # Load all limits from config
        self.limits = {
            LimitType.POSITION_SIZE: self.config.get_decimal("risk", "position_limits.max_position_size_usd"),
            LimitType.PORTFOLIO_VALUE: self.config.get_decimal("risk", "position_limits.max_portfolio_value_usd"),
            LimitType.CONCENTRATION: self.config.get_decimal("risk", "position_limits.max_concentration_pct"),
            LimitType.LEVERAGE: self.config.get_decimal("risk", "position_limits.max_leverage"),
            LimitType.DAILY_LOSS: self.config.get_decimal("risk", "loss_limits.max_daily_loss_usd"),
            LimitType.DRAWDOWN: self.config.get_decimal("risk", "loss_limits.max_drawdown_pct"),
            LimitType.VAR: self.config.get_decimal("risk", "loss_limits.var_limit_usd"),
            LimitType.CVAR: self.config.get_decimal("risk", "loss_limits.cvar_limit_usd"),
            LimitType.CORRELATION: self.config.get_decimal("risk", "risk_metrics.max_correlation")
        }

        # Track limit breaches
        self.breach_history: List[Dict] = []
        self.daily_loss_start_value: Optional[Decimal] = None
        self.daily_loss_reset_date: Optional[datetime] = None

        logger.info(
            f"RiskLimitsManager initialized with {len(self.limits)} limits: "
            f"max_position=${self.limits[LimitType.POSITION_SIZE]:,.0f}, "
            f"max_leverage={self.limits[LimitType.LEVERAGE]}x, "
            f"max_daily_loss=${self.limits[LimitType.DAILY_LOSS]:,.0f}"
        )

    async def check_limits(
        self,
        current_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Check all risk limits against current metrics

        Args:
            current_metrics: Dictionary with current risk metrics

        Returns:
            Dictionary with limit check results:
            {
                'passed': bool,
                'breaches': List[Dict],
                'warnings': List[Dict],
                'highest_severity': str,
                'timestamp': float
            }
        """
        try:
            logger.debug("Checking all risk limits")

            breaches = []
            warnings = []
            highest_severity = LimitSeverity.INFO

            # Check each limit type
            for limit_type, limit_value in self.limits.items():
                result = await self._check_single_limit(
                    limit_type,
                    limit_value,
                    current_metrics
                )

                if result['breached']:
                    severity = result['severity']

                    # Update highest severity
                    if self._compare_severity(severity, highest_severity) > 0:
                        highest_severity = severity

                    # Categorize as breach or warning
                    if severity in [LimitSeverity.CRITICAL, LimitSeverity.HALT]:
                        breaches.append(result)
                    else:
                        warnings.append(result)

            # Log results
            if breaches:
                logger.error(
                    f"LIMIT BREACH: {len(breaches)} critical breaches detected, "
                    f"severity={highest_severity.value}"
                )
                for breach in breaches:
                    logger.error(
                        f"  {breach['limit_type']}: {breach['current_value']} exceeds "
                        f"{breach['limit_value']} (severity: {breach['severity'].value})"
                    )

            if warnings:
                logger.warning(f"Risk warnings: {len(warnings)} limits approaching threshold")

            result = {
                'passed': len(breaches) == 0,
                'breaches': breaches,
                'warnings': warnings,
                'highest_severity': highest_severity.value,
                'timestamp': datetime.now().timestamp(),
                'total_checks': len(self.limits)
            }

            # Store in history
            if breaches:
                self.breach_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Error checking limits: {e}", exc_info=True)
            raise

    async def get_limit_status(
        self,
        current_metrics: Dict[str, any]
    ) -> Dict[str, Dict]:
        """
        Get detailed status of all limits

        Args:
            current_metrics: Dictionary with current risk metrics

        Returns:
            Dictionary mapping limit types to status:
            {
                'limit_type': {
                    'limit_value': Decimal,
                    'current_value': Decimal,
                    'utilization_pct': Decimal,
                    'status': str,  # 'ok', 'warning', 'critical'
                    'headroom': Decimal
                }
            }
        """
        try:
            status = {}

            for limit_type, limit_value in self.limits.items():
                current_value = await self._get_current_value(limit_type, current_metrics)

                if current_value is not None:
                    # Calculate utilization
                    utilization_pct = (current_value / limit_value * Decimal("100")) if limit_value > 0 else Decimal("0")

                    # Determine status
                    if utilization_pct >= Decimal("90"):
                        status_str = "critical"
                    elif utilization_pct >= Decimal("75"):
                        status_str = "warning"
                    else:
                        status_str = "ok"

                    # Calculate headroom
                    headroom = limit_value - current_value

                    status[limit_type.value] = {
                        'limit_value': limit_value.quantize(Decimal("0.01")),
                        'current_value': current_value.quantize(Decimal("0.01")),
                        'utilization_pct': utilization_pct.quantize(Decimal("0.01")),
                        'status': status_str,
                        'headroom': headroom.quantize(Decimal("0.01"))
                    }

            logger.debug(f"Generated limit status for {len(status)} limits")

            return status

        except Exception as e:
            logger.error(f"Error getting limit status: {e}", exc_info=True)
            raise

    async def update_limits(
        self,
        new_limits: Dict[LimitType, Decimal]
    ) -> None:
        """
        Update risk limits (requires appropriate authorization)

        Args:
            new_limits: Dictionary of limit types to new values
        """
        try:
            logger.warning(f"Updating {len(new_limits)} risk limits")

            for limit_type, new_value in new_limits.items():
                if limit_type not in self.limits:
                    logger.error(f"Unknown limit type: {limit_type}")
                    continue

                old_value = self.limits[limit_type]
                self.limits[limit_type] = new_value

                logger.info(
                    f"Updated {limit_type.value}: {old_value} -> {new_value} "
                    f"({((new_value - old_value) / old_value * Decimal('100')):.1f}% change)"
                )

            # Log to audit trail
            logger.warning(
                f"Risk limits updated: {[lt.value for lt in new_limits.keys()]}"
            )

        except Exception as e:
            logger.error(f"Error updating limits: {e}", exc_info=True)
            raise

    async def _check_single_limit(
        self,
        limit_type: LimitType,
        limit_value: Decimal,
        current_metrics: Dict[str, any]
    ) -> Dict[str, any]:
        """Check a single limit against current metrics"""
        try:
            current_value = await self._get_current_value(limit_type, current_metrics)

            if current_value is None:
                return {
                    'limit_type': limit_type.value,
                    'breached': False,
                    'severity': LimitSeverity.INFO,
                    'message': f"No data available for {limit_type.value}"
                }

            # Determine if breached and severity
            breached = current_value > limit_value
            utilization_pct = (current_value / limit_value * Decimal("100")) if limit_value > 0 else Decimal("0")

            # Determine severity based on utilization
            if utilization_pct >= Decimal("100"):
                severity = LimitSeverity.HALT
            elif utilization_pct >= Decimal("90"):
                severity = LimitSeverity.CRITICAL
            elif utilization_pct >= Decimal("75"):
                severity = LimitSeverity.WARNING
            else:
                severity = LimitSeverity.INFO

            return {
                'limit_type': limit_type.value,
                'breached': breached or severity != LimitSeverity.INFO,
                'severity': severity,
                'limit_value': limit_value,
                'current_value': current_value,
                'utilization_pct': utilization_pct,
                'message': self._format_limit_message(limit_type, current_value, limit_value, severity)
            }

        except Exception as e:
            logger.error(f"Error checking {limit_type.value}: {e}", exc_info=True)
            return {
                'limit_type': limit_type.value,
                'breached': False,
                'severity': LimitSeverity.INFO,
                'message': f"Error checking {limit_type.value}: {e}"
            }

    async def _get_current_value(
        self,
        limit_type: LimitType,
        current_metrics: Dict[str, any]
    ) -> Optional[Decimal]:
        """Extract current value for a specific limit type from metrics"""
        try:
            if limit_type == LimitType.POSITION_SIZE:
                # Get max position size from position metrics
                position_metrics = current_metrics.get('position_metrics', {})
                if position_metrics:
                    max_position = max(
                        (p['value'] for p in position_metrics.values()),
                        default=Decimal("0")
                    )
                    return max_position
                return Decimal("0")

            elif limit_type == LimitType.PORTFOLIO_VALUE:
                return current_metrics.get('portfolio_metrics', {}).get('portfolio_value', Decimal("0"))

            elif limit_type == LimitType.CONCENTRATION:
                # Get max concentration from position metrics
                position_metrics = current_metrics.get('position_metrics', {})
                if position_metrics:
                    max_weight = max(
                        (p['weight'] for p in position_metrics.values()),
                        default=Decimal("0")
                    )
                    return max_weight
                return Decimal("0")

            elif limit_type == LimitType.LEVERAGE:
                return current_metrics.get('portfolio_metrics', {}).get('leverage', Decimal("0"))

            elif limit_type == LimitType.DAILY_LOSS:
                # Calculate daily loss
                portfolio_value = current_metrics.get('portfolio_metrics', {}).get('portfolio_value', Decimal("0"))
                return await self._calculate_daily_loss(portfolio_value)

            elif limit_type == LimitType.DRAWDOWN:
                # Would need historical data - return 0 for now
                return current_metrics.get('risk_metrics', {}).get('current_drawdown', Decimal("0"))

            elif limit_type == LimitType.VAR:
                return current_metrics.get('risk_metrics', {}).get('var_1d', Decimal("0"))

            elif limit_type == LimitType.CVAR:
                return current_metrics.get('risk_metrics', {}).get('cvar_1d', Decimal("0"))

            elif limit_type == LimitType.CORRELATION:
                # Would need correlation matrix - return 0 for now
                return Decimal("0")

            else:
                logger.warning(f"Unknown limit type: {limit_type}")
                return None

        except Exception as e:
            logger.error(f"Error getting current value for {limit_type}: {e}", exc_info=True)
            return None

    async def _calculate_daily_loss(
        self,
        current_portfolio_value: Decimal
    ) -> Decimal:
        """Calculate daily loss (loss since start of day)"""
        try:
            # Reset daily tracking if new day
            current_date = datetime.now().date()

            if self.daily_loss_reset_date is None or self.daily_loss_reset_date != current_date:
                # New day - reset tracking
                self.daily_loss_start_value = current_portfolio_value
                self.daily_loss_reset_date = current_date
                return Decimal("0")

            # Calculate loss
            if self.daily_loss_start_value is None:
                self.daily_loss_start_value = current_portfolio_value
                return Decimal("0")

            loss = self.daily_loss_start_value - current_portfolio_value

            # Only return positive losses
            return max(Decimal("0"), loss)

        except Exception as e:
            logger.error(f"Error calculating daily loss: {e}", exc_info=True)
            return Decimal("0")

    def _format_limit_message(
        self,
        limit_type: LimitType,
        current_value: Decimal,
        limit_value: Decimal,
        severity: LimitSeverity
    ) -> str:
        """Format a human-readable limit message"""
        try:
            utilization = (current_value / limit_value * Decimal("100")) if limit_value > 0 else Decimal("0")

            if limit_type in [LimitType.CONCENTRATION, LimitType.DRAWDOWN]:
                # Percentage-based limits
                return (
                    f"{limit_type.value}: {current_value:.2%} / {limit_value:.2%} "
                    f"({utilization:.1f}% utilized) - {severity.value.upper()}"
                )
            else:
                # Dollar-based limits
                return (
                    f"{limit_type.value}: ${current_value:,.2f} / ${limit_value:,.2f} "
                    f"({utilization:.1f}% utilized) - {severity.value.upper()}"
                )

        except Exception as e:
            logger.error(f"Error formatting message: {e}", exc_info=True)
            return f"{limit_type.value}: Error formatting message"

    def _compare_severity(
        self,
        severity1: LimitSeverity,
        severity2: LimitSeverity
    ) -> int:
        """
        Compare two severity levels

        Returns:
            -1 if severity1 < severity2
            0 if severity1 == severity2
            1 if severity1 > severity2
        """
        severity_order = {
            LimitSeverity.INFO: 0,
            LimitSeverity.WARNING: 1,
            LimitSeverity.CRITICAL: 2,
            LimitSeverity.HALT: 3
        }

        order1 = severity_order.get(severity1, 0)
        order2 = severity_order.get(severity2, 0)

        if order1 < order2:
            return -1
        elif order1 > order2:
            return 1
        else:
            return 0

    async def get_breach_history(
        self,
        limit_count: int = 10
    ) -> List[Dict]:
        """
        Get recent limit breach history

        Args:
            limit_count: Maximum number of breaches to return

        Returns:
            List of recent breach records
        """
        try:
            # Return most recent breaches
            return self.breach_history[-limit_count:]

        except Exception as e:
            logger.error(f"Error getting breach history: {e}", exc_info=True)
            return []

    async def clear_breach_history(self) -> None:
        """Clear breach history (for testing or reset)"""
        try:
            count = len(self.breach_history)
            self.breach_history.clear()
            logger.info(f"Cleared {count} breach records from history")

        except Exception as e:
            logger.error(f"Error clearing breach history: {e}", exc_info=True)
            raise
