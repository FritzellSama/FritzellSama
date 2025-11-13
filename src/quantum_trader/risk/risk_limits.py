"""
Risk Limits Enforcement for Quantum Trader AI

Production-grade risk limits management with:
- Limit definitions and storage
- Real-time limit checking
- Limit breach alerting
- Temporary limit adjustments
- Limit override workflows
- Multi-level limits (position, portfolio, strategy)
- Historical limit breach tracking

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable
import logging

import polars as pl
import yaml

from quantum_trader.models import Position, Order, AuditLog


logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of risk limits"""
    POSITION_SIZE = "POSITION_SIZE"
    POSITION_CONCENTRATION = "POSITION_CONCENTRATION"
    PORTFOLIO_VAR = "PORTFOLIO_VAR"
    DAILY_LOSS = "DAILY_LOSS"
    LEVERAGE = "LEVERAGE"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    CORRELATION = "CORRELATION"
    LIQUIDITY = "LIQUIDITY"
    VOLATILITY = "VOLATILITY"


class LimitSeverity(Enum):
    """Severity levels for limit breaches"""
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LimitAction(Enum):
    """Actions to take on limit breach"""
    ALERT = "ALERT"
    BLOCK_TRADE = "BLOCK_TRADE"
    REDUCE_POSITION = "REDUCE_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    HALT_TRADING = "HALT_TRADING"


@dataclass
class RiskLimit:
    """Definition of a risk limit"""
    limit_id: str
    limit_type: LimitType
    description: str
    threshold_value: Decimal
    severity: LimitSeverity
    action: LimitAction
    enabled: bool
    applies_to: str  # 'portfolio', 'position', 'strategy', or specific symbol
    created_at: datetime
    modified_at: datetime


@dataclass
class LimitBreach:
    """Record of a limit breach"""
    breach_id: str
    limit: RiskLimit
    current_value: Decimal
    threshold_value: Decimal
    excess_amount: Decimal
    breach_percent: Decimal
    timestamp: datetime
    resolved: bool
    resolution_timestamp: Optional[datetime]
    actions_taken: List[str]


@dataclass
class LimitOverride:
    """Temporary limit override"""
    override_id: str
    limit_id: str
    new_threshold: Decimal
    reason: str
    approved_by: str
    start_time: datetime
    end_time: datetime
    active: bool


class RiskLimitsEnforcer:
    """
    Risk limits enforcement engine

    Manages and enforces risk limits across the trading system:
    - Define and store risk limits
    - Real-time limit checking
    - Breach detection and alerting
    - Automatic remediation actions
    - Override management
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize risk limits enforcer

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.limits: Dict[str, RiskLimit] = {}
        self.breaches: List[LimitBreach] = []
        self.overrides: Dict[str, LimitOverride] = {}
        self.alert_callbacks: List[Callable] = []

        self._load_limits_from_config(config_path, env_config_path)

        logger.info("RiskLimitsEnforcer initialized with %d limits", len(self.limits))

    def _load_limits_from_config(self, config_path: str, env_config_path: str) -> None:
        """
        Load risk limits from configuration files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            # Create limits from config
            now = datetime.now()

            # Portfolio limits
            global_config = risk_config.get('global', {})

            self.add_limit(RiskLimit(
                limit_id='portfolio_max_risk',
                limit_type=LimitType.PORTFOLIO_VAR,
                description='Maximum portfolio risk as percentage of account',
                threshold_value=Decimal(str(global_config.get('max_portfolio_risk_percent', 2.0))),
                severity=LimitSeverity.CRITICAL,
                action=LimitAction.HALT_TRADING,
                enabled=True,
                applies_to='portfolio',
                created_at=now,
                modified_at=now
            ))

            self.add_limit(RiskLimit(
                limit_id='max_daily_loss',
                limit_type=LimitType.DAILY_LOSS,
                description='Maximum daily loss in USD',
                threshold_value=Decimal(str(global_config.get('max_daily_loss_usd', 50000))),
                severity=LimitSeverity.CRITICAL,
                action=LimitAction.HALT_TRADING,
                enabled=True,
                applies_to='portfolio',
                created_at=now,
                modified_at=now
            ))

            self.add_limit(RiskLimit(
                limit_id='max_daily_loss_percent',
                limit_type=LimitType.DAILY_LOSS,
                description='Maximum daily loss as percentage',
                threshold_value=Decimal(str(global_config.get('max_daily_loss_percent', 5.0))),
                severity=LimitSeverity.CRITICAL,
                action=LimitAction.HALT_TRADING,
                enabled=True,
                applies_to='portfolio',
                created_at=now,
                modified_at=now
            ))

            # Position limits
            position_limits = risk_config.get('position_limits', {})

            self.add_limit(RiskLimit(
                limit_id='max_position_size',
                limit_type=LimitType.POSITION_SIZE,
                description='Maximum position size as percentage of account',
                threshold_value=Decimal(str(position_limits.get('max_position_size_percent', 5.0))),
                severity=LimitSeverity.ERROR,
                action=LimitAction.BLOCK_TRADE,
                enabled=True,
                applies_to='position',
                created_at=now,
                modified_at=now
            ))

            self.add_limit(RiskLimit(
                limit_id='max_leverage',
                limit_type=LimitType.LEVERAGE,
                description='Maximum leverage allowed',
                threshold_value=Decimal(str(position_limits.get('max_leverage', 2.0))),
                severity=LimitSeverity.ERROR,
                action=LimitAction.BLOCK_TRADE,
                enabled=True,
                applies_to='portfolio',
                created_at=now,
                modified_at=now
            ))

            # Concentration limits
            concentration = risk_config.get('concentration_limits', {})

            self.add_limit(RiskLimit(
                limit_id='max_single_ticker',
                limit_type=LimitType.POSITION_CONCENTRATION,
                description='Maximum percentage in single ticker',
                threshold_value=Decimal(str(concentration.get('max_single_ticker_percent', 10.0))),
                severity=LimitSeverity.WARNING,
                action=LimitAction.ALERT,
                enabled=True,
                applies_to='position',
                created_at=now,
                modified_at=now
            ))

            self.add_limit(RiskLimit(
                limit_id='max_sector_exposure',
                limit_type=LimitType.SECTOR_EXPOSURE,
                description='Maximum percentage in single sector',
                threshold_value=Decimal(str(concentration.get('max_single_sector_percent', 20.0))),
                severity=LimitSeverity.WARNING,
                action=LimitAction.ALERT,
                enabled=True,
                applies_to='sector',
                created_at=now,
                modified_at=now
            ))

            logger.info("Loaded %d risk limits from configuration", len(self.limits))

        except Exception as e:
            logger.error("Failed to load limits from configuration: %s", e)
            raise

    def add_limit(self, limit: RiskLimit) -> None:
        """
        Add a new risk limit

        Args:
            limit: Risk limit to add
        """
        self.limits[limit.limit_id] = limit

        logger.info(
            "Added risk limit: %s (%s) threshold=%s",
            limit.limit_id, limit.limit_type.value, limit.threshold_value
        )

    def update_limit(
        self,
        limit_id: str,
        new_threshold: Optional[Decimal] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """
        Update an existing risk limit

        Args:
            limit_id: ID of limit to update
            new_threshold: New threshold value
            enabled: Enable/disable limit
        """
        if limit_id not in self.limits:
            raise ValueError(f"Limit {limit_id} not found")

        limit = self.limits[limit_id]

        if new_threshold is not None:
            limit.threshold_value = new_threshold

        if enabled is not None:
            limit.enabled = enabled

        limit.modified_at = datetime.now()

        logger.info("Updated limit %s: threshold=%s, enabled=%s", limit_id, new_threshold, enabled)

    def check_position_size_limit(
        self,
        symbol: str,
        proposed_quantity: Decimal,
        current_price: Decimal,
        portfolio_value: Decimal
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """
        Check if position size is within limits

        Args:
            symbol: Asset symbol
            proposed_quantity: Proposed position quantity
            current_price: Current price
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (is_within_limits, breach_if_any)
        """
        if not isinstance(proposed_quantity, Decimal):
            raise TypeError(f"proposed_quantity must be Decimal")

        if not isinstance(current_price, Decimal):
            raise TypeError(f"current_price must be Decimal")

        if not isinstance(portfolio_value, Decimal):
            raise TypeError(f"portfolio_value must be Decimal")

        # Find position size limit
        limit = self.limits.get('max_position_size')

        if not limit or not limit.enabled:
            return True, None

        # Calculate position value
        position_value = abs(proposed_quantity) * current_price

        if portfolio_value <= Decimal('0'):
            return False, None

        # Calculate position size as percentage
        position_percent = (position_value / portfolio_value) * Decimal('100')

        # Check against limit
        if position_percent > limit.threshold_value:
            breach = LimitBreach(
                breach_id=f"breach_{limit.limit_id}_{symbol}_{datetime.now().timestamp()}",
                limit=limit,
                current_value=position_percent,
                threshold_value=limit.threshold_value,
                excess_amount=position_percent - limit.threshold_value,
                breach_percent=((position_percent - limit.threshold_value) / limit.threshold_value) * Decimal('100'),
                timestamp=datetime.now(),
                resolved=False,
                resolution_timestamp=None,
                actions_taken=[]
            )

            self.breaches.append(breach)

            logger.warning(
                "Position size limit breached for %s: %s%% exceeds %s%%",
                symbol, position_percent, limit.threshold_value
            )

            return False, breach

        return True, None

    def check_daily_loss_limit(
        self,
        current_daily_pnl: Decimal,
        portfolio_value: Decimal
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """
        Check if daily loss is within limits

        Args:
            current_daily_pnl: Current day P&L (negative = loss)
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (is_within_limits, breach_if_any)
        """
        if not isinstance(current_daily_pnl, Decimal):
            raise TypeError(f"current_daily_pnl must be Decimal")

        # Check absolute loss limit
        absolute_limit = self.limits.get('max_daily_loss')

        if absolute_limit and absolute_limit.enabled:
            if current_daily_pnl < -absolute_limit.threshold_value:
                breach = LimitBreach(
                    breach_id=f"breach_daily_loss_abs_{datetime.now().timestamp()}",
                    limit=absolute_limit,
                    current_value=abs(current_daily_pnl),
                    threshold_value=absolute_limit.threshold_value,
                    excess_amount=abs(current_daily_pnl) - absolute_limit.threshold_value,
                    breach_percent=((abs(current_daily_pnl) - absolute_limit.threshold_value) / absolute_limit.threshold_value) * Decimal('100'),
                    timestamp=datetime.now(),
                    resolved=False,
                    resolution_timestamp=None,
                    actions_taken=[]
                )

                self.breaches.append(breach)

                logger.critical(
                    "Daily loss limit breached: $%s exceeds $%s",
                    abs(current_daily_pnl), absolute_limit.threshold_value
                )

                return False, breach

        # Check percentage loss limit
        percent_limit = self.limits.get('max_daily_loss_percent')

        if percent_limit and percent_limit.enabled and portfolio_value > Decimal('0'):
            loss_percent = (abs(current_daily_pnl) / portfolio_value) * Decimal('100')

            if loss_percent > percent_limit.threshold_value:
                breach = LimitBreach(
                    breach_id=f"breach_daily_loss_pct_{datetime.now().timestamp()}",
                    limit=percent_limit,
                    current_value=loss_percent,
                    threshold_value=percent_limit.threshold_value,
                    excess_amount=loss_percent - percent_limit.threshold_value,
                    breach_percent=((loss_percent - percent_limit.threshold_value) / percent_limit.threshold_value) * Decimal('100'),
                    timestamp=datetime.now(),
                    resolved=False,
                    resolution_timestamp=None,
                    actions_taken=[]
                )

                self.breaches.append(breach)

                logger.critical(
                    "Daily loss percentage limit breached: %s%% exceeds %s%%",
                    loss_percent, percent_limit.threshold_value
                )

                return False, breach

        return True, None

    def check_leverage_limit(
        self,
        total_exposure: Decimal,
        portfolio_value: Decimal
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """
        Check if leverage is within limits

        Args:
            total_exposure: Total position exposure
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (is_within_limits, breach_if_any)
        """
        if not isinstance(total_exposure, Decimal):
            raise TypeError(f"total_exposure must be Decimal")

        limit = self.limits.get('max_leverage')

        if not limit or not limit.enabled:
            return True, None

        if portfolio_value <= Decimal('0'):
            return False, None

        # Calculate leverage
        leverage = abs(total_exposure) / portfolio_value

        if leverage > limit.threshold_value:
            breach = LimitBreach(
                breach_id=f"breach_leverage_{datetime.now().timestamp()}",
                limit=limit,
                current_value=leverage,
                threshold_value=limit.threshold_value,
                excess_amount=leverage - limit.threshold_value,
                breach_percent=((leverage - limit.threshold_value) / limit.threshold_value) * Decimal('100'),
                timestamp=datetime.now(),
                resolved=False,
                resolution_timestamp=None,
                actions_taken=[]
            )

            self.breaches.append(breach)

            logger.error(
                "Leverage limit breached: %sx exceeds %sx",
                leverage, limit.threshold_value
            )

            return False, breach

        return True, None

    def check_concentration_limit(
        self,
        symbol: str,
        position_value: Decimal,
        portfolio_value: Decimal
    ) -> Tuple[bool, Optional[LimitBreach]]:
        """
        Check if position concentration is within limits

        Args:
            symbol: Asset symbol
            position_value: Position value
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (is_within_limits, breach_if_any)
        """
        limit = self.limits.get('max_single_ticker')

        if not limit or not limit.enabled:
            return True, None

        if portfolio_value <= Decimal('0'):
            return False, None

        concentration_percent = (position_value / portfolio_value) * Decimal('100')

        if concentration_percent > limit.threshold_value:
            breach = LimitBreach(
                breach_id=f"breach_concentration_{symbol}_{datetime.now().timestamp()}",
                limit=limit,
                current_value=concentration_percent,
                threshold_value=limit.threshold_value,
                excess_amount=concentration_percent - limit.threshold_value,
                breach_percent=((concentration_percent - limit.threshold_value) / limit.threshold_value) * Decimal('100'),
                timestamp=datetime.now(),
                resolved=False,
                resolution_timestamp=None,
                actions_taken=[]
            )

            self.breaches.append(breach)

            logger.warning(
                "Concentration limit breached for %s: %s%% exceeds %s%%",
                symbol, concentration_percent, limit.threshold_value
            )

            return False, breach

        return True, None

    def create_override(
        self,
        limit_id: str,
        new_threshold: Decimal,
        duration_hours: int,
        reason: str,
        approved_by: str
    ) -> LimitOverride:
        """
        Create temporary limit override

        Args:
            limit_id: ID of limit to override
            new_threshold: New temporary threshold
            duration_hours: Duration in hours
            reason: Reason for override
            approved_by: Who approved the override

        Returns:
            Created override
        """
        if limit_id not in self.limits:
            raise ValueError(f"Limit {limit_id} not found")

        now = datetime.now()
        end_time = now + timedelta(hours=duration_hours)

        override = LimitOverride(
            override_id=f"override_{limit_id}_{now.timestamp()}",
            limit_id=limit_id,
            new_threshold=new_threshold,
            reason=reason,
            approved_by=approved_by,
            start_time=now,
            end_time=end_time,
            active=True
        )

        self.overrides[override.override_id] = override

        logger.warning(
            "Limit override created: %s, new threshold=%s, duration=%dh, reason=%s, approved_by=%s",
            limit_id, new_threshold, duration_hours, reason, approved_by
        )

        return override

    def get_effective_threshold(self, limit_id: str) -> Decimal:
        """
        Get effective threshold considering active overrides

        Args:
            limit_id: Limit ID

        Returns:
            Effective threshold value
        """
        if limit_id not in self.limits:
            raise ValueError(f"Limit {limit_id} not found")

        base_threshold = self.limits[limit_id].threshold_value
        now = datetime.now()

        # Check for active overrides
        for override in self.overrides.values():
            if (override.limit_id == limit_id and
                override.active and
                override.start_time <= now <= override.end_time):
                return override.new_threshold

        return base_threshold

    def expire_overrides(self) -> int:
        """
        Expire overrides that have passed their end time

        Returns:
            Number of expired overrides
        """
        now = datetime.now()
        expired_count = 0

        for override in self.overrides.values():
            if override.active and now > override.end_time:
                override.active = False
                expired_count += 1

                logger.info("Override expired: %s", override.override_id)

        return expired_count

    def resolve_breach(self, breach_id: str, actions_taken: List[str]) -> None:
        """
        Mark a breach as resolved

        Args:
            breach_id: Breach ID
            actions_taken: List of actions taken to resolve
        """
        for breach in self.breaches:
            if breach.breach_id == breach_id and not breach.resolved:
                breach.resolved = True
                breach.resolution_timestamp = datetime.now()
                breach.actions_taken = actions_taken

                logger.info("Breach resolved: %s, actions: %s", breach_id, actions_taken)
                return

        logger.warning("Breach %s not found or already resolved", breach_id)

    def get_active_breaches(self) -> List[LimitBreach]:
        """
        Get all active (unresolved) breaches

        Returns:
            List of active breaches
        """
        return [b for b in self.breaches if not b.resolved]

    def get_breach_history(
        self,
        limit_type: Optional[LimitType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[LimitBreach]:
        """
        Get breach history with optional filters

        Args:
            limit_type: Filter by limit type
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            Filtered list of breaches
        """
        breaches = self.breaches

        if limit_type is not None:
            breaches = [b for b in breaches if b.limit.limit_type == limit_type]

        if start_time is not None:
            breaches = [b for b in breaches if b.timestamp >= start_time]

        if end_time is not None:
            breaches = [b for b in breaches if b.timestamp <= end_time]

        return breaches

    def export_breach_report(self) -> pl.DataFrame:
        """
        Export breach history as DataFrame

        Returns:
            DataFrame with breach data
        """
        if not self.breaches:
            return pl.DataFrame()

        data = {
            'breach_id': [b.breach_id for b in self.breaches],
            'limit_type': [b.limit.limit_type.value for b in self.breaches],
            'limit_description': [b.limit.description for b in self.breaches],
            'threshold': [float(b.threshold_value) for b in self.breaches],
            'actual_value': [float(b.current_value) for b in self.breaches],
            'excess_amount': [float(b.excess_amount) for b in self.breaches],
            'breach_percent': [float(b.breach_percent) for b in self.breaches],
            'severity': [b.limit.severity.value for b in self.breaches],
            'timestamp': [b.timestamp for b in self.breaches],
            'resolved': [b.resolved for b in self.breaches],
            'actions_taken': [';'.join(b.actions_taken) for b in self.breaches]
        }

        return pl.DataFrame(data)

    def get_limits_summary(self) -> pl.DataFrame:
        """
        Get summary of all limits

        Returns:
            DataFrame with limits summary
        """
        if not self.limits:
            return pl.DataFrame()

        data = {
            'limit_id': [l.limit_id for l in self.limits.values()],
            'limit_type': [l.limit_type.value for l in self.limits.values()],
            'description': [l.description for l in self.limits.values()],
            'threshold': [float(l.threshold_value) for l in self.limits.values()],
            'severity': [l.severity.value for l in self.limits.values()],
            'action': [l.action.value for l in self.limits.values()],
            'enabled': [l.enabled for l in self.limits.values()],
            'applies_to': [l.applies_to for l in self.limits.values()]
        }

        return pl.DataFrame(data)

    def register_alert_callback(self, callback: Callable[[LimitBreach], None]) -> None:
        """
        Register callback function to be called on limit breach

        Args:
            callback: Function to call with breach details
        """
        self.alert_callbacks.append(callback)

        logger.info("Registered alert callback")

    def _trigger_alerts(self, breach: LimitBreach) -> None:
        """
        Trigger all registered alert callbacks

        Args:
            breach: Breach that occurred
        """
        for callback in self.alert_callbacks:
            try:
                callback(breach)
            except Exception as e:
                logger.error("Error in alert callback: %s", e)

    def create_audit_log(self, breach: LimitBreach) -> AuditLog:
        """
        Create audit log entry for breach

        Args:
            breach: Breach to log

        Returns:
            Audit log entry
        """
        return AuditLog(
            timestamp=breach.timestamp,
            operation='RISK_LIMIT_BREACH',
            user_id='system',
            component='risk_limits',
            severity=breach.limit.severity.value,
            details={
                'breach_id': breach.breach_id,
                'limit_id': breach.limit.limit_id,
                'limit_type': breach.limit.limit_type.value,
                'threshold': float(breach.threshold_value),
                'actual_value': float(breach.current_value),
                'excess_amount': float(breach.excess_amount),
                'action': breach.limit.action.value
            },
            result='BREACHED'
        )
