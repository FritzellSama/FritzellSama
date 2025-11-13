"""
Fixed Fractional Position Sizing
CRITICAL: Calculate position sizes using fixed fractional risk method
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class FixedFractionalSizer:
    """Calculate position sizes using fixed fractional method"""

    def __init__(self):
        """Initialize fixed fractional sizer with config"""
        self.config = get_config()
        self.risk_fraction = self.config.get_decimal("risk", "position_sizing.fixed_fractional_pct")
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")

        logger.info(
            f"FixedFractionalSizer initialized: risk_fraction={self.risk_fraction}, "
            f"max_size=${self.max_position_size_usd}, min_size=${self.min_position_size_usd}"
        )

    async def calculate_position_size(
        self,
        portfolio_value: Decimal,
        asset_symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        stop_loss_pct: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate position size using fixed fractional method

        Args:
            portfolio_value: Current total portfolio value in USD
            asset_symbol: Asset symbol
            entry_price: Planned entry price
            stop_loss_price: Optional stop loss price
            stop_loss_pct: Optional stop loss as percentage (e.g., 0.05 for 5%)

        Returns:
            Dictionary with position sizing details:
            {
                'position_size_usd': Decimal,
                'position_size_units': Decimal,
                'position_pct': Decimal,
                'risk_amount': Decimal,
                'stop_loss_price': Decimal,
                'risk_per_unit': Decimal
            }
        """
        try:
            # Calculate risk amount (fraction of portfolio)
            risk_amount = (portfolio_value * self.risk_fraction).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            # Determine stop loss
            if stop_loss_price is not None:
                calculated_stop_loss = stop_loss_price
            elif stop_loss_pct is not None:
                # Calculate stop loss from percentage
                calculated_stop_loss = (entry_price * (Decimal("1") - stop_loss_pct)).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
            else:
                # Default stop loss (5% for long positions)
                default_stop_pct = Decimal("0.05")
                calculated_stop_loss = (entry_price * (Decimal("1") - default_stop_pct)).quantize(
                    Decimal("0.00000001"), rounding=ROUND_HALF_UP
                )
                logger.warning(
                    f"No stop loss provided for {asset_symbol}, using default {default_stop_pct*100:.0f}%"
                )

            # Calculate risk per unit
            risk_per_unit = abs(entry_price - calculated_stop_loss)

            if risk_per_unit == Decimal("0"):
                logger.error("Risk per unit is zero - stop loss equals entry price")
                raise ValueError("Invalid stop loss: risk per unit cannot be zero")

            # Calculate position size in units: risk_amount / risk_per_unit
            position_units = (risk_amount / risk_per_unit).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )

            # Calculate position size in USD
            position_size_usd = (position_units * entry_price).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            # Apply limits
            position_size_usd, position_units = self._apply_limits(
                position_size_usd, position_units, entry_price, portfolio_value
            )

            # Recalculate actual risk after applying limits
            actual_risk = (position_units * risk_per_unit).quantize(Decimal("0.01"))

            # Calculate portfolio percentage
            position_pct = ((position_size_usd / portfolio_value) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if portfolio_value > 0 else Decimal("0")

            result = {
                "position_size_usd": position_size_usd,
                "position_size_units": position_units,
                "position_pct": position_pct,
                "risk_amount": actual_risk,
                "planned_risk_amount": risk_amount,
                "stop_loss_price": calculated_stop_loss,
                "entry_price": entry_price,
                "risk_per_unit": risk_per_unit,
                "risk_fraction": self.risk_fraction,
                "asset_symbol": asset_symbol,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Fixed fractional position for {asset_symbol}: ${position_size_usd:,.2f} "
                f"({position_pct:.2f}%), {position_units:.8f} units, "
                f"risk=${actual_risk:,.2f}, stop=${calculated_stop_loss:.8f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating fixed fractional position size: {e}", exc_info=True)
            raise

    async def set_risk_fraction(self, new_fraction: Decimal) -> None:
        """
        Update risk fraction

        Args:
            new_fraction: New risk fraction (e.g., 0.02 for 2%)
        """
        try:
            if new_fraction <= Decimal("0") or new_fraction > Decimal("0.1"):
                logger.error(f"Invalid risk fraction: {new_fraction}")
                raise ValueError("Risk fraction must be between 0 and 0.1 (10%)")

            old_fraction = self.risk_fraction
            self.risk_fraction = new_fraction

            logger.info(f"Risk fraction updated: {old_fraction} -> {new_fraction}")

        except Exception as e:
            logger.error(f"Error setting risk fraction: {e}", exc_info=True)
            raise

    async def validate_size(
        self,
        position_size_usd: Decimal,
        portfolio_value: Decimal,
        entry_price: Decimal,
        stop_loss_price: Decimal
    ) -> Dict[str, any]:
        """
        Validate a proposed position size

        Args:
            position_size_usd: Proposed position size in USD
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'violations': List[str],
                'actual_risk_pct': Decimal,
                'position_pct': Decimal
            }
        """
        try:
            violations = []

            # Check minimum size
            if position_size_usd < self.min_position_size_usd:
                violations.append(
                    f"Position size ${position_size_usd:,.2f} below minimum ${self.min_position_size_usd:,.2f}"
                )

            # Check maximum USD size
            if position_size_usd > self.max_position_size_usd:
                violations.append(
                    f"Position size ${position_size_usd:,.2f} exceeds maximum ${self.max_position_size_usd:,.2f}"
                )

            # Check maximum percentage
            position_pct = (position_size_usd / portfolio_value) if portfolio_value > 0 else Decimal("0")
            if position_pct > self.max_position_pct:
                violations.append(
                    f"Position {position_pct*100:.2f}% exceeds maximum {self.max_position_pct*100:.2f}%"
                )

            # Calculate actual risk
            position_units = position_size_usd / entry_price if entry_price > 0 else Decimal("0")
            risk_per_unit = abs(entry_price - stop_loss_price)
            actual_risk = position_units * risk_per_unit
            actual_risk_pct = (actual_risk / portfolio_value) if portfolio_value > 0 else Decimal("0")

            # Check if risk exceeds target
            if actual_risk_pct > self.risk_fraction * Decimal("1.5"):  # Allow 50% over-risk
                violations.append(
                    f"Actual risk {actual_risk_pct*100:.2f}% significantly exceeds "
                    f"target {self.risk_fraction*100:.2f}%"
                )

            is_valid = len(violations) == 0

            result = {
                "is_valid": is_valid,
                "violations": violations,
                "actual_risk_pct": actual_risk_pct.quantize(Decimal("0.0001")),
                "target_risk_pct": self.risk_fraction,
                "position_pct": position_pct.quantize(Decimal("0.0001")),
                "position_size_usd": position_size_usd,
                "timestamp": datetime.now().isoformat()
            }

            if not is_valid:
                logger.warning(
                    f"Position validation failed: {len(violations)} violations - {violations}"
                )
            else:
                logger.debug(f"Position validation passed: ${position_size_usd:,.2f}")

            return result

        except Exception as e:
            logger.error(f"Error validating position size: {e}", exc_info=True)
            raise

    async def calculate_risk_reward_ratio(
        self,
        entry_price: Decimal,
        stop_loss_price: Decimal,
        target_price: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate risk-reward ratio for a trade

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            target_price: Profit target price

        Returns:
            Dictionary with risk-reward metrics
        """
        try:
            # Calculate risk (distance to stop)
            risk = abs(entry_price - stop_loss_price)

            # Calculate reward (distance to target)
            reward = abs(target_price - entry_price)

            # Calculate ratio
            if risk > Decimal("0"):
                risk_reward_ratio = (reward / risk).quantize(Decimal("0.01"))
            else:
                logger.error("Risk is zero, cannot calculate risk-reward ratio")
                raise ValueError("Risk cannot be zero")

            # Calculate expected value (assume 50% win rate for simplicity)
            win_rate = Decimal("0.5")
            expected_value = (win_rate * reward - (Decimal("1") - win_rate) * risk).quantize(
                Decimal("0.00000001")
            )

            result = {
                "risk": risk.quantize(Decimal("0.00000001")),
                "reward": reward.quantize(Decimal("0.00000001")),
                "risk_reward_ratio": risk_reward_ratio,
                "expected_value": expected_value,
                "is_favorable": risk_reward_ratio >= Decimal("2.0"),  # Require at least 2:1
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "target_price": target_price,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Risk-reward: {risk_reward_ratio:.2f}:1, "
                f"risk={risk:.8f}, reward={reward:.8f}, favorable={result['is_favorable']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {e}", exc_info=True)
            raise

    def _apply_limits(
        self,
        position_size_usd: Decimal,
        position_units: Decimal,
        entry_price: Decimal,
        portfolio_value: Decimal
    ) -> tuple[Decimal, Decimal]:
        """Apply position size limits and return adjusted values"""
        original_size = position_size_usd

        # Apply minimum limit
        if position_size_usd < self.min_position_size_usd:
            position_size_usd = self.min_position_size_usd
            position_units = (position_size_usd / entry_price).quantize(Decimal("0.00000001"))

        # Apply maximum USD limit
        if position_size_usd > self.max_position_size_usd:
            position_size_usd = self.max_position_size_usd
            position_units = (position_size_usd / entry_price).quantize(Decimal("0.00000001"))

        # Apply maximum percentage limit
        max_size_by_pct = portfolio_value * self.max_position_pct
        if position_size_usd > max_size_by_pct:
            position_size_usd = max_size_by_pct
            position_units = (position_size_usd / entry_price).quantize(Decimal("0.00000001"))

        if position_size_usd != original_size:
            logger.debug(
                f"Position size adjusted: ${original_size:,.2f} -> ${position_size_usd:,.2f}"
            )

        return (
            position_size_usd.quantize(Decimal("0.01")),
            position_units
        )
