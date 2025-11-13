"""
Kelly Criterion Position Sizing
CRITICAL: Calculate optimal position sizes using Kelly Criterion for maximum long-term growth
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class KellyCriterionSizer:
    """Calculate position sizes using Kelly Criterion"""

    def __init__(self):
        """Initialize Kelly Criterion sizer with config"""
        self.config = get_config()
        self.kelly_fraction = self.config.get_decimal("risk", "position_sizing.kelly_fraction")
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")

        logger.info(
            f"KellyCriterionSizer initialized: kelly_fraction={self.kelly_fraction}, "
            f"max_size=${self.max_position_size_usd}, min_size=${self.min_position_size_usd}"
        )

    async def calculate_kelly_fraction(
        self,
        win_rate: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal
    ) -> Decimal:
        """
        Calculate Kelly Criterion fraction

        Formula: f* = (p * b - q) / b
        where:
        - p = win probability
        - q = loss probability (1 - p)
        - b = win/loss ratio (avg_win / avg_loss)

        Args:
            win_rate: Win probability (0-1)
            avg_win: Average win amount (positive)
            avg_loss: Average loss amount (positive)

        Returns:
            Kelly fraction as Decimal
        """
        try:
            # Validate inputs
            if win_rate < Decimal("0") or win_rate > Decimal("1"):
                logger.error(f"Invalid win rate: {win_rate}")
                raise ValueError("Win rate must be between 0 and 1")

            if avg_win <= Decimal("0") or avg_loss <= Decimal("0"):
                logger.error(f"Invalid avg_win ({avg_win}) or avg_loss ({avg_loss})")
                raise ValueError("Average win and loss must be positive")

            # Calculate Kelly Criterion
            loss_rate = Decimal("1") - win_rate
            win_loss_ratio = avg_win / avg_loss

            # Kelly formula: f* = (p * b - q) / b
            kelly = ((win_rate * win_loss_ratio) - loss_rate) / win_loss_ratio

            # Cap Kelly at 0 (no position if negative edge)
            kelly = max(Decimal("0"), kelly)

            # Cap Kelly at 1.0 (don't bet more than 100%)
            kelly = min(Decimal("1.0"), kelly)

            logger.info(
                f"Kelly Criterion: win_rate={win_rate:.4f}, "
                f"win_loss_ratio={win_loss_ratio:.4f}, kelly={kelly:.4f}"
            )

            return kelly.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}", exc_info=True)
            raise

    async def fractional_kelly(
        self,
        full_kelly: Decimal,
        fraction: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate fractional Kelly (conservative Kelly)

        Args:
            full_kelly: Full Kelly Criterion fraction
            fraction: Fraction of Kelly to use (default from config)

        Returns:
            Fractional Kelly as Decimal
        """
        try:
            if fraction is None:
                fraction = self.kelly_fraction

            # Validate fraction
            if fraction <= Decimal("0") or fraction > Decimal("1"):
                logger.error(f"Invalid Kelly fraction: {fraction}")
                raise ValueError("Kelly fraction must be between 0 and 1")

            fractional = (full_kelly * fraction).quantize(Decimal("0.000001"))

            logger.debug(
                f"Fractional Kelly: full={full_kelly:.4f}, "
                f"fraction={fraction:.2f}, result={fractional:.4f}"
            )

            return fractional

        except Exception as e:
            logger.error(f"Error calculating fractional Kelly: {e}", exc_info=True)
            raise

    async def validate_probability(
        self,
        win_rate: Decimal,
        historical_trades: pl.DataFrame
    ) -> Dict[str, any]:
        """
        Validate win rate against historical trades

        Args:
            win_rate: Claimed win rate
            historical_trades: DataFrame with columns ['trade_id', 'pnl']

        Returns:
            Dictionary with validation results:
            {
                'is_valid': bool,
                'actual_win_rate': Decimal,
                'claimed_win_rate': Decimal,
                'difference': Decimal,
                'num_trades': int,
                'confidence_interval': Tuple[Decimal, Decimal]
            }
        """
        try:
            if historical_trades.is_empty():
                logger.warning("No historical trades for validation")
                return {
                    "is_valid": False,
                    "actual_win_rate": Decimal("0"),
                    "claimed_win_rate": win_rate,
                    "difference": win_rate,
                    "num_trades": 0,
                    "confidence_interval": (Decimal("0"), Decimal("0")),
                    "message": "No historical data"
                }

            # Calculate actual win rate
            winning_trades = historical_trades.filter(pl.col("pnl") > 0)
            num_wins = len(winning_trades)
            num_total = len(historical_trades)

            actual_win_rate = Decimal(str(num_wins / num_total)) if num_total > 0 else Decimal("0")

            # Calculate difference
            difference = abs(win_rate - actual_win_rate)

            # Calculate confidence interval (95% using normal approximation)
            if num_total > 30:
                p = float(actual_win_rate)
                se = np.sqrt(p * (1 - p) / num_total)
                margin = Decimal(str(1.96 * se))  # 95% confidence

                ci_lower = max(Decimal("0"), actual_win_rate - margin)
                ci_upper = min(Decimal("1"), actual_win_rate + margin)
            else:
                ci_lower = Decimal("0")
                ci_upper = Decimal("1")

            # Validate: claimed win rate should be within confidence interval
            is_valid = (ci_lower <= win_rate <= ci_upper)

            result = {
                "is_valid": is_valid,
                "actual_win_rate": actual_win_rate.quantize(Decimal("0.0001")),
                "claimed_win_rate": win_rate,
                "difference": difference.quantize(Decimal("0.0001")),
                "num_trades": num_total,
                "confidence_interval": (
                    ci_lower.quantize(Decimal("0.0001")),
                    ci_upper.quantize(Decimal("0.0001"))
                ),
                "timestamp": datetime.now().isoformat()
            }

            if not is_valid:
                logger.warning(
                    f"Win rate validation failed: claimed={win_rate:.4f}, "
                    f"actual={actual_win_rate:.4f}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
                )
            else:
                logger.info(
                    f"Win rate validation passed: claimed={win_rate:.4f}, "
                    f"actual={actual_win_rate:.4f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error validating probability: {e}", exc_info=True)
            raise

    async def calculate_position_size(
        self,
        portfolio_value: Decimal,
        asset_symbol: str,
        current_price: Decimal,
        win_rate: Decimal,
        avg_win_pct: Decimal,
        avg_loss_pct: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate position size using Kelly Criterion

        Args:
            portfolio_value: Current portfolio value
            asset_symbol: Asset symbol
            current_price: Current asset price
            win_rate: Win probability (0-1)
            avg_win_pct: Average win as percentage (e.g., 0.05 for 5%)
            avg_loss_pct: Average loss as percentage (e.g., 0.03 for 3%)

        Returns:
            Dictionary with Kelly-based position sizing
        """
        try:
            # Calculate full Kelly
            full_kelly = await self.calculate_kelly_fraction(
                win_rate, avg_win_pct, avg_loss_pct
            )

            # Apply fractional Kelly
            kelly_fraction = await self.fractional_kelly(full_kelly)

            # Calculate position size
            position_size_usd = (portfolio_value * kelly_fraction).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            # Apply limits
            position_size_usd = self._apply_limits(position_size_usd, portfolio_value)

            # Calculate position size in units
            position_units = (position_size_usd / current_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )

            # Calculate portfolio percentage
            position_pct = ((position_size_usd / portfolio_value) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if portfolio_value > 0 else Decimal("0")

            # Calculate expected value
            expected_value = (win_rate * avg_win_pct - (Decimal("1") - win_rate) * avg_loss_pct).quantize(
                Decimal("0.000001")
            )

            result = {
                "position_size_usd": position_size_usd,
                "position_size_units": position_units,
                "position_pct": position_pct,
                "full_kelly": full_kelly,
                "fractional_kelly": kelly_fraction,
                "kelly_fraction_used": self.kelly_fraction,
                "win_rate": win_rate,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "expected_value": expected_value,
                "has_edge": expected_value > Decimal("0"),
                "asset_symbol": asset_symbol,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat()
            }

            if not result["has_edge"]:
                logger.warning(
                    f"Kelly position for {asset_symbol} has negative edge: EV={expected_value:.6f}"
                )

            logger.info(
                f"Kelly position for {asset_symbol}: ${position_size_usd:,.2f} "
                f"({position_pct:.2f}%), full_kelly={full_kelly:.4f}, "
                f"fractional_kelly={kelly_fraction:.4f}, EV={expected_value:.6f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating Kelly position size: {e}", exc_info=True)
            raise

    async def estimate_parameters_from_trades(
        self,
        historical_trades: pl.DataFrame
    ) -> Dict[str, Decimal]:
        """
        Estimate Kelly parameters from historical trades

        Args:
            historical_trades: DataFrame with columns ['trade_id', 'pnl']

        Returns:
            Dictionary with estimated parameters:
            {
                'win_rate': Decimal,
                'avg_win': Decimal,
                'avg_loss': Decimal,
                'num_trades': int
            }
        """
        try:
            if historical_trades.is_empty():
                logger.error("Cannot estimate parameters: no historical trades")
                raise ValueError("Historical trades DataFrame is empty")

            # Separate winning and losing trades
            winning_trades = historical_trades.filter(pl.col("pnl") > 0)
            losing_trades = historical_trades.filter(pl.col("pnl") < 0)

            # Calculate win rate
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)
            num_total = len(historical_trades)

            win_rate = Decimal(str(num_wins / num_total)) if num_total > 0 else Decimal("0")

            # Calculate average win and loss
            if not winning_trades.is_empty():
                avg_win = Decimal(str(winning_trades.select("pnl").mean().item())).quantize(
                    Decimal("0.000001")
                )
            else:
                avg_win = Decimal("0")

            if not losing_trades.is_empty():
                avg_loss = Decimal(str(abs(losing_trades.select("pnl").mean().item()))).quantize(
                    Decimal("0.000001")
                )
            else:
                avg_loss = Decimal("0")

            result = {
                "win_rate": win_rate.quantize(Decimal("0.0001")),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "num_trades": num_total,
                "num_wins": num_wins,
                "num_losses": num_losses,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Estimated Kelly parameters: win_rate={win_rate:.4f}, "
                f"avg_win={avg_win:.6f}, avg_loss={avg_loss:.6f} "
                f"({num_total} trades)"
            )

            return result

        except Exception as e:
            logger.error(f"Error estimating parameters from trades: {e}", exc_info=True)
            raise

    async def calculate_growth_rate(
        self,
        kelly_fraction: Decimal,
        win_rate: Decimal,
        avg_win_pct: Decimal,
        avg_loss_pct: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate expected geometric growth rate using Kelly fraction

        Args:
            kelly_fraction: Kelly fraction to use
            win_rate: Win probability
            avg_win_pct: Average win percentage
            avg_loss_pct: Average loss percentage

        Returns:
            Dictionary with growth rate metrics
        """
        try:
            # Expected geometric growth rate
            # g = p * ln(1 + f*b) + q * ln(1 - f)
            # where f = kelly_fraction, b = win/loss ratio

            loss_rate = Decimal("1") - win_rate
            win_loss_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else Decimal("1")

            # Calculate growth rate components
            win_component = float(win_rate) * np.log(
                1.0 + float(kelly_fraction * win_loss_ratio)
            )
            loss_component = float(loss_rate) * np.log(
                max(0.01, 1.0 - float(kelly_fraction))  # Prevent log(0)
            )

            growth_rate = Decimal(str(win_component + loss_component)).quantize(
                Decimal("0.000001")
            )

            # Annualized growth rate (assuming 252 trading days)
            annualized_growth = (growth_rate * Decimal("252")).quantize(Decimal("0.000001"))

            result = {
                "growth_rate": growth_rate,
                "annualized_growth": annualized_growth,
                "kelly_fraction": kelly_fraction,
                "win_rate": win_rate,
                "win_loss_ratio": win_loss_ratio.quantize(Decimal("0.0001")),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Growth rate: {growth_rate:.6f} per trade, "
                f"{annualized_growth:.6f} annualized (kelly={kelly_fraction:.4f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating growth rate: {e}", exc_info=True)
            raise

    def _apply_limits(
        self,
        position_size_usd: Decimal,
        portfolio_value: Decimal
    ) -> Decimal:
        """Apply position size limits"""
        # Apply minimum limit
        if position_size_usd < self.min_position_size_usd:
            position_size_usd = self.min_position_size_usd

        # Apply maximum USD limit
        if position_size_usd > self.max_position_size_usd:
            position_size_usd = self.max_position_size_usd

        # Apply maximum percentage limit
        max_size_by_pct = portfolio_value * self.max_position_pct
        if position_size_usd > max_size_by_pct:
            position_size_usd = max_size_by_pct

        return position_size_usd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
