"""
Dynamic Position Sizing
CRITICAL: Calculate position sizes dynamically based on volatility, risk, and market conditions
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class DynamicPositionSizer:
    """Calculate dynamic position sizes based on volatility and risk metrics"""

    def __init__(self):
        """Initialize dynamic position sizer with config"""
        self.config = get_config()
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")
        self.min_position_size_usd = self.config.get_decimal("risk", "position_sizing.min_position_size_usd")
        self.max_position_pct = self.config.get_decimal("risk", "position_sizing.max_position_size_pct")
        self.portfolio_value_usd = self.config.get_decimal("risk", "position_limits.max_portfolio_value_usd")

        logger.info(
            f"DynamicPositionSizer initialized: max_size=${self.max_position_size_usd}, "
            f"min_size=${self.min_position_size_usd}, max_pct={self.max_position_pct}"
        )

    async def calculate_position_size(
        self,
        portfolio_value: Decimal,
        asset_symbol: str,
        current_price: Decimal,
        volatility: Decimal,
        signal_confidence: Optional[Decimal] = None,
        account_risk_pct: Decimal = Decimal("0.02")
    ) -> Dict[str, Decimal]:
        """
        Calculate optimal position size based on multiple factors

        Args:
            portfolio_value: Current total portfolio value in USD
            asset_symbol: Asset symbol
            current_price: Current asset price
            volatility: Asset volatility (standard deviation of returns)
            signal_confidence: Optional signal confidence score (0-1)
            account_risk_pct: Percentage of account to risk per trade (default 2%)

        Returns:
            Dictionary with position sizing details:
            {
                'position_size_usd': Decimal,
                'position_size_units': Decimal,
                'position_pct': Decimal,
                'risk_amount': Decimal,
                'volatility_adjusted': bool
            }
        """
        try:
            # Base position size (account risk method)
            base_size_usd = portfolio_value * account_risk_pct

            # Adjust for volatility (inverse relationship)
            volatility_adjusted_size = await self.adjust_for_volatility(
                base_size_usd, volatility
            )

            # Adjust for signal confidence if provided
            if signal_confidence is not None:
                final_size_usd = await self.scale_by_confidence(
                    volatility_adjusted_size, signal_confidence
                )
            else:
                final_size_usd = volatility_adjusted_size

            # Apply limits
            final_size_usd = self._apply_position_limits(
                final_size_usd, portfolio_value
            )

            # Calculate position size in units
            position_units = (final_size_usd / current_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_HALF_UP
            )

            # Calculate portfolio percentage
            position_pct = ((final_size_usd / portfolio_value) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if portfolio_value > 0 else Decimal("0")

            # Calculate actual risk amount
            risk_amount = (final_size_usd * volatility).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            result = {
                "position_size_usd": final_size_usd.quantize(Decimal("0.01")),
                "position_size_units": position_units,
                "position_pct": position_pct,
                "risk_amount": risk_amount,
                "volatility": volatility,
                "volatility_adjusted": True,
                "confidence_adjusted": signal_confidence is not None,
                "signal_confidence": signal_confidence,
                "asset_symbol": asset_symbol,
                "current_price": current_price,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Dynamic position size for {asset_symbol}: ${final_size_usd:,.2f} "
                f"({position_pct:.2f}%), {position_units:.8f} units, "
                f"risk=${risk_amount:,.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            raise

    async def adjust_for_volatility(
        self,
        base_size: Decimal,
        volatility: Decimal,
        target_volatility: Decimal = Decimal("0.02")
    ) -> Decimal:
        """
        Adjust position size inversely to volatility

        Args:
            base_size: Base position size in USD
            volatility: Asset volatility (std dev of returns)
            target_volatility: Target volatility level (default 2%)

        Returns:
            Volatility-adjusted position size
        """
        try:
            if volatility <= Decimal("0"):
                logger.warning("Volatility <= 0, using base size without adjustment")
                return base_size

            # Inverse volatility scaling: size = base_size * (target_vol / actual_vol)
            adjustment_factor = target_volatility / volatility

            # Cap adjustment factor (don't increase size more than 2x or reduce below 0.25x)
            adjustment_factor = max(Decimal("0.25"), min(Decimal("2.0"), adjustment_factor))

            adjusted_size = (base_size * adjustment_factor).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            logger.debug(
                f"Volatility adjustment: base=${base_size:,.2f}, "
                f"vol={volatility:.4f}, target_vol={target_volatility:.4f}, "
                f"factor={adjustment_factor:.2f}, adjusted=${adjusted_size:,.2f}"
            )

            return adjusted_size

        except Exception as e:
            logger.error(f"Error adjusting for volatility: {e}", exc_info=True)
            raise

    async def scale_by_confidence(
        self,
        position_size: Decimal,
        confidence: Decimal
    ) -> Decimal:
        """
        Scale position size by signal confidence

        Args:
            position_size: Base position size in USD
            confidence: Signal confidence score (0-1)

        Returns:
            Confidence-adjusted position size
        """
        try:
            # Validate confidence range
            if confidence < Decimal("0") or confidence > Decimal("1"):
                logger.warning(f"Confidence {confidence} out of range [0,1], capping")
                confidence = max(Decimal("0"), min(Decimal("1"), confidence))

            # Scale position size linearly with confidence
            # At 0.5 confidence: 50% of base size
            # At 1.0 confidence: 100% of base size
            # At 0.0 confidence: 0% of base size
            scaled_size = (position_size * confidence).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            logger.debug(
                f"Confidence scaling: base=${position_size:,.2f}, "
                f"confidence={confidence:.2f}, scaled=${scaled_size:,.2f}"
            )

            return scaled_size

        except Exception as e:
            logger.error(f"Error scaling by confidence: {e}", exc_info=True)
            raise

    async def calculate_volatility_from_returns(
        self,
        returns_df: pl.DataFrame,
        asset_symbol: str,
        window_days: int = 20
    ) -> Decimal:
        """
        Calculate asset volatility from historical returns

        Args:
            returns_df: DataFrame with columns ['timestamp', 'asset', 'return']
            asset_symbol: Asset symbol to calculate volatility for
            window_days: Rolling window in days (default 20)

        Returns:
            Annualized volatility as Decimal
        """
        try:
            # Filter for asset
            asset_returns = returns_df.filter(pl.col("asset") == asset_symbol)

            if asset_returns.is_empty():
                logger.error(f"No returns data for {asset_symbol}")
                raise ValueError(f"No returns data for {asset_symbol}")

            # Get recent returns (within window)
            cutoff_timestamp = datetime.now().timestamp() - (window_days * 86400)
            recent_returns = asset_returns.filter(
                pl.col("timestamp") >= cutoff_timestamp
            )

            if len(recent_returns) < 5:
                logger.warning(
                    f"Limited returns data for {asset_symbol}: {len(recent_returns)} observations"
                )

            # Calculate standard deviation
            returns_array = recent_returns.select("return").to_numpy().flatten()
            std_dev = float(np.std(returns_array, ddof=1))

            # Annualize volatility (assuming 252 trading days)
            annualized_vol = Decimal(str(std_dev * np.sqrt(252))).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            logger.debug(
                f"Calculated volatility for {asset_symbol}: {annualized_vol:.6f} "
                f"({len(recent_returns)} observations)"
            )

            return annualized_vol

        except Exception as e:
            logger.error(f"Error calculating volatility from returns: {e}", exc_info=True)
            raise

    async def calculate_adaptive_position_size(
        self,
        portfolio_value: Decimal,
        asset_symbol: str,
        current_price: Decimal,
        returns_df: pl.DataFrame,
        market_regime: str = "normal",
        signal_confidence: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate adaptive position size based on market regime and recent volatility

        Args:
            portfolio_value: Current portfolio value
            asset_symbol: Asset symbol
            current_price: Current asset price
            returns_df: Historical returns DataFrame
            market_regime: Market regime ('low_vol', 'normal', 'high_vol', 'extreme')
            signal_confidence: Optional signal confidence

        Returns:
            Dictionary with adaptive position sizing details
        """
        try:
            # Calculate recent volatility
            volatility = await self.calculate_volatility_from_returns(
                returns_df, asset_symbol, window_days=20
            )

            # Adjust account risk based on market regime
            regime_risk_adjustments = {
                "low_vol": Decimal("0.03"),      # 3% risk in low volatility
                "normal": Decimal("0.02"),       # 2% risk in normal conditions
                "high_vol": Decimal("0.01"),     # 1% risk in high volatility
                "extreme": Decimal("0.005")      # 0.5% risk in extreme conditions
            }

            account_risk_pct = regime_risk_adjustments.get(
                market_regime, Decimal("0.02")
            )

            # Calculate position size with adjusted risk
            position_result = await self.calculate_position_size(
                portfolio_value=portfolio_value,
                asset_symbol=asset_symbol,
                current_price=current_price,
                volatility=volatility,
                signal_confidence=signal_confidence,
                account_risk_pct=account_risk_pct
            )

            # Add regime information
            position_result["market_regime"] = market_regime
            position_result["account_risk_pct"] = account_risk_pct

            logger.info(
                f"Adaptive position size for {asset_symbol}: regime={market_regime}, "
                f"risk_pct={account_risk_pct:.2%}, size=${position_result['position_size_usd']:,.2f}"
            )

            return position_result

        except Exception as e:
            logger.error(f"Error calculating adaptive position size: {e}", exc_info=True)
            raise

    async def calculate_portfolio_heat(
        self,
        current_positions: pl.DataFrame,
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate portfolio heat (total risk exposure)

        Args:
            current_positions: DataFrame with ['asset', 'position_size', 'volatility']
            portfolio_value: Total portfolio value

        Returns:
            Dictionary with portfolio heat metrics
        """
        try:
            if current_positions.is_empty():
                return {
                    "total_heat": Decimal("0"),
                    "heat_pct": Decimal("0"),
                    "num_positions": 0,
                    "avg_position_heat": Decimal("0")
                }

            # Calculate risk for each position
            positions_with_risk = current_positions.with_columns(
                (pl.col("position_size") * pl.col("volatility")).alias("position_risk")
            )

            # Sum total risk
            total_risk = positions_with_risk.select("position_risk").sum().item()
            total_risk_decimal = Decimal(str(total_risk)).quantize(Decimal("0.01"))

            # Calculate heat as percentage of portfolio
            heat_pct = ((total_risk_decimal / portfolio_value) * Decimal("100")).quantize(
                Decimal("0.01")
            ) if portfolio_value > 0 else Decimal("0")

            # Average position heat
            num_positions = len(positions_with_risk)
            avg_heat = (total_risk_decimal / Decimal(str(num_positions))).quantize(
                Decimal("0.01")
            ) if num_positions > 0 else Decimal("0")

            result = {
                "total_heat": total_risk_decimal,
                "heat_pct": heat_pct,
                "num_positions": num_positions,
                "avg_position_heat": avg_heat,
                "portfolio_value": portfolio_value,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Portfolio heat: ${total_risk_decimal:,.2f} ({heat_pct:.2f}%), "
                f"{num_positions} positions, avg=${avg_heat:,.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}", exc_info=True)
            raise

    def _apply_position_limits(
        self,
        position_size: Decimal,
        portfolio_value: Decimal
    ) -> Decimal:
        """Apply position size limits from config"""
        # Apply minimum limit
        if position_size < self.min_position_size_usd:
            logger.debug(
                f"Position size ${position_size:,.2f} below minimum ${self.min_position_size_usd:,.2f}, "
                "adjusting to minimum"
            )
            position_size = self.min_position_size_usd

        # Apply maximum USD limit
        if position_size > self.max_position_size_usd:
            logger.debug(
                f"Position size ${position_size:,.2f} exceeds maximum ${self.max_position_size_usd:,.2f}, "
                "capping to maximum"
            )
            position_size = self.max_position_size_usd

        # Apply maximum percentage limit
        max_size_by_pct = portfolio_value * self.max_position_pct
        if position_size > max_size_by_pct:
            logger.debug(
                f"Position size ${position_size:,.2f} exceeds {self.max_position_pct*100:.1f}% "
                f"of portfolio (${max_size_by_pct:,.2f}), capping"
            )
            position_size = max_size_by_pct

        return position_size.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
