"""
Risk Calculator
CRITICAL: Centralized risk calculation engine for all risk metrics
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Centralized engine for calculating all risk metrics"""

    def __init__(self):
        """Initialize risk calculator with config"""
        self.config = get_config()

        # Load risk parameters
        self.max_position_size_usd = self.config.get_decimal("risk", "position_limits.max_position_size_usd")
        self.max_portfolio_value_usd = self.config.get_decimal("risk", "position_limits.max_portfolio_value_usd")
        self.max_concentration_pct = self.config.get_decimal("risk", "position_limits.max_concentration_pct")
        self.max_leverage = self.config.get_decimal("risk", "position_limits.max_leverage")
        self.max_daily_loss_usd = self.config.get_decimal("risk", "loss_limits.max_daily_loss_usd")
        self.max_drawdown_pct = self.config.get_decimal("risk", "loss_limits.max_drawdown_pct")
        self.var_confidence = self.config.get_decimal("risk", "risk_metrics.var_confidence")

        logger.info(
            f"RiskCalculator initialized: max_position=${self.max_position_size_usd:,.0f}, "
            f"max_leverage={self.max_leverage}x, var_confidence={self.var_confidence:.2%}"
        )

    async def calculate_all_metrics(
        self,
        portfolio_data: Dict[str, any],
        market_data: pl.DataFrame,
        historical_returns: pl.DataFrame
    ) -> Dict[str, any]:
        """
        Calculate all risk metrics for portfolio

        Args:
            portfolio_data: Dictionary with:
                {
                    'positions': Dict[str, Decimal],  # symbol -> quantity
                    'prices': Dict[str, Decimal],     # symbol -> current price
                    'portfolio_value': Decimal,
                    'cash': Decimal,
                    'margin_used': Decimal
                }
            market_data: DataFrame with market data
            historical_returns: DataFrame with historical returns

        Returns:
            Dictionary with all risk metrics
        """
        try:
            logger.info("Calculating all risk metrics")

            # Extract portfolio information
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            portfolio_value = portfolio_data.get('portfolio_value', Decimal("0"))
            cash = portfolio_data.get('cash', Decimal("0"))
            margin_used = portfolio_data.get('margin_used', Decimal("0"))

            # Position-level metrics
            position_metrics = await self._calculate_position_metrics(
                positions, prices, portfolio_value
            )

            # Portfolio-level metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(
                portfolio_value, cash, margin_used, positions, prices
            )

            # Risk metrics from historical data
            risk_metrics = await self._calculate_risk_metrics(
                historical_returns, portfolio_value
            )

            # Exposure metrics
            exposure_metrics = await self._calculate_exposure_metrics(
                positions, prices, portfolio_value
            )

            # Combine all metrics
            all_metrics = {
                'timestamp': datetime.now().timestamp(),
                'position_metrics': position_metrics,
                'portfolio_metrics': portfolio_metrics,
                'risk_metrics': risk_metrics,
                'exposure_metrics': exposure_metrics,
                'summary': {
                    'portfolio_value': portfolio_value,
                    'total_positions': len(positions),
                    'cash_pct': (cash / portfolio_value * Decimal("100")) if portfolio_value > 0 else Decimal("0"),
                    'leverage': portfolio_metrics['leverage'],
                    'var_utilization_pct': (risk_metrics.get('var_1d', Decimal("0")) / self.max_daily_loss_usd * Decimal("100")) if self.max_daily_loss_usd > 0 else Decimal("0")
                }
            }

            logger.info(
                f"Risk metrics calculated: portfolio_value=${portfolio_value:,.2f}, "
                f"positions={len(positions)}, leverage={portfolio_metrics['leverage']:.2f}x, "
                f"VaR={risk_metrics.get('var_1d', 0):,.2f}"
            )

            return all_metrics

        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}", exc_info=True)
            raise

    async def get_risk_summary(
        self,
        portfolio_data: Dict[str, any],
        historical_returns: pl.DataFrame
    ) -> Dict[str, any]:
        """
        Get high-level risk summary

        Args:
            portfolio_data: Portfolio data dictionary
            historical_returns: Historical returns DataFrame

        Returns:
            Dictionary with risk summary
        """
        try:
            portfolio_value = portfolio_data.get('portfolio_value', Decimal("0"))
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})

            # Calculate key metrics
            total_exposure = sum(
                abs(qty * prices.get(symbol, Decimal("0")))
                for symbol, qty in positions.items()
            )

            concentration = await self._calculate_concentration(positions, prices, portfolio_value)

            # Calculate VaR
            returns_array = historical_returns.select("return").to_numpy().flatten()
            var_1d = await self._calculate_var_parametric(returns_array, portfolio_value)

            # Calculate volatility
            volatility = float(np.std(returns_array, ddof=1)) * np.sqrt(252)

            summary = {
                'timestamp': datetime.now().timestamp(),
                'portfolio_value': portfolio_value.quantize(Decimal("0.01")),
                'total_exposure': total_exposure.quantize(Decimal("0.01")),
                'num_positions': len(positions),
                'max_concentration_pct': concentration['max_concentration'].quantize(Decimal("0.0001")),
                'var_1d_usd': var_1d.quantize(Decimal("0.01")),
                'var_1d_pct': (var_1d / portfolio_value * Decimal("100")).quantize(Decimal("0.01")) if portfolio_value > 0 else Decimal("0"),
                'annual_volatility': Decimal(str(volatility)).quantize(Decimal("0.0001")),
                'risk_status': await self._determine_risk_status(var_1d, concentration['max_concentration'])
            }

            logger.info(
                f"Risk summary: value=${portfolio_value:,.2f}, VaR=${var_1d:,.2f}, "
                f"vol={volatility:.2%}, status={summary['risk_status']}"
            )

            return summary

        except Exception as e:
            logger.error(f"Error generating risk summary: {e}", exc_info=True)
            raise

    async def evaluate_trade_risk(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        current_portfolio: Dict[str, any],
        action: str = "buy"
    ) -> Dict[str, any]:
        """
        Evaluate risk of a proposed trade

        Args:
            symbol: Asset symbol
            quantity: Trade quantity
            price: Trade price
            current_portfolio: Current portfolio data
            action: 'buy' or 'sell'

        Returns:
            Dictionary with trade risk assessment:
            {
                'approved': bool,
                'risk_score': float (0-100),
                'violations': List[str],
                'warnings': List[str],
                'impact': Dict
            }
        """
        try:
            logger.info(f"Evaluating trade risk: {action.upper()} {quantity} {symbol} @ ${price}")

            violations = []
            warnings = []

            # Extract current portfolio info
            portfolio_value = current_portfolio.get('portfolio_value', Decimal("0"))
            positions = current_portfolio.get('positions', {})
            prices = current_portfolio.get('prices', {})

            # Calculate trade value
            trade_value = quantity * price

            # Check position size limits
            if action == "buy":
                # Check maximum position size
                if trade_value > self.max_position_size_usd:
                    violations.append(
                        f"Trade value ${trade_value:,.2f} exceeds max position size "
                        f"${self.max_position_size_usd:,.2f}"
                    )

                # Calculate new position concentration
                current_position = positions.get(symbol, Decimal("0"))
                new_position_value = (current_position + quantity) * price
                new_concentration = new_position_value / portfolio_value if portfolio_value > 0 else Decimal("1")

                if new_concentration > self.max_concentration_pct:
                    violations.append(
                        f"New concentration {new_concentration:.2%} exceeds max "
                        f"{self.max_concentration_pct:.2%}"
                    )
                elif new_concentration > self.max_concentration_pct * Decimal("0.8"):
                    warnings.append(
                        f"New concentration {new_concentration:.2%} approaching limit "
                        f"{self.max_concentration_pct:.2%}"
                    )

            # Calculate impact on portfolio
            impact = {
                'trade_value': trade_value.quantize(Decimal("0.01")),
                'trade_value_pct': (trade_value / portfolio_value * Decimal("100")).quantize(Decimal("0.01")) if portfolio_value > 0 else Decimal("0"),
                'new_exposure': trade_value if action == "buy" else -trade_value
            }

            # Calculate risk score (0-100, higher = riskier)
            risk_score = 0.0

            # Factor 1: Trade size relative to portfolio (0-40 points)
            size_pct = float(trade_value / portfolio_value) if portfolio_value > 0 else 1.0
            risk_score += min(40.0, size_pct * 100 * 4)

            # Factor 2: Concentration risk (0-30 points)
            if action == "buy":
                concentration_factor = float(new_concentration) / float(self.max_concentration_pct)
                risk_score += min(30.0, concentration_factor * 30)

            # Factor 3: Violations (30 points if any violations)
            if violations:
                risk_score += 30.0

            # Determine approval
            approved = len(violations) == 0

            result = {
                'approved': approved,
                'risk_score': round(risk_score, 2),
                'violations': violations,
                'warnings': warnings,
                'impact': impact,
                'timestamp': datetime.now().timestamp()
            }

            if not approved:
                logger.warning(
                    f"Trade REJECTED: {symbol} - {len(violations)} violations, "
                    f"risk_score={risk_score:.1f}"
                )
                for v in violations:
                    logger.warning(f"  VIOLATION: {v}")
            else:
                logger.info(
                    f"Trade APPROVED: {symbol} - risk_score={risk_score:.1f}, "
                    f"{len(warnings)} warnings"
                )

            return result

        except Exception as e:
            logger.error(f"Error evaluating trade risk: {e}", exc_info=True)
            raise

    async def _calculate_position_metrics(
        self,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, any]:
        """Calculate metrics for individual positions"""
        try:
            position_metrics = {}

            for symbol, quantity in positions.items():
                price = prices.get(symbol, Decimal("0"))
                position_value = quantity * price
                weight = position_value / portfolio_value if portfolio_value > 0 else Decimal("0")

                position_metrics[symbol] = {
                    'quantity': quantity.quantize(Decimal("0.00000001")),
                    'price': price.quantize(Decimal("0.00000001")),
                    'value': position_value.quantize(Decimal("0.01")),
                    'weight': weight.quantize(Decimal("0.000001"))
                }

            return position_metrics

        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}", exc_info=True)
            raise

    async def _calculate_portfolio_metrics(
        self,
        portfolio_value: Decimal,
        cash: Decimal,
        margin_used: Decimal,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """Calculate portfolio-level metrics"""
        try:
            # Calculate total position value
            total_position_value = sum(
                qty * prices.get(symbol, Decimal("0"))
                for symbol, qty in positions.items()
            )

            # Calculate leverage
            leverage = total_position_value / portfolio_value if portfolio_value > 0 else Decimal("0")

            # Calculate cash percentage
            cash_pct = cash / portfolio_value * Decimal("100") if portfolio_value > 0 else Decimal("0")

            return {
                'portfolio_value': portfolio_value.quantize(Decimal("0.01")),
                'cash': cash.quantize(Decimal("0.01")),
                'cash_pct': cash_pct.quantize(Decimal("0.01")),
                'margin_used': margin_used.quantize(Decimal("0.01")),
                'total_position_value': total_position_value.quantize(Decimal("0.01")),
                'leverage': leverage.quantize(Decimal("0.01"))
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}", exc_info=True)
            raise

    async def _calculate_risk_metrics(
        self,
        historical_returns: pl.DataFrame,
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate risk metrics from historical returns"""
        try:
            if historical_returns.is_empty():
                return {
                    'var_1d': Decimal("0"),
                    'cvar_1d': Decimal("0"),
                    'volatility_daily': Decimal("0"),
                    'volatility_annual': Decimal("0")
                }

            returns_array = historical_returns.select("return").to_numpy().flatten()

            # Calculate VaR
            var_1d = await self._calculate_var_parametric(returns_array, portfolio_value)

            # Calculate CVaR (Expected Shortfall)
            var_percentile = float(self.var_confidence)
            var_threshold = np.percentile(returns_array, (1 - var_percentile) * 100)
            cvar_returns = returns_array[returns_array <= var_threshold]
            cvar_1d = abs(float(np.mean(cvar_returns)) * float(portfolio_value)) if len(cvar_returns) > 0 else var_1d

            # Calculate volatility
            daily_vol = float(np.std(returns_array, ddof=1))
            annual_vol = daily_vol * np.sqrt(252)

            return {
                'var_1d': Decimal(str(var_1d)).quantize(Decimal("0.01")),
                'cvar_1d': Decimal(str(cvar_1d)).quantize(Decimal("0.01")),
                'volatility_daily': Decimal(str(daily_vol)).quantize(Decimal("0.000001")),
                'volatility_annual': Decimal(str(annual_vol)).quantize(Decimal("0.0001"))
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}", exc_info=True)
            raise

    async def _calculate_exposure_metrics(
        self,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, any]:
        """Calculate exposure metrics"""
        try:
            # Calculate total exposure (sum of absolute position values)
            total_exposure = sum(
                abs(qty * prices.get(symbol, Decimal("0")))
                for symbol, qty in positions.items()
            )

            # Calculate long/short exposure
            long_exposure = sum(
                qty * prices.get(symbol, Decimal("0"))
                for symbol, qty in positions.items()
                if qty > 0
            )

            short_exposure = abs(sum(
                qty * prices.get(symbol, Decimal("0"))
                for symbol, qty in positions.items()
                if qty < 0
            ))

            # Net exposure
            net_exposure = long_exposure - short_exposure

            return {
                'total_exposure': total_exposure.quantize(Decimal("0.01")),
                'long_exposure': long_exposure.quantize(Decimal("0.01")),
                'short_exposure': short_exposure.quantize(Decimal("0.01")),
                'net_exposure': net_exposure.quantize(Decimal("0.01")),
                'exposure_ratio': (total_exposure / portfolio_value).quantize(Decimal("0.01")) if portfolio_value > 0 else Decimal("0")
            }

        except Exception as e:
            logger.error(f"Error calculating exposure metrics: {e}", exc_info=True)
            raise

    async def _calculate_var_parametric(
        self,
        returns: np.ndarray,
        portfolio_value: Decimal
    ) -> Decimal:
        """Calculate parametric VaR"""
        try:
            if len(returns) < 2:
                return Decimal("0")

            # Calculate mean and std
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            # Calculate VaR using normal distribution
            from scipy import stats
            z_score = stats.norm.ppf(1 - float(self.var_confidence))
            var_return = mean_return + z_score * std_return

            # Convert to dollar amount
            var_usd = abs(var_return * float(portfolio_value))

            return Decimal(str(var_usd))

        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}", exc_info=True)
            return Decimal("0")

    async def _calculate_concentration(
        self,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """Calculate concentration metrics"""
        try:
            if not positions or portfolio_value <= 0:
                return {
                    'max_concentration': Decimal("0"),
                    'herfindahl_index': Decimal("0")
                }

            # Calculate position weights
            weights = []
            max_concentration = Decimal("0")

            for symbol, qty in positions.items():
                price = prices.get(symbol, Decimal("0"))
                position_value = abs(qty * price)
                weight = position_value / portfolio_value
                weights.append(weight)
                max_concentration = max(max_concentration, weight)

            # Calculate Herfindahl-Hirschman Index (sum of squared weights)
            hhi = sum(float(w) ** 2 for w in weights)

            return {
                'max_concentration': max_concentration,
                'herfindahl_index': Decimal(str(hhi)).quantize(Decimal("0.0001"))
            }

        except Exception as e:
            logger.error(f"Error calculating concentration: {e}", exc_info=True)
            raise

    async def _determine_risk_status(
        self,
        var_1d: Decimal,
        max_concentration: Decimal
    ) -> str:
        """Determine overall risk status"""
        try:
            # Check VaR
            var_utilization = var_1d / self.max_daily_loss_usd if self.max_daily_loss_usd > 0 else Decimal("0")

            # Check concentration
            concentration_ratio = max_concentration / self.max_concentration_pct if self.max_concentration_pct > 0 else Decimal("0")

            # Determine status
            if var_utilization >= Decimal("0.9") or concentration_ratio >= Decimal("0.9"):
                return "CRITICAL"
            elif var_utilization >= Decimal("0.7") or concentration_ratio >= Decimal("0.7"):
                return "WARNING"
            elif var_utilization >= Decimal("0.5") or concentration_ratio >= Decimal("0.5"):
                return "ELEVATED"
            else:
                return "NORMAL"

        except Exception as e:
            logger.error(f"Error determining risk status: {e}", exc_info=True)
            return "UNKNOWN"
