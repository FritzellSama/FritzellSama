"""
Sharpe Ratio Calculator
CRITICAL: Comprehensive Sharpe ratio calculations with refinements
"""

import logging
import numpy as np
import polars as pl
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple
from datetime import datetime

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class SharpeRatioCalculator:
    """Calculate Sharpe ratio with various refinements and adjustments"""

    def __init__(self):
        """Initialize Sharpe ratio calculator with config"""
        self.config = get_config()
        self.target_sharpe = self.config.get_decimal("risk", "risk_metrics.target_sharpe_ratio")

        # Default parameters
        self.risk_free_rate = Decimal("0.02")  # 2% annual
        self.trading_days = 252

        logger.info(
            f"SharpeRatioCalculator initialized: target_sharpe={self.target_sharpe}, "
            f"risk_free_rate={self.risk_free_rate:.2%}"
        )

    async def calculate_sharpe(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        annualize: bool = True
    ) -> Decimal:
        """
        Calculate standard Sharpe ratio

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate (default: 2%)
            annualize: Whether to annualize the result

        Returns:
            Sharpe ratio as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate Sharpe: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # Extract returns array
            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for Sharpe ratio")
                return Decimal("0")

            # Calculate daily risk-free rate
            daily_rf = float(risk_free_rate) / self.trading_days

            # Calculate excess returns
            excess_returns = returns_array - daily_rf

            # Calculate mean and std
            mean_excess = np.mean(excess_returns)
            std_excess = np.std(excess_returns, ddof=1)

            if std_excess < 1e-10:
                logger.warning("Standard deviation too low for Sharpe ratio")
                return Decimal("0")

            # Calculate Sharpe ratio
            sharpe = mean_excess / std_excess

            # Annualize if requested
            if annualize:
                sharpe = sharpe * np.sqrt(self.trading_days)

            sharpe_decimal = Decimal(str(sharpe)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(f"Sharpe ratio: {sharpe_decimal:.4f}")

            return sharpe_decimal

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}", exc_info=True)
            raise

    async def rolling_sharpe(
        self,
        returns: pl.DataFrame,
        window_days: int = 252,
        risk_free_rate: Optional[Decimal] = None
    ) -> pl.DataFrame:
        """
        Calculate rolling Sharpe ratio

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            window_days: Rolling window size in days
            risk_free_rate: Annual risk-free rate

        Returns:
            DataFrame with columns ['timestamp', 'rolling_sharpe']
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate rolling Sharpe: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            daily_rf = float(risk_free_rate) / self.trading_days

            # Sort by timestamp
            returns_sorted = returns.sort("timestamp")

            # Calculate excess returns
            returns_with_excess = returns_sorted.with_columns(
                (pl.col("return") - daily_rf).alias("excess_return")
            )

            # Calculate rolling mean and std
            rolling_df = returns_with_excess.with_columns([
                pl.col("excess_return").rolling_mean(window_size=window_days).alias("mean_excess"),
                pl.col("excess_return").rolling_std(window_size=window_days).alias("std_excess")
            ])

            # Calculate rolling Sharpe (annualized)
            rolling_df = rolling_df.with_columns(
                ((pl.col("mean_excess") / pl.col("std_excess")) * np.sqrt(self.trading_days)).alias("rolling_sharpe")
            )

            # Select only timestamp and rolling_sharpe
            result = rolling_df.select(["timestamp", "rolling_sharpe"]).drop_nulls()

            logger.info(
                f"Calculated rolling Sharpe ratio: window={window_days}d, "
                f"samples={len(result)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating rolling Sharpe: {e}", exc_info=True)
            raise

    async def annualized_sharpe(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None,
        compounding: bool = True
    ) -> Decimal:
        """
        Calculate annualized Sharpe ratio with optional compounding

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate
            compounding: Use geometric (compound) returns if True

        Returns:
            Annualized Sharpe ratio as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate annualized Sharpe: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 2:
                logger.warning("Insufficient data for annualized Sharpe")
                return Decimal("0")

            daily_rf = float(risk_free_rate) / self.trading_days

            if compounding:
                # Geometric (compound) return
                cumulative_return = np.prod(1 + returns_array) - 1
                n_days = len(returns_array)
                annualized_return = (1 + cumulative_return) ** (self.trading_days / n_days) - 1

                # Geometric risk-free rate
                annualized_rf = (1 + daily_rf) ** self.trading_days - 1

                # Excess return
                excess_return = annualized_return - annualized_rf

                # Annualized volatility
                annual_vol = np.std(returns_array, ddof=1) * np.sqrt(self.trading_days)

                if annual_vol < 1e-10:
                    return Decimal("0")

                sharpe = excess_return / annual_vol

            else:
                # Arithmetic (simple) return
                mean_return = np.mean(returns_array)
                annualized_return = mean_return * self.trading_days

                excess_return = annualized_return - float(risk_free_rate)

                annual_vol = np.std(returns_array, ddof=1) * np.sqrt(self.trading_days)

                if annual_vol < 1e-10:
                    return Decimal("0")

                sharpe = excess_return / annual_vol

            sharpe_decimal = Decimal(str(sharpe)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Annualized Sharpe ratio: {sharpe_decimal:.4f} "
                f"(compounding={compounding})"
            )

            return sharpe_decimal

        except Exception as e:
            logger.error(f"Error calculating annualized Sharpe: {e}", exc_info=True)
            raise

    async def adjusted_sharpe(
        self,
        returns: pl.DataFrame,
        risk_free_rate: Optional[Decimal] = None
    ) -> Dict[str, Decimal]:
        """
        Calculate Sharpe ratio with adjustments for skewness and kurtosis

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary with standard and adjusted Sharpe ratios
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate adjusted Sharpe: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Calculate standard Sharpe
            standard_sharpe = await self.calculate_sharpe(returns, risk_free_rate)

            returns_array = returns.select("return").to_numpy().flatten()

            if len(returns_array) < 4:
                logger.warning("Insufficient data for adjusted Sharpe")
                return {
                    'standard_sharpe': standard_sharpe,
                    'adjusted_sharpe': standard_sharpe,
                    'skewness': Decimal("0"),
                    'kurtosis': Decimal("0")
                }

            # Calculate moments
            from scipy import stats
            skewness = stats.skew(returns_array)
            kurtosis_excess = stats.kurtosis(returns_array)  # Excess kurtosis (0 for normal)

            # Adjustment factor (Pezier-White adjustment)
            # Adjusted Sharpe = Sharpe * [1 + (skew / 6) * Sharpe - ((kurtosis - 3) / 24) * Sharpe^2]
            sr = float(standard_sharpe)
            adjustment = 1 + (skewness / 6) * sr - (kurtosis_excess / 24) * sr ** 2

            adjusted_sharpe_value = sr * adjustment

            result = {
                'standard_sharpe': standard_sharpe,
                'adjusted_sharpe': Decimal(str(adjusted_sharpe_value)).quantize(Decimal("0.0001")),
                'skewness': Decimal(str(skewness)).quantize(Decimal("0.0001")),
                'kurtosis': Decimal(str(kurtosis_excess + 3)).quantize(Decimal("0.0001")),  # Total kurtosis
                'adjustment_factor': Decimal(str(adjustment)).quantize(Decimal("0.0001"))
            }

            logger.info(
                f"Adjusted Sharpe: standard={standard_sharpe:.4f}, "
                f"adjusted={result['adjusted_sharpe']:.4f}, "
                f"skew={skewness:.4f}, kurtosis={kurtosis_excess + 3:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error calculating adjusted Sharpe: {e}", exc_info=True)
            raise

    async def probabilistic_sharpe_ratio(
        self,
        returns: pl.DataFrame,
        benchmark_sharpe: Decimal,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Probabilistic Sharpe Ratio (PSR)
        PSR is the probability that the Sharpe ratio exceeds a benchmark

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            benchmark_sharpe: Benchmark Sharpe ratio to compare against
            risk_free_rate: Annual risk-free rate

        Returns:
            Probability (0 to 1) as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate PSR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Calculate observed Sharpe ratio
            observed_sharpe = await self.calculate_sharpe(returns, risk_free_rate)

            returns_array = returns.select("return").to_numpy().flatten()
            n = len(returns_array)

            if n < 2:
                return Decimal("0.5")  # 50% probability with no data

            # Calculate skewness and kurtosis
            from scipy import stats
            skewness = stats.skew(returns_array)
            kurtosis_excess = stats.kurtosis(returns_array)

            # Calculate standard error of Sharpe ratio
            # SE = sqrt((1 + 0.5 * SR^2 - skew * SR + (kurtosis - 3) / 4 * SR^2) / (n - 1))
            sr = float(observed_sharpe)
            sr_benchmark = float(benchmark_sharpe)

            variance_term = (
                1 + 0.5 * sr ** 2 - skewness * sr +
                (kurtosis_excess / 4) * sr ** 2
            )

            se_sharpe = np.sqrt(variance_term / (n - 1))

            if se_sharpe < 1e-10:
                return Decimal("1.0") if observed_sharpe >= benchmark_sharpe else Decimal("0.0")

            # Z-score
            z_score = (sr - sr_benchmark) / se_sharpe

            # Calculate probability using normal CDF
            psr = stats.norm.cdf(z_score)

            psr_decimal = Decimal(str(psr)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Probabilistic Sharpe Ratio: {psr_decimal:.4f} "
                f"(observed={observed_sharpe:.4f}, benchmark={benchmark_sharpe:.4f})"
            )

            return psr_decimal

        except Exception as e:
            logger.error(f"Error calculating PSR: {e}", exc_info=True)
            raise

    async def deflated_sharpe_ratio(
        self,
        returns: pl.DataFrame,
        num_trials: int,
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate Deflated Sharpe Ratio (DSR) to account for multiple testing

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            num_trials: Number of strategies tested (for multiple testing adjustment)
            risk_free_rate: Annual risk-free rate

        Returns:
            Deflated Sharpe ratio as Decimal
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate DSR: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Calculate observed Sharpe
            observed_sharpe = await self.calculate_sharpe(returns, risk_free_rate)

            if num_trials <= 1:
                # No multiple testing adjustment needed
                return observed_sharpe

            returns_array = returns.select("return").to_numpy().flatten()
            n = len(returns_array)

            # Calculate expected maximum Sharpe under null hypothesis
            # Using formula from Bailey and López de Prado (2014)
            from scipy import stats

            # Euler-Mascheroni constant
            euler_gamma = 0.5772156649

            # Expected maximum Sharpe ratio
            expected_max_sr = np.sqrt(2 * np.log(num_trials)) - (
                (np.log(np.log(num_trials)) + np.log(4 * np.pi)) /
                (2 * np.sqrt(2 * np.log(num_trials)))
            )

            # Calculate PSR against expected maximum
            psr = await self.probabilistic_sharpe_ratio(
                returns,
                Decimal(str(expected_max_sr)),
                risk_free_rate
            )

            # Deflated Sharpe is essentially the PSR-adjusted Sharpe
            # If PSR is high, DSR ≈ observed Sharpe
            # If PSR is low, DSR is reduced
            dsr = float(observed_sharpe) * float(psr)

            dsr_decimal = Decimal(str(dsr)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            logger.info(
                f"Deflated Sharpe Ratio: {dsr_decimal:.4f} "
                f"(observed={observed_sharpe:.4f}, trials={num_trials}, "
                f"expected_max={expected_max_sr:.4f})"
            )

            return dsr_decimal

        except Exception as e:
            logger.error(f"Error calculating DSR: {e}", exc_info=True)
            raise

    async def sharpe_confidence_interval(
        self,
        returns: pl.DataFrame,
        confidence_level: Decimal = Decimal("0.95"),
        risk_free_rate: Optional[Decimal] = None
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """
        Calculate confidence interval for Sharpe ratio

        Args:
            returns: DataFrame with columns ['timestamp', 'return']
            confidence_level: Confidence level (default: 0.95 for 95%)
            risk_free_rate: Annual risk-free rate

        Returns:
            Tuple of (sharpe_ratio, lower_bound, upper_bound)
        """
        try:
            if returns.is_empty():
                logger.error("Cannot calculate confidence interval: empty returns")
                raise ValueError("Returns DataFrame cannot be empty")

            # Calculate Sharpe ratio
            sharpe = await self.calculate_sharpe(returns, risk_free_rate)

            returns_array = returns.select("return").to_numpy().flatten()
            n = len(returns_array)

            if n < 2:
                return sharpe, sharpe, sharpe

            # Calculate standard error
            from scipy import stats

            # Simplified SE (assuming normal returns)
            se_sharpe = np.sqrt((1 + 0.5 * float(sharpe) ** 2) / n)

            # Z-score for confidence level
            alpha = float(1 - confidence_level)
            z_score = stats.norm.ppf(1 - alpha / 2)

            # Confidence interval
            lower_bound = float(sharpe) - z_score * se_sharpe
            upper_bound = float(sharpe) + z_score * se_sharpe

            lower_decimal = Decimal(str(lower_bound)).quantize(Decimal("0.0001"))
            upper_decimal = Decimal(str(upper_bound)).quantize(Decimal("0.0001"))

            logger.info(
                f"Sharpe ratio {confidence_level:.0%} CI: {sharpe:.4f} "
                f"[{lower_decimal:.4f}, {upper_decimal:.4f}]"
            )

            return sharpe, lower_decimal, upper_decimal

        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}", exc_info=True)
            raise
