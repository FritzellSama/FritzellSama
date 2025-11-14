"""Mathematical Utility Functions.

Production-ready mathematical utilities for trading calculations,
statistical analysis, and financial mathematics using Decimal for precision.
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from typing import List, Union, Optional, Tuple
import math
from structlog import get_logger

logger = get_logger(__name__)


def calculate_percentage(
    part: Union[Decimal, float, int],
    whole: Union[Decimal, float, int]
) -> Decimal:
    """Calculate percentage.

    Args:
        part: Part value
        whole: Whole value

    Returns:
        Percentage as decimal (e.g., 0.25 for 25%)

    Raises:
        ValueError: If whole is zero
    """
    try:
        whole_dec = Decimal(str(whole))
        if whole_dec == Decimal("0"):
            raise ValueError("Cannot calculate percentage of zero")

        part_dec = Decimal(str(part))
        return part_dec / whole_dec

    except Exception as e:
        logger.error("percentage_calc_failed", part=part, whole=whole, error=str(e))
        raise


def calculate_return(
    initial: Union[Decimal, float, int],
    final: Union[Decimal, float, int]
) -> Decimal:
    """Calculate return percentage.

    Args:
        initial: Initial value
        final: Final value

    Returns:
        Return as decimal

    Raises:
        ValueError: If initial is zero
    """
    try:
        initial_dec = Decimal(str(initial))
        if initial_dec == Decimal("0"):
            raise ValueError("Cannot calculate return from zero")

        final_dec = Decimal(str(final))
        return (final_dec - initial_dec) / initial_dec

    except Exception as e:
        logger.error("return_calc_failed", initial=initial, final=final, error=str(e))
        raise


def compound_return(
    returns: List[Union[Decimal, float]]
) -> Decimal:
    """Calculate compound return from series of returns.

    Args:
        returns: List of period returns (as decimals)

    Returns:
        Compound return

    Raises:
        ValueError: If returns invalid
    """
    try:
        result = Decimal("1")

        for ret in returns:
            ret_dec = Decimal(str(ret))
            result *= (Decimal("1") + ret_dec)

        return result - Decimal("1")

    except Exception as e:
        logger.error("compound_return_failed", error=str(e))
        raise


def annualized_return(
    total_return: Union[Decimal, float],
    days: int
) -> Decimal:
    """Calculate annualized return.

    Args:
        total_return: Total return (as decimal)
        days: Number of days

    Returns:
        Annualized return

    Raises:
        ValueError: If days invalid
    """
    try:
        if days <= 0:
            raise ValueError("Days must be positive")

        total_ret = Decimal(str(total_return))
        years = Decimal(str(days)) / Decimal("365")

        # (1 + total_return) ^ (1/years) - 1
        base = float(Decimal("1") + total_ret)
        exponent = float(Decimal("1") / years)
        annualized = Decimal(str(base ** exponent)) - Decimal("1")

        return annualized

    except Exception as e:
        logger.error("annualized_return_failed", days=days, error=str(e))
        raise


def sharpe_ratio(
    returns: List[Union[Decimal, float]],
    risk_free_rate: Union[Decimal, float] = 0
) -> Decimal:
    """Calculate Sharpe ratio.

    Args:
        returns: List of period returns
        risk_free_rate: Risk-free rate (same period as returns)

    Returns:
        Sharpe ratio

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for Sharpe ratio")

        # Convert to Decimal
        returns_dec = [Decimal(str(r)) for r in returns]
        rfr = Decimal(str(risk_free_rate))

        # Calculate excess returns
        excess_returns = [r - rfr for r in returns_dec]

        # Calculate mean and std
        mean_excess = sum(excess_returns) / Decimal(str(len(excess_returns)))
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / Decimal(str(len(excess_returns)))
        std_dev = Decimal(str(math.sqrt(float(variance))))

        if std_dev == Decimal("0"):
            return Decimal("0")

        sharpe = mean_excess / std_dev

        return sharpe

    except Exception as e:
        logger.error("sharpe_ratio_failed", error=str(e))
        raise


def sortino_ratio(
    returns: List[Union[Decimal, float]],
    target_return: Union[Decimal, float] = 0
) -> Decimal:
    """Calculate Sortino ratio (downside deviation).

    Args:
        returns: List of period returns
        target_return: Target/minimum acceptable return

    Returns:
        Sortino ratio

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for Sortino ratio")

        returns_dec = [Decimal(str(r)) for r in returns]
        target = Decimal(str(target_return))

        # Calculate excess returns
        excess_returns = [r - target for r in returns_dec]
        mean_excess = sum(excess_returns) / Decimal(str(len(excess_returns)))

        # Calculate downside deviation
        downside_returns = [min(r - target, Decimal("0")) for r in returns_dec]
        downside_variance = sum(r ** 2 for r in downside_returns) / Decimal(str(len(downside_returns)))
        downside_dev = Decimal(str(math.sqrt(float(downside_variance))))

        if downside_dev == Decimal("0"):
            return Decimal("0")

        sortino = mean_excess / downside_dev

        return sortino

    except Exception as e:
        logger.error("sortino_ratio_failed", error=str(e))
        raise


def max_drawdown(
    values: List[Union[Decimal, float]]
) -> Tuple[Decimal, int, int]:
    """Calculate maximum drawdown.

    Args:
        values: List of portfolio values

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(values) < 2:
            raise ValueError("Need at least 2 values for drawdown")

        values_dec = [Decimal(str(v)) for v in values]

        max_dd = Decimal("0")
        peak_idx = 0
        trough_idx = 0
        running_max = values_dec[0]
        running_max_idx = 0

        for i, value in enumerate(values_dec):
            if value > running_max:
                running_max = value
                running_max_idx = i

            drawdown = (running_max - value) / running_max if running_max > 0 else Decimal("0")

            if drawdown > max_dd:
                max_dd = drawdown
                peak_idx = running_max_idx
                trough_idx = i

        return max_dd, peak_idx, trough_idx

    except Exception as e:
        logger.error("max_drawdown_failed", error=str(e))
        raise


def calmar_ratio(
    annual_return: Union[Decimal, float],
    max_dd: Union[Decimal, float]
) -> Decimal:
    """Calculate Calmar ratio.

    Args:
        annual_return: Annualized return
        max_dd: Maximum drawdown

    Returns:
        Calmar ratio

    Raises:
        ValueError: If max_dd is zero
    """
    try:
        max_dd_dec = Decimal(str(max_dd))
        if max_dd_dec == Decimal("0"):
            raise ValueError("Maximum drawdown cannot be zero")

        annual_ret = Decimal(str(annual_return))
        return annual_ret / max_dd_dec

    except Exception as e:
        logger.error("calmar_ratio_failed", error=str(e))
        raise


def value_at_risk(
    returns: List[Union[Decimal, float]],
    confidence_level: float = 0.95
) -> Decimal:
    """Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: List of historical returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(returns) < 10:
            raise ValueError("Need at least 10 returns for VaR")

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        sorted_returns = sorted([Decimal(str(r)) for r in returns])

        # Find percentile index
        index = int(len(sorted_returns) * (1 - confidence_level))
        var = abs(sorted_returns[index])

        return var

    except Exception as e:
        logger.error("var_calc_failed", error=str(e))
        raise


def conditional_var(
    returns: List[Union[Decimal, float]],
    confidence_level: float = 0.95
) -> Decimal:
    """Calculate Conditional VaR (Expected Shortfall).

    Args:
        returns: List of historical returns
        confidence_level: Confidence level

    Returns:
        CVaR value

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(returns) < 10:
            raise ValueError("Need at least 10 returns for CVaR")

        sorted_returns = sorted([Decimal(str(r)) for r in returns])

        # Find VaR cutoff
        index = int(len(sorted_returns) * (1 - confidence_level))

        # Calculate average of returns worse than VaR
        tail_returns = sorted_returns[:index+1]
        if not tail_returns:
            return Decimal("0")

        cvar = abs(sum(tail_returns) / Decimal(str(len(tail_returns))))

        return cvar

    except Exception as e:
        logger.error("cvar_calc_failed", error=str(e))
        raise


def kelly_criterion(
    win_rate: Union[Decimal, float],
    avg_win: Union[Decimal, float],
    avg_loss: Union[Decimal, float]
) -> Decimal:
    """Calculate Kelly Criterion optimal position size.

    Args:
        win_rate: Win rate (0-1)
        avg_win: Average win amount
        avg_loss: Average loss amount

    Returns:
        Optimal position size fraction

    Raises:
        ValueError: If inputs invalid
    """
    try:
        win_rate_dec = Decimal(str(win_rate))
        avg_win_dec = Decimal(str(avg_win))
        avg_loss_dec = Decimal(str(avg_loss))

        if not Decimal("0") <= win_rate_dec <= Decimal("1"):
            raise ValueError("Win rate must be between 0 and 1")

        if avg_loss_dec == Decimal("0"):
            raise ValueError("Average loss cannot be zero")

        # Kelly = W - (1-W) / (W/L)
        # Where W is win rate, W is avg win, L is avg loss
        win_loss_ratio = avg_win_dec / avg_loss_dec
        loss_rate = Decimal("1") - win_rate_dec

        kelly = win_rate_dec - (loss_rate / win_loss_ratio)

        # Cap at 0 (never go negative)
        return max(kelly, Decimal("0"))

    except Exception as e:
        logger.error("kelly_criterion_failed", error=str(e))
        raise


def position_size_fixed_fractional(
    account_balance: Union[Decimal, float],
    risk_fraction: Union[Decimal, float],
    entry_price: Union[Decimal, float],
    stop_loss: Union[Decimal, float]
) -> Decimal:
    """Calculate position size using fixed fractional method.

    Args:
        account_balance: Total account balance
        risk_fraction: Fraction of account to risk (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_loss: Stop loss price

    Returns:
        Position size in units

    Raises:
        ValueError: If inputs invalid
    """
    try:
        balance = Decimal(str(account_balance))
        risk_frac = Decimal(str(risk_fraction))
        entry = Decimal(str(entry_price))
        stop = Decimal(str(stop_loss))

        if balance <= Decimal("0"):
            raise ValueError("Account balance must be positive")

        if entry == stop:
            raise ValueError("Entry and stop loss cannot be equal")

        # Calculate risk amount
        risk_amount = balance * risk_frac

        # Calculate risk per unit
        risk_per_unit = abs(entry - stop)

        # Calculate position size
        position_size = risk_amount / risk_per_unit

        return position_size

    except Exception as e:
        logger.error("position_size_failed", error=str(e))
        raise


def volatility_historical(
    returns: List[Union[Decimal, float]],
    annualize: bool = True,
    periods_per_year: int = 365
) -> Decimal:
    """Calculate historical volatility (standard deviation of returns).

    Args:
        returns: List of returns
        annualize: Annualize the volatility
        periods_per_year: Number of periods per year

    Returns:
        Volatility

    Raises:
        ValueError: If insufficient data
    """
    try:
        if len(returns) < 2:
            raise ValueError("Need at least 2 returns for volatility")

        returns_dec = [Decimal(str(r)) for r in returns]

        # Calculate mean
        mean = sum(returns_dec) / Decimal(str(len(returns_dec)))

        # Calculate variance
        variance = sum((r - mean) ** 2 for r in returns_dec) / Decimal(str(len(returns_dec) - 1))

        # Standard deviation
        std_dev = Decimal(str(math.sqrt(float(variance))))

        # Annualize if requested
        if annualize:
            std_dev *= Decimal(str(math.sqrt(periods_per_year)))

        return std_dev

    except Exception as e:
        logger.error("volatility_calc_failed", error=str(e))
        raise


def correlation(
    values1: List[Union[Decimal, float]],
    values2: List[Union[Decimal, float]]
) -> Decimal:
    """Calculate Pearson correlation coefficient.

    Args:
        values1: First series
        values2: Second series

    Returns:
        Correlation coefficient (-1 to 1)

    Raises:
        ValueError: If series lengths don't match or insufficient data
    """
    try:
        if len(values1) != len(values2):
            raise ValueError("Series must have same length")

        if len(values1) < 2:
            raise ValueError("Need at least 2 values for correlation")

        v1 = [Decimal(str(v)) for v in values1]
        v2 = [Decimal(str(v)) for v in values2]

        # Calculate means
        mean1 = sum(v1) / Decimal(str(len(v1)))
        mean2 = sum(v2) / Decimal(str(len(v2)))

        # Calculate covariance and standard deviations
        covariance = sum((x - mean1) * (y - mean2) for x, y in zip(v1, v2))

        variance1 = sum((x - mean1) ** 2 for x in v1)
        variance2 = sum((y - mean2) ** 2 for y in v2)

        std1 = Decimal(str(math.sqrt(float(variance1))))
        std2 = Decimal(str(math.sqrt(float(variance2))))

        if std1 == Decimal("0") or std2 == Decimal("0"):
            return Decimal("0")

        corr = covariance / (std1 * std2)

        return corr

    except Exception as e:
        logger.error("correlation_failed", error=str(e))
        raise


def round_to_precision(
    value: Union[Decimal, float],
    precision: int,
    rounding: str = ROUND_HALF_UP
) -> Decimal:
    """Round value to specified precision.

    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode

    Returns:
        Rounded Decimal
    """
    try:
        dec_value = Decimal(str(value))
        quantizer = Decimal('0.1') ** precision
        return dec_value.quantize(quantizer, rounding=rounding)

    except Exception as e:
        logger.error("rounding_failed", value=value, precision=precision, error=str(e))
        raise


def weighted_average(
    values: List[Union[Decimal, float]],
    weights: List[Union[Decimal, float]]
) -> Decimal:
    """Calculate weighted average.

    Args:
        values: List of values
        weights: List of weights

    Returns:
        Weighted average

    Raises:
        ValueError: If lists don't match or weights sum to zero
    """
    try:
        if len(values) != len(weights):
            raise ValueError("Values and weights must have same length")

        values_dec = [Decimal(str(v)) for v in values]
        weights_dec = [Decimal(str(w)) for w in weights]

        total_weight = sum(weights_dec)
        if total_weight == Decimal("0"):
            raise ValueError("Weights cannot sum to zero")

        weighted_sum = sum(v * w for v, w in zip(values_dec, weights_dec))

        return weighted_sum / total_weight

    except Exception as e:
        logger.error("weighted_avg_failed", error=str(e))
        raise


def linear_interpolate(
    x: Union[Decimal, float],
    x1: Union[Decimal, float],
    y1: Union[Decimal, float],
    x2: Union[Decimal, float],
    y2: Union[Decimal, float]
) -> Decimal:
    """Linear interpolation between two points.

    Args:
        x: X value to interpolate at
        x1: First point X
        y1: First point Y
        x2: Second point X
        y2: Second point Y

    Returns:
        Interpolated Y value

    Raises:
        ValueError: If x1 == x2
    """
    try:
        x_dec = Decimal(str(x))
        x1_dec = Decimal(str(x1))
        y1_dec = Decimal(str(y1))
        x2_dec = Decimal(str(x2))
        y2_dec = Decimal(str(y2))

        if x1_dec == x2_dec:
            raise ValueError("x1 and x2 cannot be equal")

        # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        result = y1_dec + (x_dec - x1_dec) * (y2_dec - y1_dec) / (x2_dec - x1_dec)

        return result

    except Exception as e:
        logger.error("interpolation_failed", error=str(e))
        raise
