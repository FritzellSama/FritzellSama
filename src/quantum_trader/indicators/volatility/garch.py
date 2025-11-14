"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) Volatility Indicator.

This module implements GARCH(1,1) model for volatility forecasting in financial markets.
Used for risk management and volatility-based trading strategies.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
import polars as pl
from structlog import get_logger
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

logger = get_logger(__name__)

# Set high precision for Decimal calculations
getcontext().prec = 28


@dataclass
class GARCHParams:
    """GARCH model parameters."""
    omega: Decimal  # Constant term
    alpha: Decimal  # ARCH coefficient
    beta: Decimal   # GARCH coefficient


class GARCHVolatilityIndicator:
    """
    GARCH(1,1) volatility forecasting indicator.

    The GARCH model captures volatility clustering and persistence in financial markets.
    Model: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)

    Attributes:
        config: Configuration dictionary with model parameters
        params: Fitted GARCH parameters
        conditional_volatility: Current conditional volatility estimate

    Example:
        >>> config = {"lookback_period": 252, "initial_volatility": "0.02"}
        >>> garch = GARCHVolatilityIndicator(config)
        >>> volatility = await garch.calculate(price_data)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize GARCH volatility indicator.

        Args:
            config: Configuration dictionary containing:
                - lookback_period: Number of periods for estimation
                - initial_volatility: Initial volatility estimate
                - max_iterations: Maximum optimization iterations
                - convergence_tolerance: Convergence tolerance for optimization
                - min_observations: Minimum observations required

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.lookback_period: int = int(config["lookback_period"])
        self.initial_volatility: Decimal = Decimal(str(config["initial_volatility"]))
        self.max_iterations: int = int(config.get("max_iterations", 1000))
        self.convergence_tolerance: Decimal = Decimal(str(config.get("convergence_tolerance", "0.000001")))
        self.min_observations: int = int(config.get("min_observations", 100))

        self.params: Optional[GARCHParams] = None
        self.conditional_volatility: Optional[pl.DataFrame] = None

        logger.info(
            "GARCH indicator initialized",
            lookback_period=self.lookback_period,
            initial_volatility=str(self.initial_volatility)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ["lookback_period", "initial_volatility"]

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if int(self.config["lookback_period"]) < 50:
            raise ValueError("lookback_period must be at least 50")

        if Decimal(str(self.config["initial_volatility"])) <= Decimal("0"):
            raise ValueError("initial_volatility must be positive")

        logger.debug("GARCH configuration validated")

    async def calculate(
        self,
        data: pl.DataFrame,
        fit_model: bool = True
    ) -> pl.DataFrame:
        """
        Calculate GARCH volatility forecast.

        Args:
            data: Polars DataFrame with columns: timestamp, close
            fit_model: Whether to re-fit the model parameters

        Returns:
            DataFrame with columns: timestamp, conditional_volatility, forecast_volatility

        Raises:
            ValueError: If data is invalid or insufficient
        """
        try:
            # Validate input data
            self._validate_data(data)

            # Calculate returns
            returns_df = await self._calculate_returns(data)

            if len(returns_df) < self.min_observations:
                raise ValueError(
                    f"Insufficient data: {len(returns_df)} < {self.min_observations}"
                )

            # Fit GARCH model if requested
            if fit_model or self.params is None:
                self.params = await self._fit_garch_model(returns_df)

            # Calculate conditional volatility
            volatility_df = await self._calculate_conditional_volatility(returns_df)

            # Generate forecast
            forecast_df = await self._generate_forecast(volatility_df)

            self.conditional_volatility = forecast_df

            logger.info(
                "GARCH volatility calculated",
                observations=len(forecast_df),
                current_volatility=str(forecast_df["conditional_volatility"][-1])
            )

            return forecast_df

        except Exception as e:
            logger.error("GARCH calculation failed", error=str(e))
            raise

    async def _calculate_returns(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate log returns from price data."""
        try:
            # Convert prices to Decimal and calculate log returns
            returns = data.with_columns([
                pl.col("close").cast(pl.Utf8).alias("close_str")
            ]).with_columns([
                pl.col("close_str").map_elements(
                    lambda x: str(Decimal(x).ln()) if x else None,
                    return_dtype=pl.Utf8
                ).alias("log_price")
            ]).with_columns([
                (pl.col("log_price") - pl.col("log_price").shift(1)).alias("return_str")
            ]).with_columns([
                pl.col("return_str").map_elements(
                    lambda x: Decimal(x) if x else Decimal("0"),
                    return_dtype=pl.Utf8
                ).alias("return")
            ]).select([
                "timestamp",
                "return"
            ]).filter(pl.col("return") != "0")

            return returns

        except Exception as e:
            logger.error("Return calculation failed", error=str(e))
            raise

    async def _fit_garch_model(self, returns_df: pl.DataFrame) -> GARCHParams:
        """
        Fit GARCH(1,1) model parameters using maximum likelihood estimation.

        Args:
            returns_df: DataFrame with returns

        Returns:
            Fitted GARCH parameters
        """
        try:
            # Convert returns to numpy for optimization
            returns = np.array([
                float(r) for r in returns_df["return"].to_list()
            ])

            # Initial parameter guess
            initial_params = np.array([0.00001, 0.05, 0.90])

            # Optimization bounds: ω > 0, α ≥ 0, β ≥ 0, α + β < 1
            bounds = ((1e-10, 1.0), (0.0, 1.0), (0.0, 1.0))

            # Constraint: α + β < 1 for stationarity
            constraints = {
                'type': 'ineq',
                'fun': lambda x: 0.999 - x[1] - x[2]
            }

            # Optimize log-likelihood
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: minimize(
                    self._negative_log_likelihood,
                    initial_params,
                    args=(returns,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': self.max_iterations}
                )
            )

            if not result.success:
                logger.warning("GARCH optimization did not converge", message=result.message)

            params = GARCHParams(
                omega=Decimal(str(result.x[0])),
                alpha=Decimal(str(result.x[1])),
                beta=Decimal(str(result.x[2]))
            )

            logger.info(
                "GARCH model fitted",
                omega=str(params.omega),
                alpha=str(params.alpha),
                beta=str(params.beta)
            )

            return params

        except Exception as e:
            logger.error("GARCH model fitting failed", error=str(e))
            raise

    def _negative_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Calculate negative log-likelihood for GARCH(1,1) model.

        Args:
            params: [omega, alpha, beta]
            returns: Array of returns

        Returns:
            Negative log-likelihood value
        """
        omega, alpha, beta = params

        # Initialize conditional variance
        variance = np.zeros(len(returns))
        variance[0] = np.var(returns)

        # Calculate conditional variances
        for t in range(1, len(returns)):
            variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]

        # Calculate log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi) + np.log(variance) + returns**2 / variance
        )

        return -log_likelihood

    async def _calculate_conditional_volatility(
        self,
        returns_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate conditional volatility series.

        Args:
            returns_df: DataFrame with returns

        Returns:
            DataFrame with conditional volatility
        """
        try:
            returns = [Decimal(str(r)) for r in returns_df["return"].to_list()]

            # Initialize variance
            variance = [self.initial_volatility ** 2]

            # Calculate conditional variances
            for t in range(1, len(returns)):
                var_t = (
                    self.params.omega +
                    self.params.alpha * returns[t-1] ** 2 +
                    self.params.beta * variance[t-1]
                )
                variance.append(var_t)

            # Convert to volatility (standard deviation)
            volatility = [v.sqrt() for v in variance]

            # Create result DataFrame
            result = returns_df.with_columns([
                pl.Series("conditional_volatility", [str(v) for v in volatility])
            ])

            return result

        except Exception as e:
            logger.error("Conditional volatility calculation failed", error=str(e))
            raise

    async def _generate_forecast(self, volatility_df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate one-step-ahead volatility forecast.

        Args:
            volatility_df: DataFrame with conditional volatility

        Returns:
            DataFrame with volatility forecast
        """
        try:
            # Get last return and variance
            last_return = Decimal(str(volatility_df["return"][-1]))
            last_variance = Decimal(str(volatility_df["conditional_volatility"][-1])) ** 2

            # One-step-ahead forecast
            forecast_variance = (
                self.params.omega +
                self.params.alpha * last_return ** 2 +
                self.params.beta * last_variance
            )
            forecast_volatility = forecast_variance.sqrt()

            # Add forecast to DataFrame
            result = volatility_df.with_columns([
                pl.lit(str(forecast_volatility)).alias("forecast_volatility")
            ])

            logger.debug(
                "Volatility forecast generated",
                forecast=str(forecast_volatility)
            )

            return result

        except Exception as e:
            logger.error("Forecast generation failed", error=str(e))
            raise

    def _validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ["timestamp", "close"]

        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(data) < self.min_observations:
            raise ValueError(
                f"Insufficient data points: {len(data)} < {self.min_observations}"
            )

        # Check for null values
        if data["close"].null_count() > 0:
            raise ValueError("Data contains null values in 'close' column")

    async def get_current_volatility(self) -> Optional[Decimal]:
        """
        Get current conditional volatility estimate.

        Returns:
            Current volatility as Decimal, or None if not calculated
        """
        if self.conditional_volatility is None:
            return None

        return Decimal(str(self.conditional_volatility["conditional_volatility"][-1]))

    async def get_forecast(self) -> Optional[Decimal]:
        """
        Get one-step-ahead volatility forecast.

        Returns:
            Forecast volatility as Decimal, or None if not calculated
        """
        if self.conditional_volatility is None:
            return None

        return Decimal(str(self.conditional_volatility["forecast_volatility"][-1]))


async def create_garch_indicator(config: Dict[str, Any]) -> GARCHVolatilityIndicator:
    """
    Factory function to create GARCH indicator instance.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized GARCH indicator
    """
    return GARCHVolatilityIndicator(config)
