"""Cointegration features for statistical arbitrage and pairs trading.

This module provides functionality for calculating cointegration relationships
between trading pairs, which is essential for mean reversion strategies and
statistical arbitrage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from structlog import get_logger

logger = get_logger(__name__)


class CointegrationAnalyzer:
    """Analyzes cointegration relationships between trading pairs.

    This class provides methods for detecting and quantifying cointegration
    between price series, which is useful for pairs trading and statistical
    arbitrage strategies.

    Attributes:
        config: Configuration dictionary containing analysis parameters
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the cointegration analyzer.

        Args:
            config: Configuration dictionary with parameters:
                - significance_level: P-value threshold for cointegration test
                - lookback_period: Number of periods to analyze
                - min_half_life: Minimum half-life for mean reversion (in periods)
                - max_half_life: Maximum half-life for mean reversion (in periods)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.significance_level = Decimal(str(config.get("significance_level", "0.05")))
        self.lookback_period = int(config.get("lookback_period", 252))
        self.min_half_life = Decimal(str(config.get("min_half_life", "1")))
        self.max_half_life = Decimal(str(config.get("max_half_life", "100")))

        logger.info(
            "cointegration_analyzer_initialized",
            significance_level=str(self.significance_level),
            lookback_period=self.lookback_period,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

        required_keys = ["significance_level", "lookback_period"]
        for key in required_keys:
            if key not in self.config:
                logger.warning("missing_config_key", key=key, using_default=True)

    def calculate_cointegration(
        self,
        data: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        price_column: str = "close",
    ) -> Dict[str, Any]:
        """Calculate cointegration between two price series.

        Args:
            data: Polars DataFrame containing price data
            symbol1: First symbol identifier
            symbol2: Second symbol identifier
            price_column: Column name containing prices

        Returns:
            Dictionary containing:
                - is_cointegrated: Boolean indicating cointegration
                - p_value: P-value from cointegration test
                - test_statistic: Test statistic value
                - critical_values: Critical values for different confidence levels
                - hedge_ratio: Optimal hedge ratio
                - half_life: Mean reversion half-life
                - spread_mean: Mean of the spread
                - spread_std: Standard deviation of the spread

        Raises:
            ValueError: If data is invalid or symbols not found
        """
        try:
            logger.info(
                "calculating_cointegration",
                symbol1=symbol1,
                symbol2=symbol2,
                rows=len(data),
            )

            # Validate input data
            self._validate_data(data, [symbol1, symbol2], price_column)

            # Extract price series
            prices1 = self._extract_prices(data, symbol1, price_column)
            prices2 = self._extract_prices(data, symbol2, price_column)

            # Perform cointegration test
            test_stat, p_value, critical_values = coint(prices1, prices2)

            # Calculate hedge ratio using OLS
            hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)

            # Calculate spread
            spread = prices1 - hedge_ratio * prices2

            # Calculate half-life of mean reversion
            half_life = self._calculate_half_life(spread)

            # Calculate spread statistics
            spread_mean = Decimal(str(np.mean(spread)))
            spread_std = Decimal(str(np.std(spread)))

            # Determine if cointegrated
            is_cointegrated = float(p_value) < float(self.significance_level)

            result = {
                "is_cointegrated": is_cointegrated,
                "p_value": Decimal(str(p_value)),
                "test_statistic": Decimal(str(test_stat)),
                "critical_values": {
                    "1%": Decimal(str(critical_values[0])),
                    "5%": Decimal(str(critical_values[1])),
                    "10%": Decimal(str(critical_values[2])),
                },
                "hedge_ratio": Decimal(str(hedge_ratio)),
                "half_life": Decimal(str(half_life)) if half_life is not None else None,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
            }

            logger.info(
                "cointegration_calculated",
                symbol1=symbol1,
                symbol2=symbol2,
                is_cointegrated=is_cointegrated,
                p_value=str(result["p_value"]),
            )

            return result

        except Exception as e:
            logger.error(
                "cointegration_calculation_failed",
                error=str(e),
                symbol1=symbol1,
                symbol2=symbol2,
            )
            raise

    def _extract_prices(
        self, data: pl.DataFrame, symbol: str, price_column: str
    ) -> np.ndarray:
        """Extract price series for a symbol.

        Args:
            data: Polars DataFrame with price data
            symbol: Symbol identifier
            price_column: Price column name

        Returns:
            Numpy array of prices

        Raises:
            ValueError: If symbol or column not found
        """
        try:
            # Filter for symbol and extract prices
            symbol_data = data.filter(pl.col("symbol") == symbol)
            if len(symbol_data) == 0:
                raise ValueError(f"No data found for symbol: {symbol}")

            prices = symbol_data[price_column].to_numpy()
            return prices.astype(float)

        except Exception as e:
            logger.error("price_extraction_failed", symbol=symbol, error=str(e))
            raise

    def _calculate_hedge_ratio(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate optimal hedge ratio using OLS regression.

        Args:
            prices1: First price series
            prices2: Second price series

        Returns:
            Hedge ratio (slope of regression)
        """
        try:
            # Perform linear regression: prices1 = hedge_ratio * prices2 + intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                prices2, prices1
            )
            return float(slope)

        except Exception as e:
            logger.error("hedge_ratio_calculation_failed", error=str(e))
            raise

    def _calculate_half_life(self, spread: np.ndarray) -> Optional[float]:
        """Calculate mean reversion half-life using AR(1) model.

        Args:
            spread: Spread time series

        Returns:
            Half-life in periods, or None if not mean-reverting

        Raises:
            ValueError: If calculation fails
        """
        try:
            # Fit AR(1) model: spread[t] = alpha + beta * spread[t-1] + error
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)

            # OLS regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                spread_lag, spread_diff
            )

            # Half-life formula: -log(2) / log(1 + beta)
            if slope >= 0:
                # Not mean-reverting
                logger.warning("spread_not_mean_reverting", slope=slope)
                return None

            half_life = -np.log(2) / np.log(1 + slope)

            # Validate half-life is within acceptable range
            if not (float(self.min_half_life) <= half_life <= float(self.max_half_life)):
                logger.warning(
                    "half_life_out_of_range",
                    half_life=half_life,
                    min_allowed=str(self.min_half_life),
                    max_allowed=str(self.max_half_life),
                )
                return None

            return float(half_life)

        except Exception as e:
            logger.error("half_life_calculation_failed", error=str(e))
            return None

    def _validate_data(
        self, data: pl.DataFrame, symbols: List[str], price_column: str
    ) -> None:
        """Validate input data.

        Args:
            data: DataFrame to validate
            symbols: List of required symbols
            price_column: Required price column

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, pl.DataFrame):
            raise ValueError(f"Data must be Polars DataFrame, got {type(data)}")

        if len(data) < self.lookback_period:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need {self.lookback_period}"
            )

        required_columns = ["symbol", price_column]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

    def find_cointegrated_pairs(
        self,
        data: pl.DataFrame,
        symbols: List[str],
        price_column: str = "close",
    ) -> List[Dict[str, Any]]:
        """Find all cointegrated pairs from a list of symbols.

        Args:
            data: Polars DataFrame containing price data
            symbols: List of symbols to analyze
            price_column: Column name containing prices

        Returns:
            List of dictionaries containing cointegrated pair information

        Raises:
            ValueError: If data is invalid
        """
        cointegrated_pairs = []

        try:
            logger.info(
                "finding_cointegrated_pairs",
                num_symbols=len(symbols),
                total_combinations=len(symbols) * (len(symbols) - 1) // 2,
            )

            # Test all pairs
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i + 1 :]:
                    try:
                        result = self.calculate_cointegration(
                            data, symbol1, symbol2, price_column
                        )

                        if result["is_cointegrated"]:
                            pair_info = {
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                **result,
                            }
                            cointegrated_pairs.append(pair_info)

                    except Exception as e:
                        logger.warning(
                            "pair_test_failed",
                            symbol1=symbol1,
                            symbol2=symbol2,
                            error=str(e),
                        )
                        continue

            logger.info(
                "cointegrated_pairs_found",
                num_pairs=len(cointegrated_pairs),
                total_tested=len(symbols) * (len(symbols) - 1) // 2,
            )

            return cointegrated_pairs

        except Exception as e:
            logger.error("find_pairs_failed", error=str(e))
            raise

    def calculate_spread_zscore(
        self,
        data: pl.DataFrame,
        symbol1: str,
        symbol2: str,
        hedge_ratio: Decimal,
        price_column: str = "close",
        window: Optional[int] = None,
    ) -> pl.DataFrame:
        """Calculate z-score of spread for trading signals.

        Args:
            data: Polars DataFrame containing price data
            symbol1: First symbol
            symbol2: Second symbol
            hedge_ratio: Hedge ratio between symbols
            price_column: Price column name
            window: Rolling window for mean/std calculation (None = full series)

        Returns:
            DataFrame with spread and z-score columns

        Raises:
            ValueError: If calculation fails
        """
        try:
            # Extract prices
            prices1 = self._extract_prices(data, symbol1, price_column)
            prices2 = self._extract_prices(data, symbol2, price_column)

            # Calculate spread
            spread = prices1 - float(hedge_ratio) * prices2

            # Calculate z-score
            if window is None:
                mean = np.mean(spread)
                std = np.std(spread)
                zscore = (spread - mean) / std
            else:
                # Rolling z-score
                zscore = np.zeros_like(spread)
                for i in range(window, len(spread)):
                    window_data = spread[i - window : i]
                    mean = np.mean(window_data)
                    std = np.std(window_data)
                    zscore[i] = (spread[i] - mean) / (std if std > 0 else 1)

            # Create result DataFrame
            result_df = pl.DataFrame(
                {
                    "timestamp": data.filter(pl.col("symbol") == symbol1)[
                        "timestamp"
                    ].to_list(),
                    "spread": [Decimal(str(s)) for s in spread],
                    "zscore": [Decimal(str(z)) for z in zscore],
                }
            )

            logger.info(
                "spread_zscore_calculated",
                symbol1=symbol1,
                symbol2=symbol2,
                rows=len(result_df),
            )

            return result_df

        except Exception as e:
            logger.error("zscore_calculation_failed", error=str(e))
            raise


__all__ = ["CointegrationAnalyzer"]
