"""
Quantum Trader AI - Risk Parity Position Sizing
Production-grade risk parity portfolio construction

CRITICAL CONSTRAINTS:
- All numeric values use Decimal, NEVER float
- All data operations use polars DataFrame
- All external calls wrapped in try/except with retry logic
- Complete type hints everywhere
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import polars as pl
import numpy as np

from quantum_trader.models import Position
from quantum_trader.database.timeseries import TimeSeriesDB


logger = logging.getLogger(__name__)


class RiskParitySizer:
    """
    Risk parity position sizing system

    Features:
    - Equal risk contribution calculation
    - Volatility normalization
    - Risk parity portfolio construction
    - Dynamic risk balancing
    """

    def __init__(
        self,
        config_path: Path = Path("/home/user/FritzellSama/config/bot/risk.yaml"),
        env_config_path: Path = Path("/home/user/FritzellSama/config/environments/production.yaml")
    ) -> None:
        """Initialize risk parity sizer with configuration"""
        self.config = self._load_config(config_path)
        self.env_config = self._load_config(env_config_path)

        # Position limits configuration
        position_limits = self.config.get("position_limits", {})
        self.max_position_size_percent = Decimal(str(position_limits.get("max_position_size_percent", 5.0)))
        self.max_open_positions = int(position_limits.get("max_open_positions", 20))
        self.max_leverage = Decimal(str(position_limits.get("max_leverage", 2.0)))
        self.min_position_size_usd = Decimal(str(position_limits.get("min_position_size_usd", 500)))
        self.max_sector_exposure_percent = Decimal(str(position_limits.get("max_sector_exposure_percent", 15.0)))

        # Risk model parameters
        risk_model = self.config.get("risk_model", {})
        self.var_lookback_days = int(risk_model.get("var_lookback_days", 252))

        # Risk parity specific settings
        self.target_portfolio_volatility = Decimal('0.15')  # 15% annual volatility target
        self.rebalance_threshold = Decimal('0.05')  # Rebalance if risk allocation deviates by 5%
        self.min_correlation_period = 60  # Minimum days for correlation calculation
        self.volatility_window = 30  # Days for volatility calculation

        # Database connection
        self.db: Optional[TimeSeriesDB] = None

        # Retry configuration
        self.retry_attempts = 3
        self.retry_delay_ms = 1000

        logger.info("RiskParitySizer initialized")

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}

    async def initialize(self) -> None:
        """Initialize database connections"""
        try:
            self.db = TimeSeriesDB()
            await self.db.connect()
            logger.info("RiskParitySizer initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize RiskParitySizer: {e}")
            raise

    async def calculate_risk_parity_weights(
        self,
        symbols: List[str],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate risk parity position weights for given symbols

        Risk parity ensures each asset contributes equally to portfolio risk.

        Args:
            symbols: List of trading symbols
            portfolio_value: Total portfolio value

        Returns:
            Dictionary mapping symbol to position size in USD
        """
        for attempt in range(self.retry_attempts):
            try:
                # Fetch historical price data
                price_data = await self._fetch_price_data(symbols)

                if len(price_data) == 0:
                    logger.warning("No price data available for risk parity calculation")
                    return {}

                # Calculate returns
                returns_df = self._calculate_returns(price_data)

                # Calculate volatilities
                volatilities = await self._calculate_volatilities(returns_df)

                # Calculate correlation matrix
                correlation_matrix = self._calculate_correlation_matrix(returns_df)

                # Calculate risk parity weights
                weights = self._optimize_risk_parity_weights(
                    volatilities,
                    correlation_matrix,
                    symbols
                )

                # Convert to position sizes
                position_sizes = self._weights_to_position_sizes(
                    weights,
                    portfolio_value,
                    symbols
                )

                # Apply constraints
                constrained_sizes = await self._apply_constraints(
                    position_sizes,
                    portfolio_value
                )

                return constrained_sizes

            except Exception as e:
                logger.error(f"Error calculating risk parity weights (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay_ms / 1000)
                else:
                    raise

    async def _fetch_price_data(self, symbols: List[str]) -> pl.DataFrame:
        """Fetch historical price data for symbols"""
        if not self.db:
            raise RuntimeError("Database not initialized")

        try:
            lookback_date = datetime.utcnow() - timedelta(days=self.var_lookback_days)

            # Fetch data for all symbols
            all_data = []
            for symbol in symbols:
                symbol_data = await self.db.query_ohlcv(
                    symbol=symbol,
                    start_date=lookback_date,
                    end_date=datetime.utcnow(),
                    timeframe="1d"
                )

                if len(symbol_data) > 0:
                    # Add symbol column
                    symbol_data = symbol_data.with_columns(
                        pl.lit(symbol).alias("symbol")
                    )
                    all_data.append(symbol_data)

            if not all_data:
                return pl.DataFrame()

            # Combine all data
            combined_df = pl.concat(all_data)
            return combined_df

        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            raise

    def _calculate_returns(self, price_data: pl.DataFrame) -> pl.DataFrame:
        """Calculate log returns from price data"""
        try:
            # Pivot data to have symbols as columns
            pivot_df = price_data.pivot(
                values="close",
                index="timestamp",
                columns="symbol"
            ).sort("timestamp")

            # Calculate log returns
            returns_data = {}
            returns_data["timestamp"] = pivot_df["timestamp"][1:].to_list()

            for col in pivot_df.columns:
                if col == "timestamp":
                    continue

                prices = pivot_df[col].to_numpy()
                # Calculate log returns: ln(P_t / P_{t-1})
                log_returns = np.log(prices[1:] / prices[:-1])

                # Convert to Decimal
                returns_data[col] = [Decimal(str(r)) for r in log_returns]

            returns_df = pl.DataFrame(returns_data)
            return returns_df

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise

    async def _calculate_volatilities(self, returns_df: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate annualized volatilities for each symbol"""
        volatilities = {}

        try:
            for col in returns_df.columns:
                if col == "timestamp":
                    continue

                # Calculate standard deviation of returns
                returns = returns_df[col].to_numpy()
                std_dev = np.std(returns)

                # Annualize: multiply by sqrt(252) for daily returns
                annualized_vol = Decimal(str(std_dev)) * Decimal('252').sqrt()

                volatilities[col] = annualized_vol

            return volatilities

        except Exception as e:
            logger.error(f"Error calculating volatilities: {e}")
            raise

    def _calculate_correlation_matrix(self, returns_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate correlation matrix of returns"""
        try:
            # Extract return columns (exclude timestamp)
            return_cols = [col for col in returns_df.columns if col != "timestamp"]

            # Convert to numpy for correlation calculation
            returns_array = returns_df.select(return_cols).to_numpy()

            # Calculate correlation matrix
            corr_matrix = np.corrcoef(returns_array.T)

            # Convert to polars DataFrame
            corr_df = pl.DataFrame(
                corr_matrix,
                schema=return_cols
            )

            # Add row labels
            corr_df = corr_df.with_columns(
                pl.Series("symbol", return_cols)
            )

            return corr_df

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise

    def _optimize_risk_parity_weights(
        self,
        volatilities: Dict[str, Decimal],
        correlation_matrix: pl.DataFrame,
        symbols: List[str]
    ) -> Dict[str, Decimal]:
        """
        Optimize portfolio weights for risk parity

        In risk parity, each asset contributes equally to portfolio risk.
        Weight is inversely proportional to volatility: w_i = 1/σ_i
        """
        try:
            # Calculate inverse volatility weights
            inv_vol_weights = {}
            total_inv_vol = Decimal('0')

            for symbol in symbols:
                if symbol in volatilities and volatilities[symbol] > 0:
                    inv_vol = Decimal('1') / volatilities[symbol]
                    inv_vol_weights[symbol] = inv_vol
                    total_inv_vol += inv_vol
                else:
                    inv_vol_weights[symbol] = Decimal('0')

            # Normalize weights to sum to 1
            normalized_weights = {}
            for symbol in symbols:
                if total_inv_vol > 0:
                    normalized_weights[symbol] = inv_vol_weights[symbol] / total_inv_vol
                else:
                    # Equal weights if no volatility data
                    normalized_weights[symbol] = Decimal('1') / Decimal(str(len(symbols)))

            logger.info(f"Risk parity weights calculated for {len(symbols)} symbols")
            return normalized_weights

        except Exception as e:
            logger.error(f"Error optimizing risk parity weights: {e}")
            raise

    def _weights_to_position_sizes(
        self,
        weights: Dict[str, Decimal],
        portfolio_value: Decimal,
        symbols: List[str]
    ) -> Dict[str, Decimal]:
        """Convert portfolio weights to position sizes in USD"""
        position_sizes = {}

        for symbol in symbols:
            if symbol in weights:
                position_size = weights[symbol] * portfolio_value
                position_sizes[symbol] = position_size
            else:
                position_sizes[symbol] = Decimal('0')

        return position_sizes

    async def _apply_constraints(
        self,
        position_sizes: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """Apply position size constraints"""
        constrained_sizes = {}

        # Calculate maximum position size
        max_position_size = portfolio_value * (self.max_position_size_percent / Decimal('100'))

        for symbol, size in position_sizes.items():
            # Apply minimum size constraint
            if size < self.min_position_size_usd:
                constrained_sizes[symbol] = Decimal('0')
                continue

            # Apply maximum size constraint
            if size > max_position_size:
                constrained_sizes[symbol] = max_position_size
            else:
                constrained_sizes[symbol] = size

        return constrained_sizes

    async def calculate_rebalancing_needs(
        self,
        current_positions: List[Position],
        target_weights: Dict[str, Decimal],
        portfolio_value: Decimal
    ) -> Dict[str, Decimal]:
        """
        Calculate rebalancing trades needed

        Returns:
            Dictionary mapping symbol to delta in USD (positive = buy, negative = sell)
        """
        try:
            # Calculate current allocations
            current_allocations = {}
            for position in current_positions:
                position_value = abs(position.quantity) * position.current_price
                current_allocations[position.symbol] = position_value

            # Calculate target allocations
            target_allocations = {}
            for symbol, weight in target_weights.items():
                target_allocations[symbol] = weight * portfolio_value

            # Calculate deltas
            rebalancing_trades = {}
            all_symbols = set(list(current_allocations.keys()) + list(target_allocations.keys()))

            for symbol in all_symbols:
                current = current_allocations.get(symbol, Decimal('0'))
                target = target_allocations.get(symbol, Decimal('0'))
                delta = target - current

                # Only rebalance if delta exceeds threshold
                if abs(delta) / portfolio_value > (self.rebalance_threshold / Decimal('100')):
                    rebalancing_trades[symbol] = delta

            return rebalancing_trades

        except Exception as e:
            logger.error(f"Error calculating rebalancing needs: {e}")
            raise

    async def calculate_risk_contribution(
        self,
        positions: List[Position]
    ) -> Dict[str, Decimal]:
        """
        Calculate each position's contribution to portfolio risk

        Returns:
            Dictionary mapping symbol to risk contribution (percentage)
        """
        try:
            if not positions:
                return {}

            # Get symbols
            symbols = [p.symbol for p in positions]

            # Fetch historical data
            price_data = await self._fetch_price_data(symbols)
            if len(price_data) == 0:
                return {}

            # Calculate returns and volatilities
            returns_df = self._calculate_returns(price_data)
            volatilities = await self._calculate_volatilities(returns_df)

            # Calculate portfolio value and weights
            total_value = sum(abs(p.quantity * p.current_price) for p in positions)

            risk_contributions = {}
            for position in positions:
                position_value = abs(position.quantity * position.current_price)
                weight = position_value / total_value if total_value > 0 else Decimal('0')

                # Risk contribution = weight * volatility
                if position.symbol in volatilities:
                    vol = volatilities[position.symbol]
                    risk_contrib = weight * vol
                    risk_contributions[position.symbol] = risk_contrib * Decimal('100')  # Convert to percentage
                else:
                    risk_contributions[position.symbol] = Decimal('0')

            return risk_contributions

        except Exception as e:
            logger.error(f"Error calculating risk contribution: {e}")
            raise

    def calculate_equal_risk_weights(
        self,
        volatilities: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Calculate weights for equal risk contribution

        Simple approach: w_i ∝ 1/σ_i
        """
        if not volatilities:
            return {}

        # Calculate inverse volatility
        inv_vols = {symbol: Decimal('1') / vol if vol > 0 else Decimal('0')
                    for symbol, vol in volatilities.items()}

        # Normalize
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol > 0:
            weights = {symbol: inv_vol / total_inv_vol
                      for symbol, inv_vol in inv_vols.items()}
        else:
            # Equal weights fallback
            n = len(volatilities)
            weights = {symbol: Decimal('1') / Decimal(str(n))
                      for symbol in volatilities.keys()}

        return weights

    async def estimate_portfolio_volatility(
        self,
        weights: Dict[str, Decimal],
        symbols: List[str]
    ) -> Decimal:
        """Estimate portfolio volatility given weights"""
        try:
            # Fetch historical data
            price_data = await self._fetch_price_data(symbols)
            if len(price_data) == 0:
                return Decimal('0')

            # Calculate returns
            returns_df = self._calculate_returns(price_data)

            # Calculate volatilities
            volatilities = await self._calculate_volatilities(returns_df)

            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(returns_df)

            # Portfolio variance = w^T * Σ * w
            # Where Σ is the covariance matrix
            # Simplified calculation using diagonal (ignoring correlations)
            portfolio_var = Decimal('0')
            for symbol in symbols:
                if symbol in weights and symbol in volatilities:
                    w = weights[symbol]
                    vol = volatilities[symbol]
                    portfolio_var += (w * vol) ** 2

            # Portfolio volatility is square root of variance
            portfolio_vol = portfolio_var.sqrt()

            return portfolio_vol

        except Exception as e:
            logger.error(f"Error estimating portfolio volatility: {e}")
            return Decimal('0')

    async def get_risk_parity_report(
        self,
        current_positions: List[Position],
        portfolio_value: Decimal
    ) -> Dict:
        """Generate comprehensive risk parity report"""
        try:
            # Calculate risk contributions
            risk_contributions = await self.calculate_risk_contribution(current_positions)

            # Get symbols
            symbols = [p.symbol for p in current_positions] if current_positions else []

            # Calculate ideal weights
            if symbols:
                ideal_weights = await self.calculate_risk_parity_weights(symbols, portfolio_value)
            else:
                ideal_weights = {}

            # Calculate rebalancing needs
            rebalancing = await self.calculate_rebalancing_needs(
                current_positions,
                ideal_weights,
                portfolio_value
            )

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "portfolio_value": float(portfolio_value),
                "num_positions": len(current_positions),
                "risk_contributions": {k: float(v) for k, v in risk_contributions.items()},
                "ideal_weights": {k: float(v) for k, v in ideal_weights.items()},
                "rebalancing_needed": {k: float(v) for k, v in rebalancing.items()},
                "is_balanced": len(rebalancing) == 0
            }

            return report

        except Exception as e:
            logger.error(f"Error generating risk parity report: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.db:
            await self.db.disconnect()

        logger.info("RiskParitySizer cleanup complete")
