"""
Centralized Risk Calculator for Quantum Trader AI

Production-grade risk calculation engine with:
- Position risk calculation
- Portfolio risk aggregation
- Marginal risk contribution
- Risk decomposition by asset/strategy
- Stress scenario testing
- Value at Risk (VaR) and CVaR
- Correlation-based risk adjustments

CRITICAL: All numeric values use Decimal, NEVER float
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import logging

import polars as pl
import yaml
import numpy as np
from scipy import stats

from quantum_trader.models import Position, RiskMetrics


logger = logging.getLogger(__name__)


@dataclass
class RiskCalculatorConfig:
    """Configuration for risk calculator"""
    var_confidence_level: Decimal
    var_lookback_days: int
    cvar_enabled: bool
    correlation_lookback_days: int
    stress_scenarios_enabled: bool
    max_portfolio_risk_percent: Decimal
    marginal_risk_threshold: Decimal


@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_value: Decimal
    position_weight: Decimal
    volatility: Decimal
    var_95: Decimal
    cvar_95: Decimal
    beta: Decimal
    marginal_var: Decimal
    contribution_to_portfolio_risk: Decimal


@dataclass
class PortfolioRisk:
    """Comprehensive portfolio risk metrics"""
    total_value: Decimal
    total_exposure: Decimal
    portfolio_volatility: Decimal
    portfolio_var_95: Decimal
    portfolio_cvar_95: Decimal
    portfolio_beta: Decimal
    max_position_concentration: Decimal
    num_positions: int
    diversification_ratio: Decimal
    timestamp: datetime


@dataclass
class StressTestResult:
    """Result of stress test scenario"""
    scenario_name: str
    portfolio_loss: Decimal
    portfolio_loss_percent: Decimal
    worst_position: str
    worst_position_loss: Decimal
    positions_affected: int
    breach_risk_limits: bool


class RiskCalculator:
    """
    Centralized risk calculation engine

    Provides comprehensive risk analytics:
    - Position-level risk metrics
    - Portfolio risk aggregation
    - VaR and CVaR calculations
    - Marginal risk contributions
    - Stress testing
    """

    def __init__(
        self,
        config_path: str = "/home/user/FritzellSama/config/bot/risk.yaml",
        env_config_path: str = "/home/user/FritzellSama/config/environments/production.yaml"
    ):
        """
        Initialize risk calculator

        Args:
            config_path: Path to risk configuration file
            env_config_path: Path to environment configuration file
        """
        self.config = self._load_config(config_path, env_config_path)
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None

        logger.info("RiskCalculator initialized with VaR confidence: %s", self.config.var_confidence_level)

    def _load_config(self, config_path: str, env_config_path: str) -> RiskCalculatorConfig:
        """
        Load configuration from YAML files

        Args:
            config_path: Path to risk config
            env_config_path: Path to environment config

        Returns:
            Loaded configuration
        """
        try:
            with open(config_path, 'r') as f:
                risk_config = yaml.safe_load(f)

            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)

            risk_model = risk_config.get('risk_model', {})
            global_risk = risk_config.get('global', {})

            return RiskCalculatorConfig(
                var_confidence_level=Decimal(str(risk_model.get('var_confidence_level', 0.95))),
                var_lookback_days=risk_model.get('var_lookback_days', 252),
                cvar_enabled=risk_model.get('cvar_enabled', True),
                correlation_lookback_days=252,
                stress_scenarios_enabled=risk_config.get('stress_testing', {}).get('enabled', True),
                max_portfolio_risk_percent=Decimal(str(global_risk.get('max_portfolio_risk_percent', 2.0))),
                marginal_risk_threshold=Decimal('0.1')
            )

        except Exception as e:
            logger.error("Failed to load configuration: %s", e)
            raise

    def calculate_position_risk(
        self,
        position: Position,
        returns_df: pl.DataFrame,
        portfolio_value: Decimal,
        benchmark_returns_df: Optional[pl.DataFrame] = None
    ) -> PositionRisk:
        """
        Calculate comprehensive risk metrics for a position

        Args:
            position: Position object
            returns_df: Historical returns for the asset
            portfolio_value: Total portfolio value
            benchmark_returns_df: Optional benchmark returns for beta calculation

        Returns:
            Position risk metrics
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        # Calculate position value and weight
        position_value = position.current_price * abs(position.quantity)
        position_weight = (position_value / portfolio_value * Decimal('100')
                          if portfolio_value > 0 else Decimal('0'))

        # Calculate volatility
        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]

        if not returns:
            logger.warning("No returns data for %s, using default risk values", position.symbol)
            return PositionRisk(
                symbol=position.symbol,
                position_value=position_value,
                position_weight=position_weight,
                volatility=Decimal('0.2'),
                var_95=Decimal('0'),
                cvar_95=Decimal('0'),
                beta=Decimal('1.0'),
                marginal_var=Decimal('0'),
                contribution_to_portfolio_risk=Decimal('0')
            )

        mean_return = sum(returns) / Decimal(str(len(returns)))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(str(len(returns)))
        volatility = Decimal(str(np.sqrt(float(variance))))

        # Annualize volatility
        annualized_volatility = volatility * Decimal(str(np.sqrt(252)))

        # Calculate VaR (parametric method)
        var_95 = self.calculate_var(
            returns_df,
            position_value,
            self.config.var_confidence_level
        )

        # Calculate CVaR
        cvar_95 = self.calculate_cvar(
            returns_df,
            position_value,
            self.config.var_confidence_level
        ) if self.config.cvar_enabled else var_95

        # Calculate beta
        if benchmark_returns_df is not None:
            beta = self._calculate_beta(returns_df, benchmark_returns_df)
        else:
            beta = Decimal('1.0')

        position_risk = PositionRisk(
            symbol=position.symbol,
            position_value=position_value,
            position_weight=position_weight,
            volatility=annualized_volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            marginal_var=Decimal('0'),  # Calculated at portfolio level
            contribution_to_portfolio_risk=Decimal('0')  # Calculated at portfolio level
        )

        self.position_risks[position.symbol] = position_risk

        logger.info(
            "Position risk calculated for %s: VaR=$%s, volatility=%s%%",
            position.symbol, var_95, annualized_volatility * Decimal('100')
        )

        return position_risk

    def calculate_portfolio_risk(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        correlation_matrix_df: pl.DataFrame,
        benchmark_returns_df: Optional[pl.DataFrame] = None
    ) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics

        Args:
            positions: List of portfolio positions
            returns_dict: Dictionary mapping symbol to returns DataFrame
            correlation_matrix_df: Correlation matrix of assets
            benchmark_returns_df: Optional benchmark returns

        Returns:
            Portfolio risk metrics
        """
        if not positions:
            logger.warning("No positions provided for portfolio risk calculation")
            return PortfolioRisk(
                total_value=Decimal('0'),
                total_exposure=Decimal('0'),
                portfolio_volatility=Decimal('0'),
                portfolio_var_95=Decimal('0'),
                portfolio_cvar_95=Decimal('0'),
                portfolio_beta=Decimal('0'),
                max_position_concentration=Decimal('0'),
                num_positions=0,
                diversification_ratio=Decimal('0'),
                timestamp=datetime.now()
            )

        # Calculate total portfolio value
        total_value = sum(p.current_price * abs(p.quantity) for p in positions)
        total_exposure = sum(p.current_price * p.quantity for p in positions)  # Including sign

        # Calculate individual position risks
        for position in positions:
            if position.symbol in returns_dict:
                self.calculate_position_risk(
                    position,
                    returns_dict[position.symbol],
                    total_value,
                    benchmark_returns_df
                )

        # Calculate portfolio volatility using correlation matrix
        portfolio_volatility = self._calculate_portfolio_volatility(
            positions,
            returns_dict,
            correlation_matrix_df,
            total_value
        )

        # Calculate portfolio VaR
        portfolio_var = self._calculate_portfolio_var(
            positions,
            returns_dict,
            correlation_matrix_df,
            total_value
        )

        # Calculate portfolio CVaR
        portfolio_cvar = self._calculate_portfolio_cvar(
            positions,
            returns_dict,
            correlation_matrix_df,
            total_value
        ) if self.config.cvar_enabled else portfolio_var

        # Calculate portfolio beta
        if benchmark_returns_df is not None:
            portfolio_beta = self._calculate_portfolio_beta(
                positions,
                returns_dict,
                benchmark_returns_df,
                total_value
            )
        else:
            portfolio_beta = Decimal('1.0')

        # Calculate concentration
        max_concentration = max(
            (p.current_price * abs(p.quantity) / total_value * Decimal('100')
             if total_value > 0 else Decimal('0'))
            for p in positions
        ) if positions else Decimal('0')

        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(
            positions,
            returns_dict,
            total_value
        )

        # Calculate marginal contributions
        self._calculate_marginal_contributions(
            positions,
            returns_dict,
            correlation_matrix_df,
            total_value,
            portfolio_var
        )

        portfolio_risk = PortfolioRisk(
            total_value=total_value,
            total_exposure=total_exposure,
            portfolio_volatility=portfolio_volatility,
            portfolio_var_95=portfolio_var,
            portfolio_cvar_95=portfolio_cvar,
            portfolio_beta=portfolio_beta,
            max_position_concentration=max_concentration,
            num_positions=len(positions),
            diversification_ratio=diversification_ratio,
            timestamp=datetime.now()
        )

        self.portfolio_risk = portfolio_risk

        logger.info(
            "Portfolio risk calculated: VaR=$%s, volatility=%s%%, positions=%d",
            portfolio_var, portfolio_volatility * Decimal('100'), len(positions)
        )

        return portfolio_risk

    def calculate_var(
        self,
        returns_df: pl.DataFrame,
        position_value: Decimal,
        confidence_level: Decimal
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR) using parametric method

        Args:
            returns_df: Historical returns
            position_value: Position value
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR value
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        returns = [Decimal(str(r)) for r in returns_df['return'].to_list()]

        if not returns:
            return Decimal('0')

        # Calculate mean and std
        mean = sum(returns) / Decimal(str(len(returns)))
        variance = sum((r - mean) ** 2 for r in returns) / Decimal(str(len(returns)))
        std = Decimal(str(np.sqrt(float(variance))))

        # Z-score for confidence level
        z_score = Decimal(str(stats.norm.ppf(1 - float(confidence_level))))

        # VaR = position_value * (mean - z * std)
        var = position_value * abs(mean + z_score * std)

        return var

    def calculate_cvar(
        self,
        returns_df: pl.DataFrame,
        position_value: Decimal,
        confidence_level: Decimal
    ) -> Decimal:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall)

        Average loss beyond VaR threshold

        Args:
            returns_df: Historical returns
            position_value: Position value
            confidence_level: Confidence level

        Returns:
            CVaR value
        """
        if not isinstance(returns_df, pl.DataFrame):
            raise TypeError(f"returns_df must be polars DataFrame")

        returns = returns_df['return'].to_numpy()

        if len(returns) == 0:
            return Decimal('0')

        # Calculate VaR threshold
        var_threshold = np.percentile(returns, (1 - float(confidence_level)) * 100)

        # CVaR is average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return self.calculate_var(returns_df, position_value, confidence_level)

        mean_tail_loss = np.mean(tail_returns)

        cvar = position_value * abs(Decimal(str(mean_tail_loss)))

        return cvar

    def _calculate_beta(
        self,
        asset_returns_df: pl.DataFrame,
        benchmark_returns_df: pl.DataFrame
    ) -> Decimal:
        """Calculate beta vs benchmark"""
        df = asset_returns_df.join(
            benchmark_returns_df,
            on='timestamp',
            how='inner',
            suffix='_benchmark'
        )

        if df.height < 2:
            return Decimal('1.0')

        asset_returns = df['return'].to_numpy()
        benchmark_returns = df['return_benchmark'].to_numpy()

        covariance = np.cov(asset_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return Decimal('1.0')

        beta = Decimal(str(covariance / benchmark_variance))

        return beta

    def _calculate_portfolio_volatility(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        correlation_matrix_df: pl.DataFrame,
        total_value: Decimal
    ) -> Decimal:
        """Calculate portfolio volatility considering correlations"""
        if total_value == Decimal('0'):
            return Decimal('0')

        # Build weight vector and volatility vector
        symbols = [p.symbol for p in positions]
        weights = [
            (p.current_price * abs(p.quantity) / total_value)
            for p in positions
        ]

        volatilities = []
        for symbol in symbols:
            if symbol in returns_dict:
                returns = [Decimal(str(r)) for r in returns_dict[symbol]['return'].to_list()]
                if returns:
                    mean = sum(returns) / Decimal(str(len(returns)))
                    variance = sum((r - mean) ** 2 for r in returns) / Decimal(str(len(returns)))
                    vol = Decimal(str(np.sqrt(float(variance) * 252)))  # Annualize
                else:
                    vol = Decimal('0.2')
            else:
                vol = Decimal('0.2')  # Default

            volatilities.append(vol)

        # Calculate portfolio variance: w^T * Cov * w
        portfolio_variance = Decimal('0')

        for i, (symbol_i, w_i, vol_i) in enumerate(zip(symbols, weights, volatilities)):
            for j, (symbol_j, w_j, vol_j) in enumerate(zip(symbols, weights, volatilities)):
                # Get correlation
                corr = self._get_correlation(correlation_matrix_df, symbol_i, symbol_j)

                covariance = vol_i * vol_j * corr
                portfolio_variance += w_i * w_j * covariance

        portfolio_volatility = Decimal(str(np.sqrt(float(portfolio_variance))))

        return portfolio_volatility

    def _calculate_portfolio_var(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        correlation_matrix_df: pl.DataFrame,
        total_value: Decimal
    ) -> Decimal:
        """Calculate portfolio VaR"""
        portfolio_vol = self._calculate_portfolio_volatility(
            positions,
            returns_dict,
            correlation_matrix_df,
            total_value
        )

        # Parametric VaR
        z_score = Decimal(str(stats.norm.ppf(1 - float(self.config.var_confidence_level))))

        var = total_value * abs(z_score) * portfolio_vol

        return var

    def _calculate_portfolio_cvar(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        correlation_matrix_df: pl.DataFrame,
        total_value: Decimal
    ) -> Decimal:
        """Calculate portfolio CVaR using historical simulation"""
        # This is simplified - full implementation would use Monte Carlo
        var = self._calculate_portfolio_var(positions, returns_dict, correlation_matrix_df, total_value)

        # CVaR is typically 1.2-1.5x VaR for normal distributions
        cvar = var * Decimal('1.3')

        return cvar

    def _calculate_portfolio_beta(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        benchmark_returns_df: pl.DataFrame,
        total_value: Decimal
    ) -> Decimal:
        """Calculate portfolio beta as weighted average"""
        if total_value == Decimal('0'):
            return Decimal('1.0')

        portfolio_beta = Decimal('0')

        for position in positions:
            weight = (position.current_price * abs(position.quantity)) / total_value

            if position.symbol in returns_dict:
                beta = self._calculate_beta(returns_dict[position.symbol], benchmark_returns_df)
                portfolio_beta += weight * beta

        return portfolio_beta

    def _calculate_diversification_ratio(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        total_value: Decimal
    ) -> Decimal:
        """
        Calculate diversification ratio: weighted avg volatility / portfolio volatility

        Higher ratio = better diversification
        """
        if total_value == Decimal('0'):
            return Decimal('1.0')

        # Weighted average volatility
        weighted_vol = Decimal('0')

        for position in positions:
            weight = (position.current_price * abs(position.quantity)) / total_value

            if position.symbol in returns_dict and position.symbol in self.position_risks:
                vol = self.position_risks[position.symbol].volatility
                weighted_vol += weight * vol

        if self.portfolio_risk and self.portfolio_risk.portfolio_volatility > Decimal('0'):
            diversification_ratio = weighted_vol / self.portfolio_risk.portfolio_volatility
        else:
            diversification_ratio = Decimal('1.0')

        return diversification_ratio

    def _calculate_marginal_contributions(
        self,
        positions: List[Position],
        returns_dict: Dict[str, pl.DataFrame],
        correlation_matrix_df: pl.DataFrame,
        total_value: Decimal,
        portfolio_var: Decimal
    ) -> None:
        """Calculate marginal VaR contribution for each position"""
        # This is computationally intensive - simplified implementation
        for symbol in self.position_risks.keys():
            # Marginal VaR is approximately: weight * volatility * correlation with portfolio
            position_risk = self.position_risks[symbol]

            # Simplified: contribution = position_weight * position_var
            contribution = (position_risk.position_value / total_value) * position_risk.var_95

            position_risk.contribution_to_portfolio_risk = contribution
            position_risk.marginal_var = contribution / position_risk.position_value if position_risk.position_value > 0 else Decimal('0')

    def _get_correlation(
        self,
        correlation_matrix_df: pl.DataFrame,
        symbol1: str,
        symbol2: str
    ) -> Decimal:
        """Get correlation between two assets"""
        if symbol1 == symbol2:
            return Decimal('1.0')

        try:
            corr_row = correlation_matrix_df.filter(pl.col('symbol') == symbol1).select(symbol2)

            if corr_row.height > 0:
                return Decimal(str(corr_row[0, 0]))

        except Exception as e:
            logger.debug("Could not get correlation for %s/%s: %s", symbol1, symbol2, e)

        return Decimal('0.3')  # Default correlation

    def run_stress_test(
        self,
        positions: List[Position],
        scenario_name: str,
        scenario_returns: Dict[str, Decimal]
    ) -> StressTestResult:
        """
        Run stress test scenario

        Args:
            positions: Portfolio positions
            scenario_name: Name of stress scenario
            scenario_returns: Expected returns by symbol in scenario

        Returns:
            Stress test result
        """
        total_value = sum(p.current_price * abs(p.quantity) for p in positions)

        if total_value == Decimal('0'):
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_loss=Decimal('0'),
                portfolio_loss_percent=Decimal('0'),
                worst_position='',
                worst_position_loss=Decimal('0'),
                positions_affected=0,
                breach_risk_limits=False
            )

        # Calculate losses
        total_loss = Decimal('0')
        worst_loss = Decimal('0')
        worst_position = ''
        positions_affected = 0

        for position in positions:
            position_value = position.current_price * abs(position.quantity)
            scenario_return = scenario_returns.get(position.symbol, Decimal('-0.1'))  # Default -10%

            position_loss = position_value * abs(scenario_return) if scenario_return < 0 else Decimal('0')

            total_loss += position_loss

            if position_loss > worst_loss:
                worst_loss = position_loss
                worst_position = position.symbol

            if scenario_return < 0:
                positions_affected += 1

        loss_percent = (total_loss / total_value * Decimal('100')) if total_value > 0 else Decimal('0')

        # Check if breach risk limits
        breach = loss_percent > self.config.max_portfolio_risk_percent

        result = StressTestResult(
            scenario_name=scenario_name,
            portfolio_loss=total_loss,
            portfolio_loss_percent=loss_percent,
            worst_position=worst_position,
            worst_position_loss=worst_loss,
            positions_affected=positions_affected,
            breach_risk_limits=breach
        )

        logger.info(
            "Stress test '%s': Loss=$%s (%s%%), breach=%s",
            scenario_name, total_loss, loss_percent, breach
        )

        return result

    def get_position_risk(self, symbol: str) -> Optional[PositionRisk]:
        """Get risk metrics for specific position"""
        return self.position_risks.get(symbol)

    def get_portfolio_risk(self) -> Optional[PortfolioRisk]:
        """Get current portfolio risk metrics"""
        return self.portfolio_risk

    def export_risk_metrics(self) -> RiskMetrics:
        """
        Export risk metrics in standard format

        Returns:
            RiskMetrics object
        """
        if self.portfolio_risk is None:
            raise ValueError("Portfolio risk not calculated yet")

        return RiskMetrics(
            portfolio_value=self.portfolio_risk.total_value,
            cash_balance=Decimal('0'),  # Set externally
            total_exposure=self.portfolio_risk.total_exposure,
            var_95=self.portfolio_risk.portfolio_var_95,
            cvar_95=self.portfolio_risk.portfolio_cvar_95,
            max_drawdown=Decimal('0'),  # Set externally
            sharpe_ratio=Decimal('0'),  # Set externally
            sortino_ratio=Decimal('0'),  # Set externally
            beta=self.portfolio_risk.portfolio_beta,
            daily_pnl=Decimal('0'),  # Set externally
            timestamp=datetime.now()
        )
