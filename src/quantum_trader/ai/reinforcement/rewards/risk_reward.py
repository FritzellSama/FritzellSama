"""Risk-adjusted reward function for trading reinforcement learning.

This module implements reward functions that incorporate risk metrics like
Value at Risk (VaR), Conditional VaR (CVaR), and drawdown.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics for reward calculation.

    Attributes:
        var: Value at Risk (Decimal)
        cvar: Conditional Value at Risk (Decimal)
        max_drawdown: Maximum drawdown (Decimal)
        volatility: Volatility (Decimal)
        sharpe_ratio: Sharpe ratio (Decimal)
        sortino_ratio: Sortino ratio (Decimal)
        timestamp: UTC timestamp
    """
    var: Decimal
    cvar: Decimal
    max_drawdown: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class RiskAdjustedReward:
    """Risk-adjusted reward calculator for trading RL.

    Calculates rewards that penalize risk and encourage risk-adjusted returns
    rather than just absolute returns.

    Attributes:
        config: Configuration dictionary
        var_penalty: Penalty coefficient for VaR
        cvar_penalty: Penalty coefficient for CVaR
        drawdown_penalty: Penalty coefficient for drawdown
        volatility_penalty: Penalty coefficient for volatility
        confidence_level: Confidence level for VaR/CVaR

    Example:
        >>> config = {
        ...     'var_penalty': 0.5,
        ...     'cvar_penalty': 1.0,
        ...     'drawdown_penalty': 2.0,
        ...     'volatility_penalty': 0.3,
        ...     'confidence_level': 0.95,
        ...     'lookback_window': 100,
        ...     'reward_scale': 1000.0,
        ...     'risk_free_rate': 0.02
        ... }
        >>> risk_reward = RiskAdjustedReward(config)
        >>> reward = risk_reward.calculate(
        ...     returns_history=[...],
        ...     current_return=Decimal('0.01')
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize risk-adjusted reward calculator.

        Args:
            config: Configuration with keys:
                - var_penalty: VaR penalty coefficient
                - cvar_penalty: CVaR penalty coefficient
                - drawdown_penalty: Drawdown penalty coefficient
                - volatility_penalty: Volatility penalty coefficient
                - confidence_level: Confidence level for VaR/CVaR
                - lookback_window: Window for calculating metrics
                - reward_scale: Scaling factor
                - risk_free_rate: Risk-free rate (annualized)
        """
        self.config = config
        self._validate_config()

        self.var_penalty = Decimal(str(config['var_penalty']))
        self.cvar_penalty = Decimal(str(config['cvar_penalty']))
        self.drawdown_penalty = Decimal(str(config['drawdown_penalty']))
        self.volatility_penalty = Decimal(str(config['volatility_penalty']))
        self.confidence_level = Decimal(str(config['confidence_level']))
        self.lookback_window = config['lookback_window']
        self.reward_scale = Decimal(str(config['reward_scale']))
        self.risk_free_rate = Decimal(str(config['risk_free_rate']))

        # State tracking
        self.returns_history: List[Decimal] = []
        self.values_history: List[Decimal] = []
        self.risk_metrics_history: List[RiskMetrics] = []

        logger.info(
            "initialized_risk_adjusted_reward",
            var_penalty=str(self.var_penalty),
            cvar_penalty=str(self.cvar_penalty),
            confidence_level=str(self.confidence_level)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'var_penalty', 'cvar_penalty', 'drawdown_penalty',
            'volatility_penalty', 'confidence_level', 'lookback_window',
            'reward_scale', 'risk_free_rate'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if not (0 < self.config['confidence_level'] < 1):
            raise ValueError("confidence_level must be in (0, 1)")

        if self.config['lookback_window'] < 10:
            raise ValueError("lookback_window must be at least 10")

        if self.config['reward_scale'] <= 0:
            raise ValueError("reward_scale must be positive")

    def calculate_var(
        self,
        returns: List[Decimal],
        confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate Value at Risk (VaR).

        Args:
            returns: List of returns
            confidence_level: Confidence level (default: from config)

        Returns:
            VaR value (Decimal)
        """
        try:
            if len(returns) < 10:
                return Decimal('0')

            if confidence_level is None:
                confidence_level = self.confidence_level

            returns_array = np.array([float(r) for r in returns])

            # Calculate percentile
            percentile = (Decimal('1') - confidence_level) * Decimal('100')
            var = Decimal(str(np.percentile(returns_array, float(percentile))))

            logger.debug(
                "calculated_var",
                var=str(var),
                confidence_level=str(confidence_level),
                n_returns=len(returns)
            )

            return var

        except Exception as e:
            logger.error("failed_to_calculate_var", error=str(e))
            raise

    def calculate_cvar(
        self,
        returns: List[Decimal],
        confidence_level: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: List of returns
            confidence_level: Confidence level (default: from config)

        Returns:
            CVaR value (Decimal)
        """
        try:
            if len(returns) < 10:
                return Decimal('0')

            if confidence_level is None:
                confidence_level = self.confidence_level

            # First calculate VaR
            var = self.calculate_var(returns, confidence_level)

            # CVaR is mean of returns below VaR
            returns_below_var = [r for r in returns if r <= var]

            if not returns_below_var:
                return var

            cvar = Decimal(str(np.mean([float(r) for r in returns_below_var])))

            logger.debug(
                "calculated_cvar",
                cvar=str(cvar),
                var=str(var),
                n_below_var=len(returns_below_var)
            )

            return cvar

        except Exception as e:
            logger.error("failed_to_calculate_cvar", error=str(e))
            raise

    def calculate_max_drawdown(
        self,
        values: List[Decimal]
    ) -> Decimal:
        """Calculate maximum drawdown.

        Args:
            values: List of portfolio values

        Returns:
            Maximum drawdown as positive decimal
        """
        try:
            if len(values) < 2:
                return Decimal('0')

            values_array = np.array([float(v) for v in values])

            # Calculate running maximum
            running_max = np.maximum.accumulate(values_array)

            # Calculate drawdown at each point
            drawdown = (running_max - values_array) / running_max

            max_dd = Decimal(str(np.max(drawdown)))

            logger.debug(
                "calculated_max_drawdown",
                max_drawdown=str(max_dd),
                n_values=len(values)
            )

            return max_dd

        except Exception as e:
            logger.error("failed_to_calculate_max_drawdown", error=str(e))
            raise

    def calculate_volatility(
        self,
        returns: List[Decimal],
        annualize: bool = False
    ) -> Decimal:
        """Calculate volatility (standard deviation of returns).

        Args:
            returns: List of returns
            annualize: Whether to annualize volatility

        Returns:
            Volatility (Decimal)
        """
        try:
            if len(returns) < 2:
                return Decimal('0')

            returns_array = np.array([float(r) for r in returns])
            vol = Decimal(str(np.std(returns_array)))

            if annualize:
                # Assume daily returns
                vol = vol * Decimal(str(np.sqrt(252)))

            logger.debug(
                "calculated_volatility",
                volatility=str(vol),
                annualized=annualize,
                n_returns=len(returns)
            )

            return vol

        except Exception as e:
            logger.error("failed_to_calculate_volatility", error=str(e))
            raise

    def calculate_sharpe_ratio(
        self,
        returns: List[Decimal],
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate Sharpe ratio.

        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sharpe ratio (Decimal)
        """
        try:
            if len(returns) < 2:
                return Decimal('0')

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            returns_array = np.array([float(r) for r in returns])
            mean_return = Decimal(str(np.mean(returns_array)))
            std_return = Decimal(str(np.std(returns_array)))

            if std_return <= Decimal('1e-8'):
                return Decimal('0')

            # Annualize
            annualized_return = mean_return * Decimal('252')
            annualized_vol = std_return * Decimal(str(np.sqrt(252)))

            sharpe = (annualized_return - risk_free_rate) / annualized_vol

            logger.debug(
                "calculated_sharpe_ratio",
                sharpe=str(sharpe),
                mean_return=str(annualized_return),
                volatility=str(annualized_vol)
            )

            return sharpe

        except Exception as e:
            logger.error("failed_to_calculate_sharpe_ratio", error=str(e))
            raise

    def calculate_sortino_ratio(
        self,
        returns: List[Decimal],
        target_return: Decimal = Decimal('0'),
        risk_free_rate: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate Sortino ratio.

        Args:
            returns: List of returns
            target_return: Target return threshold
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Sortino ratio (Decimal)
        """
        try:
            if len(returns) < 2:
                return Decimal('0')

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            returns_array = np.array([float(r) for r in returns])
            mean_return = Decimal(str(np.mean(returns_array)))

            # Downside returns
            downside_returns = returns_array[returns_array < float(target_return)]

            if len(downside_returns) == 0:
                return Decimal('100')  # No downside risk

            downside_std = Decimal(str(np.std(downside_returns)))

            if downside_std <= Decimal('1e-8'):
                return Decimal('0')

            # Annualize
            annualized_return = mean_return * Decimal('252')
            annualized_downside_std = downside_std * Decimal(str(np.sqrt(252)))

            sortino = (annualized_return - risk_free_rate) / annualized_downside_std

            logger.debug(
                "calculated_sortino_ratio",
                sortino=str(sortino),
                mean_return=str(annualized_return),
                downside_std=str(annualized_downside_std)
            )

            return sortino

        except Exception as e:
            logger.error("failed_to_calculate_sortino_ratio", error=str(e))
            raise

    def calculate_risk_metrics(
        self,
        returns: List[Decimal],
        values: List[Decimal]
    ) -> RiskMetrics:
        """Calculate all risk metrics.

        Args:
            returns: Returns history
            values: Portfolio values history

        Returns:
            RiskMetrics object
        """
        try:
            var = self.calculate_var(returns)
            cvar = self.calculate_cvar(returns)
            max_dd = self.calculate_max_drawdown(values)
            vol = self.calculate_volatility(returns, annualize=True)
            sharpe = self.calculate_sharpe_ratio(returns)
            sortino = self.calculate_sortino_ratio(returns)

            metrics = RiskMetrics(
                var=var,
                cvar=cvar,
                max_drawdown=max_dd,
                volatility=vol,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino
            )

            logger.debug(
                "calculated_risk_metrics",
                var=str(var),
                cvar=str(cvar),
                max_drawdown=str(max_dd),
                sharpe=str(sharpe)
            )

            return metrics

        except Exception as e:
            logger.error("failed_to_calculate_risk_metrics", error=str(e))
            raise

    def calculate(
        self,
        current_return: Decimal,
        current_value: Decimal
    ) -> Decimal:
        """Calculate risk-adjusted reward.

        Args:
            current_return: Current step return
            current_value: Current portfolio value

        Returns:
            Risk-adjusted reward (Decimal)
        """
        try:
            # Add to history
            self.returns_history.append(current_return)
            self.values_history.append(current_value)

            # Limit history size
            if len(self.returns_history) > self.lookback_window:
                self.returns_history = self.returns_history[-self.lookback_window:]
                self.values_history = self.values_history[-self.lookback_window:]

            # Base reward: return
            reward = current_return * self.reward_scale

            # Calculate risk penalties if we have enough history
            if len(self.returns_history) >= 10:
                # Calculate risk metrics
                metrics = self.calculate_risk_metrics(
                    self.returns_history,
                    self.values_history
                )

                # Store metrics
                self.risk_metrics_history.append(metrics)
                if len(self.risk_metrics_history) > 1000:
                    self.risk_metrics_history = self.risk_metrics_history[-1000:]

                # VaR penalty (VaR is typically negative, penalty is positive)
                if metrics.var < 0:
                    var_penalty_value = abs(metrics.var) * self.var_penalty * self.reward_scale
                    reward -= var_penalty_value

                    logger.debug(
                        "applied_var_penalty",
                        var=str(metrics.var),
                        penalty=str(var_penalty_value)
                    )

                # CVaR penalty
                if metrics.cvar < 0:
                    cvar_penalty_value = abs(metrics.cvar) * self.cvar_penalty * self.reward_scale
                    reward -= cvar_penalty_value

                    logger.debug(
                        "applied_cvar_penalty",
                        cvar=str(metrics.cvar),
                        penalty=str(cvar_penalty_value)
                    )

                # Drawdown penalty
                if metrics.max_drawdown > 0:
                    dd_penalty_value = metrics.max_drawdown * self.drawdown_penalty * self.reward_scale
                    reward -= dd_penalty_value

                    logger.debug(
                        "applied_drawdown_penalty",
                        max_drawdown=str(metrics.max_drawdown),
                        penalty=str(dd_penalty_value)
                    )

                # Volatility penalty
                if metrics.volatility > 0:
                    vol_penalty_value = metrics.volatility * self.volatility_penalty * self.reward_scale
                    reward -= vol_penalty_value

                    logger.debug(
                        "applied_volatility_penalty",
                        volatility=str(metrics.volatility),
                        penalty=str(vol_penalty_value)
                    )

            logger.debug(
                "calculated_risk_adjusted_reward",
                current_return=str(current_return),
                reward=str(reward)
            )

            return reward

        except Exception as e:
            logger.error("failed_to_calculate_reward", error=str(e))
            raise

    def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get most recent risk metrics.

        Returns:
            Latest RiskMetrics or None
        """
        if self.risk_metrics_history:
            return self.risk_metrics_history[-1]
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get risk-adjusted reward statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            'returns_history_size': len(self.returns_history),
            'values_history_size': len(self.values_history),
            'metrics_history_size': len(self.risk_metrics_history)
        }

        if self.returns_history:
            returns_array = np.array([float(r) for r in self.returns_history])
            stats['mean_return'] = str(Decimal(str(np.mean(returns_array))))
            stats['std_return'] = str(Decimal(str(np.std(returns_array))))

        if self.risk_metrics_history:
            latest = self.risk_metrics_history[-1]
            stats['latest_metrics'] = {
                'var': str(latest.var),
                'cvar': str(latest.cvar),
                'max_drawdown': str(latest.max_drawdown),
                'volatility': str(latest.volatility),
                'sharpe_ratio': str(latest.sharpe_ratio),
                'sortino_ratio': str(latest.sortino_ratio)
            }

        return stats

    def reset(self) -> None:
        """Reset reward calculator state."""
        self.returns_history.clear()
        self.values_history.clear()
        self.risk_metrics_history.clear()

        logger.info("reset_risk_adjusted_reward")
