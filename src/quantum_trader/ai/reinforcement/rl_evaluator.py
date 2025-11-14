"""Reinforcement learning agent evaluator.

This module provides comprehensive evaluation of RL agents including
performance metrics, risk metrics, and statistical significance testing.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import polars as pl
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


@dataclass
class EpisodeResult:
    """Results from a single episode.

    Attributes:
        episode_id: Episode identifier
        total_reward: Total episode reward (Decimal)
        episode_length: Number of steps
        final_value: Final portfolio value (Decimal)
        max_drawdown: Maximum drawdown (Decimal)
        sharpe_ratio: Sharpe ratio (Decimal)
        win_rate: Win rate (Decimal)
        avg_trade_return: Average trade return (Decimal)
        n_trades: Number of trades
        timestamp: UTC timestamp
    """
    episode_id: int
    total_reward: Decimal
    episode_length: int
    final_value: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    win_rate: Decimal
    avg_trade_return: Decimal
    n_trades: int
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report.

    Attributes:
        n_episodes: Number of evaluation episodes
        mean_reward: Mean total reward (Decimal)
        std_reward: Std of total reward (Decimal)
        mean_episode_length: Mean episode length
        mean_final_value: Mean final portfolio value (Decimal)
        mean_sharpe: Mean Sharpe ratio (Decimal)
        mean_max_drawdown: Mean maximum drawdown (Decimal)
        mean_win_rate: Mean win rate (Decimal)
        consistency_score: Consistency score (Decimal)
        confidence_interval_95: 95% CI for mean reward
        statistical_significance: P-value vs baseline
        timestamp: UTC timestamp
    """
    n_episodes: int
    mean_reward: Decimal
    std_reward: Decimal
    mean_episode_length: float
    mean_final_value: Decimal
    mean_sharpe: Decimal
    mean_max_drawdown: Decimal
    mean_win_rate: Decimal
    consistency_score: Decimal
    confidence_interval_95: Tuple[Decimal, Decimal]
    statistical_significance: Optional[float]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class RLEvaluator:
    """Reinforcement learning agent evaluator.

    Evaluates RL agents across multiple episodes with comprehensive
    performance and risk metrics.

    Attributes:
        config: Configuration dictionary
        episodes: List of episode results
        baseline_results: Optional baseline agent results for comparison

    Example:
        >>> config = {
        ...     'n_eval_episodes': 100,
        ...     'initial_portfolio_value': Decimal('100000'),
        ...     'calculate_confidence_intervals': True,
        ...     'compare_to_baseline': True,
        ...     'significance_level': 0.05
        ... }
        >>> evaluator = RLEvaluator(config)
        >>> report = await evaluator.evaluate_agent(agent, env)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize RL evaluator.

        Args:
            config: Configuration with keys:
                - n_eval_episodes: Number of evaluation episodes
                - initial_portfolio_value: Initial portfolio value
                - calculate_confidence_intervals: Whether to calc CIs
                - compare_to_baseline: Whether to compare to baseline
                - significance_level: Statistical significance level
                - max_episode_steps: Maximum steps per episode
        """
        self.config = config
        self._validate_config()

        self.n_eval_episodes = config['n_eval_episodes']
        self.initial_portfolio_value = Decimal(str(config['initial_portfolio_value']))
        self.calculate_confidence_intervals = config['calculate_confidence_intervals']
        self.compare_to_baseline = config['compare_to_baseline']
        self.significance_level = config['significance_level']
        self.max_episode_steps = config.get('max_episode_steps', 10000)

        # State tracking
        self.episodes: List[EpisodeResult] = []
        self.baseline_results: Optional[List[EpisodeResult]] = None

        logger.info(
            "initialized_rl_evaluator",
            n_eval_episodes=self.n_eval_episodes,
            initial_value=str(self.initial_portfolio_value)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'n_eval_episodes', 'initial_portfolio_value',
            'calculate_confidence_intervals', 'compare_to_baseline',
            'significance_level'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['n_eval_episodes'] <= 0:
            raise ValueError("n_eval_episodes must be positive")

        if self.config['initial_portfolio_value'] <= 0:
            raise ValueError("initial_portfolio_value must be positive")

        if not (0 < self.config['significance_level'] < 1):
            raise ValueError("significance_level must be in (0, 1)")

    async def evaluate_episode(
        self,
        agent: Any,
        env: Any,
        episode_id: int,
        render: bool = False
    ) -> EpisodeResult:
        """Evaluate agent for one episode.

        Args:
            agent: RL agent to evaluate
            env: Trading environment
            episode_id: Episode identifier
            render: Whether to render environment

        Returns:
            EpisodeResult object
        """
        try:
            logger.debug("starting_episode_evaluation", episode_id=episode_id)

            # Reset environment
            state = env.reset()
            done = False
            step = 0

            episode_rewards: List[Decimal] = []
            portfolio_values: List[Decimal] = [self.initial_portfolio_value]
            trades: List[Dict[str, Any]] = []

            while not done and step < self.max_episode_steps:
                # Select action (no exploration during evaluation)
                action = agent.select_action(state)

                # Take step
                next_state, reward, done, info = env.step(action)

                # Track metrics
                episode_rewards.append(Decimal(str(reward)))

                if 'portfolio_value' in info:
                    portfolio_values.append(Decimal(str(info['portfolio_value'])))

                if 'trade' in info and info['trade'] is not None:
                    trades.append(info['trade'])

                state = next_state
                step += 1

                if render:
                    env.render()

            # Calculate episode metrics
            total_reward = sum(episode_rewards)
            final_value = portfolio_values[-1]

            # Calculate returns
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)

            # Max drawdown
            max_dd = self._calculate_max_drawdown(portfolio_values)

            # Sharpe ratio
            sharpe = self._calculate_sharpe(returns)

            # Win rate and avg trade return
            if trades:
                winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
                win_rate = Decimal(str(len(winning_trades) / len(trades)))

                trade_returns = [Decimal(str(t.get('return', 0))) for t in trades]
                avg_trade_return = sum(trade_returns) / Decimal(str(len(trades)))
            else:
                win_rate = Decimal('0')
                avg_trade_return = Decimal('0')

            result = EpisodeResult(
                episode_id=episode_id,
                total_reward=total_reward,
                episode_length=step,
                final_value=final_value,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                avg_trade_return=avg_trade_return,
                n_trades=len(trades)
            )

            logger.info(
                "episode_evaluation_complete",
                episode_id=episode_id,
                total_reward=str(total_reward),
                final_value=str(final_value),
                sharpe=str(sharpe)
            )

            return result

        except Exception as e:
            logger.error("failed_episode_evaluation", episode_id=episode_id, error=str(e))
            raise

    def _calculate_max_drawdown(self, values: List[Decimal]) -> Decimal:
        """Calculate maximum drawdown."""
        if len(values) < 2:
            return Decimal('0')

        values_array = np.array([float(v) for v in values])
        running_max = np.maximum.accumulate(values_array)
        drawdown = (running_max - values_array) / running_max

        return Decimal(str(np.max(drawdown)))

    def _calculate_sharpe(self, returns: List[Decimal]) -> Decimal:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return Decimal('0')

        returns_array = np.array([float(r) for r in returns])
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)

        if std_return <= 1e-8:
            return Decimal('0')

        # Annualize (assuming daily returns)
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))

        return Decimal(str(sharpe))

    async def evaluate_agent(
        self,
        agent: Any,
        env: Any,
        render: bool = False
    ) -> EvaluationReport:
        """Evaluate agent across multiple episodes.

        Args:
            agent: RL agent to evaluate
            env: Trading environment
            render: Whether to render environment

        Returns:
            EvaluationReport object
        """
        try:
            logger.info(
                "starting_agent_evaluation",
                n_episodes=self.n_eval_episodes
            )

            self.episodes.clear()

            # Evaluate episodes
            for episode_id in range(self.n_eval_episodes):
                result = await self.evaluate_episode(
                    agent, env, episode_id, render
                )
                self.episodes.append(result)

            # Generate report
            report = self._generate_report()

            logger.info(
                "agent_evaluation_complete",
                mean_reward=str(report.mean_reward),
                mean_sharpe=str(report.mean_sharpe)
            )

            return report

        except Exception as e:
            logger.error("failed_agent_evaluation", error=str(e))
            raise

    def _generate_report(self) -> EvaluationReport:
        """Generate evaluation report from episode results.

        Returns:
            EvaluationReport object
        """
        if not self.episodes:
            raise ValueError("No episodes to evaluate")

        # Extract metrics
        total_rewards = [float(e.total_reward) for e in self.episodes]
        episode_lengths = [e.episode_length for e in self.episodes]
        final_values = [float(e.final_value) for e in self.episodes]
        sharpe_ratios = [float(e.sharpe_ratio) for e in self.episodes]
        max_drawdowns = [float(e.max_drawdown) for e in self.episodes]
        win_rates = [float(e.win_rate) for e in self.episodes]

        # Calculate means
        mean_reward = Decimal(str(np.mean(total_rewards)))
        std_reward = Decimal(str(np.std(total_rewards)))
        mean_episode_length = float(np.mean(episode_lengths))
        mean_final_value = Decimal(str(np.mean(final_values)))
        mean_sharpe = Decimal(str(np.mean(sharpe_ratios)))
        mean_max_drawdown = Decimal(str(np.mean(max_drawdowns)))
        mean_win_rate = Decimal(str(np.mean(win_rates)))

        # Consistency score (1 - CV of rewards)
        cv = float(std_reward / mean_reward) if mean_reward != 0 else 1.0
        consistency_score = Decimal(str(max(0.0, 1.0 - cv)))

        # Confidence intervals
        if self.calculate_confidence_intervals:
            ci = stats.t.interval(
                1 - self.significance_level,
                len(total_rewards) - 1,
                loc=np.mean(total_rewards),
                scale=stats.sem(total_rewards)
            )
            confidence_interval_95 = (Decimal(str(ci[0])), Decimal(str(ci[1])))
        else:
            confidence_interval_95 = (mean_reward, mean_reward)

        # Statistical significance vs baseline
        statistical_significance = None
        if self.compare_to_baseline and self.baseline_results:
            baseline_rewards = [float(e.total_reward) for e in self.baseline_results]
            t_stat, p_value = stats.ttest_ind(total_rewards, baseline_rewards)
            statistical_significance = p_value

            logger.info(
                "baseline_comparison",
                p_value=p_value,
                significant=p_value < self.significance_level
            )

        report = EvaluationReport(
            n_episodes=len(self.episodes),
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_episode_length,
            mean_final_value=mean_final_value,
            mean_sharpe=mean_sharpe,
            mean_max_drawdown=mean_max_drawdown,
            mean_win_rate=mean_win_rate,
            consistency_score=consistency_score,
            confidence_interval_95=confidence_interval_95,
            statistical_significance=statistical_significance
        )

        return report

    def set_baseline_results(self, baseline_episodes: List[EpisodeResult]) -> None:
        """Set baseline results for comparison.

        Args:
            baseline_episodes: Baseline episode results
        """
        self.baseline_results = baseline_episodes

        logger.info(
            "set_baseline_results",
            n_baseline_episodes=len(baseline_episodes)
        )

    def get_episode_dataframe(self) -> pl.DataFrame:
        """Get episodes as Polars DataFrame.

        Returns:
            Polars DataFrame with episode results
        """
        if not self.episodes:
            return pl.DataFrame()

        data = {
            'episode_id': [e.episode_id for e in self.episodes],
            'total_reward': [str(e.total_reward) for e in self.episodes],
            'episode_length': [e.episode_length for e in self.episodes],
            'final_value': [str(e.final_value) for e in self.episodes],
            'max_drawdown': [str(e.max_drawdown) for e in self.episodes],
            'sharpe_ratio': [str(e.sharpe_ratio) for e in self.episodes],
            'win_rate': [str(e.win_rate) for e in self.episodes],
            'avg_trade_return': [str(e.avg_trade_return) for e in self.episodes],
            'n_trades': [e.n_trades for e in self.episodes],
            'timestamp': [e.timestamp.isoformat() for e in self.episodes]
        }

        return pl.DataFrame(data)

    def export_results(self, path: str) -> None:
        """Export evaluation results to file.

        Args:
            path: Export file path (supports .csv, .parquet, .json)
        """
        try:
            df = self.get_episode_dataframe()

            if path.endswith('.csv'):
                df.write_csv(path)
            elif path.endswith('.parquet'):
                df.write_parquet(path)
            elif path.endswith('.json'):
                df.write_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")

            logger.info("exported_results", path=path, n_episodes=len(self.episodes))

        except Exception as e:
            logger.error("failed_to_export_results", error=str(e), path=path)
            raise

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary of summary statistics
        """
        if not self.episodes:
            return {'error': 'No episodes available'}

        total_rewards = [float(e.total_reward) for e in self.episodes]
        final_values = [float(e.final_value) for e in self.episodes]
        sharpe_ratios = [float(e.sharpe_ratio) for e in self.episodes]

        return {
            'n_episodes': len(self.episodes),
            'reward_stats': {
                'mean': float(np.mean(total_rewards)),
                'std': float(np.std(total_rewards)),
                'min': float(np.min(total_rewards)),
                'max': float(np.max(total_rewards)),
                'median': float(np.median(total_rewards)),
                'q25': float(np.percentile(total_rewards, 25)),
                'q75': float(np.percentile(total_rewards, 75))
            },
            'final_value_stats': {
                'mean': float(np.mean(final_values)),
                'std': float(np.std(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values))
            },
            'sharpe_stats': {
                'mean': float(np.mean(sharpe_ratios)),
                'std': float(np.std(sharpe_ratios)),
                'min': float(np.min(sharpe_ratios)),
                'max': float(np.max(sharpe_ratios))
            }
        }
