"""
Walk-Forward Optimization - Robust parameter optimization with out-of-sample testing.

This module implements walk-forward analysis to prevent overfitting by continuously
optimizing parameters on in-sample data and validating on out-of-sample periods.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationWindow:
    """Represents a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, Any]
    train_metrics: Dict[str, Decimal]
    test_metrics: Dict[str, Decimal]

    @property
    def train_days(self) -> int:
        """Number of days in training period."""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """Number of days in testing period."""
        return (self.test_end - self.test_start).days


@dataclass
class WalkForwardResults:
    """Complete walk-forward optimization results."""

    windows: List[OptimizationWindow]
    combined_test_metrics: Dict[str, Decimal]
    parameter_stability: Dict[str, Decimal]
    total_windows: int
    avg_train_performance: Decimal
    avg_test_performance: Decimal
    performance_degradation: Decimal
    is_robust: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class WalkForwardOptimizer:
    """
    Performs walk-forward optimization to find robust strategy parameters.

    Walk-forward analysis divides historical data into multiple windows,
    optimizing parameters on each in-sample (training) period and validating
    on the subsequent out-of-sample (testing) period.

    Attributes:
        config: Optimization configuration
        objective_function: Function to optimize (strategy backtest)
        param_grid: Parameter search space

    Example:
        >>> optimizer = WalkForwardOptimizer(config, objective_func, param_grid)
        >>> results = await optimizer.optimize(market_data)
        >>> print(f"Robust parameters found: {results.parameter_stability}")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        objective_function: Callable,
        param_grid: Dict[str, List[Any]]
    ) -> None:
        """
        Initialize walk-forward optimizer.

        Args:
            config: Configuration dict with keys:
                - train_period_days: Training window size
                - test_period_days: Testing window size
                - step_days: Step size between windows
                - min_train_samples: Minimum samples for training
                - max_workers: Parallel optimization workers
                - optimization_metric: Metric to optimize (sharpe/profit_factor/etc)
                - robustness_threshold: Min performance ratio (test/train)
            objective_function: Async function(data, params) -> metrics
            param_grid: Dictionary of parameter names to list of values

        Raises:
            ValueError: If config invalid
        """
        self.config = config
        self.objective_function = objective_function
        self.param_grid = param_grid

        self._validate_config()

        self.train_period_days = int(config['train_period_days'])
        self.test_period_days = int(config['test_period_days'])
        self.step_days = int(config.get('step_days', self.test_period_days))
        self.min_train_samples = int(config.get('min_train_samples', 1000))
        self.max_workers = int(config.get('max_workers', 4))
        self.optimization_metric = config.get('optimization_metric', 'sharpe_ratio')
        self.robustness_threshold = Decimal(str(config.get('robustness_threshold', '0.7')))

        logger.info(
            "WalkForwardOptimizer initialized",
            train_days=self.train_period_days,
            test_days=self.test_period_days,
            step_days=self.step_days,
            metric=self.optimization_metric
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required = [
            'train_period_days',
            'test_period_days'
        ]
        missing = [k for k in required if k not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        if not self.param_grid:
            raise ValueError("param_grid cannot be empty")

        if not callable(self.objective_function):
            raise ValueError("objective_function must be callable")

    async def optimize(self, market_data: pl.DataFrame) -> WalkForwardResults:
        """
        Execute walk-forward optimization.

        Args:
            market_data: Historical market data with timestamp column

        Returns:
            WalkForwardResults with optimal parameters and validation metrics

        Raises:
            ValueError: If insufficient data for optimization
        """
        try:
            # Validate data
            if 'timestamp' not in market_data.columns:
                raise ValueError("market_data must contain 'timestamp' column")

            # Generate walk-forward windows
            windows_config = self._generate_windows(market_data)

            if len(windows_config) == 0:
                raise ValueError(
                    f"Insufficient data for walk-forward optimization. "
                    f"Need at least {self.train_period_days + self.test_period_days} days"
                )

            logger.info(
                "Starting walk-forward optimization",
                total_windows=len(windows_config),
                param_combinations=self._count_param_combinations()
            )

            # Optimize each window
            windows = []
            for window_config in windows_config:
                window_result = await self._optimize_window(market_data, window_config)
                windows.append(window_result)

                logger.info(
                    "Window optimized",
                    window_id=window_result.window_id,
                    train_metric=str(window_result.train_metrics.get(self.optimization_metric, Decimal('0'))),
                    test_metric=str(window_result.test_metrics.get(self.optimization_metric, Decimal('0')))
                )

            # Combine results
            results = self._aggregate_results(windows)

            logger.info(
                "Walk-forward optimization complete",
                windows=len(windows),
                avg_train_perf=str(results.avg_train_performance),
                avg_test_perf=str(results.avg_test_performance),
                is_robust=results.is_robust
            )

            return results

        except Exception as e:
            logger.error("Walk-forward optimization failed", error=str(e))
            raise

    def _generate_windows(
        self,
        market_data: pl.DataFrame
    ) -> List[Dict[str, datetime]]:
        """
        Generate walk-forward window configurations.

        Args:
            market_data: Market data with timestamps

        Returns:
            List of window configurations with train/test periods
        """
        timestamps = market_data.select('timestamp').to_series().sort()

        start_date = timestamps.min()
        end_date = timestamps.max()

        windows = []
        window_id = 0

        current_start = start_date

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            # Check minimum samples in training period
            train_samples = market_data.filter(
                (pl.col('timestamp') >= train_start) &
                (pl.col('timestamp') < train_end)
            ).height

            if train_samples < self.min_train_samples:
                logger.warning(
                    "Skipping window due to insufficient training samples",
                    window_id=window_id,
                    samples=train_samples,
                    required=self.min_train_samples
                )
                current_start = current_start + timedelta(days=self.step_days)
                continue

            windows.append({
                'window_id': window_id,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            window_id += 1
            current_start = current_start + timedelta(days=self.step_days)

        return windows

    async def _optimize_window(
        self,
        market_data: pl.DataFrame,
        window_config: Dict[str, Any]
    ) -> OptimizationWindow:
        """
        Optimize parameters for a single walk-forward window.

        Args:
            market_data: Full market data
            window_config: Window configuration with train/test periods

        Returns:
            OptimizationWindow with optimal parameters and metrics
        """
        try:
            # Extract training data
            train_data = market_data.filter(
                (pl.col('timestamp') >= window_config['train_start']) &
                (pl.col('timestamp') < window_config['train_end'])
            )

            # Extract testing data
            test_data = market_data.filter(
                (pl.col('timestamp') >= window_config['test_start']) &
                (pl.col('timestamp') < window_config['test_end'])
            )

            # Grid search on training data
            best_params, best_train_metrics = await self._grid_search(train_data)

            # Validate on test data
            test_metrics = await self.objective_function(test_data, best_params)

            window = OptimizationWindow(
                window_id=window_config['window_id'],
                train_start=window_config['train_start'],
                train_end=window_config['train_end'],
                test_start=window_config['test_start'],
                test_end=window_config['test_end'],
                optimal_params=best_params,
                train_metrics=self._convert_metrics_to_decimal(best_train_metrics),
                test_metrics=self._convert_metrics_to_decimal(test_metrics)
            )

            return window

        except Exception as e:
            logger.error(
                "Window optimization failed",
                window_id=window_config['window_id'],
                error=str(e)
            )
            raise

    async def _grid_search(
        self,
        train_data: pl.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform grid search to find optimal parameters.

        Args:
            train_data: Training data

        Returns:
            Tuple of (best_params, best_metrics)
        """
        param_combinations = self._generate_param_combinations()

        best_params = None
        best_score = Decimal('-Infinity')
        best_metrics = {}

        # Process in batches for parallel execution
        batch_size = self.max_workers

        for i in range(0, len(param_combinations), batch_size):
            batch = param_combinations[i:i + batch_size]

            # Evaluate batch concurrently
            tasks = [
                self.objective_function(train_data, params)
                for params in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Find best in batch
            for params, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Parameter evaluation failed",
                        params=params,
                        error=str(result)
                    )
                    continue

                score = Decimal(str(result.get(self.optimization_metric, 0)))

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = result

        if best_params is None:
            raise ValueError("No valid parameter combinations found")

        return best_params, best_metrics

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    def _count_param_combinations(self) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in self.param_grid.values():
            count *= len(values)
        return count

    def _convert_metrics_to_decimal(self, metrics: Dict[str, Any]) -> Dict[str, Decimal]:
        """Convert metrics dictionary values to Decimal."""
        decimal_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                try:
                    decimal_metrics[key] = Decimal(str(value))
                except:
                    decimal_metrics[key] = value
            else:
                decimal_metrics[key] = value
        return decimal_metrics

    def _aggregate_results(self, windows: List[OptimizationWindow]) -> WalkForwardResults:
        """
        Aggregate results from all windows.

        Args:
            windows: List of optimization windows

        Returns:
            Aggregated WalkForwardResults
        """
        # Calculate average performance
        train_scores = [
            w.train_metrics.get(self.optimization_metric, Decimal('0'))
            for w in windows
        ]
        test_scores = [
            w.test_metrics.get(self.optimization_metric, Decimal('0'))
            for w in windows
        ]

        avg_train = sum(train_scores) / Decimal(str(len(train_scores))) if train_scores else Decimal('0')
        avg_test = sum(test_scores) / Decimal(str(len(test_scores))) if test_scores else Decimal('0')

        # Calculate performance degradation
        degradation = (avg_test / avg_train) if avg_train > Decimal('0') else Decimal('0')

        # Check robustness
        is_robust = degradation >= self.robustness_threshold

        # Calculate parameter stability (how consistent parameters are across windows)
        param_stability = self._calculate_parameter_stability(windows)

        # Combine test metrics across all windows
        combined_test_metrics = self._combine_test_metrics(windows)

        results = WalkForwardResults(
            windows=windows,
            combined_test_metrics=combined_test_metrics,
            parameter_stability=param_stability,
            total_windows=len(windows),
            avg_train_performance=avg_train,
            avg_test_performance=avg_test,
            performance_degradation=degradation,
            is_robust=is_robust,
            metadata={
                'optimization_metric': self.optimization_metric,
                'robustness_threshold': str(self.robustness_threshold),
                'generated_at': datetime.utcnow()
            }
        )

        return results

    def _calculate_parameter_stability(
        self,
        windows: List[OptimizationWindow]
    ) -> Dict[str, Decimal]:
        """
        Calculate stability score for each parameter.

        High stability means parameter values are consistent across windows.

        Returns:
            Dictionary mapping parameter names to stability scores (0-1)
        """
        stability = {}

        for param_name in self.param_grid.keys():
            values = [w.optimal_params.get(param_name) for w in windows]

            # Calculate coefficient of variation for numeric parameters
            numeric_values = [v for v in values if isinstance(v, (int, float, Decimal))]

            if numeric_values:
                decimal_values = [Decimal(str(v)) for v in numeric_values]
                mean = sum(decimal_values) / Decimal(str(len(decimal_values)))

                if mean == Decimal('0'):
                    stability[param_name] = Decimal('0')
                else:
                    variance = sum((v - mean) ** 2 for v in decimal_values) / Decimal(str(len(decimal_values)))
                    std_dev = variance.sqrt()
                    cv = std_dev / mean  # Coefficient of variation

                    # Convert CV to stability score (0-1, higher is more stable)
                    stability[param_name] = Decimal('1') / (Decimal('1') + cv)
            else:
                # For categorical parameters, measure consistency
                unique_values = len(set(values))
                total_values = len(values)
                stability[param_name] = Decimal('1') - (Decimal(str(unique_values - 1)) / Decimal(str(total_values)))

        return stability

    def _combine_test_metrics(
        self,
        windows: List[OptimizationWindow]
    ) -> Dict[str, Decimal]:
        """Combine test metrics across all windows."""
        combined = {}

        # Get all metric keys
        all_keys = set()
        for window in windows:
            all_keys.update(window.test_metrics.keys())

        # Average each metric
        for key in all_keys:
            values = [
                w.test_metrics.get(key, Decimal('0'))
                for w in windows
                if key in w.test_metrics
            ]
            if values:
                combined[key] = sum(values) / Decimal(str(len(values)))
            else:
                combined[key] = Decimal('0')

        return combined

    def get_best_stable_params(self, results: WalkForwardResults) -> Dict[str, Any]:
        """
        Get the most stable parameter set from walk-forward results.

        Args:
            results: WalkForwardResults object

        Returns:
            Dictionary of most stable parameters
        """
        # Count frequency of each parameter combination
        param_combinations = {}

        for window in results.windows:
            params_tuple = tuple(sorted(window.optimal_params.items()))
            param_combinations[params_tuple] = param_combinations.get(params_tuple, 0) + 1

        # Get most frequent combination
        most_common = max(param_combinations.items(), key=lambda x: x[1])

        return dict(most_common[0])
