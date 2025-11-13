"""Random search hyperparameter optimization.

This module implements random search for hyperparameter tuning with support
for parallel evaluation and early stopping.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logger = structlog.get_logger(__name__)


@dataclass
class ParameterSpace:
    """Parameter space definition.

    Attributes:
        name: Parameter name
        param_type: 'int', 'float', 'choice', 'loguniform'
        low: Lower bound (for numeric types)
        high: Upper bound (for numeric types)
        choices: List of choices (for choice type)
        log_scale: Use log scale for sampling
    """
    name: str
    param_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False

    def sample(self) -> Any:
        """Sample a value from this parameter space.

        Returns:
            Sampled parameter value

        Raises:
            ValueError: If parameter space is invalid
        """
        if self.param_type == 'int':
            if self.low is None or self.high is None:
                raise ValueError(f"int parameter {self.name} requires low and high")
            if self.log_scale:
                value = int(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))))
            else:
                value = np.random.randint(int(self.low), int(self.high) + 1)
            return value

        elif self.param_type == 'float':
            if self.low is None or self.high is None:
                raise ValueError(f"float parameter {self.name} requires low and high")
            if self.log_scale:
                value = np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
            else:
                value = np.random.uniform(self.low, self.high)
            return float(value)

        elif self.param_type == 'choice':
            if self.choices is None or len(self.choices) == 0:
                raise ValueError(f"choice parameter {self.name} requires choices")
            return random.choice(self.choices)

        elif self.param_type == 'loguniform':
            if self.low is None or self.high is None:
                raise ValueError(f"loguniform parameter {self.name} requires low and high")
            value = np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
            return float(value)

        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class Trial:
    """Single trial result.

    Attributes:
        trial_id: Trial identifier
        parameters: Parameter configuration
        score: Evaluation score (Decimal)
        metrics: Additional metrics
        duration: Trial duration in seconds
        timestamp: UTC timestamp
        error: Error message if trial failed
    """
    trial_id: int
    parameters: Dict[str, Any]
    score: Optional[Decimal] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if trial succeeded."""
        return self.score is not None and self.error is None


class RandomSearch:
    """Random search hyperparameter optimization.

    Implements random search with support for parallel evaluation,
    early stopping, and comprehensive result tracking.

    Attributes:
        config: Configuration dictionary
        param_spaces: List of parameter spaces
        objective_fn: Objective function to minimize/maximize
        maximize: Whether to maximize objective (default: minimize)

    Example:
        >>> param_spaces = [
        ...     ParameterSpace('learning_rate', 'loguniform', low=1e-5, high=1e-2),
        ...     ParameterSpace('batch_size', 'choice', choices=[32, 64, 128]),
        ...     ParameterSpace('n_layers', 'int', low=2, high=5)
        ... ]
        >>>
        >>> def objective(params):
        ...     # Train model and return validation score
        ...     return validation_score
        ...
        >>> config = {
        ...     'n_trials': 100,
        ...     'n_jobs': 4,
        ...     'maximize': False,
        ...     'early_stopping_rounds': 20,
        ...     'early_stopping_threshold': 0.01,
        ...     'random_state': 42
        ... }
        >>> search = RandomSearch(config, param_spaces, objective)
        >>> best_params, best_score = search.optimize()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        param_spaces: List[ParameterSpace],
        objective_fn: Callable[[Dict[str, Any]], float]
    ) -> None:
        """Initialize random search optimizer.

        Args:
            config: Configuration with keys:
                - n_trials: Number of trials to run
                - n_jobs: Number of parallel jobs
                - maximize: Whether to maximize objective
                - early_stopping_rounds: Rounds without improvement before stopping
                - early_stopping_threshold: Minimum improvement threshold
                - random_state: Random seed
                - timeout_per_trial: Maximum seconds per trial
            param_spaces: List of ParameterSpace objects
            objective_fn: Objective function taking params dict and returning score
        """
        self.config = config
        self._validate_config()

        self.param_spaces = param_spaces
        self.objective_fn = objective_fn

        self.n_trials = config['n_trials']
        self.n_jobs = config['n_jobs']
        self.maximize = config['maximize']
        self.early_stopping_rounds = config.get('early_stopping_rounds')
        self.early_stopping_threshold = config.get('early_stopping_threshold', Decimal('0.01'))
        self.timeout_per_trial = config.get('timeout_per_trial')

        # Set random seed
        if 'random_state' in config:
            np.random.seed(config['random_state'])
            random.seed(config['random_state'])

        # State tracking
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.trials_without_improvement = 0

        logger.info(
            "initialized_random_search",
            n_trials=self.n_trials,
            n_params=len(param_spaces),
            maximize=self.maximize,
            n_jobs=self.n_jobs
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = ['n_trials', 'n_jobs', 'maximize']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['n_trials'] <= 0:
            raise ValueError("n_trials must be positive")

        if self.config['n_jobs'] <= 0:
            raise ValueError("n_jobs must be positive")

    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from parameter spaces.

        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        for param_space in self.param_spaces:
            params[param_space.name] = param_space.sample()
        return params

    def _evaluate_trial(
        self,
        trial_id: int,
        parameters: Dict[str, Any]
    ) -> Trial:
        """Evaluate a single trial.

        Args:
            trial_id: Trial identifier
            parameters: Parameter configuration

        Returns:
            Trial object with results
        """
        start_time = datetime.utcnow()

        try:
            logger.debug("starting_trial", trial_id=trial_id, parameters=parameters)

            # Evaluate objective function
            score = self.objective_fn(parameters)

            # Convert to Decimal
            score_decimal = Decimal(str(score))

            duration = (datetime.utcnow() - start_time).total_seconds()

            trial = Trial(
                trial_id=trial_id,
                parameters=parameters,
                score=score_decimal,
                duration=duration
            )

            logger.info(
                "trial_complete",
                trial_id=trial_id,
                score=str(score_decimal),
                duration=duration
            )

            return trial

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()

            trial = Trial(
                trial_id=trial_id,
                parameters=parameters,
                error=str(e),
                duration=duration
            )

            logger.error(
                "trial_failed",
                trial_id=trial_id,
                error=str(e),
                duration=duration
            )

            return trial

    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met.

        Returns:
            True if should stop early
        """
        if self.early_stopping_rounds is None:
            return False

        if self.trials_without_improvement >= self.early_stopping_rounds:
            logger.info(
                "early_stopping_triggered",
                trials_without_improvement=self.trials_without_improvement,
                threshold=self.early_stopping_rounds
            )
            return True

        return False

    def _update_best_trial(self, trial: Trial) -> bool:
        """Update best trial if current trial is better.

        Args:
            trial: Current trial

        Returns:
            True if best trial was updated
        """
        if not trial.is_success:
            return False

        if self.best_trial is None:
            self.best_trial = trial
            self.trials_without_improvement = 0
            logger.info(
                "new_best_trial",
                trial_id=trial.trial_id,
                score=str(trial.score)
            )
            return True

        # Check if improvement
        if self.maximize:
            is_better = trial.score > self.best_trial.score
            improvement = trial.score - self.best_trial.score
        else:
            is_better = trial.score < self.best_trial.score
            improvement = self.best_trial.score - trial.score

        if is_better:
            # Check if improvement is significant
            if self.early_stopping_threshold is not None:
                relative_improvement = abs(improvement / self.best_trial.score)
                if relative_improvement < self.early_stopping_threshold:
                    self.trials_without_improvement += 1
                    return False

            self.best_trial = trial
            self.trials_without_improvement = 0

            logger.info(
                "new_best_trial",
                trial_id=trial.trial_id,
                score=str(trial.score),
                improvement=str(improvement)
            )
            return True
        else:
            self.trials_without_improvement += 1
            return False

    def optimize(self) -> Tuple[Dict[str, Any], Decimal]:
        """Run random search optimization.

        Returns:
            Tuple of (best_parameters, best_score)

        Raises:
            ValueError: If optimization fails
        """
        try:
            logger.info("starting_optimization", n_trials=self.n_trials)

            if self.n_jobs == 1:
                # Sequential execution
                for trial_id in range(self.n_trials):
                    if self._check_early_stopping():
                        break

                    params = self._sample_parameters()
                    trial = self._evaluate_trial(trial_id, params)
                    self.trials.append(trial)
                    self._update_best_trial(trial)

            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futures = []

                    for trial_id in range(self.n_trials):
                        if self._check_early_stopping():
                            break

                        params = self._sample_parameters()
                        future = executor.submit(
                            self._evaluate_trial,
                            trial_id,
                            params
                        )
                        futures.append(future)

                    # Collect results as they complete
                    for future in as_completed(futures):
                        trial = future.result()
                        self.trials.append(trial)
                        self._update_best_trial(trial)

                        if self._check_early_stopping():
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break

            if self.best_trial is None:
                raise ValueError("No successful trials completed")

            logger.info(
                "optimization_complete",
                total_trials=len(self.trials),
                successful_trials=sum(1 for t in self.trials if t.is_success),
                best_score=str(self.best_trial.score)
            )

            return self.best_trial.parameters, self.best_trial.score

        except Exception as e:
            logger.error("optimization_failed", error=str(e))
            raise

    def get_results(self) -> Dict[str, Any]:
        """Get optimization results and statistics.

        Returns:
            Dictionary of results and statistics
        """
        successful_trials = [t for t in self.trials if t.is_success]

        if not successful_trials:
            return {
                'total_trials': len(self.trials),
                'successful_trials': 0,
                'failed_trials': len(self.trials),
                'best_trial': None
            }

        scores = [float(t.score) for t in successful_trials]

        results = {
            'total_trials': len(self.trials),
            'successful_trials': len(successful_trials),
            'failed_trials': len(self.trials) - len(successful_trials),
            'best_trial': {
                'trial_id': self.best_trial.trial_id,
                'parameters': self.best_trial.parameters,
                'score': str(self.best_trial.score),
                'duration': self.best_trial.duration
            },
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores))
            },
            'all_trials': [
                {
                    'trial_id': t.trial_id,
                    'parameters': t.parameters,
                    'score': str(t.score) if t.score is not None else None,
                    'duration': t.duration,
                    'error': t.error
                }
                for t in self.trials
            ]
        }

        return results

    def get_parameter_importance(self) -> Dict[str, Decimal]:
        """Estimate parameter importance using variance analysis.

        Returns:
            Dictionary mapping parameter names to importance scores

        Raises:
            ValueError: If not enough successful trials
        """
        try:
            successful_trials = [t for t in self.trials if t.is_success]

            if len(successful_trials) < 10:
                raise ValueError("Need at least 10 successful trials for importance analysis")

            importance_scores = {}

            for param_space in self.param_spaces:
                param_name = param_space.name

                # Skip choice parameters with too few values
                if param_space.param_type == 'choice' and len(param_space.choices) < 2:
                    continue

                # Group scores by parameter value bins
                param_values = []
                scores = []

                for trial in successful_trials:
                    param_values.append(trial.parameters[param_name])
                    scores.append(float(trial.score))

                param_values = np.array(param_values)
                scores = np.array(scores)

                # Calculate correlation for numeric parameters
                if param_space.param_type in ['int', 'float', 'loguniform']:
                    correlation = np.abs(np.corrcoef(param_values, scores)[0, 1])
                    importance_scores[param_name] = Decimal(str(correlation))
                else:
                    # For categorical, use ANOVA-style variance ratio
                    unique_values = np.unique(param_values)
                    if len(unique_values) > 1:
                        between_var = Decimal('0')
                        for val in unique_values:
                            mask = param_values == val
                            if np.sum(mask) > 0:
                                group_mean = np.mean(scores[mask])
                                overall_mean = np.mean(scores)
                                between_var += Decimal(str((group_mean - overall_mean) ** 2))

                        total_var = Decimal(str(np.var(scores)))
                        if total_var > Decimal('1e-8'):
                            importance_scores[param_name] = between_var / total_var
                        else:
                            importance_scores[param_name] = Decimal('0')

            logger.info(
                "calculated_parameter_importance",
                importance_scores={k: str(v) for k, v in importance_scores.items()}
            )

            return importance_scores

        except Exception as e:
            logger.error("failed_to_calculate_importance", error=str(e))
            raise
