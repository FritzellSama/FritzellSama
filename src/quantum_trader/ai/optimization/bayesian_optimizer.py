"""
Bayesian Optimization for Hyperparameter Tuning

CRITICAL: Production-grade Bayesian optimization for ML models
- Gaussian Process-based optimization
- Acquisition functions: EI, UCB, PI
- Handles high-dimensional spaces
- Parallel optimization support
"""

from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import os
import logging
from abc import ABC, abstractmethod

import numpy as np
import polars as pl

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
    from scipy.stats import norm
    from scipy.optimize import minimize
except ImportError as e:
    raise ImportError(f"Required ML libraries not installed: {e}")


logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization"""
    n_iterations: int
    n_initial_points: int
    acquisition_function: str  # 'ei', 'ucb', 'pi'
    kernel_type: str  # 'matern', 'rbf'
    kappa: Decimal  # UCB exploration parameter
    xi: Decimal  # EI/PI exploration parameter
    random_state: Optional[int] = None
    n_restarts_optimizer: int = 10
    normalize_y: bool = True


@dataclass
class OptimizationResult:
    """Result from Bayesian optimization"""
    best_params: Dict[str, Decimal]
    best_score: Decimal
    all_params: List[Dict[str, Decimal]]
    all_scores: List[Decimal]
    convergence_history: pl.DataFrame
    optimization_time: Decimal
    n_iterations: int
    timestamp: datetime


class BayesianOptimizer:
    """
    Bayesian Hyperparameter Optimization

    Uses Gaussian Process regression to model the objective function
    and selects next points using acquisition functions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize Bayesian optimizer

        Args:
            config: Configuration dictionary loaded from config files
        """
        self.config = self._load_config(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize GP kernel
        self.kernel = self._create_kernel()

        # Initialize GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=float(config.get('gp_alpha', '1e-6')),
            normalize_y=self.config.normalize_y,
            n_restarts_optimizer=self.config.n_restarts_optimizer,
            random_state=self.config.random_state
        )

        # Tracking
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.iteration = 0

        self.logger.info(f"BayesianOptimizer initialized with {self.config.acquisition_function} acquisition")

    def _load_config(self, config: Dict[str, Any]) -> OptimizationConfig:
        """Load configuration from dict"""
        try:
            return OptimizationConfig(
                n_iterations=int(config.get('n_iterations', os.getenv('BAYES_OPT_ITERATIONS', '100'))),
                n_initial_points=int(config.get('n_initial_points', os.getenv('BAYES_OPT_INITIAL', '10'))),
                acquisition_function=config.get('acquisition_function', os.getenv('BAYES_OPT_ACQ', 'ei')),
                kernel_type=config.get('kernel_type', os.getenv('BAYES_OPT_KERNEL', 'matern')),
                kappa=Decimal(str(config.get('kappa', os.getenv('BAYES_OPT_KAPPA', '2.576')))),
                xi=Decimal(str(config.get('xi', os.getenv('BAYES_OPT_XI', '0.01')))),
                random_state=config.get('random_state'),
                n_restarts_optimizer=int(config.get('n_restarts_optimizer', '10')),
                normalize_y=config.get('normalize_y', True)
            )
        except (ValueError, KeyError) as e:
            self.logger.error(f"Configuration error: {e}")
            raise ValueError(f"Invalid Bayesian optimization configuration: {e}")

    def _create_kernel(self) -> Any:
        """Create GP kernel based on configuration"""
        if self.config.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        elif self.config.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0)
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

    def optimize(
        self,
        objective_function: Callable[[Dict[str, Decimal]], Decimal],
        parameter_bounds: Dict[str, Tuple[Decimal, Decimal]],
        maximize: bool = True
    ) -> OptimizationResult:
        """
        Run Bayesian optimization

        Args:
            objective_function: Function to optimize, returns score
            parameter_bounds: Dict of parameter names to (min, max) bounds
            maximize: Whether to maximize (True) or minimize (False) objective

        Returns:
            OptimizationResult with best parameters and optimization history
        """
        start_time = datetime.utcnow()
        self.logger.info(f"Starting Bayesian optimization for {self.config.n_iterations} iterations")

        try:
            # Convert parameter bounds to arrays
            param_names = list(parameter_bounds.keys())
            bounds = np.array([[float(b[0]), float(b[1])] for b in parameter_bounds.values()])

            # Reset state
            self.X_observed = []
            self.y_observed = []
            self.iteration = 0

            # Random initialization
            self._initialize_random(objective_function, param_names, bounds, maximize)

            # Bayesian optimization loop
            for iteration in range(self.config.n_initial_points, self.config.n_iterations):
                self.iteration = iteration

                # Fit GP to observed data
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

                # Find next point using acquisition function
                next_point = self._get_next_point(bounds)

                # Evaluate objective
                params_dict = {name: Decimal(str(val)) for name, val in zip(param_names, next_point)}
                score = objective_function(params_dict)

                # Store observation (negate if minimizing)
                self.X_observed.append(next_point)
                self.y_observed.append(float(score) if maximize else -float(score))

                # Log progress
                if iteration % max(1, self.config.n_iterations // 10) == 0:
                    best_idx = np.argmax(self.y_observed)
                    best_score = self.y_observed[best_idx]
                    self.logger.info(
                        f"Iteration {iteration}/{self.config.n_iterations}: "
                        f"Current={score:.6f}, Best={best_score if maximize else -best_score:.6f}"
                    )

            # Extract results
            best_idx = np.argmax(self.y_observed)
            best_params = {
                name: Decimal(str(val))
                for name, val in zip(param_names, self.X_observed[best_idx])
            }
            best_score = Decimal(str(self.y_observed[best_idx])) if maximize else Decimal(str(-self.y_observed[best_idx]))

            # Create convergence history
            convergence_df = pl.DataFrame({
                'iteration': list(range(len(self.y_observed))),
                'score': [Decimal(str(s if maximize else -s)) for s in self.y_observed],
                'best_score': [
                    Decimal(str(max(self.y_observed[:i+1]) if maximize else -min(self.y_observed[:i+1])))
                    for i in range(len(self.y_observed))
                ]
            })

            optimization_time = Decimal(str((datetime.utcnow() - start_time).total_seconds()))

            self.logger.info(
                f"Optimization complete in {optimization_time}s. "
                f"Best score: {best_score}, Best params: {best_params}"
            )

            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_params=[
                    {name: Decimal(str(val)) for name, val in zip(param_names, x)}
                    for x in self.X_observed
                ],
                all_scores=[Decimal(str(s if maximize else -s)) for s in self.y_observed],
                convergence_history=convergence_df,
                optimization_time=optimization_time,
                n_iterations=len(self.y_observed),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise RuntimeError(f"Bayesian optimization error: {e}")

    def _initialize_random(
        self,
        objective_function: Callable[[Dict[str, Decimal]], Decimal],
        param_names: List[str],
        bounds: np.ndarray,
        maximize: bool
    ) -> None:
        """Initialize with random points"""
        self.logger.info(f"Random initialization with {self.config.n_initial_points} points")

        rng = np.random.RandomState(self.config.random_state)

        for i in range(self.config.n_initial_points):
            # Sample random point
            point = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.rand(len(bounds))

            # Evaluate
            params_dict = {name: Decimal(str(val)) for name, val in zip(param_names, point)}
            score = objective_function(params_dict)

            # Store (negate if minimizing)
            self.X_observed.append(point)
            self.y_observed.append(float(score) if maximize else -float(score))

            self.logger.debug(f"Init point {i+1}/{self.config.n_initial_points}: score={score}")

    def _get_next_point(self, bounds: np.ndarray) -> np.ndarray:
        """Get next point to evaluate using acquisition function"""
        # Define acquisition function
        if self.config.acquisition_function == 'ei':
            acq_func = self._expected_improvement
        elif self.config.acquisition_function == 'ucb':
            acq_func = self._upper_confidence_bound
        elif self.config.acquisition_function == 'pi':
            acq_func = self._probability_of_improvement
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")

        # Optimize acquisition function
        best_acq = -np.inf
        best_point = None

        rng = np.random.RandomState(self.iteration)

        for _ in range(self.config.n_restarts_optimizer):
            # Random starting point
            x0 = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.rand(len(bounds))

            # Minimize negative acquisition
            result = minimize(
                fun=lambda x: -acq_func(x.reshape(1, -1)),
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if -result.fun > best_acq:
                best_acq = -result.fun
                best_point = result.x

        return best_point

    def _expected_improvement(self, X: np.ndarray) -> float:
        """Expected Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)

        # Best observed value
        y_best = np.max(self.y_observed)

        # Calculate EI
        with np.errstate(divide='warn', invalid='warn'):
            xi = float(self.config.xi)
            Z = (mu - y_best - xi) / (sigma + 1e-9)
            ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return float(ei[0])

    def _upper_confidence_bound(self, X: np.ndarray) -> float:
        """Upper Confidence Bound acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)
        kappa = float(self.config.kappa)
        return float(mu[0] + kappa * sigma[0])

    def _probability_of_improvement(self, X: np.ndarray) -> float:
        """Probability of Improvement acquisition function"""
        mu, sigma = self.gp.predict(X, return_std=True)

        y_best = np.max(self.y_observed)

        with np.errstate(divide='warn', invalid='warn'):
            xi = float(self.config.xi)
            Z = (mu - y_best - xi) / (sigma + 1e-9)
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return float(pi[0])


async def optimize_hyperparameters_async(
    objective_function: Callable[[Dict[str, Decimal]], Decimal],
    parameter_bounds: Dict[str, Tuple[Decimal, Decimal]],
    config: Dict[str, Any],
    maximize: bool = True
) -> OptimizationResult:
    """
    Async wrapper for Bayesian optimization

    Args:
        objective_function: Objective to optimize
        parameter_bounds: Parameter search space
        config: Optimization configuration
        maximize: Whether to maximize objective

    Returns:
        OptimizationResult
    """
    optimizer = BayesianOptimizer(config)

    # Run optimization in executor to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        optimizer.optimize,
        objective_function,
        parameter_bounds,
        maximize
    )

    return result
