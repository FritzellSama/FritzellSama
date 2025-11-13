"""Advanced hyperparameter tuning using Bayesian optimization.

This module implements sophisticated hyperparameter search using
Bayesian optimization, TPE, and other advanced methods beyond grid search.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning.

    Uses Gaussian processes to model objective function and
    efficiently explore hyperparameter space.

    Attributes:
        config: Configuration dictionary
        objective_fn: Function to minimize/maximize
        param_space: Parameter search space
        acquisition_fn: Acquisition function type

    Example:
        >>> config = {
        ...     "n_iter": 50,
        ...     "n_init": 10,
        ...     "acquisition": "ei",
        ...     "maximize": True
        ... }
        >>> param_space = {
        ...     "learning_rate": ("log_uniform", "-4", "-2"),
        ...     "hidden_dim": ("int_uniform", "64", "256"),
        ...     "dropout": ("uniform", "0.1", "0.5")
        ... }
        >>> optimizer = BayesianOptimizer(config)
        >>> best_params, history = await optimizer.optimize(
        ...     objective_fn, param_space
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Bayesian optimizer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.n_iter = config.get("n_iter", 50)
        self.n_init = config.get("n_init", 10)
        self.acquisition = config.get("acquisition", "ei")  # ei, ucb, poi
        self.maximize = config.get("maximize", True)
        self.xi = Decimal(str(config.get("xi", "0.01")))  # Exploration parameter
        self.kappa = Decimal(str(config.get("kappa", "2.576")))  # UCB parameter

        # Optimization state
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[Decimal] = []
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[Decimal] = None

        logger.info(
            "Bayesian optimizer initialized",
            n_iter=self.n_iter,
            n_init=self.n_init,
            acquisition=self.acquisition
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        valid_acquisitions = ["ei", "ucb", "poi"]
        acquisition = self.config.get("acquisition", "ei")

        if acquisition not in valid_acquisitions:
            raise ValueError(
                f"Invalid acquisition '{acquisition}'. "
                f"Must be one of {valid_acquisitions}"
            )

    async def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], Decimal],
        param_space: Dict[str, Tuple[str, str, str]],
        constraints: Optional[List[Callable]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform Bayesian optimization.

        Args:
            objective_fn: Function to optimize (returns score)
            param_space: Dictionary mapping param names to (type, min, max)
            constraints: Optional list of constraint functions

        Returns:
            Tuple of (best_params, optimization_history)

        Raises:
            ValueError: If param_space is invalid
        """
        try:
            if not param_space:
                raise ValueError("param_space cannot be empty")

            logger.info(
                "Starting Bayesian optimization",
                n_iter=self.n_iter,
                param_space=param_space
            )

            # Random initialization
            for i in range(self.n_init):
                params = self._sample_params(param_space)

                # Check constraints
                if constraints and not all(c(params) for c in constraints):
                    continue

                # Evaluate
                score = await self._evaluate_params(objective_fn, params, i)

            # Bayesian optimization iterations
            for i in range(self.n_init, self.n_iter):
                # Fit surrogate model (Gaussian process)
                gp_mean, gp_std = self._fit_gp()

                # Optimize acquisition function
                next_params = self._optimize_acquisition(
                    param_space,
                    gp_mean,
                    gp_std,
                    constraints
                )

                # Evaluate
                score = await self._evaluate_params(objective_fn, next_params, i)

                # Check for early stopping
                if self._should_stop():
                    logger.info(
                        "Early stopping triggered",
                        iteration=i,
                        best_score=float(self.best_score) if self.best_score else None
                    )
                    break

            logger.info(
                "Bayesian optimization completed",
                best_params=self.best_params,
                best_score=float(self.best_score) if self.best_score else None,
                total_iterations=len(self.history)
            )

            return self.best_params, self.history

        except Exception as e:
            logger.error("Bayesian optimization failed", error=str(e))
            raise

    async def _evaluate_params(
        self,
        objective_fn: Callable,
        params: Dict[str, Any],
        iteration: int
    ) -> Decimal:
        """Evaluate parameter configuration.

        Args:
            objective_fn: Objective function
            params: Parameters to evaluate
            iteration: Current iteration

        Returns:
            Score
        """
        try:
            # Evaluate
            score = await self._async_call(objective_fn, params)
            score_decimal = Decimal(str(score))

            # Store observation
            param_vector = self._params_to_vector(params)
            self.X_observed.append(param_vector)
            self.y_observed.append(score_decimal)

            # Update best
            is_better = (
                self.best_score is None or
                (self.maximize and score_decimal > self.best_score) or
                (not self.maximize and score_decimal < self.best_score)
            )

            if is_better:
                self.best_score = score_decimal
                self.best_params = params.copy()

                logger.info(
                    "New best parameters",
                    iteration=iteration,
                    params=params,
                    score=float(score_decimal)
                )

            # Store history
            self.history.append({
                "iteration": iteration,
                "params": params.copy(),
                "score": float(score_decimal),
                "is_best": is_better,
                "timestamp": datetime.utcnow()
            })

            return score_decimal

        except Exception as e:
            logger.error(
                "Failed to evaluate parameters",
                params=params,
                error=str(e)
            )
            raise

    async def _async_call(
        self,
        fn: Callable,
        params: Dict[str, Any]
    ) -> Decimal:
        """Call function asynchronously.

        Args:
            fn: Function to call
            params: Parameters

        Returns:
            Result
        """
        if asyncio.iscoroutinefunction(fn):
            return await fn(params)
        else:
            return fn(params)

    def _sample_params(self, param_space: Dict[str, Tuple[str, str, str]]) -> Dict[str, Any]:
        """Sample random parameters from space.

        Args:
            param_space: Parameter space definition

        Returns:
            Sampled parameters
        """
        params = {}

        for name, (param_type, min_val, max_val) in param_space.items():
            if param_type == "uniform":
                value = np.random.uniform(
                    float(min_val),
                    float(max_val)
                )
                params[name] = str(Decimal(str(value)))

            elif param_type == "log_uniform":
                log_value = np.random.uniform(
                    float(min_val),
                    float(max_val)
                )
                value = 10 ** log_value
                params[name] = str(Decimal(str(value)))

            elif param_type == "int_uniform":
                value = np.random.randint(
                    int(min_val),
                    int(max_val) + 1
                )
                params[name] = value

            elif param_type == "categorical":
                # min_val contains comma-separated choices
                choices = min_val.split(",")
                params[name] = np.random.choice(choices)

        return params

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert params dict to vector.

        Args:
            params: Parameters dictionary

        Returns:
            Parameter vector
        """
        vector = []

        for key in sorted(params.keys()):
            value = params[key]

            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, str):
                try:
                    vector.append(float(value))
                except ValueError:
                    # Categorical - use hash
                    vector.append(float(hash(value) % 1000))

        return np.array(vector)

    def _fit_gp(self) -> Tuple[Callable, Callable]:
        """Fit Gaussian process to observations.

        Returns:
            Tuple of (mean_function, std_function)
        """
        # Simplified GP using RBF kernel
        X = np.array(self.X_observed)
        y = np.array([float(score) for score in self.y_observed])

        # Normalize
        y_mean = y.mean()
        y_std = y.std() + 1e-6

        y_norm = (y - y_mean) / y_std

        # RBF kernel parameters
        length_scale = Decimal(str(self.config.get("gp_length_scale", "1.0")))
        noise = Decimal(str(self.config.get("gp_noise", "0.01")))

        def mean_fn(x_new: np.ndarray) -> np.ndarray:
            """GP mean function."""
            # Compute kernel matrix
            K = self._rbf_kernel(X, X, length_scale) + float(noise) * np.eye(len(X))
            k = self._rbf_kernel(X, x_new.reshape(1, -1), length_scale)

            # Predict
            K_inv = np.linalg.inv(K + np.eye(len(K)) * 1e-6)
            mean = k.T @ K_inv @ y_norm

            # Denormalize
            return mean * y_std + y_mean

        def std_fn(x_new: np.ndarray) -> np.ndarray:
            """GP std function."""
            # Compute kernel matrix
            K = self._rbf_kernel(X, X, length_scale) + float(noise) * np.eye(len(X))
            k = self._rbf_kernel(X, x_new.reshape(1, -1), length_scale)
            k_star = self._rbf_kernel(x_new.reshape(1, -1), x_new.reshape(1, -1), length_scale)

            # Predict variance
            K_inv = np.linalg.inv(K + np.eye(len(K)) * 1e-6)
            var = k_star - k.T @ K_inv @ k

            # Denormalize
            return np.sqrt(np.maximum(var, 0)) * y_std

        return mean_fn, std_fn

    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        length_scale: Decimal
    ) -> np.ndarray:
        """RBF (Gaussian) kernel.

        Args:
            X1: First set of points
            X2: Second set of points
            length_scale: Kernel length scale

        Returns:
            Kernel matrix
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                 np.sum(X2**2, axis=1) - \
                 2 * np.dot(X1, X2.T)

        return np.exp(-0.5 * sqdist / (float(length_scale) ** 2))

    def _optimize_acquisition(
        self,
        param_space: Dict[str, Tuple[str, str, str]],
        gp_mean: Callable,
        gp_std: Callable,
        constraints: Optional[List[Callable]]
    ) -> Dict[str, Any]:
        """Optimize acquisition function.

        Args:
            param_space: Parameter space
            gp_mean: GP mean function
            gp_std: GP std function
            constraints: Constraint functions

        Returns:
            Next parameters to evaluate
        """
        best_acquisition = Decimal("-inf")
        best_params = None

        # Random search over acquisition function
        n_candidates = self.config.get("n_acquisition_samples", 1000)

        for _ in range(n_candidates):
            params = self._sample_params(param_space)

            # Check constraints
            if constraints and not all(c(params) for c in constraints):
                continue

            # Compute acquisition value
            x = self._params_to_vector(params)
            acq_value = self._acquisition_function(x, gp_mean, gp_std)

            if acq_value > best_acquisition:
                best_acquisition = acq_value
                best_params = params

        return best_params if best_params else self._sample_params(param_space)

    def _acquisition_function(
        self,
        x: np.ndarray,
        gp_mean: Callable,
        gp_std: Callable
    ) -> Decimal:
        """Compute acquisition function value.

        Args:
            x: Point to evaluate
            gp_mean: GP mean function
            gp_std: GP std function

        Returns:
            Acquisition value
        """
        mean = float(gp_mean(x)[0])
        std = float(gp_std(x)[0])

        if self.acquisition == "ei":
            # Expected improvement
            if not self.y_observed:
                return Decimal("0")

            best_y = float(max(self.y_observed) if self.maximize else min(self.y_observed))

            if std < 1e-10:
                return Decimal("0")

            z = (mean - best_y - float(self.xi)) / std
            ei = (mean - best_y - float(self.xi)) * self._normal_cdf(z) + \
                 std * self._normal_pdf(z)

            return Decimal(str(ei)) if self.maximize else -Decimal(str(ei))

        elif self.acquisition == "ucb":
            # Upper confidence bound
            ucb = mean + float(self.kappa) * std
            return Decimal(str(ucb)) if self.maximize else -Decimal(str(ucb))

        elif self.acquisition == "poi":
            # Probability of improvement
            if not self.y_observed:
                return Decimal("0")

            best_y = float(max(self.y_observed) if self.maximize else min(self.y_observed))

            if std < 1e-10:
                return Decimal("0")

            z = (mean - best_y - float(self.xi)) / std
            poi = self._normal_cdf(z)

            return Decimal(str(poi)) if self.maximize else -Decimal(str(poi))

        return Decimal("0")

    def _normal_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF (approximation)."""
        return 0.5 * (1.0 + np.tanh(x * np.sqrt(2.0 / np.pi)))

    def _should_stop(self) -> bool:
        """Check early stopping criteria.

        Returns:
            True if should stop
        """
        if not self.config.get("early_stopping", False):
            return False

        patience = self.config.get("early_stopping_patience", 10)

        if len(self.history) < patience:
            return False

        # Check if no improvement in last N iterations
        recent_best = max(
            [h["score"] for h in self.history[-patience:]],
            key=lambda x: x if self.maximize else -x
        )

        return recent_best == self.history[-patience]["score"]

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found.

        Returns:
            Best parameter dictionary
        """
        return self.best_params

    def get_optimization_history(self) -> pl.DataFrame:
        """Get optimization history as DataFrame.

        Returns:
            Polars DataFrame with history
        """
        try:
            if not self.history:
                return pl.DataFrame()

            # Flatten history
            records = []
            for entry in self.history:
                record = {
                    "iteration": entry["iteration"],
                    "score": Decimal(str(entry["score"])),
                    "is_best": entry["is_best"],
                    "timestamp": entry["timestamp"]
                }

                # Add parameters
                for key, value in entry["params"].items():
                    record[f"param_{key}"] = str(value)

                records.append(record)

            df = pl.DataFrame(records)

            logger.info("Generated history DataFrame", rows=len(df))

            return df

        except Exception as e:
            logger.error("Failed to generate history DataFrame", error=str(e))
            raise
