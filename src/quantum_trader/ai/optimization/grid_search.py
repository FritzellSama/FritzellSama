"""Grid search optimization for hyperparameter tuning.

This module implements exhaustive grid search for finding optimal
hyperparameter combinations for trading models.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from itertools import product
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class GridSearchOptimizer:
    """Grid search hyperparameter optimization.

    Performs exhaustive search over specified parameter combinations,
    evaluating model performance using cross-validation.

    Attributes:
        config: Configuration dictionary
        param_grid: Parameter grid to search
        scoring_metric: Metric to optimize
        cv_folds: Number of cross-validation folds

    Example:
        >>> config = {
        ...     "scoring_metric": "sharpe_ratio",
        ...     "cv_folds": 5,
        ...     "n_jobs": 4,
        ...     "verbose": 1
        ... }
        >>> param_grid = {
        ...     "learning_rate": ["0.001", "0.01", "0.1"],
        ...     "hidden_dim": [64, 128, 256],
        ...     "dropout": ["0.3", "0.5"]
        ... }
        >>> optimizer = GridSearchOptimizer(config)
        >>> best_params, results = await optimizer.search(
        ...     model_factory, param_grid, train_data, val_data
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize grid search optimizer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.scoring_metric = config.get("scoring_metric", "mse")
        self.cv_folds = config.get("cv_folds", 5)
        self.n_jobs = config.get("n_jobs", 1)
        self.verbose = config.get("verbose", 1)
        self.higher_is_better = config.get("higher_is_better", True)
        self.early_stopping = config.get("early_stopping", False)
        self.early_stopping_rounds = config.get("early_stopping_rounds", 10)

        # Results tracking
        self.search_results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[Decimal] = None

        logger.info(
            "Grid search optimizer initialized",
            metric=self.scoring_metric,
            cv_folds=self.cv_folds,
            n_jobs=self.n_jobs
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["scoring_metric"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    async def search(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        param_grid: Dict[str, List[Any]],
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        scoring_fn: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform grid search over parameter space.

        Args:
            model_factory: Function that creates model given parameters
            param_grid: Dictionary of parameter names to lists of values
            train_data: Tuple of (features, labels) for training
            val_data: Optional validation data for final evaluation
            scoring_fn: Optional custom scoring function

        Returns:
            Tuple of (best_params, all_results)

        Raises:
            ValueError: If param_grid is empty or invalid
        """
        try:
            if not param_grid:
                raise ValueError("param_grid cannot be empty")

            logger.info(
                "Starting grid search",
                param_grid=param_grid,
                total_combinations=self._count_combinations(param_grid)
            )

            # Generate all parameter combinations
            param_combinations = self._generate_combinations(param_grid)

            # Track best result
            best_score = Decimal("-inf") if self.higher_is_better else Decimal("inf")
            best_params = None
            no_improvement_count = 0

            # Evaluate each combination
            for idx, params in enumerate(param_combinations):
                try:
                    # Create model with current parameters
                    model_config = {**self.config, **params}

                    # Perform cross-validation
                    cv_scores = await self._cross_validate(
                        model_factory,
                        model_config,
                        train_data,
                        scoring_fn
                    )

                    # Calculate aggregate metrics
                    mean_score = Decimal(str(np.mean(cv_scores)))
                    std_score = Decimal(str(np.std(cv_scores)))

                    # Store result
                    result = {
                        "params": params,
                        "mean_score": float(mean_score),
                        "std_score": float(std_score),
                        "cv_scores": [float(s) for s in cv_scores],
                        "timestamp": datetime.utcnow()
                    }
                    self.search_results.append(result)

                    # Check if best
                    is_better = (
                        (self.higher_is_better and mean_score > best_score) or
                        (not self.higher_is_better and mean_score < best_score)
                    )

                    if is_better:
                        best_score = mean_score
                        best_params = params
                        no_improvement_count = 0

                        logger.info(
                            "New best parameters found",
                            params=params,
                            score=float(best_score),
                            std=float(std_score)
                        )
                    else:
                        no_improvement_count += 1

                    # Early stopping
                    if self.early_stopping and no_improvement_count >= self.early_stopping_rounds:
                        logger.info(
                            "Early stopping triggered",
                            rounds_without_improvement=no_improvement_count
                        )
                        break

                    # Progress logging
                    if self.verbose and (idx + 1) % self.verbose == 0:
                        logger.info(
                            "Grid search progress",
                            completed=idx + 1,
                            total=len(param_combinations),
                            current_best_score=float(best_score)
                        )

                except Exception as e:
                    logger.error(
                        "Failed to evaluate parameter combination",
                        params=params,
                        error=str(e)
                    )
                    # Continue with next combination
                    continue

            # Final validation if validation data provided
            if val_data is not None and best_params is not None:
                final_score = await self._evaluate_on_validation(
                    model_factory,
                    {**self.config, **best_params},
                    train_data,
                    val_data,
                    scoring_fn
                )

                logger.info(
                    "Final validation score",
                    score=float(final_score),
                    params=best_params
                )

            self.best_params = best_params
            self.best_score = best_score

            logger.info(
                "Grid search completed",
                best_params=best_params,
                best_score=float(best_score),
                total_evaluated=len(self.search_results)
            )

            return best_params, self.search_results

        except Exception as e:
            logger.error("Grid search failed", error=str(e))
            raise

    def _generate_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid.

        Args:
            param_grid: Parameter grid

        Returns:
            List of parameter dictionaries
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for value_combo in product(*values):
            param_dict = dict(zip(keys, value_combo))
            combinations.append(param_dict)

        return combinations

    def _count_combinations(self, param_grid: Dict[str, List[Any]]) -> int:
        """Count total number of parameter combinations.

        Args:
            param_grid: Parameter grid

        Returns:
            Total number of combinations
        """
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    async def _cross_validate(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        config: Dict[str, Any],
        data: Tuple[np.ndarray, np.ndarray],
        scoring_fn: Optional[Callable]
    ) -> List[Decimal]:
        """Perform k-fold cross-validation.

        Args:
            model_factory: Function to create model
            config: Model configuration
            data: Training data
            scoring_fn: Scoring function

        Returns:
            List of scores for each fold
        """
        features, labels = data
        fold_scores = []

        # Create folds
        fold_size = len(features) // self.cv_folds

        for fold in range(self.cv_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size

            # Validation fold
            val_features = features[val_start:val_end]
            val_labels = labels[val_start:val_end]

            # Training folds
            train_features = np.concatenate([
                features[:val_start],
                features[val_end:]
            ], axis=0)
            train_labels = np.concatenate([
                labels[:val_start],
                labels[val_end:]
            ], axis=0)

            # Train model
            model = model_factory(config)
            model.train(train_features, train_labels)

            # Evaluate
            if scoring_fn is not None:
                score = scoring_fn(model, val_features, val_labels)
            else:
                metrics = model.evaluate(val_features, val_labels)
                score = metrics.get(self.scoring_metric, 0.0)

            fold_scores.append(Decimal(str(score)))

        return fold_scores

    async def _evaluate_on_validation(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        config: Dict[str, Any],
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        scoring_fn: Optional[Callable]
    ) -> Decimal:
        """Evaluate best parameters on validation set.

        Args:
            model_factory: Function to create model
            config: Model configuration
            train_data: Training data
            val_data: Validation data
            scoring_fn: Scoring function

        Returns:
            Validation score
        """
        try:
            # Train on full training set
            train_features, train_labels = train_data
            val_features, val_labels = val_data

            model = model_factory(config)
            model.train(train_features, train_labels)

            # Evaluate on validation set
            if scoring_fn is not None:
                score = scoring_fn(model, val_features, val_labels)
            else:
                metrics = model.evaluate(val_features, val_labels)
                score = metrics.get(self.scoring_metric, 0.0)

            return Decimal(str(score))

        except Exception as e:
            logger.error("Validation evaluation failed", error=str(e))
            raise

    def get_results_dataframe(self) -> pl.DataFrame:
        """Get search results as Polars DataFrame.

        Returns:
            DataFrame with all search results
        """
        try:
            if not self.search_results:
                return pl.DataFrame()

            # Flatten results
            records = []
            for result in self.search_results:
                record = {
                    "mean_score": Decimal(str(result["mean_score"])),
                    "std_score": Decimal(str(result["std_score"])),
                    "timestamp": result["timestamp"]
                }
                # Add parameters
                for key, value in result["params"].items():
                    record[f"param_{key}"] = str(value)

                records.append(record)

            df = pl.DataFrame(records)

            # Sort by score
            if self.higher_is_better:
                df = df.sort("mean_score", descending=True)
            else:
                df = df.sort("mean_score", descending=False)

            logger.info("Generated results DataFrame", rows=len(df))

            return df

        except Exception as e:
            logger.error("Failed to generate results DataFrame", error=str(e))
            raise

    def save_results(self, path: str) -> None:
        """Save search results to file.

        Args:
            path: File path to save results
        """
        try:
            df = self.get_results_dataframe()
            df.write_parquet(path)

            logger.info("Search results saved", path=path)

        except Exception as e:
            logger.error("Failed to save results", error=str(e))
            raise

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found.

        Returns:
            Best parameter dictionary or None if search not run
        """
        return self.best_params

    def get_best_score(self) -> Optional[Decimal]:
        """Get best score achieved.

        Returns:
            Best score or None if search not run
        """
        return self.best_score

    async def parallel_search(
        self,
        model_factory: Callable[[Dict[str, Any]], Any],
        param_grid: Dict[str, List[Any]],
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        scoring_fn: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform parallel grid search using asyncio.

        Args:
            model_factory: Function to create model
            param_grid: Parameter grid
            train_data: Training data
            val_data: Optional validation data
            scoring_fn: Optional scoring function

        Returns:
            Tuple of (best_params, all_results)
        """
        try:
            if not param_grid:
                raise ValueError("param_grid cannot be empty")

            logger.info(
                "Starting parallel grid search",
                n_jobs=self.n_jobs,
                total_combinations=self._count_combinations(param_grid)
            )

            # Generate combinations
            param_combinations = self._generate_combinations(param_grid)

            # Create tasks for parallel execution
            tasks = []
            for params in param_combinations:
                model_config = {**self.config, **params}
                task = self._evaluate_params_async(
                    model_factory,
                    model_config,
                    params,
                    train_data,
                    scoring_fn
                )
                tasks.append(task)

            # Execute in batches
            batch_size = self.n_jobs
            all_results = []

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error("Task failed", error=str(result))
                    else:
                        all_results.append(result)
                        self.search_results.append(result)

            # Find best
            best_result = self._find_best_result(all_results)
            self.best_params = best_result["params"]
            self.best_score = Decimal(str(best_result["mean_score"]))

            logger.info(
                "Parallel grid search completed",
                best_params=self.best_params,
                best_score=float(self.best_score)
            )

            return self.best_params, self.search_results

        except Exception as e:
            logger.error("Parallel grid search failed", error=str(e))
            raise

    async def _evaluate_params_async(
        self,
        model_factory: Callable,
        config: Dict[str, Any],
        params: Dict[str, Any],
        train_data: Tuple[np.ndarray, np.ndarray],
        scoring_fn: Optional[Callable]
    ) -> Dict[str, Any]:
        """Evaluate single parameter combination asynchronously.

        Args:
            model_factory: Function to create model
            config: Model configuration
            params: Parameter dictionary
            train_data: Training data
            scoring_fn: Scoring function

        Returns:
            Result dictionary
        """
        try:
            cv_scores = await self._cross_validate(
                model_factory,
                config,
                train_data,
                scoring_fn
            )

            mean_score = Decimal(str(np.mean(cv_scores)))
            std_score = Decimal(str(np.std(cv_scores)))

            return {
                "params": params,
                "mean_score": float(mean_score),
                "std_score": float(std_score),
                "cv_scores": [float(s) for s in cv_scores],
                "timestamp": datetime.utcnow()
            }

        except Exception as e:
            logger.error(
                "Failed to evaluate parameters",
                params=params,
                error=str(e)
            )
            raise

    def _find_best_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best result from list.

        Args:
            results: List of result dictionaries

        Returns:
            Best result
        """
        if not results:
            raise ValueError("No results to evaluate")

        best = results[0]
        best_score = Decimal(str(best["mean_score"]))

        for result in results[1:]:
            score = Decimal(str(result["mean_score"]))

            is_better = (
                (self.higher_is_better and score > best_score) or
                (not self.higher_is_better and score < best_score)
            )

            if is_better:
                best = result
                best_score = score

        return best
