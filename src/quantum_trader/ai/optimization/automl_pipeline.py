"""
AutoML Pipeline for Automated Model Selection and Hyperparameter Tuning.

Implements automated machine learning pipeline to find optimal models
and configurations for trading strategies.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


class ModelFamily(str, Enum):
    """Available model families."""

    LINEAR = "linear"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"


@dataclass
class ModelCandidate:
    """Candidate model configuration."""

    model_family: ModelFamily
    model_class: str
    hyperparameters: Dict[str, Any]
    score: Optional[Decimal] = None
    training_time: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline."""

    max_trials: int = 100
    max_time_seconds: int = 3600
    cv_folds: int = 5
    optimization_metric: str = "f1"
    early_stopping_rounds: int = 10
    n_jobs: int = -1
    random_state: int = 42
    search_strategy: str = "bayesian"  # random, grid, bayesian


class SearchSpace:
    """Hyperparameter search space definitions."""

    @staticmethod
    def get_logistic_regression_space() -> Dict[str, List[Any]]:
        """Get search space for logistic regression."""
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "max_iter": [100, 500, 1000]
        }

    @staticmethod
    def get_random_forest_space() -> Dict[str, List[Any]]:
        """Get search space for random forest."""
        return {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }

    @staticmethod
    def get_gradient_boosting_space() -> Dict[str, List[Any]]:
        """Get search space for gradient boosting."""
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        }

    @staticmethod
    def get_svm_space() -> Dict[str, List[Any]]:
        """Get search space for SVM."""
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4]
        }


class ModelSelector:
    """Model selection and evaluation."""

    def __init__(self, config: AutoMLConfig):
        """
        Initialize model selector.

        Args:
            config: AutoML configuration
        """
        self.config = config

    def create_model(
        self,
        model_family: ModelFamily,
        hyperparameters: Dict[str, Any]
    ) -> Any:
        """
        Create model instance from family and hyperparameters.

        Args:
            model_family: Model family
            hyperparameters: Hyperparameters

        Returns:
            Model instance
        """
        if model_family == ModelFamily.LINEAR:
            return LogisticRegression(
                random_state=self.config.random_state,
                **hyperparameters
            )
        elif model_family == ModelFamily.TREE:
            return RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                **hyperparameters
            )
        elif model_family == ModelFamily.ENSEMBLE:
            return GradientBoostingClassifier(
                random_state=self.config.random_state,
                **hyperparameters
            )
        elif model_family == ModelFamily.SVM:
            return SVC(
                random_state=self.config.random_state,
                **hyperparameters
            )
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

    async def evaluate_candidate(
        self,
        candidate: ModelCandidate,
        X_train: Any,
        y_train: Any
    ) -> ModelCandidate:
        """
        Evaluate a candidate model using cross-validation.

        Args:
            candidate: Model candidate
            X_train: Training features
            y_train: Training labels

        Returns:
            Updated candidate with score
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Create model
            model = self.create_model(
                candidate.model_family,
                candidate.hyperparameters
            )

            # Cross-validation
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=self.config.cv_folds,
                scoring=self.config.optimization_metric,
                n_jobs=self.config.n_jobs
            )

            # Average score
            candidate.score = Decimal(str(scores.mean()))

            # Training time
            end_time = datetime.now(timezone.utc)
            training_time = (end_time - start_time).total_seconds()
            candidate.training_time = Decimal(str(training_time))

            logger.info(
                "Evaluated candidate",
                extra={
                    "model_family": candidate.model_family.value,
                    "score": str(candidate.score),
                    "training_time": str(candidate.training_time),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        except Exception as e:
            logger.error(
                "Candidate evaluation failed",
                extra={
                    "model_family": candidate.model_family.value,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                exc_info=True
            )
            candidate.score = Decimal("-999")

        return candidate


class HyperparameterSampler:
    """Hyperparameter sampling strategies."""

    def __init__(self, search_space: Dict[str, List[Any]], random_state: int = 42):
        """
        Initialize sampler.

        Args:
            search_space: Hyperparameter search space
            random_state: Random seed
        """
        self.search_space = search_space
        self.random_state = random_state

        import numpy as np
        self.rng = np.random.RandomState(random_state)

    def random_sample(self) -> Dict[str, Any]:
        """
        Random sampling from search space.

        Returns:
            Sampled hyperparameters
        """
        hyperparameters = {}

        for param, values in self.search_space.items():
            if isinstance(values, list):
                hyperparameters[param] = self.rng.choice(values)
            else:
                hyperparameters[param] = values

        return hyperparameters

    def grid_sample(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Grid sampling from search space.

        Args:
            index: Grid index

        Returns:
            Sampled hyperparameters or None if exhausted
        """
        import itertools

        # Generate all combinations
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]

        all_combinations = list(itertools.product(*param_values))

        if index >= len(all_combinations):
            return None

        combination = all_combinations[index]
        hyperparameters = dict(zip(param_names, combination))

        return hyperparameters


class AutoMLPipeline:
    """Automated machine learning pipeline."""

    def __init__(self, config: AutoMLConfig):
        """
        Initialize AutoML pipeline.

        Args:
            config: AutoML configuration
        """
        self.config = config
        self.model_selector = ModelSelector(config)

        # Results
        self.candidates: List[ModelCandidate] = []
        self.best_candidate: Optional[ModelCandidate] = None
        self.best_model: Optional[Any] = None

        logger.info(
            "Initialized AutoML pipeline",
            extra={
                "max_trials": config.max_trials,
                "optimization_metric": config.optimization_metric,
                "search_strategy": config.search_strategy,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def search(
        self,
        train_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str,
        model_families: Optional[List[ModelFamily]] = None
    ) -> ModelCandidate:
        """
        Search for best model and hyperparameters.

        Args:
            train_df: Training data
            feature_cols: Feature column names
            label_col: Label column name
            model_families: Model families to search (None for all)

        Returns:
            Best model candidate
        """
        if model_families is None:
            model_families = [
                ModelFamily.LINEAR,
                ModelFamily.TREE,
                ModelFamily.ENSEMBLE,
                ModelFamily.SVM
            ]

        # Prepare data
        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(label_col).to_numpy().ravel()

        start_time = datetime.now(timezone.utc)
        trials = 0
        best_score = Decimal("-999")
        no_improvement_count = 0

        logger.info(
            "Starting AutoML search",
            extra={
                "num_samples": len(train_df),
                "num_features": len(feature_cols),
                "num_families": len(model_families),
                "timestamp": start_time.isoformat()
            }
        )

        # Search loop
        while trials < self.config.max_trials:
            # Check time limit
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            if elapsed > self.config.max_time_seconds:
                logger.info(
                    "Time limit reached",
                    extra={
                        "trials": trials,
                        "elapsed_seconds": elapsed,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                break

            # Select model family
            import random
            model_family = random.choice(model_families)

            # Get search space
            if model_family == ModelFamily.LINEAR:
                search_space = SearchSpace.get_logistic_regression_space()
            elif model_family == ModelFamily.TREE:
                search_space = SearchSpace.get_random_forest_space()
            elif model_family == ModelFamily.ENSEMBLE:
                search_space = SearchSpace.get_gradient_boosting_space()
            elif model_family == ModelFamily.SVM:
                search_space = SearchSpace.get_svm_space()
            else:
                continue

            # Sample hyperparameters
            sampler = HyperparameterSampler(
                search_space,
                self.config.random_state + trials
            )

            if self.config.search_strategy == "random":
                hyperparameters = sampler.random_sample()
            elif self.config.search_strategy == "grid":
                hyperparameters = sampler.grid_sample(trials)
                if hyperparameters is None:
                    break
            else:
                # Default to random
                hyperparameters = sampler.random_sample()

            # Create candidate
            candidate = ModelCandidate(
                model_family=model_family,
                model_class=str(model_family.value),
                hyperparameters=hyperparameters
            )

            # Evaluate candidate
            candidate = await self.model_selector.evaluate_candidate(
                candidate,
                X_train,
                y_train
            )

            self.candidates.append(candidate)

            # Update best
            if candidate.score > best_score:
                best_score = candidate.score
                self.best_candidate = candidate
                no_improvement_count = 0

                logger.info(
                    "New best candidate found",
                    extra={
                        "trial": trials,
                        "model_family": candidate.model_family.value,
                        "score": str(best_score),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= self.config.early_stopping_rounds:
                logger.info(
                    "Early stopping triggered",
                    extra={
                        "trials": trials,
                        "no_improvement_rounds": no_improvement_count,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )
                break

            trials += 1

        # Train best model on full data
        if self.best_candidate:
            self.best_model = self.model_selector.create_model(
                self.best_candidate.model_family,
                self.best_candidate.hyperparameters
            )
            self.best_model.fit(X_train, y_train)

            logger.info(
                "AutoML search completed",
                extra={
                    "total_trials": trials,
                    "best_score": str(best_score),
                    "best_model": self.best_candidate.model_family.value,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        return self.best_candidate

    def predict(self, test_df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """
        Make predictions using best model.

        Args:
            test_df: Test data
            feature_cols: Feature column names

        Returns:
            DataFrame with predictions
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call search() first.")

        X_test = test_df.select(feature_cols).to_numpy()
        predictions = self.best_model.predict(X_test)

        # Get probabilities if available
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(X_test)
            max_probs = probabilities.max(axis=1)
        else:
            max_probs = [1.0] * len(predictions)

        result_df = test_df.with_columns([
            pl.Series("prediction", predictions.tolist()),
            pl.Series("confidence", max_probs.tolist())
        ])

        return result_df

    def evaluate(
        self,
        test_df: pl.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> Dict[str, Decimal]:
        """
        Evaluate best model on test data.

        Args:
            test_df: Test data
            feature_cols: Feature column names
            label_col: Label column name

        Returns:
            Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call search() first.")

        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df.select(label_col).to_numpy().ravel()

        predictions = self.best_model.predict(X_test)

        metrics = {
            "accuracy": Decimal(str(accuracy_score(y_test, predictions))),
            "precision": Decimal(str(precision_score(y_test, predictions, average="weighted", zero_division=0))),
            "recall": Decimal(str(recall_score(y_test, predictions, average="weighted", zero_division=0))),
            "f1": Decimal(str(f1_score(y_test, predictions, average="weighted", zero_division=0)))
        }

        logger.info(
            "Model evaluation completed",
            extra={
                "metrics": {k: str(v) for k, v in metrics.items()},
                "num_samples": len(test_df),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    def get_leaderboard(self, top_k: int = 10) -> pl.DataFrame:
        """
        Get leaderboard of top candidates.

        Args:
            top_k: Number of top candidates to return

        Returns:
            DataFrame with leaderboard
        """
        # Sort candidates by score
        sorted_candidates = sorted(
            self.candidates,
            key=lambda c: c.score if c.score is not None else Decimal("-999"),
            reverse=True
        )

        # Create leaderboard data
        leaderboard_data = {
            "rank": list(range(1, min(top_k, len(sorted_candidates)) + 1)),
            "model_family": [
                c.model_family.value for c in sorted_candidates[:top_k]
            ],
            "score": [
                str(c.score) if c.score is not None else "N/A"
                for c in sorted_candidates[:top_k]
            ],
            "training_time": [
                str(c.training_time) if c.training_time is not None else "N/A"
                for c in sorted_candidates[:top_k]
            ]
        }

        return pl.DataFrame(leaderboard_data)
