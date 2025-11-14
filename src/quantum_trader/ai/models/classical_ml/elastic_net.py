"""Elastic Net regression model for trading predictions.

This module implements an Elastic Net model that combines L1 and L2 regularization
for robust feature selection and prediction in trading scenarios.
"""

import asyncio
import pickle
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import ElasticNet as SKElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from structlog import get_logger

logger = get_logger(__name__)


class ElasticNetModel:
    """Elastic Net regression model for trading predictions.

    Combines L1 (Lasso) and L2 (Ridge) regularization for feature selection
    and robust predictions. Inherits from BaseMLModel interface.

    Attributes:
        config: Configuration dictionary
        alpha: Regularization strength
        l1_ratio: Balance between L1 and L2 (0=Ridge, 1=Lasso)
        model: Scikit-learn ElasticNet model
        scaler: Feature scaler for normalization
        feature_importance: Feature importance scores
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Elastic Net model.

        Args:
            config: Configuration dictionary containing:
                - alpha: Regularization strength
                - l1_ratio: L1/L2 balance
                - max_iter: Maximum iterations
                - tol: Convergence tolerance
                - positive: Force positive coefficients
                - fit_intercept: Fit intercept
                - normalize_features: Normalize input features
                - cv_folds: Cross-validation folds
                - random_state: Random seed

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.alpha = float(config.get('alpha', 1.0))
        self.l1_ratio = float(config.get('l1_ratio', 0.5))
        self.max_iter = config.get('max_iter', 10000)
        self.tol = float(config.get('tol', 0.0001))
        self.positive = config.get('positive', False)
        self.fit_intercept = config.get('fit_intercept', True)
        self.normalize_features = config.get('normalize_features', True)
        self.cv_folds = config.get('cv_folds', 5)
        self.random_state = config.get('random_state', 42)

        # Initialize model
        self.model = SKElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            positive=self.positive,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state
        )

        # Initialize scaler
        self.scaler = StandardScaler() if self.normalize_features else None
        self.feature_importance: Optional[np.ndarray] = None
        self.is_trained = False

        logger.info(
            "elastic_net_model_initialized",
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config parameters are invalid
        """
        if 'alpha' in self.config and self.config['alpha'] < 0:
            raise ValueError("alpha must be non-negative")

        if 'l1_ratio' in self.config:
            if not 0 <= self.config['l1_ratio'] <= 1:
                raise ValueError("l1_ratio must be between 0 and 1")

        if 'max_iter' in self.config and self.config['max_iter'] <= 0:
            raise ValueError("max_iter must be positive")

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> None:
        """Train the Elastic Net model.

        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels (n_samples,)
            sample_weights: Optional sample weights

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_training_data(features, labels)

            logger.info(
                "elastic_net_training_started",
                n_samples=features.shape[0],
                n_features=features.shape[1]
            )

            # Scale features if configured
            if self.scaler is not None:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features

            # Train model
            self.model.fit(
                features_scaled,
                labels.ravel(),
                sample_weight=sample_weights
            )

            # Store feature importance (absolute coefficients)
            self.feature_importance = np.abs(self.model.coef_)

            self.is_trained = True

            logger.info(
                "elastic_net_training_completed",
                n_features=features.shape[1],
                n_nonzero_coefs=np.sum(self.model.coef_ != 0),
                intercept=float(self.model.intercept_)
            )

        except Exception as e:
            logger.error("elastic_net_training_failed", error=str(e))
            raise

    def _validate_training_data(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Validate training data.

        Args:
            features: Training features
            labels: Training labels

        Raises:
            ValueError: If data is invalid
        """
        if features.shape[0] == 0:
            raise ValueError("features array is empty")

        if labels.shape[0] == 0:
            raise ValueError("labels array is empty")

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features ({features.shape[0]}) and labels ({labels.shape[0]}) "
                "must have same number of samples"
            )

        if np.any(np.isnan(features)):
            raise ValueError("features contain NaN values")

        if np.any(np.isnan(labels)):
            raise ValueError("labels contain NaN values")

        if np.any(np.isinf(features)):
            raise ValueError("features contain infinite values")

        if np.any(np.isinf(labels)):
            raise ValueError("labels contain infinite values")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using trained model.

        Args:
            features: Input features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            RuntimeError: If model is not trained
            ValueError: If input is invalid
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")

            if features.shape[0] == 0:
                raise ValueError("features array is empty")

            # Scale features if configured
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

            # Make predictions
            predictions = self.model.predict(features_scaled)

            logger.debug(
                "predictions_made",
                n_samples=features.shape[0],
                mean_prediction=float(np.mean(predictions)),
                std_prediction=float(np.std(predictions))
            )

            return predictions

        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            raise

    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            features: Test features
            labels: True labels

        Returns:
            Dictionary with evaluation metrics:
                - mse: Mean squared error
                - rmse: Root mean squared error
                - mae: Mean absolute error
                - r2: R-squared score
                - mape: Mean absolute percentage error

        Raises:
            RuntimeError: If model is not trained
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Model must be trained before evaluation")

            predictions = self.predict(features)

            # Calculate metrics
            mse = float(mean_squared_error(labels, predictions))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(labels, predictions))
            r2 = float(r2_score(labels, predictions))

            # Mean Absolute Percentage Error
            labels_nonzero = labels != 0
            if np.any(labels_nonzero):
                mape = float(
                    np.mean(
                        np.abs((labels[labels_nonzero] - predictions[labels_nonzero]) /
                               labels[labels_nonzero])
                    ) * 100
                )
            else:
                mape = float('inf')

            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'n_samples': int(features.shape[0]),
                'n_features': int(features.shape[1])
            }

            logger.info("model_evaluated", **metrics)

            return metrics

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model

        Raises:
            RuntimeError: If model is not trained
            Exception: If save operation fails
        """
        try:
            if not self.is_trained:
                raise RuntimeError("Cannot save untrained model")

            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare model state
            model_state = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'config': self.config,
                'is_trained': self.is_trained,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio
            }

            # Save using pickle
            with open(save_path, 'wb') as f:
                pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("model_saved", path=str(save_path))

        except Exception as e:
            logger.error("model_save_failed", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If load operation fails
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")

            # Load model state
            with open(load_path, 'rb') as f:
                model_state = pickle.load(f)

            # Restore model state
            self.model = model_state['model']
            self.scaler = model_state['scaler']
            self.feature_importance = model_state['feature_importance']
            self.config = model_state['config']
            self.is_trained = model_state['is_trained']
            self.alpha = model_state['alpha']
            self.l1_ratio = model_state['l1_ratio']

            logger.info("model_loaded", path=str(load_path))

        except Exception as e:
            logger.error("model_load_failed", error=str(e), path=path)
            raise

    def optimize_hyperparameters(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search.

        Args:
            features: Training features
            labels: Training labels
            param_grid: Parameter grid for search (uses default if None)

        Returns:
            Best parameters found

        Raises:
            ValueError: If input data is invalid
        """
        try:
            self._validate_training_data(features, labels)

            # Default parameter grid
            if param_grid is None:
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }

            logger.info(
                "hyperparameter_optimization_started",
                param_grid=param_grid
            )

            # Scale features if configured
            if self.scaler is not None:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)

            # Grid search
            grid_search = GridSearchCV(
                SKElasticNet(
                    max_iter=self.max_iter,
                    tol=self.tol,
                    positive=self.positive,
                    fit_intercept=self.fit_intercept,
                    random_state=self.random_state
                ),
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=self.config.get('n_jobs', -1),
                verbose=self.config.get('grid_search_verbose', 0)
            )

            grid_search.fit(features_scaled, labels.ravel())

            # Update model with best parameters
            self.alpha = grid_search.best_params_['alpha']
            self.l1_ratio = grid_search.best_params_['l1_ratio']

            self.model = grid_search.best_estimator_
            self.feature_importance = np.abs(self.model.coef_)
            self.is_trained = True

            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Negate for MSE

            logger.info(
                "hyperparameter_optimization_completed",
                best_params=best_params,
                best_mse=float(best_score)
            )

            return {
                'best_params': best_params,
                'best_score': float(best_score),
                'cv_results': grid_search.cv_results_
            }

        except Exception as e:
            logger.error("hyperparameter_optimization_failed", error=str(e))
            raise

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Get feature importance scores.

        Args:
            feature_names: Optional feature names for output

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained or self.feature_importance is None:
            raise RuntimeError("Model must be trained before getting feature importance")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]

        if len(feature_names) != len(self.feature_importance):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) must match "
                f"number of features ({len(self.feature_importance)})"
            )

        # Sort by importance
        importance_dict = {
            name: float(importance)
            for name, importance in zip(feature_names, self.feature_importance)
        }

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_model_coefficients(self) -> Dict[str, Any]:
        """Get model coefficients.

        Returns:
            Dictionary with coefficients and intercept

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained")

        return {
            'coefficients': self.model.coef_.tolist(),
            'intercept': float(self.model.intercept_),
            'n_nonzero': int(np.sum(self.model.coef_ != 0)),
            'n_features': int(len(self.model.coef_)),
            'sparsity': float(np.sum(self.model.coef_ == 0) / len(self.model.coef_))
        }

    async def train_async(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> None:
        """Async wrapper for training.

        Args:
            features: Training features
            labels: Training labels
            sample_weights: Optional sample weights
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.train,
            features,
            labels,
            sample_weights
        )

    async def predict_async(self, features: np.ndarray) -> np.ndarray:
        """Async wrapper for predictions.

        Args:
            features: Input features

        Returns:
            Predictions
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.predict,
            features
        )
