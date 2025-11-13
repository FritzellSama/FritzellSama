"""Hidden Markov Model for market regime detection.

This module implements HMM for identifying distinct market regimes
(trending, ranging, volatile, calm) from price and volume data.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class HMMModel(BaseMLModel):
    """Hidden Markov Model for regime detection.

    Identifies latent market regimes and their transitions, enabling
    regime-adaptive trading strategies.

    Attributes:
        config: Model configuration
        n_states: Number of hidden states (regimes)
        transition_matrix: State transition probabilities
        emission_params: Emission distribution parameters
        initial_probs: Initial state probabilities

    Example:
        >>> config = {
        ...     "n_states": 4,
        ...     "covariance_type": "full",
        ...     "n_iter": 100,
        ...     "tol": "0.01"
        ... }
        >>> model = HMMModel(config)
        >>> model.train(market_features, None)
        >>> regimes = model.predict(test_features)
        >>> probs = model.predict_proba(test_features)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize HMM model.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        super().__init__(config)

        self._validate_config()

        # Model parameters
        self.n_states = config["n_states"]
        self.covariance_type = config.get("covariance_type", "full")
        self.n_iter = config.get("n_iter", 100)
        self.tol = Decimal(str(config.get("tol", "0.01")))
        self.random_state = config.get("random_state", 42)

        # Model components
        self.transition_matrix: Optional[np.ndarray] = None
        self.initial_probs: Optional[np.ndarray] = None
        self.emission_params: Dict[str, Any] = {}

        # Gaussian mixture for emissions
        self.gmm = GaussianMixture(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            max_iter=self.n_iter,
            tol=float(self.tol),
            random_state=self.random_state
        )

        # Training state
        self.is_fitted = False
        self.feature_dim: Optional[int] = None
        self.training_history: List[Dict[str, float]] = []

        logger.info(
            "HMM model initialized",
            n_states=self.n_states,
            covariance_type=self.covariance_type
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["n_states"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config["n_states"] < 2:
            raise ValueError("n_states must be at least 2")

    def train(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> None:
        """Train HMM using Baum-Welch algorithm.

        Args:
            features: Observation sequences [num_samples, feature_dim]
            labels: Not used (unsupervised learning)

        Raises:
            ValueError: If features shape is invalid
        """
        try:
            if len(features.shape) != 2:
                raise ValueError(
                    f"Expected 2D features array, got shape {features.shape}"
                )

            self.feature_dim = features.shape[1]

            logger.info(
                "Starting HMM training",
                num_samples=len(features),
                feature_dim=self.feature_dim,
                n_states=self.n_states
            )

            # Initialize parameters
            self._initialize_parameters(features)

            # EM algorithm iterations
            prev_log_likelihood = Decimal("-inf")

            for iteration in range(self.n_iter):
                # E-step: Forward-backward algorithm
                alpha, beta, log_likelihood = self._forward_backward(features)

                # M-step: Update parameters
                self._update_parameters(features, alpha, beta)

                # Check convergence
                log_likelihood_decimal = Decimal(str(log_likelihood))
                improvement = log_likelihood_decimal - prev_log_likelihood

                self.training_history.append({
                    "iteration": iteration,
                    "log_likelihood": float(log_likelihood)
                })

                if iteration % 10 == 0:
                    logger.info(
                        "Training progress",
                        iteration=iteration,
                        log_likelihood=float(log_likelihood),
                        improvement=float(improvement)
                    )

                if abs(improvement) < self.tol:
                    logger.info(
                        "Converged",
                        iteration=iteration,
                        improvement=float(improvement)
                    )
                    break

                prev_log_likelihood = log_likelihood_decimal

            # Fit GMM for emissions
            self.gmm.fit(features)
            self.emission_params = {
                "means": self.gmm.means_,
                "covariances": self.gmm.covariances_,
                "weights": self.gmm.weights_
            }

            self.is_fitted = True

            logger.info("HMM training completed", iterations=len(self.training_history))

        except Exception as e:
            logger.error("Failed to train HMM", error=str(e))
            raise

    def _initialize_parameters(self, features: np.ndarray) -> None:
        """Initialize model parameters.

        Args:
            features: Training features
        """
        # Uniform initial probabilities
        self.initial_probs = np.ones(self.n_states) / self.n_states

        # Random transition matrix (row-stochastic)
        self.transition_matrix = np.random.rand(self.n_states, self.n_states)
        self.transition_matrix = (
            self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        )

        # Add self-transition bias
        self.transition_matrix += np.eye(self.n_states) * 0.5
        self.transition_matrix = (
            self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        )

        logger.debug("Parameters initialized")

    def _forward_backward(
        self,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm.

        Args:
            features: Observation sequence

        Returns:
            Tuple of (alpha, beta, log_likelihood)
        """
        T = len(features)
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))

        # Emission probabilities
        emission_probs = self._compute_emission_probs(features)

        # Forward pass
        alpha[0] = self.initial_probs * emission_probs[0]
        alpha[0] /= alpha[0].sum() + 1e-10

        for t in range(1, T):
            alpha[t] = emission_probs[t] * (alpha[t-1] @ self.transition_matrix)
            alpha[t] /= alpha[t].sum() + 1e-10

        # Backward pass
        beta[-1] = 1.0

        for t in range(T-2, -1, -1):
            beta[t] = self.transition_matrix @ (emission_probs[t+1] * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-10

        # Log likelihood
        log_likelihood = float(np.log(alpha[-1].sum() + 1e-10))

        return alpha, beta, log_likelihood

    def _compute_emission_probs(self, features: np.ndarray) -> np.ndarray:
        """Compute emission probabilities.

        Args:
            features: Observation features

        Returns:
            Emission probability matrix [T, n_states]
        """
        if not self.emission_params:
            # Use GMM if available, else uniform
            if hasattr(self.gmm, 'means_'):
                return self.gmm.predict_proba(features)
            else:
                return np.ones((len(features), self.n_states)) / self.n_states

        # Compute Gaussian probabilities
        T = len(features)
        emission_probs = np.zeros((T, self.n_states))

        for state in range(self.n_states):
            mean = self.emission_params["means"][state]
            cov = self.emission_params["covariances"][state]

            # Multivariate Gaussian
            diff = features - mean
            if len(cov.shape) == 1:  # Diagonal covariance
                inv_cov = 1.0 / (cov + 1e-10)
                exponent = -0.5 * np.sum(diff * inv_cov * diff, axis=1)
                normalizer = 1.0 / np.sqrt((2 * np.pi) ** self.feature_dim * np.prod(cov + 1e-10))
            else:  # Full covariance
                inv_cov = np.linalg.inv(cov + np.eye(self.feature_dim) * 1e-6)
                exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                normalizer = 1.0 / np.sqrt(
                    (2 * np.pi) ** self.feature_dim * (np.linalg.det(cov) + 1e-10)
                )

            emission_probs[:, state] = normalizer * np.exp(exponent)

        # Normalize
        emission_probs /= emission_probs.sum(axis=1, keepdims=True) + 1e-10

        return emission_probs

    def _update_parameters(
        self,
        features: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> None:
        """Update model parameters (M-step).

        Args:
            features: Observations
            alpha: Forward probabilities
            beta: Backward probabilities
        """
        T = len(features)

        # State posterior probabilities
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10

        # Update initial probabilities
        self.initial_probs = gamma[0]

        # Update transition matrix
        emission_probs = self._compute_emission_probs(features)

        xi = np.zeros((T-1, self.n_states, self.n_states))

        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        self.transition_matrix[i, j] *
                        emission_probs[t+1, j] *
                        beta[t+1, j]
                    )

            xi[t] /= xi[t].sum() + 1e-10

        # Update transition matrix
        self.transition_matrix = xi.sum(axis=0)
        self.transition_matrix /= gamma[:-1].sum(axis=0, keepdims=True).T + 1e-10

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict most likely state sequence using Viterbi algorithm.

        Args:
            features: Input features

        Returns:
            State sequence [num_samples]
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model not fitted")

            T = len(features)
            delta = np.zeros((T, self.n_states))
            psi = np.zeros((T, self.n_states), dtype=int)

            # Emission probabilities
            emission_probs = self._compute_emission_probs(features)

            # Initialization
            delta[0] = self.initial_probs * emission_probs[0]

            # Recursion
            for t in range(1, T):
                for j in range(self.n_states):
                    prob = delta[t-1] * self.transition_matrix[:, j]
                    psi[t, j] = np.argmax(prob)
                    delta[t, j] = np.max(prob) * emission_probs[t, j]

            # Backtracking
            states = np.zeros(T, dtype=int)
            states[-1] = np.argmax(delta[-1])

            for t in range(T-2, -1, -1):
                states[t] = psi[t+1, states[t+1]]

            logger.info("Generated state predictions", num_samples=T)

            return states

        except Exception as e:
            logger.error("Failed to generate predictions", error=str(e))
            raise

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict state probabilities.

        Args:
            features: Input features

        Returns:
            State probabilities [num_samples, n_states]
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model not fitted")

            # Use forward algorithm
            alpha, _, _ = self._forward_backward(features)

            # Normalize to get probabilities
            probs = alpha / alpha.sum(axis=1, keepdims=True)

            logger.info("Generated probability predictions", shape=probs.shape)

            return probs

        except Exception as e:
            logger.error("Failed to generate probability predictions", error=str(e))
            raise

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model (unsupervised metrics).

        Args:
            features: Input features
            labels: Not used (unsupervised)

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            _, _, log_likelihood = self._forward_backward(features)

            # AIC and BIC
            n_params = (
                self.n_states - 1 +  # Initial probs
                self.n_states * (self.n_states - 1) +  # Transition matrix
                self.n_states * self.feature_dim +  # Means
                self.n_states * self.feature_dim  # Covariances (simplified)
            )

            aic = float(2 * n_params - 2 * log_likelihood)
            bic = float(n_params * np.log(len(features)) - 2 * log_likelihood)

            metrics = {
                "log_likelihood": log_likelihood,
                "aic": aic,
                "bic": bic,
                "n_params": n_params
            }

            logger.info("HMM evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Failed to evaluate HMM", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                save_path,
                transition_matrix=self.transition_matrix,
                initial_probs=self.initial_probs,
                emission_params=self.emission_params,
                config=self.config,
                feature_dim=self.feature_dim,
                training_history=self.training_history
            )

            logger.info("HMM model saved", path=path)

        except Exception as e:
            logger.error("Failed to save HMM model", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model from
        """
        try:
            data = np.load(path, allow_pickle=True)

            self.transition_matrix = data["transition_matrix"]
            self.initial_probs = data["initial_probs"]
            self.emission_params = data["emission_params"].item()
            self.feature_dim = int(data["feature_dim"])
            self.training_history = data["training_history"].tolist()

            self.is_fitted = True

            logger.info("HMM model loaded", path=path)

        except Exception as e:
            logger.error("Failed to load HMM model", error=str(e))
            raise
