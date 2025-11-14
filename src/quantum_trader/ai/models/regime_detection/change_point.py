"""Change point detection for market regime identification.

This module implements advanced change point detection algorithms for identifying
regime shifts in financial markets. Detecting regime changes is critical for:
- Adapting trading strategies to market conditions
- Risk management and position sizing
- Portfolio rebalancing
- Signal filtering

The module provides multiple detection algorithms:
- CUSUM (Cumulative Sum) for online detection
- Bayesian change point detection for probabilistic analysis
- PELT (Pruned Exact Linear Time) for offline detection
- Kernel change point detection for nonlinear regimes

Reference:
    Killick, R., Fearnhead, P., & Eckley, I. A. (2012).
    Optimal detection of changepoints with a linear computational cost.
    Journal of the American Statistical Association.

Example:
    ```python
    from quantum_trader.ai.models.regime_detection.change_point import ChangePointDetector
    import numpy as np

    config = {
        "method": "pelt",
        "penalty": 10.0,
        "min_size": 20,
        "model": "rbf",
    }

    detector = ChangePointDetector(config)
    time_series = np.random.randn(1000)
    change_points = detector.detect(time_series)
    regimes = detector.get_regimes(time_series)
    ```
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.signal import find_peaks
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class ChangePointDetector(BaseMLModel):
    """Change point detection for market regime identification.

    Production-ready implementation with multiple detection algorithms:
    - CUSUM: Online cumulative sum control chart
    - Bayesian: Probabilistic change point detection
    - PELT: Pruned exact linear time algorithm
    - Kernel: Kernel-based nonlinear detection

    Attributes:
        method: Detection method ('cusum', 'bayesian', 'pelt', 'kernel')
        change_points: Detected change point indices
        regimes: Regime labels for each data point
        regime_stats: Statistics for each detected regime
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize change point detector.

        Args:
            config: Configuration dictionary with keys:
                - method: Detection method ('cusum', 'bayesian', 'pelt', 'kernel')
                  Default: 'pelt'
                - penalty: Penalty for change point detection (higher = fewer changes)
                  Default: 10.0
                - threshold: Threshold for CUSUM detection
                  Default: 5.0
                - min_size: Minimum regime size
                  Default: 20
                - model: Statistical model ('l1', 'l2', 'rbf', 'normal')
                  Default: 'rbf'
                - prior_scale: Prior scale for Bayesian method
                  Default: 1.0
        """
        super().__init__(config)

        # Detection parameters
        self.method = config.get("method", "pelt")
        if self.method not in ["cusum", "bayesian", "pelt", "kernel"]:
            raise ValueError(f"Invalid method: {self.method}")

        self.penalty = float(config.get("penalty", 10.0))
        self.threshold = float(config.get("threshold", 5.0))
        self.min_size = int(config.get("min_size", 20))
        self.model = config.get("model", "rbf")
        self.prior_scale = float(config.get("prior_scale", 1.0))

        # Detection results
        self.change_points: List[int] = []
        self.regimes: Optional[np.ndarray] = None
        self.regime_stats: List[Dict[str, Any]] = []

        logger.info(
            "Change point detector initialized",
            method=self.method,
            penalty=self.penalty
        )

    def _detect_cusum(self, data: np.ndarray) -> List[int]:
        """Detect change points using CUSUM algorithm.

        Args:
            data: Time series data

        Returns:
            List of change point indices
        """
        n = len(data)
        change_points = []

        # Calculate cumulative sum
        mean_est = np.mean(data[:self.min_size])
        std_est = np.std(data[:self.min_size])

        if std_est == 0:
            std_est = 1.0

        s_pos = 0.0
        s_neg = 0.0

        for i in range(self.min_size, n):
            # Normalized deviation
            dev = (data[i] - mean_est) / std_est

            # Update CUSUM
            s_pos = max(0, s_pos + dev - 0.5)
            s_neg = max(0, s_neg - dev - 0.5)

            # Check for change point
            if s_pos > self.threshold or s_neg > self.threshold:
                change_points.append(i)

                # Reset and update estimates
                s_pos = 0.0
                s_neg = 0.0

                # Update mean and std for next segment
                if i + self.min_size < n:
                    window_data = data[i:min(i + self.min_size, n)]
                    mean_est = np.mean(window_data)
                    std_est = np.std(window_data)
                    if std_est == 0:
                        std_est = 1.0

        logger.debug("CUSUM detection completed", num_changes=len(change_points))

        return change_points

    def _detect_bayesian(self, data: np.ndarray) -> List[int]:
        """Detect change points using Bayesian method.

        Args:
            data: Time series data

        Returns:
            List of change point indices
        """
        n = len(data)

        # Compute growth probabilities
        Q = np.zeros((n, n))

        for s in range(n):
            for t in range(s + self.min_size, n):
                # Compute log likelihood ratio
                segment = data[s:t]
                segment_next = data[t:min(t + self.min_size, n)]

                if len(segment_next) == 0:
                    continue

                mean1 = np.mean(segment)
                mean2 = np.mean(segment_next)
                var1 = np.var(segment) + 1e-8
                var2 = np.var(segment_next) + 1e-8

                # Log likelihood ratio
                ll_ratio = (
                    -0.5 * len(segment) * np.log(var1)
                    - 0.5 * len(segment_next) * np.log(var2)
                    + 0.5 * (t - s) * np.log(np.var(data[s:t]) + 1e-8)
                )

                Q[s, t] = ll_ratio

        # Find change points via dynamic programming
        P = np.zeros(n)
        cp = np.zeros(n, dtype=int)

        for t in range(self.min_size, n):
            candidates = []
            for s in range(0, t - self.min_size + 1):
                cost = P[s] + Q[s, t] - self.penalty
                candidates.append((cost, s))

            if candidates:
                P[t], cp[t] = max(candidates)

        # Backtrack to find change points
        change_points = []
        t = n - 1
        while t > 0:
            if cp[t] > 0:
                change_points.append(cp[t])
                t = cp[t]
            else:
                break

        change_points.reverse()

        logger.debug("Bayesian detection completed", num_changes=len(change_points))

        return change_points

    def _detect_pelt(self, data: np.ndarray) -> List[int]:
        """Detect change points using PELT algorithm.

        Args:
            data: Time series data

        Returns:
            List of change point indices
        """
        n = len(data)

        # Cost function for segments
        def segment_cost(start: int, end: int) -> float:
            if end <= start:
                return 0.0

            segment = data[start:end]

            if self.model == "l1":
                # L1 cost (median)
                center = np.median(segment)
                cost = np.sum(np.abs(segment - center))
            elif self.model == "l2":
                # L2 cost (mean)
                center = np.mean(segment)
                cost = np.sum((segment - center) ** 2)
            elif self.model == "normal":
                # Negative log likelihood for normal distribution
                mu = np.mean(segment)
                sigma = np.std(segment) + 1e-8
                cost = len(segment) * np.log(sigma) + np.sum((segment - mu) ** 2) / (2 * sigma ** 2)
            else:  # rbf (kernel-based)
                # RBF kernel cost
                center = np.mean(segment)
                cost = len(segment) - np.sum(np.exp(-((segment - center) ** 2) / (2 * self.prior_scale ** 2)))

            return cost

        # Dynamic programming with pruning
        F = np.zeros(n + 1)
        F[0] = -self.penalty
        cp = np.zeros(n + 1, dtype=int)
        R = [0]  # Candidate change points

        for t in range(self.min_size, n + 1):
            candidates = []
            pruned_R = []

            for s in R:
                if t - s >= self.min_size:
                    cost = F[s] + segment_cost(s, t) + self.penalty
                    candidates.append((cost, s))

                    # Pruning: keep only if cost is competitive
                    if not pruned_R or cost <= F[t] + segment_cost(t, n):
                        pruned_R.append(s)

            if candidates:
                F[t], cp[t] = min(candidates)
                pruned_R.append(t)
                R = pruned_R
            else:
                F[t] = F[t - 1] + segment_cost(t - 1, t)
                cp[t] = cp[t - 1]

        # Backtrack to find change points
        change_points = []
        t = n
        while t > 0:
            if cp[t] > 0 and cp[t] != t:
                change_points.append(cp[t])
                t = cp[t]
            else:
                break

        change_points.reverse()

        logger.debug("PELT detection completed", num_changes=len(change_points))

        return change_points

    def _detect_kernel(self, data: np.ndarray) -> List[int]:
        """Detect change points using kernel method.

        Args:
            data: Time series data

        Returns:
            List of change point indices
        """
        n = len(data)

        # Compute kernel change detection statistic
        window_size = self.min_size
        statistics = np.zeros(n - 2 * window_size)

        for t in range(window_size, n - window_size):
            # Data before and after
            before = data[t - window_size:t]
            after = data[t:t + window_size]

            # Compute MMD (Maximum Mean Discrepancy) statistic
            mean_before = np.mean(before)
            mean_after = np.mean(after)

            var_before = np.var(before) + 1e-8
            var_after = np.var(after) + 1e-8

            # Kernel statistic (simplified RBF-based)
            mmd = (mean_after - mean_before) ** 2 / (var_before + var_after)
            statistics[t - window_size] = mmd

        # Find peaks in the statistic
        peaks, properties = find_peaks(
            statistics,
            height=self.threshold,
            distance=self.min_size
        )

        # Convert peaks to change point indices
        change_points = [p + window_size for p in peaks]

        logger.debug("Kernel detection completed", num_changes=len(change_points))

        return change_points

    def detect(self, data: np.ndarray) -> List[int]:
        """Detect change points in time series data.

        Args:
            data: Time series data of shape (n_samples,)

        Returns:
            List of change point indices

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If detection fails
        """
        try:
            logger.info("Starting change point detection", method=self.method)

            # Validate input
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            if len(data.shape) != 1:
                raise ValueError("Data must be 1-dimensional")

            if len(data) < self.min_size * 2:
                raise ValueError(f"Data too short, need at least {self.min_size * 2} points")

            # Detect change points based on method
            if self.method == "cusum":
                self.change_points = self._detect_cusum(data)
            elif self.method == "bayesian":
                self.change_points = self._detect_bayesian(data)
            elif self.method == "pelt":
                self.change_points = self._detect_pelt(data)
            elif self.method == "kernel":
                self.change_points = self._detect_kernel(data)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Mark as trained
            self.is_trained = True
            self.last_trained = datetime.utcnow()

            logger.info(
                "Detection completed",
                num_changes=len(self.change_points),
                change_points=self.change_points[:10]  # Log first 10
            )

            return self.change_points

        except Exception as e:
            logger.error("Detection failed", error=str(e))
            raise RuntimeError(f"Detection failed: {e}")

    def get_regimes(self, data: np.ndarray) -> np.ndarray:
        """Get regime labels for each data point.

        Args:
            data: Time series data

        Returns:
            Array of regime labels (integers starting from 0)

        Raises:
            RuntimeError: If detect() has not been called
        """
        if not self.is_trained:
            raise RuntimeError("Must call detect() before get_regimes()")

        n = len(data)
        regimes = np.zeros(n, dtype=int)

        if not self.change_points:
            # No change points, all one regime
            self.regimes = regimes
            return regimes

        # Assign regime labels
        regime_id = 0
        prev_cp = 0

        for cp in self.change_points:
            regimes[prev_cp:cp] = regime_id
            regime_id += 1
            prev_cp = cp

        regimes[prev_cp:] = regime_id

        self.regimes = regimes

        # Calculate regime statistics
        self._calculate_regime_stats(data)

        logger.debug("Regimes computed", num_regimes=regime_id + 1)

        return regimes

    def _calculate_regime_stats(self, data: np.ndarray) -> None:
        """Calculate statistics for each regime.

        Args:
            data: Time series data
        """
        if self.regimes is None:
            return

        self.regime_stats = []
        num_regimes = np.max(self.regimes) + 1

        for regime_id in range(num_regimes):
            mask = self.regimes == regime_id
            regime_data = data[mask]

            if len(regime_data) == 0:
                continue

            stats = {
                "regime_id": int(regime_id),
                "size": int(len(regime_data)),
                "mean": float(np.mean(regime_data)),
                "std": float(np.std(regime_data)),
                "min": float(np.min(regime_data)),
                "max": float(np.max(regime_data)),
                "median": float(np.median(regime_data)),
            }

            self.regime_stats.append(stats)

        logger.debug("Regime statistics calculated", num_regimes=num_regimes)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train is equivalent to detect for this model.

        Args:
            features: Time series data
            labels: Not used for unsupervised detection
        """
        # For change point detection, training is just detection
        if len(features.shape) > 1:
            # Use first column if multidimensional
            data = features[:, 0]
        else:
            data = features

        self.detect(data)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime for new data points.

        Args:
            features: Time series data

        Returns:
            Regime labels for each point

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # For new data, detect change points and return regimes
        if len(features.shape) > 1:
            data = features[:, 0]
        else:
            data = features

        self.detect(data)
        return self.get_regimes(data)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate change point detection.

        Args:
            features: Time series data
            labels: True change point locations (binary array)

        Returns:
            Dictionary of evaluation metrics

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        try:
            # Detect change points
            detected_cps = set(self.change_points)

            # Convert labels to change points
            if len(labels.shape) > 1:
                labels = labels.ravel()

            true_cps = set(np.where(labels == 1)[0])

            # Calculate metrics with tolerance
            tolerance = self.min_size // 2

            true_positives = 0
            for true_cp in true_cps:
                if any(abs(true_cp - det_cp) <= tolerance for det_cp in detected_cps):
                    true_positives += 1

            false_positives = len(detected_cps) - true_positives
            false_negatives = len(true_cps) - true_positives

            # Metrics
            precision = true_positives / max(len(detected_cps), 1)
            recall = true_positives / max(len(true_cps), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": int(true_positives),
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
                "num_detected": len(detected_cps),
                "num_true": len(true_cps),
            }

            logger.info("Evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise RuntimeError(f"Evaluation failed: {e}")

    def save(self, path: str) -> None:
        """Save detector to disk.

        Args:
            path: File path for saving
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save detection results
            save_data = {
                "change_points": self.change_points,
                "regimes": self.regimes.tolist() if self.regimes is not None else None,
                "regime_stats": self.regime_stats,
            }

            np.save(save_path, save_data, allow_pickle=True)

            # Save metadata
            self.save_metadata(str(save_path))

            logger.info("Detector saved successfully", path=str(save_path))

        except Exception as e:
            logger.error("Failed to save detector", error=str(e))
            raise IOError(f"Failed to save detector: {e}")

    def load(self, path: str) -> None:
        """Load detector from disk.

        Args:
            path: File path for loading
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise FileNotFoundError(f"Detector file not found: {load_path}")

            # Load detection results
            save_data = np.load(load_path, allow_pickle=True).item()

            self.change_points = save_data.get("change_points", [])

            regimes = save_data.get("regimes")
            if regimes is not None:
                self.regimes = np.array(regimes)

            self.regime_stats = save_data.get("regime_stats", [])

            # Load metadata
            self.load_metadata(str(load_path))

            logger.info("Detector loaded successfully", path=str(load_path))

        except Exception as e:
            logger.error("Failed to load detector", error=str(e))
            raise IOError(f"Failed to load detector: {e}")
