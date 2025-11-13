"""
Change Point Detection for Market Regime Changes.

Implements algorithms to detect structural breaks and regime changes
in financial time series data.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """Detected change point."""

    index: int
    timestamp: datetime
    score: Decimal
    confidence: Decimal
    change_type: str  # mean, variance, trend
    before_stats: Dict[str, Decimal]
    after_stats: Dict[str, Decimal]


@dataclass
class ChangePointConfig:
    """Configuration for change point detection."""

    method: str = "cusum"  # cusum, bayesian, ruptures
    min_segment_length: int = 50
    threshold: Decimal = Decimal("5.0")
    penalty: Decimal = Decimal("1.0")
    max_change_points: int = 10


class CUSUMDetector:
    """Cumulative Sum (CUSUM) change point detector."""

    def __init__(self, threshold: Decimal = Decimal("5.0")):
        """
        Initialize CUSUM detector.

        Args:
            threshold: Detection threshold
        """
        self.threshold = float(threshold)

    def detect(
        self,
        data: np.ndarray,
        timestamps: List[datetime]
    ) -> List[ChangePoint]:
        """
        Detect change points using CUSUM.

        Args:
            data: Time series data
            timestamps: Timestamps for data points

        Returns:
            List of detected change points
        """
        # Compute mean and std
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return []

        # Standardize data
        standardized = (data - mean) / std

        # Compute CUSUM statistics
        cumsum_pos = np.zeros(len(data))
        cumsum_neg = np.zeros(len(data))

        for i in range(1, len(data)):
            cumsum_pos[i] = max(0, cumsum_pos[i - 1] + standardized[i])
            cumsum_neg[i] = min(0, cumsum_neg[i - 1] + standardized[i])

        # Detect change points
        change_points = []

        for i in range(len(data)):
            if abs(cumsum_pos[i]) > self.threshold or abs(cumsum_neg[i]) > self.threshold:
                # Compute statistics before and after
                before_data = data[:i] if i > 0 else np.array([data[0]])
                after_data = data[i:] if i < len(data) else np.array([data[-1]])

                before_stats = {
                    "mean": Decimal(str(np.mean(before_data))),
                    "std": Decimal(str(np.std(before_data)))
                }
                after_stats = {
                    "mean": Decimal(str(np.mean(after_data))),
                    "std": Decimal(str(np.std(after_data)))
                }

                # Determine change type
                mean_diff = abs(before_stats["mean"] - after_stats["mean"])
                std_diff = abs(before_stats["std"] - after_stats["std"])

                if mean_diff > std_diff:
                    change_type = "mean"
                else:
                    change_type = "variance"

                change_point = ChangePoint(
                    index=i,
                    timestamp=timestamps[i] if i < len(timestamps) else timestamps[-1],
                    score=Decimal(str(max(abs(cumsum_pos[i]), abs(cumsum_neg[i])))),
                    confidence=Decimal("0.8"),
                    change_type=change_type,
                    before_stats=before_stats,
                    after_stats=after_stats
                )

                change_points.append(change_point)

                # Reset CUSUM after detection
                cumsum_pos[i:] = 0
                cumsum_neg[i:] = 0

        logger.info(
            "CUSUM detection completed",
            extra={
                "num_change_points": len(change_points),
                "threshold": self.threshold,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return change_points


class BayesianDetector:
    """Bayesian change point detector."""

    def __init__(self, prior_prob: Decimal = Decimal("0.01")):
        """
        Initialize Bayesian detector.

        Args:
            prior_prob: Prior probability of change point
        """
        self.prior_prob = float(prior_prob)

    def detect(
        self,
        data: np.ndarray,
        timestamps: List[datetime]
    ) -> List[ChangePoint]:
        """
        Detect change points using Bayesian approach.

        Args:
            data: Time series data
            timestamps: Timestamps for data points

        Returns:
            List of detected change points
        """
        n = len(data)
        change_probs = np.zeros(n)

        # Compute change probabilities
        for t in range(1, n - 1):
            before_data = data[:t]
            after_data = data[t:]

            # Compute likelihood ratio
            before_mean = np.mean(before_data)
            after_mean = np.mean(after_data)

            before_var = np.var(before_data) + 1e-10
            after_var = np.var(after_data) + 1e-10

            # Log likelihood
            ll_before = -0.5 * np.sum((before_data - before_mean) ** 2 / before_var)
            ll_after = -0.5 * np.sum((after_data - after_mean) ** 2 / after_var)

            combined_mean = np.mean(data)
            combined_var = np.var(data) + 1e-10
            ll_combined = -0.5 * np.sum((data - combined_mean) ** 2 / combined_var)

            # Bayes factor
            log_bf = (ll_before + ll_after) - ll_combined

            # Posterior probability
            odds = np.exp(log_bf) * self.prior_prob / (1 - self.prior_prob)
            change_probs[t] = odds / (1 + odds)

        # Detect peaks in change probabilities
        change_points = []
        threshold = 0.5

        for i in range(1, n - 1):
            if (change_probs[i] > threshold and
                change_probs[i] > change_probs[i - 1] and
                change_probs[i] > change_probs[i + 1]):

                # Compute statistics
                before_data = data[:i]
                after_data = data[i:]

                before_stats = {
                    "mean": Decimal(str(np.mean(before_data))),
                    "std": Decimal(str(np.std(before_data)))
                }
                after_stats = {
                    "mean": Decimal(str(np.mean(after_data))),
                    "std": Decimal(str(np.std(after_data)))
                }

                change_point = ChangePoint(
                    index=i,
                    timestamp=timestamps[i],
                    score=Decimal(str(change_probs[i])),
                    confidence=Decimal(str(change_probs[i])),
                    change_type="bayesian",
                    before_stats=before_stats,
                    after_stats=after_stats
                )

                change_points.append(change_point)

        logger.info(
            "Bayesian detection completed",
            extra={
                "num_change_points": len(change_points),
                "prior_prob": self.prior_prob,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return change_points


class ChangePointDetector:
    """Main change point detection coordinator."""

    def __init__(self, config: ChangePointConfig):
        """
        Initialize change point detector.

        Args:
            config: Detection configuration
        """
        self.config = config

        # Initialize detectors
        if config.method == "cusum":
            self.detector = CUSUMDetector(config.threshold)
        elif config.method == "bayesian":
            self.detector = BayesianDetector()
        else:
            self.detector = CUSUMDetector(config.threshold)

        logger.info(
            "Initialized change point detector",
            extra={
                "method": config.method,
                "threshold": str(config.threshold),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def detect(
        self,
        data_df: pl.DataFrame,
        value_col: str,
        timestamp_col: str = "timestamp"
    ) -> List[ChangePoint]:
        """
        Detect change points in time series.

        Args:
            data_df: DataFrame with time series data
            value_col: Column with values
            timestamp_col: Column with timestamps

        Returns:
            List of detected change points
        """
        # Extract data
        values = data_df.select(value_col).to_numpy().ravel()
        timestamps = data_df.select(timestamp_col).to_series().to_list()

        # Parse timestamps if strings
        if timestamps and isinstance(timestamps[0], str):
            from dateutil import parser
            timestamps = [parser.parse(ts) for ts in timestamps]

        # Detect change points
        change_points = self.detector.detect(values, timestamps)

        # Filter by minimum segment length
        filtered_points = []
        last_idx = 0

        for cp in change_points:
            if cp.index - last_idx >= self.config.min_segment_length:
                filtered_points.append(cp)
                last_idx = cp.index

        # Limit number of change points
        filtered_points = sorted(
            filtered_points,
            key=lambda x: x.score,
            reverse=True
        )[:self.config.max_change_points]

        # Sort by index
        filtered_points = sorted(filtered_points, key=lambda x: x.index)

        logger.info(
            "Change point detection completed",
            extra={
                "total_detected": len(change_points),
                "filtered": len(filtered_points),
                "method": self.config.method,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return filtered_points

    def to_dataframe(self, change_points: List[ChangePoint]) -> pl.DataFrame:
        """
        Convert change points to DataFrame.

        Args:
            change_points: List of change points

        Returns:
            DataFrame with change points
        """
        if not change_points:
            return pl.DataFrame()

        data = {
            "index": [cp.index for cp in change_points],
            "timestamp": [cp.timestamp.isoformat() for cp in change_points],
            "score": [str(cp.score) for cp in change_points],
            "confidence": [str(cp.confidence) for cp in change_points],
            "change_type": [cp.change_type for cp in change_points],
            "before_mean": [str(cp.before_stats["mean"]) for cp in change_points],
            "after_mean": [str(cp.after_stats["mean"]) for cp in change_points]
        }

        return pl.DataFrame(data)

    async def segment_data(
        self,
        data_df: pl.DataFrame,
        change_points: List[ChangePoint],
        value_col: str
    ) -> List[pl.DataFrame]:
        """
        Segment data based on detected change points.

        Args:
            data_df: Original DataFrame
            change_points: Detected change points
            value_col: Value column name

        Returns:
            List of segmented DataFrames
        """
        if not change_points:
            return [data_df]

        segments = []
        indices = [0] + [cp.index for cp in change_points] + [len(data_df)]

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]

            segment = data_df[start:end]
            segments.append(segment)

        logger.info(
            "Data segmentation completed",
            extra={
                "num_segments": len(segments),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return segments

    def analyze_segments(
        self,
        segments: List[pl.DataFrame],
        value_col: str
    ) -> pl.DataFrame:
        """
        Analyze statistical properties of segments.

        Args:
            segments: List of data segments
            value_col: Value column name

        Returns:
            DataFrame with segment analysis
        """
        analysis_data = {
            "segment_id": [],
            "length": [],
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "trend": []
        }

        for i, segment in enumerate(segments):
            values = segment.select(value_col).to_numpy().ravel()

            # Compute statistics
            analysis_data["segment_id"].append(i)
            analysis_data["length"].append(len(values))
            analysis_data["mean"].append(str(Decimal(str(np.mean(values)))))
            analysis_data["std"].append(str(Decimal(str(np.std(values)))))
            analysis_data["min"].append(str(Decimal(str(np.min(values)))))
            analysis_data["max"].append(str(Decimal(str(np.max(values)))))

            # Compute trend (linear regression slope)
            if len(values) > 1:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                analysis_data["trend"].append(str(Decimal(str(slope))))
            else:
                analysis_data["trend"].append("0.0")

        return pl.DataFrame(analysis_data)
