"""Concept drift detection for online learning systems.

This module provides methods for detecting concept drift in streaming data,
which is critical for maintaining model performance in non-stationary markets.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from scipy import stats
from structlog import get_logger

logger = get_logger(__name__)


class DriftDetectionMethod:
    """Drift Detection Method (DDM) for concept drift detection.

    This class implements the DDM algorithm which monitors error rates
    to detect concept drift in online learning scenarios.

    Attributes:
        config: Configuration dictionary for drift detection
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the drift detection method.

        Args:
            config: Configuration dictionary with parameters:
                - warning_level: Standard deviations for warning (default: 2.0)
                - drift_level: Standard deviations for drift (default: 3.0)
                - min_instances: Minimum instances before detection (default: 30)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.warning_level = Decimal(str(config.get("warning_level", "2.0")))
        self.drift_level = Decimal(str(config.get("drift_level", "3.0")))
        self.min_instances = int(config.get("min_instances", 30))

        # State variables
        self.error_rate = Decimal("0.0")
        self.std_dev = Decimal("0.0")
        self.min_error_rate = Decimal("1.0")
        self.min_std_dev = Decimal("1.0")
        self.instance_count = 0
        self.drift_detected = False
        self.warning_detected = False

        logger.info(
            "ddm_initialized",
            warning_level=str(self.warning_level),
            drift_level=str(self.drift_level),
            min_instances=self.min_instances,
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def add_instance(self, prediction_correct: bool) -> Dict[str, Any]:
        """Add a new prediction instance and check for drift.

        Args:
            prediction_correct: Whether the prediction was correct

        Returns:
            Dictionary containing:
                - drift_detected: Whether drift was detected
                - warning_detected: Whether warning level reached
                - error_rate: Current error rate
                - std_dev: Current standard deviation

        Raises:
            ValueError: If instance is invalid
        """
        try:
            self.instance_count += 1
            error = Decimal("0.0") if prediction_correct else Decimal("1.0")

            # Update error rate (exponential moving average)
            alpha = Decimal("1.0") / Decimal(str(self.instance_count))
            self.error_rate = self.error_rate * (Decimal("1.0") - alpha) + error * alpha

            # Update standard deviation
            self.std_dev = (
                self.error_rate
                * (Decimal("1.0") - self.error_rate)
                / Decimal(str(self.instance_count))
            ) ** Decimal("0.5")

            # Reset drift and warning flags
            self.drift_detected = False
            self.warning_detected = False

            # Only check after minimum instances
            if self.instance_count >= self.min_instances:
                # Update minimum error rate and std dev
                if self.error_rate + self.std_dev < self.min_error_rate + self.min_std_dev:
                    self.min_error_rate = self.error_rate
                    self.min_std_dev = self.std_dev

                # Check for drift
                drift_threshold = self.min_error_rate + self.drift_level * self.min_std_dev
                warning_threshold = (
                    self.min_error_rate + self.warning_level * self.min_std_dev
                )

                if self.error_rate + self.std_dev >= drift_threshold:
                    self.drift_detected = True
                    logger.warning(
                        "concept_drift_detected",
                        instance_count=self.instance_count,
                        error_rate=str(self.error_rate),
                        threshold=str(drift_threshold),
                    )
                    self._reset()
                elif self.error_rate + self.std_dev >= warning_threshold:
                    self.warning_detected = True
                    logger.info(
                        "drift_warning",
                        instance_count=self.instance_count,
                        error_rate=str(self.error_rate),
                        threshold=str(warning_threshold),
                    )

            return {
                "drift_detected": self.drift_detected,
                "warning_detected": self.warning_detected,
                "error_rate": self.error_rate,
                "std_dev": self.std_dev,
                "instance_count": self.instance_count,
            }

        except Exception as e:
            logger.error("add_instance_failed", error=str(e))
            raise

    def _reset(self) -> None:
        """Reset the detector after drift detection."""
        self.error_rate = Decimal("0.0")
        self.std_dev = Decimal("0.0")
        self.min_error_rate = Decimal("1.0")
        self.min_std_dev = Decimal("1.0")
        self.instance_count = 0

        logger.info("ddm_reset")


class ADWIN:
    """Adaptive Windowing (ADWIN) drift detection algorithm.

    ADWIN dynamically adjusts window size to detect changes in data distribution.

    Attributes:
        config: Configuration dictionary for ADWIN
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ADWIN detector.

        Args:
            config: Configuration dictionary with parameters:
                - delta: Confidence level (default: 0.002)
                - max_buckets: Maximum number of buckets (default: 5)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.delta = Decimal(str(config.get("delta", "0.002")))
        self.max_buckets = int(config.get("max_buckets", 5))

        # Window state
        self.window: deque = deque()
        self.total = Decimal("0.0")
        self.variance = Decimal("0.0")
        self.width = 0

        logger.info("adwin_initialized", delta=str(self.delta), max_buckets=self.max_buckets)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def add_element(self, value: Decimal) -> bool:
        """Add element to window and check for drift.

        Args:
            value: New data point value

        Returns:
            True if drift detected, False otherwise

        Raises:
            ValueError: If value is invalid
        """
        try:
            if not isinstance(value, Decimal):
                value = Decimal(str(value))

            # Add to window
            self.window.append(value)
            self.width += 1
            self.total += value

            # Update variance
            if self.width > 1:
                mean = self.total / Decimal(str(self.width))
                variance_sum = sum((x - mean) ** 2 for x in self.window)
                self.variance = variance_sum / Decimal(str(self.width))

            # Check for drift
            drift_detected = self._detect_change()

            if drift_detected:
                logger.warning(
                    "adwin_drift_detected",
                    window_size=self.width,
                    mean=str(self.total / Decimal(str(self.width))),
                )

            # Limit window size
            while self.width > self.max_buckets * 100:
                removed = self.window.popleft()
                self.width -= 1
                self.total -= removed

            return drift_detected

        except Exception as e:
            logger.error("adwin_add_element_failed", error=str(e))
            raise

    def _detect_change(self) -> bool:
        """Detect change in window using statistical test.

        Returns:
            True if change detected, False otherwise
        """
        if self.width < 2:
            return False

        try:
            # Split window and compare means
            split_point = self.width // 2
            window_list = list(self.window)

            left_window = window_list[:split_point]
            right_window = window_list[split_point:]

            if len(left_window) == 0 or len(right_window) == 0:
                return False

            left_mean = sum(left_window) / Decimal(str(len(left_window)))
            right_mean = sum(right_window) / Decimal(str(len(right_window)))

            # Calculate difference threshold
            n0 = Decimal(str(len(left_window)))
            n1 = Decimal(str(len(right_window)))
            m = Decimal("1.0") / n0 + Decimal("1.0") / n1

            # Hoeffding bound
            epsilon = (
                Decimal("2.0")
                * self.variance
                * m
                * (Decimal(str(np.log(2.0 / float(self.delta)))) ** Decimal("0.5"))
            )

            # Check if means differ significantly
            if abs(left_mean - right_mean) > epsilon:
                # Remove old data
                for _ in range(split_point):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                return True

            return False

        except Exception as e:
            logger.error("change_detection_failed", error=str(e))
            return False


class PageHinkleyTest:
    """Page-Hinkley test for drift detection.

    Detects changes in the mean of a signal by accumulating
    differences from the mean.

    Attributes:
        config: Configuration dictionary for Page-Hinkley test
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Page-Hinkley test.

        Args:
            config: Configuration dictionary with parameters:
                - threshold: Detection threshold (default: 50.0)
                - alpha: Magnitude of changes to detect (default: 0.9999)
                - min_instances: Minimum instances before detection (default: 30)

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.threshold = Decimal(str(config.get("threshold", "50.0")))
        self.alpha = Decimal(str(config.get("alpha", "0.9999")))
        self.min_instances = int(config.get("min_instances", 30))

        # State variables
        self.sum = Decimal("0.0")
        self.mean = Decimal("0.0")
        self.min_sum = Decimal("0.0")
        self.instance_count = 0

        logger.info(
            "page_hinkley_initialized",
            threshold=str(self.threshold),
            alpha=str(self.alpha),
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def add_element(self, value: Decimal) -> bool:
        """Add element and check for drift.

        Args:
            value: New data point value

        Returns:
            True if drift detected, False otherwise

        Raises:
            ValueError: If value is invalid
        """
        try:
            if not isinstance(value, Decimal):
                value = Decimal(str(value))

            self.instance_count += 1

            # Update mean
            if self.instance_count == 1:
                self.mean = value
            else:
                self.mean = (
                    self.alpha * self.mean + (Decimal("1.0") - self.alpha) * value
                )

            # Update cumulative sum
            self.sum += value - self.mean - self.alpha

            # Track minimum
            if self.sum < self.min_sum:
                self.min_sum = self.sum

            # Check for drift
            if self.instance_count >= self.min_instances:
                ph_value = self.sum - self.min_sum

                if ph_value > self.threshold:
                    logger.warning(
                        "page_hinkley_drift_detected",
                        instance_count=self.instance_count,
                        ph_value=str(ph_value),
                        threshold=str(self.threshold),
                    )
                    self._reset()
                    return True

            return False

        except Exception as e:
            logger.error("page_hinkley_add_element_failed", error=str(e))
            raise

    def _reset(self) -> None:
        """Reset the detector after drift detection."""
        self.sum = Decimal("0.0")
        self.min_sum = Decimal("0.0")
        self.instance_count = 0

        logger.info("page_hinkley_reset")


class ConceptDriftDetector:
    """Main concept drift detection system.

    Combines multiple drift detection methods for robust detection.

    Attributes:
        config: Configuration dictionary for drift detection
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize concept drift detector.

        Args:
            config: Configuration dictionary with method-specific configs

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        # Initialize detectors based on config
        self.detectors: Dict[str, Any] = {}

        if config.get("ddm_enabled", True):
            self.detectors["ddm"] = DriftDetectionMethod(config.get("ddm", {}))

        if config.get("adwin_enabled", True):
            self.detectors["adwin"] = ADWIN(config.get("adwin", {}))

        if config.get("page_hinkley_enabled", True):
            self.detectors["page_hinkley"] = PageHinkleyTest(
                config.get("page_hinkley", {})
            )

        logger.info(
            "concept_drift_detector_initialized",
            num_detectors=len(self.detectors),
            detectors=list(self.detectors.keys()),
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError(f"Config must be a dictionary, got {type(self.config)}")

    def detect_drift(
        self,
        prediction_correct: Optional[bool] = None,
        value: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """Detect drift using configured methods.

        Args:
            prediction_correct: Whether prediction was correct (for DDM)
            value: Numeric value to monitor (for ADWIN, Page-Hinkley)

        Returns:
            Dictionary with drift detection results from all methods

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            results = {}

            # Run DDM if prediction provided
            if prediction_correct is not None and "ddm" in self.detectors:
                results["ddm"] = self.detectors["ddm"].add_instance(prediction_correct)

            # Run ADWIN if value provided
            if value is not None and "adwin" in self.detectors:
                drift = self.detectors["adwin"].add_element(value)
                results["adwin"] = {"drift_detected": drift}

            # Run Page-Hinkley if value provided
            if value is not None and "page_hinkley" in self.detectors:
                drift = self.detectors["page_hinkley"].add_element(value)
                results["page_hinkley"] = {"drift_detected": drift}

            # Aggregate results
            any_drift = any(
                r.get("drift_detected", False) for r in results.values()
            )
            any_warning = any(
                r.get("warning_detected", False) for r in results.values()
            )

            results["summary"] = {
                "drift_detected": any_drift,
                "warning_detected": any_warning,
                "num_detectors": len(results) - 1,
            }

            if any_drift:
                logger.warning("drift_detected_aggregate", methods=list(results.keys()))

            return results

        except Exception as e:
            logger.error("drift_detection_failed", error=str(e))
            raise


__all__ = [
    "DriftDetectionMethod",
    "ADWIN",
    "PageHinkleyTest",
    "ConceptDriftDetector",
]
