"""Concept Drift Detection for Online Learning."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Result from drift detection."""
    drift_detected: bool
    drift_score: Decimal
    timestamp: datetime
    window_performance: Decimal
    baseline_performance: Decimal

class DDMDetector:
    """Drift Detection Method (DDM)."""

    def __init__(self, warning_level: Decimal = Decimal("2.0"), drift_level: Decimal = Decimal("3.0")):
        self.warning_level = float(warning_level)
        self.drift_level = float(drift_level)
        self.errors = []
        self.baseline_error = None
        self.baseline_std = None

    async def detect(self, predictions: np.ndarray, actuals: np.ndarray) -> DriftDetectionResult:
        """Detect concept drift."""
        errors = (predictions != actuals).astype(float)
        self.errors.extend(errors)

        current_error = np.mean(errors)
        current_std = np.std(errors)

        if self.baseline_error is None:
            self.baseline_error = current_error
            self.baseline_std = current_std
            drift_detected = False
            drift_score = Decimal("0")
        else:
            z_score = (current_error - self.baseline_error) / (self.baseline_std + 1e-10)
            drift_score = Decimal(str(abs(z_score)))
            drift_detected = abs(z_score) > self.drift_level

            if drift_detected:
                self.baseline_error = current_error
                self.baseline_std = current_std

        logger.info("Drift detection completed", extra={"drift_detected": drift_detected, "timestamp": datetime.now(timezone.utc).isoformat()})

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=drift_score,
            timestamp=datetime.now(timezone.utc),
            window_performance=Decimal(str(1.0 - current_error)),
            baseline_performance=Decimal(str(1.0 - self.baseline_error)) if self.baseline_error else Decimal("0")
        )
