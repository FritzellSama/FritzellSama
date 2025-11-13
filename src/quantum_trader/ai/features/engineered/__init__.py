"""Engineered features for machine learning models.

This submodule provides custom engineered features including:
- Interaction features between market variables
- Polynomial features for non-linear relationships
- Rolling window statistics (mean, std, min, max)
- Lag features for temporal dependencies
- Differencing and returns calculations
- Composite indicators combining multiple signals
- Factor exposure features from fundamental data
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = []

__version__ = "1.0.0"
