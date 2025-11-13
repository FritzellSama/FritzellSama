"""Model interpretability and explainability module.

This submodule provides tools for understanding AI model decisions including:
- SHAP (SHapley Additive exPlanations) value computation
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis and attribution
- Decision tree visualization
- Attention mechanism visualization
- Gradient-based explanations
- Counterfactual explanations
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__all__: list[str] = []

__version__ = "1.0.0"
