"""LIME Explainer for Model Interpretability."""
import logging
from datetime import datetime, timezone
import numpy as np
import polars as pl
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)

class LIMEExplainer:
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    def fit(self, training_data: np.ndarray):
        self.explainer = LimeTabularExplainer(training_data, feature_names=self.feature_names, mode='classification')
        logger.info("LIME explainer fitted")
    async def explain(self, instance: np.ndarray) -> dict:
        if self.explainer is None:
            raise ValueError("Explainer not fitted")
        exp = self.explainer.explain_instance(instance, self.model.predict_proba, num_features=5)
        return {"explanation": exp.as_list(), "timestamp": datetime.now(timezone.utc).isoformat()}
