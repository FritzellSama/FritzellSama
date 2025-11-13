"""Elastic Net Model."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
import polars as pl
from sklearn.linear_model import ElasticNet
from quantum_trader.ai.models.base_model import SklearnModel

logger = logging.getLogger(__name__)

class ElasticNetModel(SklearnModel):
    """Elastic Net regression model."""
    
    def __init__(self, model_id: str, alpha: Decimal = Decimal("1.0"), l1_ratio: Decimal = Decimal("0.5")):
        estimator = ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), random_state=42)
        super().__init__(model_id, "elastic_net", estimator)
        logger.info(f"Initialized ElasticNet model: {model_id}")
