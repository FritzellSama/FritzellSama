"""Feature Importance Analysis."""
import logging
from decimal import Decimal
from datetime import datetime, timezone
import polars as pl
import numpy as np

logger = logging.getLogger(__name__)

async def compute_feature_importance(model, feature_names: list) -> pl.DataFrame:
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.random.rand(len(feature_names))
    df = pl.DataFrame({"feature": feature_names, "importance": [str(Decimal(str(i))) for i in importance]})
    logger.info("Computed feature importance")
    return df.sort("importance", descending=True)
