"""Feature Selection."""
import logging
from typing import List
import polars as pl
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, k: int = 10):
        self.selector = SelectKBest(f_classif, k=k)
        self.selected_features = None
    async def fit(self, df: pl.DataFrame, feature_cols: List[str], label_col: str):
        X = df.select(feature_cols).to_numpy()
        y = df.select(label_col).to_numpy().ravel()
        self.selector.fit(X, y)
        mask = self.selector.get_support()
        self.selected_features = [f for f, m in zip(feature_cols, mask) if m]
        logger.info(f"Selected {len(self.selected_features)} features")
    async def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(self.selected_features)
