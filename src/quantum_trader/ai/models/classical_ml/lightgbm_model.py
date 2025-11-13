"""LightGBM Model."""
import logging
from decimal import Decimal
from typing import Dict, List
import polars as pl
import lightgbm as lgb
from quantum_trader.ai.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class LightGBMModel(BaseModel):
    def __init__(self, model_id: str):
        super().__init__(model_id, "lightgbm")
        self.model = None
    async def train(self, train_df: pl.DataFrame, feature_cols: List[str], label_col: str, **kwargs) -> Dict[str, Decimal]:
        X = train_df.select(feature_cols).to_numpy()
        y = train_df.select(label_col).to_numpy().ravel()
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train({'objective': 'binary'}, train_data, num_boost_round=100)
        logger.info(f"Trained LightGBM model: {model_id}")
        return {"trees": Decimal("100")}
    async def predict(self, data_df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        X = data_df.select(feature_cols).to_numpy()
        predictions = (self.model.predict(X) > 0.5).astype(int)
        return data_df.with_columns([pl.Series("prediction", predictions.tolist())])
    def save(self, path: str):
        self.model.save_model(path)
    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
