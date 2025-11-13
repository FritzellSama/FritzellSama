"""Inference Engine for Production."""
import logging
from datetime import datetime, timezone
from decimal import Decimal
import torch
import polars as pl

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    async def predict(self, df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
        X = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.model(X)
            predictions = outputs.argmax(dim=-1).cpu().numpy()
        return df.with_columns([pl.Series("prediction", predictions.tolist())])
