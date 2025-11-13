"""Polynomial feature engineering."""
from decimal import Decimal
from typing import Dict
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class PolynomialFeatureGenerator:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.degree = config.get("degree", 2)
        
    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns([(pl.col("close") ** 2).alias("close_squared")])
