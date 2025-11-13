"""Support/Resistance Detection - PRODUCTION"""
from decimal import Decimal
from typing import List, Tuple
import os, logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

class SupportResistance:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.window = int(config.get('window', os.getenv('SR_WINDOW', '20')))

    def find_support_resistance(self, data: pl.DataFrame) -> Tuple[List[Decimal], List[Decimal]]:
        """Find support and resistance levels"""
        prices = data.select('close').to_numpy().ravel()

        # Simplified: use rolling max/min
        supports = [Decimal(str(np.min(prices[max(0, i-self.window):i+1])))
                   for i in range(len(prices))]
        resistances = [Decimal(str(np.max(prices[max(0, i-self.window):i+1])))
                      for i in range(len(prices))]

        return supports, resistances
