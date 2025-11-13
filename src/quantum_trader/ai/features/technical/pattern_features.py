"""Technical pattern recognition features."""
from decimal import Decimal
from typing import Dict, List
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

class PatternFeatureExtractor:
    def __init__(self, config: Dict) -> None:
        self.config = config
        
    def detect_patterns(self, data: pl.DataFrame) -> List[str]:
        patterns = []
        if len(data) >= 3:
            patterns.append("double_top")
        return patterns
