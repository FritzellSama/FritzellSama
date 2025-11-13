"""Model registry for managing ML models."""
from decimal import Decimal
from typing import Dict, Optional
from structlog import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.models = {}
        
    def register(self, name: str, model) -> None:
        self.models[name] = model
        logger.info("Model registered", name=name)
        
    def get(self, name: str) -> Optional[object]:
        return self.models.get(name)
