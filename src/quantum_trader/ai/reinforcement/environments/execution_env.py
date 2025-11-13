"""Execution Environment for RL."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExecutionEnv:
    """Trading execution environment."""
    initial_capital: Decimal = Decimal("100000")
    
    def __post_init__(self):
        self.capital = self.initial_capital
        self.position = Decimal("0")
        self.step_count = 0
        
    def reset(self):
        """Reset environment."""
        self.capital = self.initial_capital
        self.position = Decimal("0")
        self.step_count = 0
        return np.array([float(self.capital), float(self.position)])
        
    def step(self, action: int):
        """Execute action."""
        reward = Decimal("0.01") if action == 1 else Decimal("-0.01")
        self.capital += reward
        self.step_count += 1
        done = self.step_count >= 1000
        state = np.array([float(self.capital), float(self.position)])
        return state, float(reward), done, {}
        
    def render(self):
        """Render environment."""
        logger.info(f"Step: {self.step_count}, Capital: {self.capital}")
