"""Prioritized experience replay buffer."""
from decimal import Decimal
from typing import Dict, List, Tuple
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)

class PrioritizedReplayBuffer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.capacity = config.get("capacity", 100000)
        self.buffer: List = []
        self.priorities = np.zeros(self.capacity)
        
    def add(self, experience: Tuple, priority: float = 1.0) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            idx = np.argmin(self.priorities)
            self.buffer[idx] = experience
            self.priorities[idx] = priority
            
    def sample(self, batch_size: int) -> List[Tuple]:
        if len(self.buffer) < batch_size:
            return []
        probs = self.priorities[:len(self.buffer)] / self.priorities[:len(self.buffer)].sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
