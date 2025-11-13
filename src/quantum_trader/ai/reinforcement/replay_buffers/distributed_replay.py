"""Distributed Experience Replay Buffer."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class DistributedReplayBuffer:
    """Distributed replay buffer for multi-agent RL."""
    capacity: int = 100000
    
    def __post_init__(self):
        self.buffer = []
        self.position = 0
        
    async def add(self, experience: Dict):
        """Add experience."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    async def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch."""
        import random
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
