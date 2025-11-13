"""Experience Replay Buffer."""
import logging
from collections import deque
from decimal import Decimal
import random

logger = logging.getLogger(__name__)

class ExperienceReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, float(reward), next_state, done))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self):
        return len(self.buffer)
