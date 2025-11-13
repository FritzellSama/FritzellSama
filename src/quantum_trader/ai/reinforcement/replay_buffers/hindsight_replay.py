"""Hindsight Experience Replay."""
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)

class HindsightReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    def add(self, episode):
        for transition in episode:
            self.buffer.append(transition)
    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
