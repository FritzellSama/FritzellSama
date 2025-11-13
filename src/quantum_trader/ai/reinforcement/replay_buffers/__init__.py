"""Experience replay buffers for reinforcement learning.

This module provides efficient implementations of experience replay buffers
used in deep reinforcement learning algorithms. It includes standard replay
buffers, prioritized experience replay, and distributed variants.

Replay buffers enable sample efficiency by storing and replaying past
experiences, which is crucial for off-policy learning algorithms like
DQN, DDPG, and SAC.
"""

from typing import Dict, List, Tuple, Any

__all__ = [
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "DistributedReplayBuffer",
    "ExperienceTransition",
]

# Placeholder exports - import actual classes when available
# from .replay_buffer import ReplayBuffer
# from .prioritized_buffer import PrioritizedReplayBuffer
# from .distributed_buffer import DistributedReplayBuffer
# from .transition import ExperienceTransition

__version__ = "1.0.0"
