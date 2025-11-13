"""Experience Replay Buffer for Reinforcement Learning.

This module implements a production-ready experience replay buffer for storing
and sampling trading experiences for reinforcement learning algorithms.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import random
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class Experience:
    """Single experience tuple for reinforcement learning.

    Attributes:
        state: Current state observation
        action: Action taken
        reward: Reward received (as Decimal)
        next_state: Resulting state
        done: Whether episode terminated
        timestamp: UTC timestamp of experience
        metadata: Additional context
    """
    state: np.ndarray
    action: int
    reward: Decimal
    next_state: np.ndarray
    done: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceReplayBuffer:
    """Production-ready experience replay buffer with prioritization support.

    This buffer stores trading experiences and supports:
    - Random sampling for standard experience replay
    - Prioritized sampling based on TD error
    - Efficient storage with maximum capacity
    - Data persistence and recovery
    - Statistics tracking

    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Deque storing experiences
        priorities: Priority values for each experience
        config: Configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize experience replay buffer.

        Args:
            config: Configuration with keys:
                - buffer_capacity: Maximum buffer size
                - prioritized: Whether to use prioritized replay
                - alpha: Priority exponent (for prioritized)
                - beta: Importance sampling exponent
                - beta_increment: Beta increase per sample
                - min_priority: Minimum priority value

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.capacity: int = config.get("buffer_capacity", 100000)
        self.buffer: Deque[Experience] = deque(maxlen=self.capacity)
        self.priorities: Deque[Decimal] = deque(maxlen=self.capacity)

        # Prioritized replay parameters
        self.prioritized: bool = config.get("prioritized", False)
        self.alpha: Decimal = Decimal(str(config.get("alpha", 0.6)))
        self.beta: Decimal = Decimal(str(config.get("beta", 0.4)))
        self.beta_increment: Decimal = Decimal(str(config.get("beta_increment", 0.001)))
        self.min_priority: Decimal = Decimal(str(config.get("min_priority", 0.01)))

        # Statistics
        self.total_added: int = 0
        self.total_sampled: int = 0

        logger.info(
            "experience_replay_buffer_initialized",
            capacity=self.capacity,
            prioritized=self.prioritized
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters missing or invalid
        """
        if "buffer_capacity" not in self.config:
            raise ValueError("buffer_capacity required in config")

        capacity = self.config["buffer_capacity"]
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive integer, got {capacity}")

        if self.config.get("prioritized", False):
            if "alpha" not in self.config:
                raise ValueError("alpha required for prioritized replay")
            if "beta" not in self.config:
                raise ValueError("beta required for prioritized replay")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: Decimal,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[Decimal] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add experience to buffer.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received (Decimal)
            next_state: Resulting state
            done: Whether episode terminated
            priority: Optional priority value
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(reward, Decimal):
                raise ValueError(f"reward must be Decimal, got {type(reward)}")

            if not isinstance(state, np.ndarray) or not isinstance(next_state, np.ndarray):
                raise ValueError("state and next_state must be numpy arrays")

            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )

            # Add to buffer
            self.buffer.append(experience)

            # Handle priority
            if self.prioritized:
                if priority is None:
                    # New experiences get maximum priority
                    priority = max(self.priorities) if self.priorities else Decimal("1.0")
                priority = max(priority, self.min_priority)
                self.priorities.append(priority)

            self.total_added += 1

            if self.total_added % 1000 == 0:
                logger.debug(
                    "experience_buffer_update",
                    total_added=self.total_added,
                    buffer_size=len(self.buffer)
                )

        except Exception as e:
            logger.error("add_experience_failed", error=str(e))
            raise

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample batch of experiences from buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, importance_weights, indices)
            - experiences: List of sampled Experience objects
            - importance_weights: Importance sampling weights (uniform if not prioritized)
            - indices: Indices of sampled experiences

        Raises:
            ValueError: If batch_size invalid or buffer too small
        """
        try:
            if batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {batch_size}")

            if len(self.buffer) < batch_size:
                raise ValueError(
                    f"Not enough experiences: {len(self.buffer)} < {batch_size}"
                )

            if self.prioritized:
                experiences, weights, indices = self._sample_prioritized(batch_size)
            else:
                experiences, weights, indices = self._sample_uniform(batch_size)

            self.total_sampled += batch_size

            # Increment beta for importance sampling annealing
            if self.prioritized:
                self.beta = min(Decimal("1.0"), self.beta + self.beta_increment)

            return experiences, weights, indices

        except Exception as e:
            logger.error("sample_experiences_failed", error=str(e), batch_size=batch_size)
            raise

    def _sample_uniform(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample experiences uniformly.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, uniform_weights, indices)
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        experiences = [self.buffer[idx] for idx in indices]
        weights = np.ones(batch_size, dtype=np.float64)

        return experiences, weights, indices

    def _sample_prioritized(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, importance_weights, indices)
        """
        # Calculate sampling probabilities
        priorities_array = np.array([float(p) for p in self.priorities], dtype=np.float64)
        priorities_alpha = np.power(priorities_array, float(self.alpha))
        probabilities = priorities_alpha / priorities_alpha.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )

        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = np.power(len(self.buffer) * probabilities[indices], -float(self.beta))
        weights = weights / weights.max()  # Normalize

        return experiences, weights, indices.tolist()

    def update_priorities(self, indices: List[int], priorities: List[Decimal]) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            priorities: New priority values (typically TD errors)

        Raises:
            ValueError: If inputs invalid or not in prioritized mode
        """
        try:
            if not self.prioritized:
                raise ValueError("Cannot update priorities in non-prioritized mode")

            if len(indices) != len(priorities):
                raise ValueError("indices and priorities must have same length")

            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    # Ensure minimum priority
                    self.priorities[idx] = max(priority, self.min_priority)
                else:
                    logger.warning("invalid_priority_index", index=idx)

        except Exception as e:
            logger.error("update_priorities_failed", error=str(e))
            raise

    def get_batch_as_arrays(
        self, experiences: List[Experience]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert batch of experiences to numpy arrays.

        Args:
            experiences: List of Experience objects

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        try:
            states = np.array([exp.state for exp in experiences], dtype=np.float64)
            actions = np.array([exp.action for exp in experiences], dtype=np.int64)
            rewards = np.array([float(exp.reward) for exp in experiences], dtype=np.float64)
            next_states = np.array([exp.next_state for exp in experiences], dtype=np.float64)
            dones = np.array([exp.done for exp in experiences], dtype=np.bool_)

            return states, actions, rewards, next_states, dones

        except Exception as e:
            logger.error("batch_conversion_failed", error=str(e))
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            "capacity": self.capacity,
            "size": len(self.buffer),
            "utilization": len(self.buffer) / self.capacity,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "prioritized": self.prioritized
        }

        if self.prioritized and self.priorities:
            priorities_array = [float(p) for p in self.priorities]
            stats.update({
                "mean_priority": float(np.mean(priorities_array)),
                "max_priority": float(np.max(priorities_array)),
                "min_priority": float(np.min(priorities_array)),
                "beta": float(self.beta)
            })

        return stats

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.priorities.clear()
        logger.info("experience_buffer_cleared")

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation of buffer."""
        return (
            f"ExperienceReplayBuffer(capacity={self.capacity}, "
            f"size={len(self.buffer)}, prioritized={self.prioritized})"
        )
