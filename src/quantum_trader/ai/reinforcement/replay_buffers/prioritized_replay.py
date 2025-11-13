"""Prioritized Experience Replay Buffer for reinforcement learning.

This module implements a prioritized experience replay buffer using sum-tree data structure
for efficient sampling based on TD-error priorities.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
    """Single experience tuple for replay buffer.

    Attributes:
        state: Current state representation
        action: Action taken
        reward: Reward received (as Decimal)
        next_state: Next state after action
        done: Whether episode terminated
        timestamp: UTC timestamp of experience
    """
    state: np.ndarray
    action: int
    reward: Decimal
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class SumTree:
    """Binary sum tree for efficient prioritized sampling.

    The sum tree allows O(log n) updates and O(log n) sampling based on priorities.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize sum tree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def add(self, priority: float, experience: Experience) -> None:
        """Add experience with given priority.

        Args:
            priority: Priority value for this experience
            experience: Experience tuple to store
        """
        tree_idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = experience
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority of experience at tree index.

        Args:
            tree_idx: Index in tree structure
            priority: New priority value
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> Tuple[int, float, Experience]:
        """Retrieve experience based on cumulative priority value.

        Args:
            value: Cumulative priority value to search for

        Returns:
            Tuple of (tree_idx, priority, experience)
        """
        parent_idx = 0

        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1

            # Reached leaf node
            if left_idx >= len(self.tree):
                tree_idx = parent_idx
                break

            # Traverse tree based on cumulative priorities
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx

        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Get sum of all priorities."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Get maximum priority in tree."""
        return np.max(self.tree[-self.capacity:])

    @property
    def min_priority(self) -> float:
        """Get minimum non-zero priority in tree."""
        priorities = self.tree[-self.capacity:]
        non_zero = priorities[priorities > 0]
        return np.min(non_zero) if len(non_zero) > 0 else 0.0


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for RL training.

    Implements prioritized experience replay (PER) which samples experiences
    based on their TD-error, giving more weight to surprising experiences.

    Attributes:
        config: Configuration dictionary
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        beta_increment: Amount to increase beta per sample
        epsilon: Small constant to ensure non-zero priorities

    Example:
        >>> config = {
        ...     'capacity': 100000,
        ...     'alpha': 0.6,
        ...     'beta': 0.4,
        ...     'beta_increment_per_sampling': 0.001,
        ...     'epsilon': 0.01
        ... }
        >>> buffer = PrioritizedReplayBuffer(config)
        >>> state = np.array([1, 2, 3])
        >>> next_state = np.array([4, 5, 6])
        >>> buffer.add(state, 0, Decimal('1.5'), next_state, False)
        >>> batch = buffer.sample(32)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize prioritized replay buffer.

        Args:
            config: Configuration with keys:
                - capacity: Buffer size
                - alpha: Priority exponent
                - beta: IS exponent
                - beta_increment_per_sampling: Beta increment
                - epsilon: Minimum priority
        """
        self.config = config
        self._validate_config()

        self.capacity = config['capacity']
        self.alpha = Decimal(str(config['alpha']))
        self.beta = Decimal(str(config['beta']))
        self.beta_increment = Decimal(str(config['beta_increment_per_sampling']))
        self.epsilon = Decimal(str(config['epsilon']))

        self.tree = SumTree(self.capacity)
        self.max_priority = Decimal('1.0')

        logger.info(
            "initialized_prioritized_replay_buffer",
            capacity=self.capacity,
            alpha=str(self.alpha),
            beta=str(self.beta)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'capacity', 'alpha', 'beta',
            'beta_increment_per_sampling', 'epsilon'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['capacity'] <= 0:
            raise ValueError("Capacity must be positive")

        if not (0 <= self.config['alpha'] <= 1):
            raise ValueError("Alpha must be in [0, 1]")

        if not (0 <= self.config['beta'] <= 1):
            raise ValueError("Beta must be in [0, 1]")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: Decimal,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add experience to buffer with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received (Decimal)
            next_state: Next state
            done: Terminal flag
        """
        try:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            # New experiences get maximum priority
            priority = float(self.max_priority ** self.alpha)
            self.tree.add(priority, experience)

            logger.debug(
                "added_experience",
                action=action,
                reward=str(reward),
                priority=priority,
                buffer_size=self.tree.size
            )

        except Exception as e:
            logger.error("failed_to_add_experience", error=str(e))
            raise

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch of experiences based on priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary containing:
                - states: Batch of states
                - actions: Batch of actions
                - rewards: Batch of rewards (Decimal)
                - next_states: Batch of next states
                - dones: Batch of terminal flags
                - indices: Tree indices for updating priorities
                - weights: Importance sampling weights

        Raises:
            ValueError: If buffer has fewer experiences than batch_size
        """
        if self.tree.size < batch_size:
            raise ValueError(
                f"Not enough experiences: {self.tree.size} < {batch_size}"
            )

        try:
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            indices = []
            priorities = []

            # Divide priority range into segments
            segment_size = self.tree.total_priority / batch_size

            for i in range(batch_size):
                # Sample uniformly from each segment
                segment_start = segment_size * i
                segment_end = segment_size * (i + 1)
                value = np.random.uniform(segment_start, segment_end)

                tree_idx, priority, experience = self.tree.get(value)

                batch_states.append(experience.state)
                batch_actions.append(experience.action)
                batch_rewards.append(experience.reward)
                batch_next_states.append(experience.next_state)
                batch_dones.append(experience.done)
                indices.append(tree_idx)
                priorities.append(priority)

            # Calculate importance sampling weights
            priorities_array = np.array(priorities, dtype=np.float64)
            sampling_probs = priorities_array / self.tree.total_priority

            # IS weights: (N * P(i))^(-beta)
            weights = np.power(
                self.tree.size * sampling_probs,
                -float(self.beta)
            )
            # Normalize weights
            weights = weights / np.max(weights)

            # Anneal beta
            self.beta = min(
                Decimal('1.0'),
                self.beta + self.beta_increment
            )

            logger.debug(
                "sampled_batch",
                batch_size=batch_size,
                beta=str(self.beta),
                mean_weight=float(np.mean(weights))
            )

            return {
                'states': np.array(batch_states),
                'actions': np.array(batch_actions),
                'rewards': batch_rewards,  # Keep as list of Decimals
                'next_states': np.array(batch_next_states),
                'dones': np.array(batch_dones),
                'indices': indices,
                'weights': weights
            }

        except Exception as e:
            logger.error("failed_to_sample_batch", error=str(e))
            raise

    def update_priorities(
        self,
        indices: List[int],
        td_errors: np.ndarray
    ) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Tree indices of experiences
            td_errors: TD-errors for priority update
        """
        try:
            for idx, td_error in zip(indices, td_errors):
                # Priority = |TD-error| + epsilon
                priority = (abs(float(td_error)) + float(self.epsilon)) ** float(self.alpha)
                self.tree.update(idx, priority)

                # Track maximum priority
                self.max_priority = max(
                    self.max_priority,
                    Decimal(str(abs(float(td_error)) + float(self.epsilon)))
                )

            logger.debug(
                "updated_priorities",
                count=len(indices),
                mean_td_error=float(np.mean(np.abs(td_errors))),
                max_priority=str(self.max_priority)
            )

        except Exception as e:
            logger.error("failed_to_update_priorities", error=str(e))
            raise

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.tree.size

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary of buffer statistics
        """
        return {
            'size': self.tree.size,
            'capacity': self.capacity,
            'alpha': str(self.alpha),
            'beta': str(self.beta),
            'max_priority': str(self.max_priority),
            'total_priority': self.tree.total_priority,
            'tree_max_priority': self.tree.max_priority,
            'tree_min_priority': self.tree.min_priority
        }
