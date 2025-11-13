"""Hindsight Experience Replay buffer for reinforcement learning.

This module implements HER for sparse reward environments, enabling
learning from failed episodes by relabeling goals in hindsight.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import deque
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class HindsightReplayBuffer:
    """Hindsight Experience Replay buffer.

    Stores transitions and generates additional training samples by
    relabeling goals in hindsight, useful for sparse reward trading scenarios.

    Attributes:
        config: Configuration dictionary
        buffer: Deque storing transitions
        capacity: Maximum buffer size
        strategy: Hindsight strategy ('final', 'future', 'episode', 'random')

    Example:
        >>> config = {
        ...     "capacity": 100000,
        ...     "strategy": "future",
        ...     "k": 4,
        ...     "reward_fn": custom_reward_function
        ... }
        >>> buffer = HindsightReplayBuffer(config)
        >>> buffer.store_transition(state, action, reward, next_state, goal, done)
        >>> batch = buffer.sample(batch_size=64)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize hindsight replay buffer.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.capacity = config["capacity"]
        self.strategy = config.get("strategy", "future")
        self.k = config.get("k", 4)  # Number of hindsight goals per transition
        self.reward_fn = config.get("reward_fn", self._default_reward_fn)

        # Buffer storage
        self.buffer: deque = deque(maxlen=self.capacity)
        self.episode_buffer: List[Dict[str, Any]] = []

        # Statistics
        self.num_episodes = 0
        self.num_transitions = 0
        self.num_hindsight_samples = 0

        logger.info(
            "Hindsight replay buffer initialized",
            capacity=self.capacity,
            strategy=self.strategy,
            k=self.k
        )

    def _validate_config(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If required config missing
        """
        required_keys = ["capacity"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        valid_strategies = ["final", "future", "episode", "random"]
        strategy = self.config.get("strategy", "future")

        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Must be one of {valid_strategies}"
            )

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Decimal,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store single transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            goal: Goal state
            done: Whether episode is done
            info: Optional additional info
        """
        try:
            transition = {
                "state": state.copy(),
                "action": action.copy(),
                "reward": reward,
                "next_state": next_state.copy(),
                "goal": goal.copy(),
                "done": done,
                "info": info or {},
                "timestamp": datetime.utcnow()
            }

            self.episode_buffer.append(transition)
            self.num_transitions += 1

            # If episode is done, process and store
            if done:
                self._process_episode()

        except Exception as e:
            logger.error("Failed to store transition", error=str(e))
            raise

    def _process_episode(self) -> None:
        """Process completed episode and generate hindsight samples."""
        try:
            if not self.episode_buffer:
                return

            # Store original episode
            for transition in self.episode_buffer:
                self.buffer.append(transition)

            # Generate hindsight samples
            hindsight_samples = self._generate_hindsight_samples()

            for sample in hindsight_samples:
                self.buffer.append(sample)
                self.num_hindsight_samples += 1

            self.num_episodes += 1

            logger.debug(
                "Episode processed",
                episode_length=len(self.episode_buffer),
                hindsight_samples=len(hindsight_samples),
                total_transitions=len(self.buffer)
            )

            # Clear episode buffer
            self.episode_buffer = []

        except Exception as e:
            logger.error("Failed to process episode", error=str(e))
            raise

    def _generate_hindsight_samples(self) -> List[Dict[str, Any]]:
        """Generate hindsight experience replay samples.

        Returns:
            List of hindsight transitions
        """
        try:
            hindsight_samples = []
            episode_length = len(self.episode_buffer)

            for t in range(episode_length):
                transition = self.episode_buffer[t]

                # Generate k hindsight goals
                hindsight_goals = self._select_hindsight_goals(t, episode_length)

                for new_goal in hindsight_goals:
                    # Compute new reward with hindsight goal
                    new_reward = self.reward_fn(
                        transition["state"],
                        transition["action"],
                        transition["next_state"],
                        new_goal
                    )

                    # Check if goal achieved
                    new_done = self._check_goal_achieved(
                        transition["next_state"],
                        new_goal
                    )

                    # Create hindsight sample
                    hindsight_sample = {
                        "state": transition["state"].copy(),
                        "action": transition["action"].copy(),
                        "reward": new_reward,
                        "next_state": transition["next_state"].copy(),
                        "goal": new_goal.copy(),
                        "done": new_done,
                        "info": {**transition["info"], "hindsight": True},
                        "timestamp": transition["timestamp"]
                    }

                    hindsight_samples.append(hindsight_sample)

            return hindsight_samples

        except Exception as e:
            logger.error("Failed to generate hindsight samples", error=str(e))
            raise

    def _select_hindsight_goals(
        self,
        current_idx: int,
        episode_length: int
    ) -> List[np.ndarray]:
        """Select hindsight goals based on strategy.

        Args:
            current_idx: Current transition index
            episode_length: Total episode length

        Returns:
            List of hindsight goals
        """
        goals = []

        if self.strategy == "final":
            # Use final state as goal
            final_state = self.episode_buffer[-1]["next_state"]
            goals = [self._extract_goal(final_state)] * self.k

        elif self.strategy == "future":
            # Use future states as goals
            future_indices = np.random.randint(
                current_idx + 1,
                episode_length,
                size=min(self.k, episode_length - current_idx - 1)
            )

            for idx in future_indices:
                future_state = self.episode_buffer[idx]["next_state"]
                goals.append(self._extract_goal(future_state))

        elif self.strategy == "episode":
            # Use random states from episode
            random_indices = np.random.randint(
                0,
                episode_length,
                size=self.k
            )

            for idx in random_indices:
                episode_state = self.episode_buffer[idx]["next_state"]
                goals.append(self._extract_goal(episode_state))

        elif self.strategy == "random":
            # Use random goals from buffer
            if len(self.buffer) > 0:
                random_samples = np.random.choice(
                    len(self.buffer),
                    size=min(self.k, len(self.buffer)),
                    replace=False
                )

                for idx in random_samples:
                    random_transition = list(self.buffer)[idx]
                    goals.append(random_transition["goal"].copy())

        # If no goals generated, use current goal
        if not goals:
            goals = [self.episode_buffer[current_idx]["goal"].copy()]

        return goals

    def _extract_goal(self, state: np.ndarray) -> np.ndarray:
        """Extract goal representation from state.

        Args:
            state: State array

        Returns:
            Goal array
        """
        # By default, use entire state as goal
        # Override this method for custom goal extraction
        return state.copy()

    def _check_goal_achieved(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        tolerance: Optional[Decimal] = None
    ) -> bool:
        """Check if goal is achieved.

        Args:
            state: Current state
            goal: Goal state
            tolerance: Achievement tolerance

        Returns:
            True if goal achieved
        """
        if tolerance is None:
            tolerance = Decimal(str(self.config.get("goal_tolerance", "0.01")))

        distance = Decimal(str(np.linalg.norm(state - goal)))
        return distance <= tolerance

    def _default_reward_fn(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        goal: np.ndarray
    ) -> Decimal:
        """Default reward function.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            goal: Goal state

        Returns:
            Reward value
        """
        # Simple distance-based reward
        distance = Decimal(str(np.linalg.norm(next_state - goal)))
        achieved = self._check_goal_achieved(next_state, goal)

        if achieved:
            return Decimal("1.0")
        else:
            # Negative distance as reward
            return -distance

    def sample(
        self,
        batch_size: int,
        prioritized: bool = False
    ) -> Dict[str, np.ndarray]:
        """Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            prioritized: Whether to use prioritized sampling

        Returns:
            Dictionary of batched arrays

        Raises:
            ValueError: If buffer is too small
        """
        try:
            if len(self.buffer) < batch_size:
                raise ValueError(
                    f"Buffer has {len(self.buffer)} samples, "
                    f"but batch_size is {batch_size}"
                )

            # Sample indices
            if prioritized:
                indices = self._prioritized_sample(batch_size)
            else:
                indices = np.random.choice(
                    len(self.buffer),
                    size=batch_size,
                    replace=False
                )

            # Gather samples
            batch = {
                "states": [],
                "actions": [],
                "rewards": [],
                "next_states": [],
                "goals": [],
                "dones": []
            }

            for idx in indices:
                transition = list(self.buffer)[idx]

                batch["states"].append(transition["state"])
                batch["actions"].append(transition["action"])
                batch["rewards"].append(float(transition["reward"]))
                batch["next_states"].append(transition["next_state"])
                batch["goals"].append(transition["goal"])
                batch["dones"].append(float(transition["done"]))

            # Convert to arrays
            for key in batch:
                batch[key] = np.array(batch[key])

            logger.debug("Sampled batch", batch_size=batch_size)

            return batch

        except Exception as e:
            logger.error("Failed to sample batch", error=str(e))
            raise

    def _prioritized_sample(self, batch_size: int) -> np.ndarray:
        """Sample with priority based on reward magnitude.

        Args:
            batch_size: Number of samples

        Returns:
            Sampled indices
        """
        # Calculate priorities based on absolute reward
        priorities = np.array([
            abs(float(t["reward"])) + 0.01  # Small epsilon
            for t in self.buffer
        ])

        # Normalize to probabilities
        probabilities = priorities / priorities.sum()

        # Sample
        indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
            p=probabilities
        )

        return indices

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.episode_buffer.clear()
        self.num_transitions = 0
        self.num_hindsight_samples = 0
        self.num_episodes = 0

        logger.info("Buffer cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "num_episodes": self.num_episodes,
            "num_transitions": self.num_transitions,
            "num_hindsight_samples": self.num_hindsight_samples,
            "hindsight_ratio": (
                float(self.num_hindsight_samples) / max(self.num_transitions, 1)
            )
        }

    def save(self, path: str) -> None:
        """Save buffer to disk.

        Args:
            path: File path to save buffer
        """
        try:
            # Convert buffer to list for serialization
            buffer_list = list(self.buffer)

            # Convert Decimal to float for serialization
            for transition in buffer_list:
                transition["reward"] = float(transition["reward"])

            np.savez_compressed(
                path,
                buffer=buffer_list,
                num_episodes=self.num_episodes,
                num_transitions=self.num_transitions,
                num_hindsight_samples=self.num_hindsight_samples
            )

            logger.info("Buffer saved", path=path)

        except Exception as e:
            logger.error("Failed to save buffer", error=str(e))
            raise

    def load(self, path: str) -> None:
        """Load buffer from disk.

        Args:
            path: File path to load buffer from
        """
        try:
            data = np.load(path, allow_pickle=True)

            self.buffer = deque(data["buffer"], maxlen=self.capacity)
            self.num_episodes = int(data["num_episodes"])
            self.num_transitions = int(data["num_transitions"])
            self.num_hindsight_samples = int(data["num_hindsight_samples"])

            # Convert rewards back to Decimal
            for transition in self.buffer:
                transition["reward"] = Decimal(str(transition["reward"]))

            logger.info("Buffer loaded", path=path, size=len(self.buffer))

        except Exception as e:
            logger.error("Failed to load buffer", error=str(e))
            raise
