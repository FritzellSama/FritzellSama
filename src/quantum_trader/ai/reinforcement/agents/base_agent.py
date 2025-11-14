"""Base reinforcement learning agent for trading.

This module provides the abstract base class for all reinforcement learning agents
in the trading system. RL agents learn optimal trading policies through interaction
with market environments and reward signals.

Key capabilities:
- Policy learning and optimization
- Experience replay and sampling
- Action selection (exploration vs exploitation)
- Reward processing and discounting
- Model saving and loading

The base agent provides a standard interface for:
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A3C (Asynchronous Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- Custom trading agents

Example:
    ```python
    from quantum_trader.ai.reinforcement.agents.base_agent import BaseRLAgent

    class MyAgent(BaseRLAgent):
        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            # Action selection logic
            pass

        def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
            # Learning update logic
            pass
    ```
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from structlog import get_logger

from quantum_trader.ai.models.base_model import BaseMLModel

logger = get_logger(__name__)


class ReplayBuffer:
    """Experience replay buffer for RL agents.

    Stores and samples transitions (state, action, reward, next_state, done)
    for off-policy learning algorithms.

    Attributes:
        capacity: Maximum buffer size
        buffer: Deque storing transitions
        position: Current position in buffer
    """

    def __init__(self, capacity: int) -> None:
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.position = 0

        logger.debug("Replay buffer initialized", capacity=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add transition to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batched transitions

        Raises:
            ValueError: If batch_size exceeds buffer size
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Batch size {batch_size} exceeds buffer size {len(self.buffer)}")

        # Random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'dones': np.array(dones),
        }

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.position = 0
        logger.debug("Replay buffer cleared")


class BaseRLAgent(BaseMLModel, ABC):
    """Abstract base class for reinforcement learning agents.

    Provides common functionality for RL agents including:
    - Experience replay
    - Epsilon-greedy exploration
    - Reward processing
    - Training loops
    - Model persistence

    Attributes:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        gamma: Discount factor for future rewards
        epsilon: Exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration decay rate
        replay_buffer: Experience replay buffer
        device: Computation device (CPU/CUDA)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize RL agent.

        Args:
            config: Configuration dictionary with keys:
                - state_dim: State space dimension (required)
                - action_dim: Action space dimension (required)
                - gamma: Discount factor (default: 0.99)
                - epsilon: Initial exploration rate (default: 1.0)
                - epsilon_min: Minimum exploration rate (default: 0.01)
                - epsilon_decay: Exploration decay rate (default: 0.995)
                - buffer_size: Replay buffer capacity (default: 100000)
                - batch_size: Training batch size (default: 64)
                - learning_rate: Learning rate (default: 0.001)
                - device: Device ('cuda' or 'cpu', default: auto)
                - target_update_freq: Frequency of target network updates (default: 1000)
        """
        super().__init__(config)

        # Environment parameters
        self.state_dim = int(config.get("state_dim"))
        self.action_dim = int(config.get("action_dim"))

        if not self.state_dim or not self.action_dim:
            raise ValueError("state_dim and action_dim are required")

        # Learning parameters
        self.gamma = float(config.get("gamma", 0.99))
        self.epsilon = float(config.get("epsilon", 1.0))
        self.epsilon_min = float(config.get("epsilon_min", 0.01))
        self.epsilon_decay = float(config.get("epsilon_decay", 0.995))

        # Training parameters
        self.batch_size = int(config.get("batch_size", 64))
        self.learning_rate = float(config.get("learning_rate", 0.001))
        self.target_update_freq = int(config.get("target_update_freq", 1000))

        # Replay buffer
        buffer_size = int(config.get("buffer_size", 100000))
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Device configuration
        device_config = config.get("device", "auto")
        if device_config == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_config)

        # Training statistics
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []

        logger.info(
            "RL agent initialized",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=str(self.device)
        )

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """Select action given current state.

        Args:
            state: Current state observation
            training: Whether in training mode (enables exploration)

        Returns:
            Selected action index

        Note:
            In training mode, should use epsilon-greedy or other exploration strategy.
            In evaluation mode, should use greedy policy.
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update agent parameters using batch of transitions.

        Args:
            batch: Dictionary containing:
                - states: Batch of states
                - actions: Batch of actions
                - rewards: Batch of rewards
                - next_states: Batch of next states
                - dones: Batch of done flags

        Returns:
            Dictionary of training metrics (e.g., loss, q_values)

        Note:
            This method implements the core learning algorithm (DQN, PPO, etc.)
        """
        pass

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def process_reward(self, reward: float) -> float:
        """Process and normalize reward.

        Args:
            reward: Raw reward value

        Returns:
            Processed reward

        Note:
            Can be overridden for custom reward shaping
        """
        # Default: clip rewards to [-1, 1]
        return np.clip(reward, -1.0, 1.0)

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train agent on transitions.

        For RL agents, this typically means updating policy/value networks
        using experience replay.

        Args:
            features: Not used directly (RL uses environment interaction)
            labels: Not used directly (RL uses rewards)

        Note:
            This is a simplified interface. Real training happens through
            environment interaction via train_episode().
        """
        logger.info("Training RL agent")

        # Validate buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            logger.warning(
                "Not enough samples in buffer",
                buffer_size=len(self.replay_buffer),
                batch_size=self.batch_size
            )
            return

        # Perform multiple update steps
        num_updates = int(config.get("num_updates", 100))

        for i in range(num_updates):
            # Sample batch
            batch = self.replay_buffer.sample(self.batch_size)

            # Update agent
            metrics = self.update(batch)
            self.losses.append(metrics.get('loss', 0.0))

            # Decay epsilon
            self.decay_epsilon()

            if (i + 1) % 10 == 0:
                logger.debug(
                    "Training progress",
                    update=i + 1,
                    loss=f"{metrics.get('loss', 0.0):.6f}",
                    epsilon=f"{self.epsilon:.4f}"
                )

        self.is_trained = True
        self.last_trained = datetime.utcnow()

        logger.info("Training completed", num_updates=num_updates)

    def train_episode(
        self,
        env: Any,
        max_steps: Optional[int] = None
    ) -> Tuple[float, int]:
        """Train agent for one episode in environment.

        Args:
            env: Trading environment (must have reset(), step() methods)
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (total_reward, episode_length)

        Raises:
            RuntimeError: If training fails
        """
        try:
            # Reset environment
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0

            done = False
            steps = 0

            while not done:
                # Select action
                action = self.select_action(state, training=True)

                # Take action in environment
                next_state, reward, done, info = env.step(action)

                # Process reward
                processed_reward = self.process_reward(reward)

                # Store transition
                self.store_transition(
                    state,
                    action,
                    processed_reward,
                    next_state,
                    done
                )

                # Update agent if buffer is ready
                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    metrics = self.update(batch)
                    self.losses.append(metrics.get('loss', 0.0))

                # Update state
                state = next_state
                episode_reward += reward
                episode_length += 1
                steps += 1

                # Check max steps
                if max_steps and steps >= max_steps:
                    break

            # Decay epsilon
            self.decay_epsilon()

            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            logger.info(
                "Episode completed",
                reward=f"{episode_reward:.2f}",
                length=episode_length,
                epsilon=f"{self.epsilon:.4f}"
            )

            return episode_reward, episode_length

        except Exception as e:
            logger.error("Training episode failed", error=str(e))
            raise RuntimeError(f"Training episode failed: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict actions for given states.

        Args:
            features: States array of shape (n_samples, state_dim)

        Returns:
            Actions array of shape (n_samples,)

        Raises:
            RuntimeError: If agent is not trained
        """
        if not self.is_trained:
            logger.warning("Agent not trained, using random policy")

        try:
            actions = []

            for state in features:
                action = self.select_action(state, training=False)
                actions.append(action)

            return np.array(actions)

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {e}")

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate agent performance.

        Args:
            features: States
            labels: Optimal actions (for supervised evaluation)

        Returns:
            Dictionary of evaluation metrics

        Note:
            For RL agents, true evaluation happens in environment.
            This provides a supervised accuracy metric if optimal actions are known.
        """
        try:
            predictions = self.predict(features)

            # Flatten labels if needed
            if len(labels.shape) > 1:
                labels = labels.ravel()

            # Compute accuracy
            accuracy = np.mean(predictions == labels)

            metrics = {
                "accuracy": float(accuracy),
                "avg_episode_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0,
                "avg_episode_length": float(np.mean(self.episode_lengths[-100:])) if self.episode_lengths else 0.0,
                "total_episodes": len(self.episode_rewards),
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
            }

            logger.info("Evaluation completed", metrics=metrics)

            return metrics

        except Exception as e:
            logger.error("Evaluation failed", error=str(e))
            raise RuntimeError(f"Evaluation failed: {e}")

    def save(self, path: str) -> None:
        """Save agent to disk.

        Args:
            path: File path for saving

        Note:
            Subclasses should override to save network parameters
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save training statistics
            save_data = {
                "total_steps": self.total_steps,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "losses": self.losses,
                "epsilon": self.epsilon,
            }

            np.save(save_path, save_data, allow_pickle=True)

            # Save metadata
            self.save_metadata(str(save_path))

            logger.info("Agent saved successfully", path=str(save_path))

        except Exception as e:
            logger.error("Failed to save agent", error=str(e))
            raise IOError(f"Failed to save agent: {e}")

    def load(self, path: str) -> None:
        """Load agent from disk.

        Args:
            path: File path for loading

        Note:
            Subclasses should override to load network parameters
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise FileNotFoundError(f"Agent file not found: {load_path}")

            # Load training statistics
            save_data = np.load(load_path, allow_pickle=True).item()

            self.total_steps = save_data.get("total_steps", 0)
            self.episode_rewards = save_data.get("episode_rewards", [])
            self.episode_lengths = save_data.get("episode_lengths", [])
            self.losses = save_data.get("losses", [])
            self.epsilon = save_data.get("epsilon", self.epsilon_min)

            # Load metadata
            self.load_metadata(str(load_path))

            logger.info("Agent loaded successfully", path=str(load_path))

        except Exception as e:
            logger.error("Failed to load agent", error=str(e))
            raise IOError(f"Failed to load agent: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent training statistics.

        Returns:
            Dictionary of training statistics
        """
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else []
        recent_losses = self.losses[-100:] if self.losses else []

        return {
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "avg_reward_recent": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "avg_length_recent": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
            "avg_loss_recent": float(np.mean(recent_losses)) if recent_losses else 0.0,
            "best_reward": float(np.max(self.episode_rewards)) if self.episode_rewards else 0.0,
            "buffer_size": len(self.replay_buffer),
        }

    def reset(self) -> None:
        """Reset agent state for new training run."""
        self.replay_buffer.clear()
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon = self.config.get("epsilon", 1.0)
        self.is_trained = False

        logger.info("Agent reset")
