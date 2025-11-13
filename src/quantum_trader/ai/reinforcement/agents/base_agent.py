"""
Base Agent for Reinforcement Learning.

Provides abstract base class and common functionality for all RL agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Base configuration for RL agents."""

    state_dim: int
    action_dim: int
    learning_rate: Decimal = Decimal("0.0001")
    gamma: Decimal = Decimal("0.99")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 100


@dataclass
class AgentState:
    """State of an RL agent."""

    episode: int = 0
    total_steps: int = 0
    total_reward: Decimal = Decimal("0")
    best_reward: Decimal = Decimal("-inf")
    episode_rewards: List[Decimal] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.state = AgentState()

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized base agent",
            extra={
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "device": str(self.device),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    @abstractmethod
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """
        Select action given current state.

        Args:
            state: Current state tensor
            training: Whether in training mode

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    async def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: Decimal,
        next_state: torch.Tensor,
        done: bool
    ) -> Dict[str, Decimal]:
        """
        Update agent with new experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done

        Returns:
            Update metrics
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        pass

    async def train_episode(
        self,
        env: Any,
        max_steps: Optional[int] = None
    ) -> Tuple[Decimal, int]:
        """
        Train for one episode.

        Args:
            env: Training environment
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (episode reward, num steps)
        """
        state = env.reset()
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        )

        episode_reward = Decimal("0")
        step = 0

        while True:
            # Select action
            action = self.select_action(state_tensor, training=True)

            # Execute action
            next_state, reward, done, _ = env.step(action)
            reward = Decimal(str(reward))

            next_state_tensor = torch.tensor(
                next_state,
                dtype=torch.float32,
                device=self.device
            )

            # Update agent
            await self.update(state_tensor, action, reward, next_state_tensor, done)

            episode_reward += reward
            state_tensor = next_state_tensor
            step += 1

            if done or (max_steps and step >= max_steps):
                break

        # Update state
        self.state.episode += 1
        self.state.total_steps += step
        self.state.total_reward += episode_reward
        self.state.episode_rewards.append(episode_reward)

        if episode_reward > self.state.best_reward:
            self.state.best_reward = episode_reward

        # Log progress
        if self.state.episode % self.config.log_interval == 0:
            avg_reward = (
                sum(self.state.episode_rewards[-100:]) /
                min(100, len(self.state.episode_rewards))
            )

            logger.info(
                "Training progress",
                extra={
                    "episode": self.state.episode,
                    "episode_reward": str(episode_reward),
                    "avg_reward": str(avg_reward),
                    "total_steps": self.state.total_steps,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        return episode_reward, step

    async def evaluate(
        self,
        env: Any,
        num_episodes: int = 10,
        max_steps: Optional[int] = None
    ) -> Dict[str, Decimal]:
        """
        Evaluate agent performance.

        Args:
            env: Evaluation environment
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state = env.reset()
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device
            )

            episode_reward = Decimal("0")
            step = 0

            while True:
                # Select action (no exploration)
                action = self.select_action(state_tensor, training=False)

                # Execute action
                next_state, reward, done, _ = env.step(action)
                reward = Decimal(str(reward))

                next_state_tensor = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=self.device
                )

                episode_reward += reward
                state_tensor = next_state_tensor
                step += 1

                if done or (max_steps and step >= max_steps):
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)

        # Compute metrics
        metrics = {
            "mean_reward": sum(episode_rewards) / len(episode_rewards),
            "std_reward": Decimal(str(torch.tensor([float(r) for r in episode_rewards]).std().item())),
            "max_reward": max(episode_rewards),
            "min_reward": min(episode_rewards),
            "mean_length": Decimal(str(sum(episode_lengths) / len(episode_lengths)))
        }

        logger.info(
            "Evaluation completed",
            extra={
                "num_episodes": num_episodes,
                "mean_reward": str(metrics["mean_reward"]),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    def get_metrics(self) -> pl.DataFrame:
        """
        Get agent metrics as DataFrame.

        Returns:
            DataFrame with training metrics
        """
        data = {
            "episode": list(range(len(self.state.episode_rewards))),
            "reward": [str(r) for r in self.state.episode_rewards],
            "timestamp": [datetime.now(timezone.utc).isoformat()] * len(self.state.episode_rewards)
        }

        return pl.DataFrame(data)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.

        Returns:
            Dictionary with agent state
        """
        return {
            "episode": self.state.episode,
            "total_steps": self.state.total_steps,
            "total_reward": str(self.state.total_reward),
            "best_reward": str(self.state.best_reward),
            "num_episodes": len(self.state.episode_rewards),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.state = AgentState()

        logger.info(
            "Agent state reset",
            extra={"timestamp": datetime.now(timezone.utc).isoformat()}
        )


class DQNAgent(BaseAgent):
    """Deep Q-Network agent implementation."""

    def __init__(
        self,
        config: AgentConfig,
        network: nn.Module,
        target_network: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: Decimal = Decimal("1.0"),
        epsilon_decay: Decimal = Decimal("0.995"),
        epsilon_min: Decimal = Decimal("0.01"),
        target_update_freq: int = 100
    ):
        """
        Initialize DQN agent.

        Args:
            config: Agent configuration
            network: Q-network
            target_network: Target Q-network
            optimizer: Optimizer
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            target_update_freq: Target network update frequency
        """
        super().__init__(config)

        self.network = network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optimizer
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        self.update_count = 0

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and torch.rand(1).item() < float(self.epsilon):
            # Random action
            return torch.randint(0, self.config.action_dim, (1,)).item()
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.network(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()

    async def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: Decimal,
        next_state: torch.Tensor,
        done: bool
    ) -> Dict[str, Decimal]:
        """Update Q-network."""
        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.target_network(next_state.unsqueeze(0))
            max_next_q = next_q_values.max(dim=1)[0]

            if done:
                target_q = float(reward)
            else:
                target_q = float(reward) + float(self.config.gamma) * max_next_q.item()

        # Compute current Q-value
        current_q = self.network(state.unsqueeze(0))[0, action]

        # Compute loss
        loss = nn.functional.mse_loss(current_q, torch.tensor(target_q, device=self.device))

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        # Decay epsilon
        if training:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

        return {
            "loss": Decimal(str(loss.item())),
            "epsilon": self.epsilon,
            "q_value": Decimal(str(current_q.item()))
        }

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "network": self.network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "agent_state": self.state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        torch.save(checkpoint, path)

        logger.info(
            "Saved agent checkpoint",
            extra={
                "path": path,
                "episode": self.state.episode,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.state = checkpoint["agent_state"]

        logger.info(
            "Loaded agent checkpoint",
            extra={
                "path": path,
                "episode": self.state.episode,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
