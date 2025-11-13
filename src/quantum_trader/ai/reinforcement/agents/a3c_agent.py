"""
A3C (Asynchronous Advantage Actor-Critic) Agent Implementation.

Production-ready A3C agent for distributed reinforcement learning in trading.
Implements asynchronous training with multiple worker processes.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


@dataclass
class A3CConfig:
    """Configuration for A3C agent."""

    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: Decimal = Decimal("0.0001")
    gamma: Decimal = Decimal("0.99")
    entropy_coef: Decimal = Decimal("0.01")
    value_loss_coef: Decimal = Decimal("0.5")
    max_grad_norm: Decimal = Decimal("0.5")
    num_workers: int = 4
    t_max: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Experience:
    """Single experience tuple."""

    state: torch.Tensor
    action: int
    reward: Decimal
    next_state: torch.Tensor
    done: bool
    log_prob: torch.Tensor
    value: torch.Tensor
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ActorCriticNetwork(nn.Module):
    """Actor-Critic neural network for A3C."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action_probs, value)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor output (action probabilities)
        action_logits = self.actor(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic output (state value)
        value = self.critic(x)

        return action_probs, value


class A3CAgent:
    """A3C Agent for asynchronous reinforcement learning."""

    def __init__(self, config: A3CConfig):
        """
        Initialize A3C agent.

        Args:
            config: A3C configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Global network (shared across workers)
        self.global_network = ActorCriticNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)

        # Global optimizer
        self.global_optimizer = optim.Adam(
            self.global_network.parameters(),
            lr=float(config.learning_rate)
        )

        # Lock for thread-safe updates
        self.lock = asyncio.Lock()

        # Training metrics
        self.global_step = 0
        self.episode_rewards: List[Decimal] = []

        logger.info(
            "Initialized A3C agent",
            extra={
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "hidden_dim": config.hidden_dim,
                "num_workers": config.num_workers,
                "device": str(self.device),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def select_action(
        self,
        state: torch.Tensor,
        network: ActorCriticNetwork
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using the policy network.

        Args:
            state: Current state tensor
            network: Actor-Critic network to use

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            action_probs, value = network(state)

        # Sample action from distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value

    def compute_returns(
        self,
        rewards: List[Decimal],
        values: List[torch.Tensor],
        next_value: torch.Tensor,
        dones: List[bool]
    ) -> torch.Tensor:
        """
        Compute discounted returns using TD(λ).

        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value of next state
            dones: List of done flags

        Returns:
            Tensor of computed returns
        """
        returns = []
        R = next_value
        gamma = float(self.config.gamma)

        for i in reversed(range(len(rewards))):
            if dones[i]:
                R = torch.zeros(1, device=self.device)
            else:
                R = float(rewards[i]) + gamma * R
            returns.insert(0, R)

        return torch.stack(returns).squeeze()

    def compute_advantages(
        self,
        returns: torch.Tensor,
        values: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).

        Args:
            returns: Computed returns
            values: Value estimates

        Returns:
            Tensor of advantages
        """
        values_tensor = torch.stack(values).squeeze()
        advantages = returns - values_tensor

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    async def update_global_network(
        self,
        experiences: List[Experience],
        local_network: ActorCriticNetwork,
        next_state: torch.Tensor,
        done: bool
    ) -> Dict[str, float]:
        """
        Update global network using accumulated experiences.

        Args:
            experiences: List of experiences
            local_network: Local worker network
            next_state: Next state after experiences
            done: Whether episode is done

        Returns:
            Dictionary of training metrics
        """
        if not experiences:
            return {}

        # Extract components from experiences
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], device=self.device)
        rewards = [exp.reward for exp in experiences]
        dones = [exp.done for exp in experiences]
        old_log_probs = torch.stack([exp.log_prob for exp in experiences])
        old_values = [exp.value for exp in experiences]

        # Compute next state value
        with torch.no_grad():
            if done:
                next_value = torch.zeros(1, device=self.device)
            else:
                _, next_value = local_network(next_state.unsqueeze(0))

        # Compute returns and advantages
        returns = self.compute_returns(rewards, old_values, next_value, dones)
        advantages = self.compute_advantages(returns, old_values)

        # Forward pass through local network
        action_probs, values = local_network(states)

        # Compute policy loss
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(log_probs * advantages.detach()).mean()

        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns.detach())

        # Total loss
        entropy_coef = float(self.config.entropy_coef)
        value_loss_coef = float(self.config.value_loss_coef)

        total_loss = (
            policy_loss +
            value_loss_coef * value_loss -
            entropy_coef * entropy
        )

        # Update global network (thread-safe)
        async with self.lock:
            self.global_optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients
            max_grad_norm = float(self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(
                self.global_network.parameters(),
                max_grad_norm
            )

            self.global_optimizer.step()
            self.global_step += 1

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
            "mean_return": returns.mean().item(),
            "mean_advantage": advantages.mean().item()
        }

        logger.info(
            "Updated global network",
            extra={
                "global_step": self.global_step,
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return metrics

    async def worker_process(
        self,
        worker_id: int,
        env: Any,
        max_episodes: int
    ) -> None:
        """
        Worker process for asynchronous training.

        Args:
            worker_id: Unique worker identifier
            env: Training environment
            max_episodes: Maximum episodes to run
        """
        # Create local network (copy of global)
        local_network = ActorCriticNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        logger.info(
            "Starting worker process",
            extra={
                "worker_id": worker_id,
                "max_episodes": max_episodes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        for episode in range(max_episodes):
            # Sync with global network
            local_network.load_state_dict(self.global_network.state_dict())

            # Reset environment
            state = env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            experiences: List[Experience] = []
            episode_reward = Decimal("0")
            done = False
            step = 0

            while not done and step < self.config.t_max:
                # Select action
                action, log_prob, value = self.select_action(
                    state_tensor.unsqueeze(0),
                    local_network
                )

                # Execute action
                next_state, reward, done, _ = env.step(action)
                reward = Decimal(str(reward))

                next_state_tensor = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=self.device
                )

                # Store experience
                experience = Experience(
                    state=state_tensor,
                    action=action,
                    reward=reward,
                    next_state=next_state_tensor,
                    done=done,
                    log_prob=log_prob,
                    value=value
                )
                experiences.append(experience)

                episode_reward += reward
                state_tensor = next_state_tensor
                step += 1

            # Update global network
            if experiences:
                await self.update_global_network(
                    experiences,
                    local_network,
                    state_tensor,
                    done
                )

            # Track episode reward
            async with self.lock:
                self.episode_rewards.append(episode_reward)

            if episode % 10 == 0:
                logger.info(
                    "Worker episode completed",
                    extra={
                        "worker_id": worker_id,
                        "episode": episode,
                        "reward": str(episode_reward),
                        "steps": step,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                )

    async def train(
        self,
        envs: List[Any],
        max_episodes: int
    ) -> pl.DataFrame:
        """
        Train A3C agent with multiple workers.

        Args:
            envs: List of training environments (one per worker)
            max_episodes: Maximum episodes per worker

        Returns:
            DataFrame with training metrics
        """
        logger.info(
            "Starting A3C training",
            extra={
                "num_workers": len(envs),
                "max_episodes": max_episodes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        # Create worker tasks
        tasks = [
            self.worker_process(i, env, max_episodes)
            for i, env in enumerate(envs)
        ]

        # Run all workers concurrently
        await asyncio.gather(*tasks)

        # Create metrics DataFrame
        metrics_data = {
            "episode": list(range(len(self.episode_rewards))),
            "reward": [str(r) for r in self.episode_rewards],
            "timestamp": [datetime.now(timezone.utc).isoformat()] * len(self.episode_rewards)
        }

        df = pl.DataFrame(metrics_data)

        logger.info(
            "A3C training completed",
            extra={
                "total_episodes": len(self.episode_rewards),
                "mean_reward": str(sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else Decimal("0")),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return df

    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "global_network": self.global_network.state_dict(),
            "optimizer": self.global_optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        torch.save(checkpoint, path)

        logger.info(
            "Saved A3C agent",
            extra={
                "path": path,
                "global_step": self.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.global_network.load_state_dict(checkpoint["global_network"])
        self.global_optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]

        logger.info(
            "Loaded A3C agent",
            extra={
                "path": path,
                "global_step": self.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
