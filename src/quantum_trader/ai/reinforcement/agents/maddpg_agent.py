"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Agent.

This module implements MADDPG for multi-agent trading scenarios where
multiple trading strategies interact in the same market environment.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from collections import deque

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

from quantum_trader.exceptions import AgentError, ValidationError

logger = get_logger(__name__)


class Actor(nn.Module):
    """Actor network for MADDPG.

    Maps state to action with continuous action space.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_action: float
    ) -> None:
        """Initialize actor network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            max_action: Maximum action value
        """
        super(Actor, self).__init__()

        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: State tensor

        Returns:
            Action tensor
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc3(x)) * self.max_action

        return action


class Critic(nn.Module):
    """Critic network for MADDPG.

    Evaluates state-action pairs considering all agents' actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int
    ) -> None:
        """Initialize critic network.

        Args:
            state_dim: State space dimension per agent
            action_dim: Action space dimension per agent
            num_agents: Number of agents
            hidden_dim: Hidden layer dimension
        """
        super(Critic, self).__init__()

        total_input_dim = (state_dim + action_dim) * num_agents

        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            states: State tensor [batch, num_agents * state_dim]
            actions: Action tensor [batch, num_agents * action_dim]

        Returns:
            Q-value tensor
        """
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)

        return q_value


@dataclass
class ReplayBuffer:
    """Experience replay buffer for MADDPG.

    Attributes:
        capacity: Maximum buffer size
        batch_size: Batch size for sampling
        buffer: Experience storage
    """
    capacity: int
    batch_size: int
    buffer: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        """Initialize buffer with max length."""
        self.buffer = deque(maxlen=self.capacity)

    def push(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """Add experience to buffer.

        Args:
            states: State array [num_agents, state_dim]
            actions: Action array [num_agents, action_dim]
            rewards: Reward array [num_agents,]
            next_states: Next state array [num_agents, state_dim]
            dones: Done flags [num_agents,]
        """
        self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self) -> Tuple[torch.Tensor, ...]:
        """Sample batch from buffer.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        experiences = [self.buffer[i] for i in batch]

        states = torch.FloatTensor(np.array([e[0] for e in experiences]))
        actions = torch.FloatTensor(np.array([e[1] for e in experiences]))
        rewards = torch.FloatTensor(np.array([e[2] for e in experiences]))
        next_states = torch.FloatTensor(np.array([e[3] for e in experiences]))
        dones = torch.FloatTensor(np.array([e[4] for e in experiences]))

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)


class MADDPGAgent:
    """Multi-Agent DDPG for coordinated trading strategies.

    Implements MADDPG algorithm for multi-agent reinforcement learning
    in trading environments with continuous action spaces.

    Attributes:
        config: Agent configuration
        num_agents: Number of agents
        state_dim: State space dimension per agent
        action_dim: Action space dimension per agent
        actors: Actor networks for each agent
        critics: Critic networks for each agent
        target_actors: Target actor networks
        target_critics: Target critic networks

    Example:
        >>> config = {
        ...     "num_agents": 3,
        ...     "state_dim": 50,
        ...     "action_dim": 10,
        ...     "hidden_dim": 256,
        ...     "actor_lr": 0.0001,
        ...     "critic_lr": 0.001,
        ...     "gamma": 0.99,
        ...     "tau": 0.01,
        ...     "buffer_size": 1000000,
        ...     "batch_size": 256,
        ...     "max_action": 1.0
        ... }
        >>> agent = MADDPGAgent(config)
        >>> actions = agent.select_actions(states)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MADDPG agent.

        Args:
            config: Agent configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.num_agents = config["num_agents"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.actor_lr = Decimal(str(config.get("actor_lr", 0.0001)))
        self.critic_lr = Decimal(str(config.get("critic_lr", 0.001)))
        self.gamma = Decimal(str(config.get("gamma", 0.99)))
        self.tau = Decimal(str(config.get("tau", 0.01)))
        self.buffer_size = config.get("buffer_size", 1000000)
        self.batch_size = config.get("batch_size", 256)
        self.max_action = config.get("max_action", 1.0)
        self.noise_std = Decimal(str(config.get("noise_std", 0.1)))

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize networks
        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_optimizers: List[optim.Optimizer] = []
        self.critic_optimizers: List[optim.Optimizer] = []

        for _ in range(self.num_agents):
            # Actor
            actor = Actor(
                self.state_dim,
                self.action_dim,
                self.hidden_dim,
                self.max_action
            ).to(self.device)

            target_actor = Actor(
                self.state_dim,
                self.action_dim,
                self.hidden_dim,
                self.max_action
            ).to(self.device)
            target_actor.load_state_dict(actor.state_dict())

            # Critic
            critic = Critic(
                self.state_dim,
                self.action_dim,
                self.num_agents,
                self.hidden_dim
            ).to(self.device)

            target_critic = Critic(
                self.state_dim,
                self.action_dim,
                self.num_agents,
                self.hidden_dim
            ).to(self.device)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)

            # Optimizers
            self.actor_optimizers.append(
                optim.Adam(actor.parameters(), lr=float(self.actor_lr))
            )
            self.critic_optimizers.append(
                optim.Adam(critic.parameters(), lr=float(self.critic_lr))
            )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            batch_size=self.batch_size
        )

        self._training_step = 0

        logger.info(
            "MADDPG agent initialized",
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate agent configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        required_fields = ["num_agents", "state_dim", "action_dim"]
        for field in required_fields:
            if field not in self.config:
                raise ValidationError(f"Missing required config field: {field}")

        if self.config["num_agents"] < 1:
            raise ValidationError("num_agents must be >= 1")

        if self.config["state_dim"] < 1:
            raise ValidationError("state_dim must be >= 1")

        if self.config["action_dim"] < 1:
            raise ValidationError("action_dim must be >= 1")

    def select_actions(
        self,
        states: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """Select actions for all agents.

        Args:
            states: State array [num_agents, state_dim]
            add_noise: Whether to add exploration noise

        Returns:
            Action array [num_agents, action_dim]

        Raises:
            ValidationError: If states are invalid
        """
        try:
            self._validate_states(states)

            actions = []

            for i, actor in enumerate(self.actors):
                actor.eval()
                with torch.no_grad():
                    state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                    action = actor(state).cpu().numpy()[0]

                    if add_noise:
                        noise = np.random.normal(0, float(self.noise_std), size=action.shape)
                        action = np.clip(
                            action + noise,
                            -self.max_action,
                            self.max_action
                        )

                    actions.append(action)

            return np.array(actions)

        except Exception as e:
            logger.error("Action selection failed", error=str(e))
            raise AgentError(f"Action selection failed: {e}") from e

    def update(self) -> Dict[str, Decimal]:
        """Update all agents using MADDPG algorithm.

        Returns:
            Dictionary of training metrics

        Raises:
            AgentError: If update fails
        """
        try:
            if len(self.replay_buffer) < self.batch_size:
                return {}

            # Sample batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            metrics = {}

            # Update each agent
            for agent_idx in range(self.num_agents):
                # Flatten states and actions for critic
                states_flat = states.view(states.size(0), -1)
                actions_flat = actions.view(actions.size(0), -1)
                next_states_flat = next_states.view(next_states.size(0), -1)

                # Get next actions from target actors
                next_actions = []
                for i, target_actor in enumerate(self.target_actors):
                    next_action = target_actor(next_states[:, i, :])
                    next_actions.append(next_action)

                next_actions_flat = torch.cat(next_actions, dim=1)

                # Update critic
                target_q = self.target_critics[agent_idx](
                    next_states_flat,
                    next_actions_flat
                )

                target_value = rewards[:, agent_idx].unsqueeze(1) + \
                               float(self.gamma) * target_q * (1 - dones[:, agent_idx].unsqueeze(1))

                current_q = self.critics[agent_idx](states_flat, actions_flat)
                critic_loss = F.mse_loss(current_q, target_value.detach())

                self.critic_optimizers[agent_idx].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
                self.critic_optimizers[agent_idx].step()

                # Update actor
                current_actions = []
                for i, actor in enumerate(self.actors):
                    if i == agent_idx:
                        current_action = actor(states[:, i, :])
                    else:
                        current_action = actions[:, i, :]
                    current_actions.append(current_action)

                current_actions_flat = torch.cat(current_actions, dim=1)

                actor_loss = -self.critics[agent_idx](
                    states_flat,
                    current_actions_flat
                ).mean()

                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
                self.actor_optimizers[agent_idx].step()

                metrics[f"agent_{agent_idx}_critic_loss"] = Decimal(str(critic_loss.item()))
                metrics[f"agent_{agent_idx}_actor_loss"] = Decimal(str(actor_loss.item()))

            # Update target networks
            self._update_targets()

            self._training_step += 1

            if self._training_step % 100 == 0:
                logger.info(
                    "Training step completed",
                    step=self._training_step,
                    buffer_size=len(self.replay_buffer)
                )

            return metrics

        except Exception as e:
            logger.error("Update failed", error=str(e))
            raise AgentError(f"Update failed: {e}") from e

    def _update_targets(self) -> None:
        """Soft update target networks."""
        tau = float(self.tau)

        for agent_idx in range(self.num_agents):
            # Update target actor
            for target_param, param in zip(
                self.target_actors[agent_idx].parameters(),
                self.actors[agent_idx].parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

            # Update target critic
            for target_param, param in zip(
                self.target_critics[agent_idx].parameters(),
                self.critics[agent_idx].parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )

    def store_experience(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """Store experience in replay buffer.

        Args:
            states: State array [num_agents, state_dim]
            actions: Action array [num_agents, action_dim]
            rewards: Reward array [num_agents,]
            next_states: Next state array [num_agents, state_dim]
            dones: Done flags [num_agents,]
        """
        self.replay_buffer.push(states, actions, rewards, next_states, dones)

    def _validate_states(self, states: np.ndarray) -> None:
        """Validate state array.

        Args:
            states: State array to validate

        Raises:
            ValidationError: If states are invalid
        """
        if states is None or len(states) == 0:
            raise ValidationError("States cannot be empty")

        if states.shape[0] != self.num_agents:
            raise ValidationError(
                f"State count mismatch: expected {self.num_agents}, got {states.shape[0]}"
            )

        if states.shape[1] != self.state_dim:
            raise ValidationError(
                f"State dimension mismatch: expected {self.state_dim}, got {states.shape[1]}"
            )

    def save(self, path: str) -> None:
        """Save agent to disk.

        Args:
            path: Path to save agent

        Raises:
            AgentError: If save fails
        """
        try:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save each agent's networks
            for i in range(self.num_agents):
                torch.save({
                    'actor_state_dict': self.actors[i].state_dict(),
                    'critic_state_dict': self.critics[i].state_dict(),
                    'target_actor_state_dict': self.target_actors[i].state_dict(),
                    'target_critic_state_dict': self.target_critics[i].state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizers[i].state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizers[i].state_dict(),
                }, save_path / f"agent_{i}.pt")

            logger.info("Agent saved", path=path)

        except Exception as e:
            logger.error("Failed to save agent", error=str(e))
            raise AgentError(f"Failed to save agent: {e}") from e

    def load(self, path: str) -> None:
        """Load agent from disk.

        Args:
            path: Path to load agent from

        Raises:
            AgentError: If load fails
        """
        try:
            load_path = Path(path)

            if not load_path.exists():
                raise AgentError(f"Agent directory not found: {path}")

            # Load each agent's networks
            for i in range(self.num_agents):
                checkpoint_path = load_path / f"agent_{i}.pt"
                if not checkpoint_path.exists():
                    raise AgentError(f"Agent checkpoint not found: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                self.actors[i].load_state_dict(checkpoint['actor_state_dict'])
                self.critics[i].load_state_dict(checkpoint['critic_state_dict'])
                self.target_actors[i].load_state_dict(checkpoint['target_actor_state_dict'])
                self.target_critics[i].load_state_dict(checkpoint['target_critic_state_dict'])
                self.actor_optimizers[i].load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizers[i].load_state_dict(checkpoint['critic_optimizer_state_dict'])

            logger.info("Agent loaded", path=path)

        except Exception as e:
            logger.error("Failed to load agent", error=str(e))
            raise AgentError(f"Failed to load agent: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "num_agents": self.num_agents,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "training_step": self._training_step,
            "buffer_size": len(self.replay_buffer),
            "buffer_capacity": self.buffer_size
        }
