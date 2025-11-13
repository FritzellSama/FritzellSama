"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) Agent.

Production implementation for multi-asset portfolio management using
cooperative multi-agent reinforcement learning.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for MADDPG agent."""
    state_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: Decimal
    gamma: Decimal
    tau: Decimal
    batch_size: int
    buffer_size: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    """Actor network for MADDPG."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        return self.network(state)


class Critic(nn.Module):
    """Critic network for MADDPG."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dim: int) -> None:
        super().__init__()
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents

        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for MADDPG."""

    def __init__(self, buffer_size: int, num_agents: int) -> None:
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.buffer: List[Tuple] = []
        self.position = 0

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """Add experience to buffer."""
        experience = (states, actions, rewards, next_states, dones)

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MADDPGAgent:
    """
    Multi-Agent DDPG for collaborative portfolio management.

    Each agent manages a different asset, cooperating to maximize
    overall portfolio performance.

    Attributes:
        config: Agent configuration from config files
        num_agents: Number of cooperative agents
        actors: List of actor networks
        critics: List of critic networks
        target_actors: List of target actor networks
        target_critics: List of target critic networks
        replay_buffer: Shared experience replay buffer

    Example:
        >>> config = load_config('config/ai/maddpg.yaml')
        >>> agent = MADDPGAgent(config, num_agents=5)
        >>> await agent.train(market_data)
    """

    def __init__(self, config: Dict[str, Any], num_agents: int) -> None:
        """
        Initialize MADDPG agent.

        Args:
            config: Configuration dictionary from YAML
            num_agents: Number of cooperative agents

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.num_agents = num_agents
        self._validate_config()

        # Extract config parameters
        agent_config = self._build_agent_config()
        self.device = agent_config.device

        # Initialize networks for each agent
        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_optimizers: List[optim.Adam] = []
        self.critic_optimizers: List[optim.Adam] = []

        for _ in range(num_agents):
            actor = Actor(
                agent_config.state_dim,
                agent_config.action_dim,
                agent_config.hidden_dim
            ).to(self.device)
            critic = Critic(
                agent_config.state_dim,
                agent_config.action_dim,
                num_agents,
                agent_config.hidden_dim
            ).to(self.device)

            target_actor = Actor(
                agent_config.state_dim,
                agent_config.action_dim,
                agent_config.hidden_dim
            ).to(self.device)
            target_critic = Critic(
                agent_config.state_dim,
                agent_config.action_dim,
                num_agents,
                agent_config.hidden_dim
            ).to(self.device)

            # Copy weights to target networks
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)

            # Initialize optimizers
            lr = float(agent_config.learning_rate)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr))

        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(
            agent_config.buffer_size,
            num_agents
        )

        self.gamma = float(agent_config.gamma)
        self.tau = float(agent_config.tau)
        self.batch_size = agent_config.batch_size

        logger.info(
            "MADDPG agent initialized",
            num_agents=num_agents,
            device=self.device,
            state_dim=agent_config.state_dim,
            action_dim=agent_config.action_dim
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = [
            'state_dim', 'action_dim', 'hidden_dim',
            'learning_rate', 'gamma', 'tau', 'batch_size', 'buffer_size'
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _build_agent_config(self) -> AgentConfig:
        """Build agent configuration from config dict."""
        return AgentConfig(
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            hidden_dim=self.config['hidden_dim'],
            learning_rate=Decimal(str(self.config['learning_rate'])),
            gamma=Decimal(str(self.config['gamma'])),
            tau=Decimal(str(self.config['tau'])),
            batch_size=self.config['batch_size'],
            buffer_size=self.config['buffer_size']
        )

    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select actions for all agents.

        Args:
            states: States for all agents (num_agents, state_dim)
            add_noise: Whether to add exploration noise

        Returns:
            Actions for all agents (num_agents, action_dim)
        """
        actions = []

        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                action = actor(state).cpu().numpy()[0]

                if add_noise:
                    noise_scale = float(self.config.get('exploration_noise', 0.1))
                    noise = np.random.normal(0, noise_scale, size=action.shape)
                    action = np.clip(action + noise, -1, 1)

                actions.append(action)

        return np.array(actions)

    def update(self) -> Dict[str, Decimal]:
        """
        Update all agents using sampled experiences.

        Returns:
            Dictionary of training metrics

        Raises:
            RuntimeError: If update fails
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        try:
            # Sample from replay buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            total_actor_loss = Decimal('0')
            total_critic_loss = Decimal('0')

            # Update each agent
            for agent_idx in range(self.num_agents):
                # Update critic
                with torch.no_grad():
                    # Get next actions from target actors
                    next_actions = []
                    for i, target_actor in enumerate(self.target_actors):
                        next_action = target_actor(next_states[:, i])
                        next_actions.append(next_action)
                    next_actions = torch.cat(next_actions, dim=-1)

                    # Flatten states for critic
                    flat_next_states = next_states.reshape(self.batch_size, -1)

                    # Compute target Q-value
                    target_q = self.target_critics[agent_idx](flat_next_states, next_actions)
                    target_q = rewards[:, agent_idx].unsqueeze(-1) + \
                               self.gamma * target_q * (1 - dones[:, agent_idx].unsqueeze(-1))

                # Current Q-value
                flat_states = states.reshape(self.batch_size, -1)
                flat_actions = actions.reshape(self.batch_size, -1)
                current_q = self.critics[agent_idx](flat_states, flat_actions)

                # Critic loss
                critic_loss = nn.MSELoss()(current_q, target_q)

                # Update critic
                self.critic_optimizers[agent_idx].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
                self.critic_optimizers[agent_idx].step()

                # Update actor
                # Get actions from current policy
                policy_actions = []
                for i, actor in enumerate(self.actors):
                    if i == agent_idx:
                        policy_action = actor(states[:, i])
                    else:
                        with torch.no_grad():
                            policy_action = actor(states[:, i])
                    policy_actions.append(policy_action)
                policy_actions = torch.cat(policy_actions, dim=-1)

                # Actor loss (maximize Q-value)
                actor_loss = -self.critics[agent_idx](flat_states, policy_actions).mean()

                # Update actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
                self.actor_optimizers[agent_idx].step()

                total_actor_loss += Decimal(str(actor_loss.item()))
                total_critic_loss += Decimal(str(critic_loss.item()))

            # Soft update target networks
            self._soft_update_targets()

            return {
                'actor_loss': total_actor_loss / self.num_agents,
                'critic_loss': total_critic_loss / self.num_agents,
                'buffer_size': Decimal(str(len(self.replay_buffer)))
            }

        except Exception as e:
            logger.error("MADDPG update failed", error=str(e))
            raise RuntimeError(f"Update failed: {e}")

    def _soft_update_targets(self) -> None:
        """Soft update target networks."""
        tau = self.tau

        for i in range(self.num_agents):
            # Update target actor
            for target_param, param in zip(
                self.target_actors[i].parameters(),
                self.actors[i].parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            # Update target critic
            for target_param, param in zip(
                self.target_critics[i].parameters(),
                self.critics[i].parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

    def save(self, path: str) -> None:
        """
        Save agent models.

        Args:
            path: Directory to save models
        """
        os.makedirs(path, exist_ok=True)

        try:
            for i in range(self.num_agents):
                torch.save(
                    self.actors[i].state_dict(),
                    os.path.join(path, f'actor_{i}.pt')
                )
                torch.save(
                    self.critics[i].state_dict(),
                    os.path.join(path, f'critic_{i}.pt')
                )

            logger.info("MADDPG models saved", path=path)

        except Exception as e:
            logger.error("Failed to save models", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """
        Load agent models.

        Args:
            path: Directory containing saved models

        Raises:
            FileNotFoundError: If models not found
        """
        try:
            for i in range(self.num_agents):
                actor_path = os.path.join(path, f'actor_{i}.pt')
                critic_path = os.path.join(path, f'critic_{i}.pt')

                self.actors[i].load_state_dict(torch.load(actor_path, map_location=self.device))
                self.critics[i].load_state_dict(torch.load(critic_path, map_location=self.device))

                # Update target networks
                self.target_actors[i].load_state_dict(self.actors[i].state_dict())
                self.target_critics[i].load_state_dict(self.critics[i].state_dict())

            logger.info("MADDPG models loaded", path=path)

        except FileNotFoundError as e:
            logger.error("Model files not found", error=str(e), path=path)
            raise
