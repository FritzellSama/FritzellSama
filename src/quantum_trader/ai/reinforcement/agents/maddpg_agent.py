"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent.

This module implements the MADDPG algorithm for multi-asset trading scenarios
where multiple trading agents interact in the same market environment.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for MADDPG agent.

    Attributes:
        actor_lr: Learning rate for actor network
        critic_lr: Learning rate for critic network
        gamma: Discount factor for future rewards
        tau: Soft update parameter for target networks
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        hidden_dims: Hidden layer dimensions
        noise_scale: Scale of exploration noise
        noise_decay: Decay rate for exploration noise
        min_noise: Minimum noise scale
        update_frequency: Steps between network updates
        checkpoint_dir: Directory for saving checkpoints
    """

    actor_lr: Decimal
    critic_lr: Decimal
    gamma: Decimal
    tau: Decimal
    buffer_size: int
    batch_size: int
    hidden_dims: List[int]
    noise_scale: Decimal
    noise_decay: Decimal
    min_noise: Decimal
    update_frequency: int
    checkpoint_dir: str


class ActorNetwork(nn.Module):
    """Actor network for MADDPG agent.

    Maps state observations to continuous action values.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int],
        action_bound: float
    ) -> None:
        """Initialize actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            action_bound: Maximum absolute value for actions
        """
        super().__init__()

        self.action_bound = action_bound

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network.

        Args:
            state: State tensor

        Returns:
            Action tensor scaled by action_bound
        """
        return self.network(state) * self.action_bound


class CriticNetwork(nn.Module):
    """Critic network for MADDPG agent.

    Estimates Q-value for state-action pairs across all agents.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dims: List[int]
    ) -> None:
        """Initialize critic network.

        Args:
            state_dim: Dimension of state space per agent
            action_dim: Dimension of action space per agent
            num_agents: Number of agents in environment
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        # Critic sees all states and all actions
        input_dim = (state_dim + action_dim) * num_agents

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through critic network.

        Args:
            states: Combined states from all agents
            actions: Combined actions from all agents

        Returns:
            Q-value estimate
        """
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent.

    Implements MADDPG for coordinated trading across multiple assets/exchanges.
    Uses centralized training with decentralized execution.

    Example:
        >>> config = AgentConfig(
        ...     actor_lr=Decimal("0.0001"),
        ...     critic_lr=Decimal("0.001"),
        ...     gamma=Decimal("0.99"),
        ...     tau=Decimal("0.005"),
        ...     buffer_size=100000,
        ...     batch_size=64,
        ...     hidden_dims=[256, 128],
        ...     noise_scale=Decimal("0.1"),
        ...     noise_decay=Decimal("0.9995"),
        ...     min_noise=Decimal("0.01"),
        ...     update_frequency=1,
        ...     checkpoint_dir="./checkpoints"
        ... )
        >>> agent = MADDPGAgent(
        ...     agent_id=0,
        ...     num_agents=3,
        ...     state_dim=50,
        ...     action_dim=5,
        ...     config=config
        ... )
    """

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        config: AgentConfig,
        device: Optional[str] = None
    ) -> None:
        """Initialize MADDPG agent.

        Args:
            agent_id: Unique identifier for this agent
            num_agents: Total number of agents in environment
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Agent configuration
            device: Device for computation ('cuda' or 'cpu')
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Action bounds
        self.action_bound = 1.0

        # Actor networks
        self.actor = ActorNetwork(
            state_dim,
            action_dim,
            config.hidden_dims,
            self.action_bound
        ).to(self.device)

        self.actor_target = ActorNetwork(
            state_dim,
            action_dim,
            config.hidden_dims,
            self.action_bound
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = CriticNetwork(
            state_dim,
            action_dim,
            num_agents,
            config.hidden_dims
        ).to(self.device)

        self.critic_target = CriticNetwork(
            state_dim,
            action_dim,
            num_agents,
            config.hidden_dims
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=float(config.actor_lr)
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=float(config.critic_lr)
        )

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=config.buffer_size)

        # Exploration noise
        self.noise_scale = config.noise_scale

        # Training state
        self.steps = 0
        self.episodes = 0

        # Metrics
        self.metrics: Dict[str, List[float]] = {
            'actor_loss': [],
            'critic_loss': [],
            'q_values': [],
            'rewards': []
        }

        logger.info(
            "maddpg_agent_initialized",
            agent_id=agent_id,
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """Select action based on current policy.

        Args:
            state: Current state observation
            add_noise: Whether to add exploration noise

        Returns:
            Selected action
        """
        try:
            self.actor.eval()

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.actor(state_tensor).cpu().numpy()[0]

            if add_noise:
                noise = np.random.normal(
                    0,
                    float(self.noise_scale),
                    size=action.shape
                )
                action = np.clip(action + noise, -self.action_bound, self.action_bound)

            self.actor.train()

            return action

        except Exception as e:
            logger.error(
                "action_selection_failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            # Return zero action on error
            return np.zeros(self.action_dim)

    def store_transition(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool]
    ) -> None:
        """Store transition in replay buffer.

        Args:
            states: States for all agents
            actions: Actions for all agents
            rewards: Rewards for all agents
            next_states: Next states for all agents
            dones: Done flags for all agents
        """
        self.replay_buffer.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        })

        self.steps += 1

    def update(self, agent_actions: List[Any]) -> Dict[str, float]:
        """Update actor and critic networks.

        Args:
            agent_actions: List of other agents for coordinated learning

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        try:
            # Sample batch
            batch_indices = np.random.choice(
                len(self.replay_buffer),
                self.config.batch_size,
                replace=False
            )

            batch = [self.replay_buffer[i] for i in batch_indices]

            # Extract batch data
            states = torch.FloatTensor(
                [t['states'][self.agent_id] for t in batch]
            ).to(self.device)

            actions = torch.FloatTensor(
                [t['actions'][self.agent_id] for t in batch]
            ).to(self.device)

            rewards = torch.FloatTensor(
                [t['rewards'][self.agent_id] for t in batch]
            ).unsqueeze(1).to(self.device)

            next_states = torch.FloatTensor(
                [t['next_states'][self.agent_id] for t in batch]
            ).to(self.device)

            dones = torch.FloatTensor(
                [t['dones'][self.agent_id] for t in batch]
            ).unsqueeze(1).to(self.device)

            # Combined states and actions for critic
            all_states = torch.FloatTensor(
                np.array([
                    np.concatenate([t['states'][i] for i in range(self.num_agents)])
                    for t in batch
                ])
            ).to(self.device)

            all_actions = torch.FloatTensor(
                np.array([
                    np.concatenate([t['actions'][i] for i in range(self.num_agents)])
                    for t in batch
                ])
            ).to(self.device)

            all_next_states = torch.FloatTensor(
                np.array([
                    np.concatenate([t['next_states'][i] for i in range(self.num_agents)])
                    for t in batch
                ])
            ).to(self.device)

            # Update critic
            with torch.no_grad():
                next_actions = self.actor_target(next_states)

                # Build next actions for all agents
                all_next_actions_list = []
                for i in range(self.num_agents):
                    if i == self.agent_id:
                        all_next_actions_list.append(next_actions)
                    else:
                        agent_next_states = torch.FloatTensor(
                            [t['next_states'][i] for t in batch]
                        ).to(self.device)
                        all_next_actions_list.append(
                            agent_actions[i].actor_target(agent_next_states)
                        )

                all_next_actions = torch.cat(all_next_actions_list, dim=1)

                target_q = self.critic_target(all_next_states, all_next_actions)
                y = rewards + float(self.config.gamma) * target_q * (1 - dones)

            current_q = self.critic(all_states, all_actions)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # Update actor
            predicted_actions = self.actor(states)

            # Build actions for all agents
            all_actions_list = []
            for i in range(self.num_agents):
                if i == self.agent_id:
                    all_actions_list.append(predicted_actions)
                else:
                    agent_states = torch.FloatTensor(
                        [t['states'][i] for t in batch]
                    ).to(self.device)
                    all_actions_list.append(
                        agent_actions[i].actor(agent_states).detach()
                    )

            all_predicted_actions = torch.cat(all_actions_list, dim=1)

            actor_loss = -self.critic(all_states, all_predicted_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            # Decay noise
            self.noise_scale = max(
                self.config.min_noise,
                self.noise_scale * self.config.noise_decay
            )

            # Track metrics
            metrics = {
                'actor_loss': float(actor_loss.item()),
                'critic_loss': float(critic_loss.item()),
                'q_value': float(current_q.mean().item()),
                'noise_scale': float(self.noise_scale)
            }

            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)

            return metrics

        except Exception as e:
            logger.error(
                "update_failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            return {}

    def _soft_update(
        self,
        source: nn.Module,
        target: nn.Module
    ) -> None:
        """Soft update target network parameters.

        Args:
            source: Source network
            target: Target network
        """
        tau = float(self.config.tau)

        for target_param, source_param in zip(
            target.parameters(),
            source.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    async def save_checkpoint(self, episode: int) -> None:
        """Save agent checkpoint asynchronously.

        Args:
            episode: Current episode number
        """
        try:
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"agent_{self.agent_id}_ep{episode}.pt"

            checkpoint = {
                'episode': episode,
                'steps': self.steps,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'noise_scale': float(self.noise_scale),
                'metrics': self.metrics
            }

            await asyncio.to_thread(
                torch.save,
                checkpoint,
                checkpoint_path
            )

            logger.info(
                "checkpoint_saved",
                agent_id=self.agent_id,
                episode=episode,
                path=str(checkpoint_path)
            )

        except Exception as e:
            logger.error(
                "checkpoint_save_failed",
                agent_id=self.agent_id,
                error=str(e)
            )

    async def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load agent checkpoint asynchronously.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = await asyncio.to_thread(
                torch.load,
                checkpoint_path,
                map_location=self.device
            )

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            self.steps = checkpoint['steps']
            self.episodes = checkpoint['episode']
            self.noise_scale = Decimal(str(checkpoint['noise_scale']))
            self.metrics = checkpoint['metrics']

            logger.info(
                "checkpoint_loaded",
                agent_id=self.agent_id,
                episode=self.episodes,
                path=checkpoint_path
            )

        except Exception as e:
            logger.error(
                "checkpoint_load_failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            raise

    def get_metrics_dataframe(self) -> pl.DataFrame:
        """Get training metrics as Polars DataFrame.

        Returns:
            DataFrame with training metrics
        """
        try:
            data = {
                'step': list(range(len(self.metrics['actor_loss']))),
                'actor_loss': self.metrics['actor_loss'],
                'critic_loss': self.metrics['critic_loss'],
                'q_values': self.metrics['q_values']
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error(
                "metrics_dataframe_creation_failed",
                agent_id=self.agent_id,
                error=str(e)
            )
            return pl.DataFrame()
