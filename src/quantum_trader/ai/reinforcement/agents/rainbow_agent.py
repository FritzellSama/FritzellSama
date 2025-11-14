"""Rainbow DQN agent for trading reinforcement learning.

This module implements the Rainbow DQN algorithm which combines multiple improvements
to DQN: Double Q-learning, Dueling networks, Prioritized replay, Multi-step learning,
Distributional RL, and Noisy networks.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import structlog

logger = structlog.get_logger(__name__)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration.

    Implements factorized Gaussian noise for parameter space exploration.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5
    ) -> None:
        """Initialize noisy linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            std_init: Initial standard deviation for noise
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        # Noise buffers
        self.register_buffer(
            'weight_epsilon',
            torch.FloatTensor(out_features, in_features)
        )
        self.register_buffer(
            'bias_epsilon',
            torch.FloatTensor(out_features)
        )

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / np.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / np.sqrt(self.out_features)
        )

    def reset_noise(self) -> None:
        """Reset noise for both weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy parameters."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """Rainbow DQN network with distributional, dueling, and noisy layers.

    Combines:
    - Dueling architecture (separate value and advantage streams)
    - Distributional RL (C51 - categorical distribution)
    - Noisy networks (for exploration)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_size: int,
        support: torch.Tensor,
        hidden_dims: List[int]
    ) -> None:
        """Initialize Rainbow network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            atom_size: Number of atoms in distribution
            support: Support for categorical distribution
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support = support

        # Feature extraction layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        self.feature_layer = nn.Sequential(*layers)

        # Dueling architecture with noisy layers
        feature_dim = hidden_dims[-2] if len(hidden_dims) > 1 else state_dim

        # Value stream
        self.value_hidden = NoisyLinear(feature_dim, hidden_dims[-1])
        self.value_out = NoisyLinear(hidden_dims[-1], atom_size)

        # Advantage stream
        self.advantage_hidden = NoisyLinear(feature_dim, hidden_dims[-1])
        self.advantage_out = NoisyLinear(hidden_dims[-1], action_dim * atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning distribution over atoms.

        Args:
            x: State tensor

        Returns:
            Distribution tensor of shape (batch, action_dim, atom_size)
        """
        batch_size = x.size(0)

        # Extract features
        features = self.feature_layer(x)

        # Value stream
        value = self.value_hidden(features)
        value = F.relu(value)
        value = self.value_out(value).view(batch_size, 1, self.atom_size)

        # Advantage stream
        advantage = self.advantage_hidden(features)
        advantage = F.relu(advantage)
        advantage = self.advantage_out(advantage).view(
            batch_size, self.action_dim, self.atom_size
        )

        # Combine value and advantage
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        # Apply softmax to get distributions
        dist = F.softmax(q_atoms, dim=2)

        return dist

    def reset_noise(self) -> None:
        """Reset noise in all noisy layers."""
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()


class RainbowAgent:
    """Rainbow DQN agent for trading.

    Implements Rainbow DQN with all improvements for trading decision making.

    Attributes:
        config: Configuration dictionary
        state_dim: State space dimension
        action_dim: Action space dimension
        device: PyTorch device

    Example:
        >>> config = {
        ...     'state_dim': 128,
        ...     'action_dim': 3,
        ...     'hidden_dims': [256, 256, 128],
        ...     'learning_rate': 0.0001,
        ...     'gamma': 0.99,
        ...     'atom_size': 51,
        ...     'v_min': -10.0,
        ...     'v_max': 10.0,
        ...     'n_step': 3,
        ...     'target_update_freq': 1000,
        ...     'device': 'cuda'
        ... }
        >>> agent = RainbowAgent(config)
        >>> action = agent.select_action(state)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Rainbow agent.

        Args:
            config: Configuration with keys:
                - state_dim: State dimension
                - action_dim: Action dimension
                - hidden_dims: Hidden layer dimensions
                - learning_rate: Learning rate
                - gamma: Discount factor
                - atom_size: Number of atoms
                - v_min: Minimum value
                - v_max: Maximum value
                - n_step: N-step returns
                - target_update_freq: Target network update frequency
                - device: Device (cpu/cuda)
        """
        self.config = config
        self._validate_config()

        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.hidden_dims = config['hidden_dims']
        self.gamma = Decimal(str(config['gamma']))
        self.atom_size = config['atom_size']
        self.v_min = Decimal(str(config['v_min']))
        self.v_max = Decimal(str(config['v_max']))
        self.n_step = config['n_step']
        self.target_update_freq = config['target_update_freq']

        # Device setup
        self.device = torch.device(config['device'])

        # Support for categorical distribution
        self.support = torch.linspace(
            float(self.v_min),
            float(self.v_max),
            self.atom_size
        ).to(self.device)
        self.delta_z = (float(self.v_max) - float(self.v_min)) / (self.atom_size - 1)

        # Networks
        self.online_net = RainbowNetwork(
            self.state_dim,
            self.action_dim,
            self.atom_size,
            self.support,
            self.hidden_dims
        ).to(self.device)

        self.target_net = RainbowNetwork(
            self.state_dim,
            self.action_dim,
            self.atom_size,
            self.support,
            self.hidden_dims
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config['learning_rate']
        )

        # Training state
        self.training_step = 0

        logger.info(
            "initialized_rainbow_agent",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            atom_size=self.atom_size,
            device=str(self.device)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'state_dim', 'action_dim', 'hidden_dims', 'learning_rate',
            'gamma', 'atom_size', 'v_min', 'v_max', 'n_step',
            'target_update_freq', 'device'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['state_dim'] <= 0:
            raise ValueError("state_dim must be positive")

        if self.config['action_dim'] <= 0:
            raise ValueError("action_dim must be positive")

        if not (0 < self.config['gamma'] <= 1):
            raise ValueError("gamma must be in (0, 1]")

    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy.

        Args:
            state: Current state

        Returns:
            Selected action index
        """
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get distribution
                dist = self.online_net(state_tensor)

                # Calculate Q-values from distribution
                q_values = (dist * self.support).sum(dim=2)

                # Select action with highest Q-value
                action = q_values.argmax(dim=1).item()

            logger.debug("selected_action", action=action)

            return action

        except Exception as e:
            logger.error("failed_to_select_action", error=str(e))
            raise

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: List[Decimal],
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[Decimal, np.ndarray]:
        """Perform single training step.

        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards (Decimal)
            next_states: Batch of next states
            dones: Batch of done flags
            weights: Importance sampling weights

        Returns:
            Tuple of (loss, td_errors)
        """
        try:
            self.online_net.train()

            # Convert to tensors
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)

            # Convert rewards to tensor
            rewards_tensor = torch.FloatTensor(
                [float(r) for r in rewards]
            ).to(self.device)

            # Get current distribution
            dist = self.online_net(states)
            actions_expanded = actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.atom_size)
            dist = dist.gather(1, actions_expanded).squeeze(1)

            # Get next distribution using double Q-learning
            with torch.no_grad():
                # Online network selects action
                next_dist = self.online_net(next_states)
                next_q_values = (next_dist * self.support).sum(dim=2)
                next_actions = next_q_values.argmax(dim=1)

                # Target network evaluates action
                next_dist_target = self.target_net(next_states)
                next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(1).expand(
                    -1, -1, self.atom_size
                )
                next_dist = next_dist_target.gather(1, next_actions_expanded).squeeze(1)

                # Compute projected distribution
                gamma_tensor = torch.FloatTensor([float(self.gamma ** self.n_step)]).to(self.device)
                projected_support = rewards_tensor.unsqueeze(1) + \
                                  gamma_tensor * self.support.unsqueeze(0) * \
                                  (1 - dones.float().unsqueeze(1))

                # Clip to support range
                projected_support = projected_support.clamp(
                    float(self.v_min),
                    float(self.v_max)
                )

                # Compute categorical projection
                b = (projected_support - float(self.v_min)) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                # Distribute probability
                m = torch.zeros_like(next_dist)
                offset = torch.linspace(
                    0, (next_dist.size(0) - 1) * self.atom_size,
                    next_dist.size(0)
                ).long().unsqueeze(1).expand_as(next_dist).to(self.device)

                m.view(-1).index_add_(
                    0,
                    (l + offset).view(-1),
                    (next_dist * (u.float() - b)).view(-1)
                )
                m.view(-1).index_add_(
                    0,
                    (u + offset).view(-1),
                    (next_dist * (b - l.float())).view(-1)
                )

            # Compute loss
            loss = -(m * dist.clamp(min=1e-8, max=1.0).log()).sum(dim=1)

            # Apply importance sampling weights
            weighted_loss = (weights * loss).mean()

            # Optimize
            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
            self.optimizer.step()

            # Reset noise
            self.online_net.reset_noise()
            self.target_net.reset_noise()

            # Update target network
            self.training_step += 1
            if self.training_step % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                logger.info("updated_target_network", step=self.training_step)

            # Calculate TD errors for priority update
            with torch.no_grad():
                q_values = (dist * self.support).sum(dim=1)
                next_q_values_target = (m * self.support).sum(dim=1)
                td_errors = (rewards_tensor + float(self.gamma ** self.n_step) * \
                           next_q_values_target * (1 - dones.float()) - q_values).cpu().numpy()

            loss_decimal = Decimal(str(weighted_loss.item()))

            logger.debug(
                "training_step_complete",
                loss=str(loss_decimal),
                step=self.training_step
            )

            return loss_decimal, td_errors

        except Exception as e:
            logger.error("failed_training_step", error=str(e))
            raise

    def save(self, path: str) -> None:
        """Save agent state.

        Args:
            path: Save path
        """
        try:
            torch.save({
                'online_net': self.online_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'config': self.config
            }, path)

            logger.info("saved_agent", path=path)

        except Exception as e:
            logger.error("failed_to_save_agent", error=str(e), path=path)
            raise

    def load(self, path: str) -> None:
        """Load agent state.

        Args:
            path: Load path
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.online_net.load_state_dict(checkpoint['online_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_step = checkpoint['training_step']

            logger.info("loaded_agent", path=path, step=self.training_step)

        except Exception as e:
            logger.error("failed_to_load_agent", error=str(e), path=path)
            raise
