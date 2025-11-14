"""PPO Agent for Trading - PRODUCTION RL"""
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os, logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    raise ImportError("PyTorch required")

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class PPOAgent:
    """Proximal Policy Optimization for Trading"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.state_dim = int(config.get('state_dim', os.getenv('PPO_STATE_DIM', '200')))
        self.action_dim = int(config.get('action_dim', os.getenv('PPO_ACTION_DIM', '3')))  # Buy/Hold/Sell
        self.hidden_dim = int(config.get('hidden_dim', os.getenv('PPO_HIDDEN_DIM', '256')))

        # PPO hyperparameters
        self.lr = float(config.get('learning_rate', os.getenv('PPO_LR', '3e-4')))
        self.gamma = float(config.get('gamma', os.getenv('PPO_GAMMA', '0.99')))
        self.gae_lambda = float(config.get('gae_lambda', os.getenv('PPO_GAE_LAMBDA', '0.95')))
        self.clip_param = float(config.get('clip_param', os.getenv('PPO_CLIP', '0.2')))
        self.value_loss_coef = float(config.get('value_loss_coef', '0.5'))
        self.entropy_coef = float(config.get('entropy_coef', '0.01'))

        # Initialize network
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Training tracking
        self.total_steps = 0
        self.episode_count = 0

        self.logger.info(f"PPOAgent initialized: state_dim={self.state_dim}, action_dim={self.action_dim}")

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy"""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action_probs, value = self.policy(state_tensor)

            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return int(action.item()), float(log_prob.item()), float(value.item())

        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            return 1, 0.0, 0.0  # Default to HOLD

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO"""
        try:
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Multiple epochs of updates
            n_epochs = int(self.config.get('ppo_epochs', os.getenv('PPO_EPOCHS', '4')))
            batch_size = int(self.config.get('batch_size', os.getenv('PPO_BATCH_SIZE', '64')))

            for _ in range(n_epochs):
                for idx in range(0, len(states), batch_size):
                    batch_states = states[idx:idx+batch_size]
                    batch_actions = actions[idx:idx+batch_size]
                    batch_old_log_probs = old_log_probs[idx:idx+batch_size]
                    batch_returns = returns[idx:idx+batch_size]
                    batch_advantages = advantages[idx:idx+batch_size]

                    # Forward pass
                    action_probs, values = self.policy(batch_states)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    # PPO loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)

                    # Total loss
                    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

            self.total_steps += len(states)

            return {
                'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()),
                'entropy': float(entropy.item())
            }

        except Exception as e:
            self.logger.error(f"Update failed: {e}", exc_info=True)
            return {'error': str(e)}

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]

        return advantages, returns

    def save(self, path: str) -> None:
        try:
            torch.save({
                'policy_state': self.policy.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'episode_count': self.episode_count,
                'config': self.config
            }, path)
            self.logger.info(f"Agent saved to {path}")
        except Exception as e:
            raise IOError(f"Save error: {e}")

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.total_steps = checkpoint['total_steps']
            self.episode_count = checkpoint['episode_count']
            self.logger.info(f"Agent loaded from {path}")
        except Exception as e:
            raise IOError(f"Load error: {e}")
