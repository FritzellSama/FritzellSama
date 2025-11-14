"""
A3C (Asynchronous Advantage Actor-Critic) Reinforcement Learning Agent.

This module implements the A3C algorithm for trading decisions, using multiple
asynchronous workers to explore the environment and update a shared model.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class A3CConfig:
    """Configuration for A3C agent."""

    learning_rate: Decimal
    gamma: Decimal  # Discount factor
    entropy_coef: Decimal  # Entropy coefficient for exploration
    value_loss_coef: Decimal  # Value loss coefficient
    max_grad_norm: Decimal  # Gradient clipping
    num_workers: int  # Number of async workers
    update_frequency: int  # Steps before update
    hidden_size: int  # Hidden layer size
    num_actions: int  # Number of possible actions
    state_dim: int  # State dimension
    device: str  # 'cpu' or 'cuda'
    model_path: str  # Path to save/load models
    min_exploration: Decimal  # Minimum exploration rate
    max_episodes: int  # Maximum training episodes
    episode_timeout: int  # Max steps per episode


class ActorCriticNetwork(nn.Module):
    """Neural network for both actor (policy) and critic (value) functions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int) -> None:
        """Initialize actor-critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.

        Args:
            state: Input state tensor

        Returns:
            Tuple of (action_logits, state_value)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor output: action probabilities
        action_logits = self.actor(x)

        # Critic output: state value
        state_value = self.critic(x)

        return action_logits, state_value


class A3CWorker:
    """Asynchronous worker for A3C algorithm."""

    def __init__(
        self,
        worker_id: int,
        shared_model: ActorCriticNetwork,
        config: A3CConfig,
        optimizer: optim.Optimizer
    ) -> None:
        """Initialize A3C worker.

        Args:
            worker_id: Unique identifier for this worker
            shared_model: Shared global model
            config: A3C configuration
            optimizer: Shared optimizer
        """
        self.worker_id = worker_id
        self.shared_model = shared_model
        self.config = config
        self.optimizer = optimizer

        # Local model for this worker
        self.local_model = ActorCriticNetwork(
            config.state_dim,
            config.num_actions,
            config.hidden_size
        ).to(config.device)

        self.episode_rewards: List[Decimal] = []
        self.episode_lengths: List[int] = []

        logger.info(
            "A3C worker initialized",
            worker_id=worker_id,
            device=config.device
        )

    def _sync_with_shared_model(self) -> None:
        """Synchronize local model with shared model."""
        self.local_model.load_state_dict(self.shared_model.state_dict())

    async def select_action(
        self,
        state: np.ndarray
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, log_prob, state_value)
        """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                action_logits, state_value = self.local_model(state_tensor)

            # Sample action from policy
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob, state_value

        except Exception as e:
            logger.error(
                "Action selection failed",
                worker_id=self.worker_id,
                error=str(e)
            )
            raise

    def _compute_returns(
        self,
        rewards: List[Decimal],
        next_value: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute discounted returns.

        Args:
            rewards: List of rewards
            next_value: Value of next state

        Returns:
            List of discounted returns
        """
        returns = []
        R = next_value
        gamma = float(self.config.gamma)

        for reward in reversed(rewards):
            R = float(reward) + gamma * R
            returns.insert(0, R)

        return [torch.tensor([r], dtype=torch.float32).to(self.config.device) for r in returns]

    async def update_shared_model(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[Decimal],
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        next_value: torch.Tensor
    ) -> Dict[str, float]:
        """Update shared model with collected experience.

        Args:
            states: List of states
            actions: List of actions taken
            rewards: List of rewards received
            log_probs: List of log probabilities
            values: List of state values
            next_value: Value of final state

        Returns:
            Dictionary of loss metrics
        """
        try:
            # Compute returns
            returns = self._compute_returns(rewards, next_value)

            # Convert to tensors
            returns_tensor = torch.cat(returns).detach()
            values_tensor = torch.cat(values)
            log_probs_tensor = torch.cat(log_probs)

            # Compute advantages
            advantages = returns_tensor - values_tensor.detach()

            # Actor loss (policy gradient)
            actor_loss = -(log_probs_tensor * advantages).mean()

            # Critic loss (value function)
            value_loss = F.smooth_l1_loss(values_tensor, returns_tensor)

            # Entropy bonus for exploration
            states_tensor = torch.FloatTensor(np.array(states)).to(self.config.device)
            action_logits, _ = self.local_model(states_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()

            # Total loss
            total_loss = (
                actor_loss +
                float(self.config.value_loss_coef) * value_loss -
                float(self.config.entropy_coef) * entropy
            )

            # Update shared model
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.local_model.parameters(),
                float(self.config.max_grad_norm)
            )

            # Update shared model parameters
            for shared_param, local_param in zip(
                self.shared_model.parameters(),
                self.local_model.parameters()
            ):
                if shared_param.grad is None:
                    shared_param._grad = local_param.grad
                else:
                    shared_param._grad += local_param.grad

            self.optimizer.step()

            # Sync local model
            self._sync_with_shared_model()

            return {
                "actor_loss": actor_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "total_loss": total_loss.item()
            }

        except Exception as e:
            logger.error(
                "Model update failed",
                worker_id=self.worker_id,
                error=str(e)
            )
            raise

    async def train_episode(
        self,
        env: Any,
        episode: int
    ) -> Dict[str, Any]:
        """Train for one episode.

        Args:
            env: Trading environment
            episode: Episode number

        Returns:
            Episode statistics
        """
        try:
            self._sync_with_shared_model()

            state = await env.reset()
            done = False
            episode_reward = Decimal("0")
            episode_length = 0

            states: List[np.ndarray] = []
            actions: List[int] = []
            rewards: List[Decimal] = []
            log_probs: List[torch.Tensor] = []
            values: List[torch.Tensor] = []

            while not done and episode_length < self.config.episode_timeout:
                # Select action
                action, log_prob, value = await self.select_action(state)

                # Take action in environment
                next_state, reward, done, info = await env.step(action)

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(Decimal(str(reward)))
                log_probs.append(log_prob)
                values.append(value)

                episode_reward += Decimal(str(reward))
                episode_length += 1
                state = next_state

                # Update shared model periodically
                if episode_length % self.config.update_frequency == 0:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(
                        self.config.device
                    )
                    _, next_value = self.local_model(next_state_tensor)

                    loss_metrics = await self.update_shared_model(
                        states, actions, rewards, log_probs, values, next_value
                    )

                    # Clear buffers
                    states, actions, rewards, log_probs, values = [], [], [], [], []

            # Final update
            if len(states) > 0:
                next_value = torch.tensor([0.0]).to(self.config.device) if done else value
                loss_metrics = await self.update_shared_model(
                    states, actions, rewards, log_probs, values, next_value
                )

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            logger.info(
                "Episode completed",
                worker_id=self.worker_id,
                episode=episode,
                reward=float(episode_reward),
                length=episode_length
            )

            return {
                "episode": episode,
                "reward": episode_reward,
                "length": episode_length,
                "worker_id": self.worker_id
            }

        except Exception as e:
            logger.error(
                "Episode training failed",
                worker_id=self.worker_id,
                episode=episode,
                error=str(e)
            )
            raise


class A3CAgent:
    """Asynchronous Advantage Actor-Critic agent for trading."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize A3C agent.

        Args:
            config: Agent configuration dictionary

        Example:
            >>> config = {
            ...     "learning_rate": "0.0001",
            ...     "gamma": "0.99",
            ...     "num_workers": 4,
            ...     "state_dim": 50,
            ...     "num_actions": 3
            ... }
            >>> agent = A3CAgent(config)
        """
        self.config = self._build_config(config)

        # Shared global model
        self.shared_model = ActorCriticNetwork(
            self.config.state_dim,
            self.config.num_actions,
            self.config.hidden_size
        ).to(self.config.device)

        self.shared_model.share_memory()  # Enable sharing across processes

        # Shared optimizer
        self.optimizer = optim.Adam(
            self.shared_model.parameters(),
            lr=float(self.config.learning_rate)
        )

        # Workers
        self.workers: List[A3CWorker] = []
        for i in range(self.config.num_workers):
            worker = A3CWorker(i, self.shared_model, self.config, self.optimizer)
            self.workers.append(worker)

        self.training_stats: Dict[str, List[Any]] = {
            "episodes": [],
            "rewards": [],
            "lengths": []
        }

        logger.info(
            "A3C agent initialized",
            num_workers=self.config.num_workers,
            device=self.config.device,
            state_dim=self.config.state_dim,
            num_actions=self.config.num_actions
        )

    def _build_config(self, config: Dict[str, Any]) -> A3CConfig:
        """Build A3CConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            A3CConfig instance
        """
        return A3CConfig(
            learning_rate=Decimal(str(config.get("learning_rate", "0.0001"))),
            gamma=Decimal(str(config.get("gamma", "0.99"))),
            entropy_coef=Decimal(str(config.get("entropy_coef", "0.01"))),
            value_loss_coef=Decimal(str(config.get("value_loss_coef", "0.5"))),
            max_grad_norm=Decimal(str(config.get("max_grad_norm", "0.5"))),
            num_workers=config.get("num_workers", 4),
            update_frequency=config.get("update_frequency", 20),
            hidden_size=config.get("hidden_size", 256),
            num_actions=config["num_actions"],
            state_dim=config["state_dim"],
            device=config.get("device", "cpu"),
            model_path=config.get("model_path", "/tmp/a3c_model.pt"),
            min_exploration=Decimal(str(config.get("min_exploration", "0.01"))),
            max_episodes=config.get("max_episodes", 10000),
            episode_timeout=config.get("episode_timeout", 1000)
        )

    async def train(self, env_factory: Any, num_episodes: int) -> pl.DataFrame:
        """Train the agent using multiple async workers.

        Args:
            env_factory: Factory function to create environment instances
            num_episodes: Number of episodes to train

        Returns:
            DataFrame with training statistics

        Example:
            >>> async def env_factory():
            ...     return TradingEnvironment(config)
            >>> stats = await agent.train(env_factory, 1000)
        """
        try:
            logger.info("Starting A3C training", num_episodes=num_episodes)

            # Create tasks for all workers
            tasks = []
            episodes_per_worker = num_episodes // self.config.num_workers

            for worker in self.workers:
                env = await env_factory()
                task = self._worker_train_loop(worker, env, episodes_per_worker)
                tasks.append(task)

            # Run all workers concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect statistics
            all_stats = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Worker failed", error=str(result))
                else:
                    all_stats.extend(result)

            # Convert to DataFrame
            df = pl.DataFrame({
                "episode": [s["episode"] for s in all_stats],
                "reward": [float(s["reward"]) for s in all_stats],
                "length": [s["length"] for s in all_stats],
                "worker_id": [s["worker_id"] for s in all_stats]
            })

            logger.info(
                "Training completed",
                total_episodes=len(all_stats),
                avg_reward=df["reward"].mean()
            )

            return df

        except Exception as e:
            logger.error("Training failed", error=str(e))
            raise

    async def _worker_train_loop(
        self,
        worker: A3CWorker,
        env: Any,
        num_episodes: int
    ) -> List[Dict[str, Any]]:
        """Training loop for a single worker.

        Args:
            worker: A3C worker
            env: Trading environment
            num_episodes: Number of episodes

        Returns:
            List of episode statistics
        """
        stats = []
        for episode in range(num_episodes):
            try:
                episode_stats = await worker.train_episode(env, episode)
                stats.append(episode_stats)
            except Exception as e:
                logger.error(
                    "Episode failed",
                    worker_id=worker.worker_id,
                    episode=episode,
                    error=str(e)
                )
        return stats

    async def predict(self, state: np.ndarray) -> Tuple[int, Decimal]:
        """Predict action for given state.

        Args:
            state: Current state

        Returns:
            Tuple of (action, confidence)

        Example:
            >>> state = np.random.randn(50)
            >>> action, confidence = await agent.predict(state)
        """
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)

            with torch.no_grad():
                action_logits, _ = self.shared_model(state_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()
                confidence = Decimal(str(action_probs[0, action].item()))

            return action, confidence

        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise

    def save(self, path: Optional[str] = None) -> None:
        """Save model to disk.

        Args:
            path: Path to save model (uses config path if None)
        """
        try:
            save_path = path or self.config.model_path
            torch.save({
                "model_state_dict": self.shared_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config
            }, save_path)

            logger.info("Model saved", path=save_path)

        except Exception as e:
            logger.error("Model save failed", error=str(e))
            raise

    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk.

        Args:
            path: Path to load model from (uses config path if None)
        """
        try:
            load_path = path or self.config.model_path
            checkpoint = torch.load(load_path, map_location=self.config.device)

            self.shared_model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info("Model loaded", path=load_path)

        except Exception as e:
            logger.error("Model load failed", error=str(e))
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics from all workers.

        Returns:
            Dictionary of aggregated statistics
        """
        all_rewards = []
        all_lengths = []

        for worker in self.workers:
            all_rewards.extend([float(r) for r in worker.episode_rewards])
            all_lengths.extend(worker.episode_lengths)

        if not all_rewards:
            return {}

        return {
            "total_episodes": len(all_rewards),
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "mean_length": np.mean(all_lengths),
            "total_steps": np.sum(all_lengths)
        }
