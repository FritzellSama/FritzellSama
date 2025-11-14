"""Reinforcement learning trainer for trading agents.

This module implements the training loop for RL agents with support for
experience replay, target network updates, and comprehensive logging.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import polars as pl
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single step/episode.

    Attributes:
        step: Training step number
        episode: Episode number
        loss: Training loss (Decimal)
        reward: Episode reward (Decimal)
        epsilon: Exploration epsilon (Decimal)
        q_value_mean: Mean Q-value (Decimal)
        td_error_mean: Mean TD error (Decimal)
        learning_rate: Current learning rate (Decimal)
        timestamp: UTC timestamp
    """
    step: int
    episode: int
    loss: Optional[Decimal] = None
    reward: Optional[Decimal] = None
    epsilon: Optional[Decimal] = None
    q_value_mean: Optional[Decimal] = None
    td_error_mean: Optional[Decimal] = None
    learning_rate: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass
class CheckpointInfo:
    """Checkpoint information.

    Attributes:
        checkpoint_id: Checkpoint identifier
        step: Training step
        episode: Episode number
        path: Checkpoint file path
        metrics: Training metrics at checkpoint
        timestamp: UTC timestamp
    """
    checkpoint_id: str
    step: int
    episode: int
    path: str
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class RLTrainer:
    """Reinforcement learning trainer for trading agents.

    Manages the training loop, experience collection, model updates,
    and checkpointing for RL agents.

    Attributes:
        config: Configuration dictionary
        agent: RL agent to train
        env: Trading environment
        replay_buffer: Experience replay buffer

    Example:
        >>> config = {
        ...     'total_timesteps': 1000000,
        ...     'batch_size': 64,
        ...     'learning_starts': 10000,
        ...     'train_freq': 4,
        ...     'target_update_freq': 1000,
        ...     'checkpoint_freq': 10000,
        ...     'eval_freq': 5000,
        ...     'log_freq': 100,
        ...     'save_path': './checkpoints'
        ... }
        >>> trainer = RLTrainer(config, agent, env, replay_buffer)
        >>> await trainer.train()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        agent: Any,
        env: Any,
        replay_buffer: Any
    ) -> None:
        """Initialize RL trainer.

        Args:
            config: Configuration with keys:
                - total_timesteps: Total training steps
                - batch_size: Training batch size
                - learning_starts: Steps before training starts
                - train_freq: Training frequency (steps)
                - target_update_freq: Target network update frequency
                - checkpoint_freq: Checkpoint save frequency
                - eval_freq: Evaluation frequency
                - log_freq: Logging frequency
                - save_path: Path for saving checkpoints
                - max_episode_steps: Max steps per episode
            agent: RL agent
            env: Trading environment
            replay_buffer: Experience replay buffer
        """
        self.config = config
        self._validate_config()

        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer

        # Training parameters
        self.total_timesteps = config['total_timesteps']
        self.batch_size = config['batch_size']
        self.learning_starts = config['learning_starts']
        self.train_freq = config['train_freq']
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.checkpoint_freq = config['checkpoint_freq']
        self.eval_freq = config.get('eval_freq', 5000)
        self.log_freq = config['log_freq']
        self.save_path = Path(config['save_path'])
        self.max_episode_steps = config.get('max_episode_steps', 10000)

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.current_step = 0
        self.current_episode = 0
        self.episode_rewards: List[Decimal] = []
        self.episode_lengths: List[int] = []
        self.training_metrics: List[TrainingMetrics] = []
        self.checkpoints: List[CheckpointInfo] = []

        logger.info(
            "initialized_rl_trainer",
            total_timesteps=self.total_timesteps,
            batch_size=self.batch_size,
            save_path=str(self.save_path)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'total_timesteps', 'batch_size', 'learning_starts',
            'train_freq', 'checkpoint_freq', 'log_freq', 'save_path'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if self.config['total_timesteps'] <= 0:
            raise ValueError("total_timesteps must be positive")

        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

        if self.config['learning_starts'] < self.config['batch_size']:
            raise ValueError("learning_starts must be >= batch_size")

    async def train(self) -> None:
        """Execute training loop.

        Raises:
            Exception: If training fails
        """
        try:
            logger.info("starting_training_loop")

            state = self.env.reset()
            episode_reward = Decimal('0')
            episode_length = 0

            while self.current_step < self.total_timesteps:
                # Select action
                action = self.agent.select_action(state)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)
                reward_decimal = Decimal(str(reward))

                # Store transition in replay buffer
                self.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward_decimal,
                    next_state=next_state,
                    done=done
                )

                # Track episode metrics
                episode_reward += reward_decimal
                episode_length += 1

                # Training
                if self.current_step >= self.learning_starts:
                    if self.current_step % self.train_freq == 0:
                        await self._train_step()

                # Update state
                state = next_state
                self.current_step += 1

                # Episode end
                if done or episode_length >= self.max_episode_steps:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.current_episode += 1

                    logger.info(
                        "episode_complete",
                        episode=self.current_episode,
                        reward=str(episode_reward),
                        length=episode_length,
                        step=self.current_step
                    )

                    # Reset for next episode
                    state = self.env.reset()
                    episode_reward = Decimal('0')
                    episode_length = 0

                # Periodic operations
                if self.current_step % self.checkpoint_freq == 0:
                    await self._save_checkpoint()

                if self.eval_freq and self.current_step % self.eval_freq == 0:
                    await self._evaluate()

                if self.current_step % self.log_freq == 0:
                    self._log_metrics()

            logger.info(
                "training_complete",
                total_steps=self.current_step,
                total_episodes=self.current_episode
            )

        except Exception as e:
            logger.error("training_failed", error=str(e), step=self.current_step)
            raise

    async def _train_step(self) -> None:
        """Perform single training step.

        Raises:
            Exception: If training step fails
        """
        try:
            # Sample batch from replay buffer
            batch = self.replay_buffer.sample(self.batch_size)

            # Prepare tensors
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            dones = batch['dones']
            weights = batch.get('weights', np.ones(self.batch_size))

            # Train agent
            loss, td_errors = self.agent.train_step(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                weights=weights
            )

            # Update priorities if using prioritized replay
            if 'indices' in batch and hasattr(self.replay_buffer, 'update_priorities'):
                self.replay_buffer.update_priorities(
                    batch['indices'],
                    td_errors
                )

            # Store metrics
            metrics = TrainingMetrics(
                step=self.current_step,
                episode=self.current_episode,
                loss=loss,
                td_error_mean=Decimal(str(np.mean(np.abs(td_errors))))
            )
            self.training_metrics.append(metrics)

            # Limit metrics history
            if len(self.training_metrics) > 10000:
                self.training_metrics = self.training_metrics[-10000:]

            logger.debug(
                "training_step_complete",
                step=self.current_step,
                loss=str(loss),
                td_error=str(metrics.td_error_mean)
            )

        except Exception as e:
            logger.error("training_step_failed", error=str(e), step=self.current_step)
            raise

    async def _save_checkpoint(self) -> None:
        """Save training checkpoint.

        Raises:
            Exception: If checkpoint save fails
        """
        try:
            checkpoint_id = f"checkpoint_step_{self.current_step}_ep_{self.current_episode}"
            checkpoint_path = self.save_path / f"{checkpoint_id}.pt"

            # Save agent
            self.agent.save(str(checkpoint_path))

            # Collect metrics
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]
                mean_reward = sum(recent_rewards) / Decimal(str(len(recent_rewards)))
            else:
                mean_reward = Decimal('0')

            metrics = {
                'step': self.current_step,
                'episode': self.current_episode,
                'mean_reward_100': str(mean_reward),
                'buffer_size': len(self.replay_buffer)
            }

            # Store checkpoint info
            checkpoint_info = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                step=self.current_step,
                episode=self.current_episode,
                path=str(checkpoint_path),
                metrics=metrics
            )
            self.checkpoints.append(checkpoint_info)

            logger.info(
                "saved_checkpoint",
                checkpoint_id=checkpoint_id,
                path=str(checkpoint_path),
                metrics=metrics
            )

        except Exception as e:
            logger.error("failed_to_save_checkpoint", error=str(e))
            raise

    async def _evaluate(self) -> None:
        """Run evaluation.

        This is a placeholder - actual evaluation would use RLEvaluator.
        """
        logger.info(
            "evaluation_checkpoint",
            step=self.current_step,
            episode=self.current_episode
        )

    def _log_metrics(self) -> None:
        """Log current training metrics."""
        if not self.episode_rewards:
            return

        # Recent episode metrics
        recent_rewards = self.episode_rewards[-100:]
        recent_lengths = self.episode_lengths[-100:]

        mean_reward = sum(recent_rewards) / Decimal(str(len(recent_rewards)))
        mean_length = sum(recent_lengths) / len(recent_lengths)

        # Recent training metrics
        if self.training_metrics:
            recent_metrics = self.training_metrics[-100:]
            losses = [float(m.loss) for m in recent_metrics if m.loss is not None]
            if losses:
                mean_loss = Decimal(str(np.mean(losses)))
            else:
                mean_loss = Decimal('0')
        else:
            mean_loss = Decimal('0')

        logger.info(
            "training_metrics",
            step=self.current_step,
            episode=self.current_episode,
            mean_reward_100=str(mean_reward),
            mean_length_100=mean_length,
            mean_loss=str(mean_loss),
            buffer_size=len(self.replay_buffer)
        )

    def get_training_dataframe(self) -> pl.DataFrame:
        """Get training metrics as Polars DataFrame.

        Returns:
            Polars DataFrame with training metrics
        """
        if not self.training_metrics:
            return pl.DataFrame()

        data = {
            'step': [m.step for m in self.training_metrics],
            'episode': [m.episode for m in self.training_metrics],
            'loss': [str(m.loss) if m.loss else None for m in self.training_metrics],
            'td_error_mean': [str(m.td_error_mean) if m.td_error_mean else None for m in self.training_metrics],
            'timestamp': [m.timestamp.isoformat() for m in self.training_metrics]
        }

        return pl.DataFrame(data)

    def get_episode_dataframe(self) -> pl.DataFrame:
        """Get episode metrics as Polars DataFrame.

        Returns:
            Polars DataFrame with episode metrics
        """
        if not self.episode_rewards:
            return pl.DataFrame()

        data = {
            'episode': list(range(len(self.episode_rewards))),
            'reward': [str(r) for r in self.episode_rewards],
            'length': self.episode_lengths
        }

        return pl.DataFrame(data)

    def export_metrics(self, path: str) -> None:
        """Export training metrics to file.

        Args:
            path: Export file path
        """
        try:
            training_df = self.get_training_dataframe()
            episode_df = self.get_episode_dataframe()

            if path.endswith('.parquet'):
                training_df.write_parquet(path.replace('.parquet', '_training.parquet'))
                episode_df.write_parquet(path.replace('.parquet', '_episodes.parquet'))
            elif path.endswith('.csv'):
                training_df.write_csv(path.replace('.csv', '_training.csv'))
                episode_df.write_csv(path.replace('.csv', '_episodes.csv'))
            else:
                raise ValueError(f"Unsupported file format: {path}")

            logger.info("exported_metrics", path=path)

        except Exception as e:
            logger.error("failed_to_export_metrics", error=str(e), path=path)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training statistics
        """
        stats = {
            'current_step': self.current_step,
            'current_episode': self.current_episode,
            'total_timesteps': self.total_timesteps,
            'progress': self.current_step / self.total_timesteps,
            'n_checkpoints': len(self.checkpoints)
        }

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            rewards_array = np.array([float(r) for r in recent_rewards])

            stats['episode_stats'] = {
                'mean_reward_100': float(np.mean(rewards_array)),
                'std_reward_100': float(np.std(rewards_array)),
                'min_reward_100': float(np.min(rewards_array)),
                'max_reward_100': float(np.max(rewards_array)),
                'total_episodes': len(self.episode_rewards)
            }

        if self.training_metrics:
            recent_metrics = self.training_metrics[-100:]
            losses = [float(m.loss) for m in recent_metrics if m.loss is not None]

            if losses:
                stats['training_stats'] = {
                    'mean_loss_100': float(np.mean(losses)),
                    'std_loss_100': float(np.std(losses))
                }

        return stats

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            self.agent.load(checkpoint_path)

            logger.info("loaded_checkpoint", path=checkpoint_path)

        except Exception as e:
            logger.error("failed_to_load_checkpoint", error=str(e), path=checkpoint_path)
            raise
