"""Distributed replay buffer for reinforcement learning.

This module implements a distributed prioritized replay buffer for RL agents,
enabling efficient experience storage and sampling across multiple processes.
"""

import asyncio
import pickle
from collections import deque
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Deque
import numpy as np
import polars as pl
from structlog import get_logger
import redis.asyncio as aioredis

logger = get_logger(__name__)


class DistributedReplayBuffer:
    """Distributed prioritized experience replay buffer.

    Stores experiences across distributed storage with prioritized sampling
    for efficient reinforcement learning training.

    Attributes:
        config: Configuration dictionary
        buffer_size: Maximum number of experiences to store
        batch_size: Batch size for sampling
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling weight (0 = no correction, 1 = full)
        redis_client: Async Redis client for distributed storage
        local_buffer: Local deque for fast access
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize distributed replay buffer.

        Args:
            config: Configuration dictionary containing:
                - buffer_size: Maximum buffer size
                - batch_size: Sampling batch size
                - alpha: Priority exponent
                - beta: Importance sampling weight
                - beta_increment: Beta increment per sampling
                - redis_host: Redis host
                - redis_port: Redis port
                - redis_db: Redis database number
                - redis_key_prefix: Key prefix for Redis

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.alpha = Decimal(str(config.get('alpha', '0.6')))
        self.beta = Decimal(str(config.get('beta', '0.4')))
        self.beta_increment = Decimal(str(config.get('beta_increment', '0.001')))

        # Redis configuration
        self.redis_host = config['redis_host']
        self.redis_port = config['redis_port']
        self.redis_db = config.get('redis_db', 0)
        self.redis_key_prefix = config.get('redis_key_prefix', 'replay_buffer')

        # Local buffer for fast access
        self.local_buffer: Deque[Dict[str, Any]] = deque(maxlen=self.buffer_size)
        self.priorities: Deque[Decimal] = deque(maxlen=self.buffer_size)

        self.redis_client: Optional[aioredis.Redis] = None
        self._position = 0
        self._max_priority = Decimal('1.0')

        logger.info(
            "distributed_replay_buffer_initialized",
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            alpha=str(self.alpha),
            beta=str(self.beta)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = [
            'buffer_size',
            'batch_size',
            'redis_host',
            'redis_port'
        ]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("config_validation_failed", error=error_msg)
            raise ValueError(error_msg)

        if self.config['buffer_size'] <= 0:
            raise ValueError("buffer_size must be positive")

        if self.config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

        if self.config['batch_size'] > self.config['buffer_size']:
            raise ValueError("batch_size cannot exceed buffer_size")

    async def connect(self) -> None:
        """Connect to Redis for distributed storage.

        Raises:
            ConnectionError: If Redis connection fails
        """
        try:
            self.redis_client = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}",
                encoding="utf-8",
                decode_responses=False,
                socket_timeout=self.config.get('redis_timeout', 5),
                socket_connect_timeout=self.config.get('redis_connect_timeout', 5),
                retry_on_timeout=True,
                max_connections=self.config.get('redis_max_connections', 50)
            )

            # Test connection
            await self.redis_client.ping()

            logger.info(
                "redis_connected",
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db
            )

        except Exception as e:
            error_msg = f"Redis connection failed: {str(e)}"
            logger.error("redis_connection_failed", error=error_msg)
            raise ConnectionError(error_msg)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("redis_disconnected")

    async def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: Decimal,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add experience to replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            info: Additional information

        Raises:
            ValueError: If experience data is invalid
        """
        try:
            # Validate inputs
            if state.shape != next_state.shape:
                raise ValueError("state and next_state must have same shape")

            # Create experience dictionary
            experience = {
                'state': state,
                'action': action,
                'reward': float(reward),  # Convert Decimal for serialization
                'next_state': next_state,
                'done': done,
                'info': info or {},
                'timestamp': asyncio.get_event_loop().time()
            }

            # Add to local buffer with max priority
            self.local_buffer.append(experience)
            self.priorities.append(self._max_priority)

            # Store in Redis for distributed access
            if self.redis_client:
                await self._store_in_redis(experience)

            self._position = (self._position + 1) % self.buffer_size

            logger.debug(
                "experience_added",
                buffer_size=len(self.local_buffer),
                position=self._position
            )

        except Exception as e:
            logger.error("add_experience_failed", error=str(e))
            raise

    async def _store_in_redis(self, experience: Dict[str, Any]) -> None:
        """Store experience in Redis.

        Args:
            experience: Experience dictionary

        Raises:
            Exception: If Redis storage fails
        """
        try:
            # Serialize experience
            serialized = pickle.dumps(experience)

            # Store with key based on position
            key = f"{self.redis_key_prefix}:exp:{self._position}"

            await self.redis_client.set(
                key,
                serialized,
                ex=self.config.get('redis_expiry', 86400)  # 24 hour default TTL
            )

            # Update buffer metadata
            meta_key = f"{self.redis_key_prefix}:meta"
            await self.redis_client.hset(
                meta_key,
                mapping={
                    'size': str(len(self.local_buffer)),
                    'position': str(self._position),
                    'max_priority': str(self._max_priority)
                }
            )

        except Exception as e:
            logger.warning("redis_storage_failed", error=str(e))
            # Don't raise - local buffer is still valid

    async def sample_batch(
        self,
        beta: Optional[Decimal] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Sample batch of experiences using prioritized sampling.

        Args:
            beta: Importance sampling weight (uses self.beta if None)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)

        Raises:
            ValueError: If buffer doesn't have enough experiences
        """
        try:
            if len(self.local_buffer) < self.batch_size:
                raise ValueError(
                    f"Buffer has {len(self.local_buffer)} experiences, "
                    f"need {self.batch_size}"
                )

            # Use provided beta or increment internal beta
            if beta is None:
                self.beta = min(Decimal('1.0'), self.beta + self.beta_increment)
                beta = self.beta

            # Calculate sampling probabilities
            priorities_array = np.array([float(p) for p in self.priorities])
            probabilities = priorities_array ** float(self.alpha)
            probabilities /= probabilities.sum()

            # Sample indices
            indices = np.random.choice(
                len(self.local_buffer),
                size=self.batch_size,
                replace=False,
                p=probabilities
            )

            # Calculate importance sampling weights
            weights = (len(self.local_buffer) * probabilities[indices]) ** (-float(beta))
            weights /= weights.max()  # Normalize

            # Extract experiences
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []

            for idx in indices:
                exp = self.local_buffer[idx]
                batch_states.append(exp['state'])
                batch_actions.append(exp['action'])
                batch_rewards.append(exp['reward'])
                batch_next_states.append(exp['next_state'])
                batch_dones.append(exp['done'])

            logger.debug(
                "batch_sampled",
                batch_size=self.batch_size,
                beta=str(beta)
            )

            return (
                np.array(batch_states),
                np.array(batch_actions),
                np.array(batch_rewards),
                np.array(batch_next_states),
                np.array(batch_dones),
                weights,
                indices.tolist()
            )

        except Exception as e:
            logger.error("batch_sampling_failed", error=str(e))
            raise

    async def update_priorities(
        self,
        indices: List[int],
        td_errors: np.ndarray
    ) -> None:
        """Update priorities based on TD errors.

        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for priority calculation

        Raises:
            ValueError: If indices or td_errors are invalid
        """
        try:
            if len(indices) != len(td_errors):
                raise ValueError("indices and td_errors must have same length")

            epsilon = Decimal(str(self.config.get('priority_epsilon', '0.01')))

            for idx, td_error in zip(indices, td_errors):
                if idx >= len(self.priorities):
                    continue

                # New priority = |TD error| + epsilon
                priority = Decimal(str(abs(td_error))) + epsilon
                self.priorities[idx] = priority

                # Update max priority
                if priority > self._max_priority:
                    self._max_priority = priority

            logger.debug(
                "priorities_updated",
                count=len(indices),
                max_priority=str(self._max_priority)
            )

        except Exception as e:
            logger.error("priority_update_failed", error=str(e))
            raise

    async def sync_from_redis(self) -> None:
        """Synchronize local buffer from Redis.

        Loads experiences from Redis to populate local buffer.
        Useful when starting a new worker process.

        Raises:
            Exception: If sync fails
        """
        try:
            if not self.redis_client:
                logger.warning("redis_not_connected", action="skipping_sync")
                return

            # Get metadata
            meta_key = f"{self.redis_key_prefix}:meta"
            meta = await self.redis_client.hgetall(meta_key)

            if not meta:
                logger.info("no_redis_data_to_sync")
                return

            buffer_size = int(meta.get(b'size', 0))
            position = int(meta.get(b'position', 0))

            # Load experiences
            synced_count = 0
            for i in range(buffer_size):
                key = f"{self.redis_key_prefix}:exp:{i}"
                serialized = await self.redis_client.get(key)

                if serialized:
                    experience = pickle.loads(serialized)
                    self.local_buffer.append(experience)
                    # Initialize with max priority
                    self.priorities.append(self._max_priority)
                    synced_count += 1

            self._position = position

            logger.info(
                "redis_sync_completed",
                synced_experiences=synced_count,
                buffer_size=len(self.local_buffer)
            )

        except Exception as e:
            logger.error("redis_sync_failed", error=str(e))
            raise

    def get_buffer_size(self) -> int:
        """Get current buffer size.

        Returns:
            Number of experiences in buffer
        """
        return len(self.local_buffer)

    def clear_buffer(self) -> None:
        """Clear all experiences from buffer."""
        self.local_buffer.clear()
        self.priorities.clear()
        self._position = 0
        self._max_priority = Decimal('1.0')

        logger.info("buffer_cleared")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            'buffer_size': len(self.local_buffer),
            'max_size': self.buffer_size,
            'position': self._position,
            'max_priority': str(self._max_priority),
            'beta': str(self.beta),
            'alpha': str(self.alpha)
        }

        if len(self.priorities) > 0:
            priorities_array = np.array([float(p) for p in self.priorities])
            stats.update({
                'avg_priority': float(np.mean(priorities_array)),
                'min_priority': float(np.min(priorities_array)),
                'priority_std': float(np.std(priorities_array))
            })

        return stats
