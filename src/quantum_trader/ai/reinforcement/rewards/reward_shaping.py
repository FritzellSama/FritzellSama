"""Reward shaping for reinforcement learning.

This module implements reward shaping techniques to improve learning efficiency
by providing intermediate rewards and guidance signals.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ShapingPotential:
    """Potential function value for reward shaping.

    Attributes:
        state_id: State identifier
        potential: Potential value (Decimal)
        features: Feature values
        timestamp: UTC timestamp
    """
    state_id: str
    potential: Decimal
    features: Dict[str, Decimal]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


class RewardShaping:
    """Reward shaping using potential-based shaping.

    Implements potential-based reward shaping (PBRS) which adds bonus rewards
    based on state potential without changing optimal policy.

    F(s, a, s') = γ * Φ(s') - Φ(s)

    where Φ is the potential function and γ is discount factor.

    Attributes:
        config: Configuration dictionary
        gamma: Discount factor
        shaping_weight: Weight for shaped rewards
        potential_fn: Potential function

    Example:
        >>> config = {
        ...     'gamma': 0.99,
        ...     'shaping_weight': 0.1,
        ...     'potential_type': 'distance_to_goal',
        ...     'goal_features': {'portfolio_value': Decimal('110000')},
        ...     'normalize_potential': True,
        ...     'clip_shaping': True,
        ...     'clip_min': -1.0,
        ...     'clip_max': 1.0
        ... }
        >>> shaper = RewardShaping(config)
        >>> shaped_reward = shaper.shape_reward(
        ...     state_features, next_state_features, original_reward
        ... )
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize reward shaping.

        Args:
            config: Configuration with keys:
                - gamma: Discount factor
                - shaping_weight: Weight for shaping term
                - potential_type: 'distance_to_goal', 'heuristic', 'learned'
                - goal_features: Target feature values
                - normalize_potential: Whether to normalize potentials
                - clip_shaping: Whether to clip shaping bonus
                - clip_min: Minimum shaping bonus
                - clip_max: Maximum shaping bonus
        """
        self.config = config
        self._validate_config()

        self.gamma = Decimal(str(config['gamma']))
        self.shaping_weight = Decimal(str(config['shaping_weight']))
        self.potential_type = config['potential_type']
        self.goal_features = config.get('goal_features', {})
        self.normalize_potential = config['normalize_potential']
        self.clip_shaping = config['clip_shaping']
        self.clip_min = Decimal(str(config['clip_min']))
        self.clip_max = Decimal(str(config['clip_max']))

        # Potential tracking
        self.potential_history: List[ShapingPotential] = []
        self.max_potential = Decimal('1.0')
        self.min_potential = Decimal('-1.0')

        logger.info(
            "initialized_reward_shaping",
            potential_type=self.potential_type,
            gamma=str(self.gamma),
            shaping_weight=str(self.shaping_weight)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config is invalid
        """
        required_keys = [
            'gamma', 'shaping_weight', 'potential_type',
            'normalize_potential', 'clip_shaping', 'clip_min', 'clip_max'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        if not (0 < self.config['gamma'] <= 1):
            raise ValueError("gamma must be in (0, 1]")

        if self.config['shaping_weight'] < 0:
            raise ValueError("shaping_weight must be non-negative")

        if self.config['potential_type'] not in ['distance_to_goal', 'heuristic', 'learned']:
            raise ValueError("Invalid potential_type")

        if self.config['clip_min'] >= self.config['clip_max']:
            raise ValueError("clip_min must be less than clip_max")

    def calculate_potential(
        self,
        features: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate potential function value for state.

        Args:
            features: State features

        Returns:
            Potential value (Decimal)
        """
        try:
            if self.potential_type == 'distance_to_goal':
                potential = self._distance_to_goal_potential(features)

            elif self.potential_type == 'heuristic':
                potential = self._heuristic_potential(features)

            elif self.potential_type == 'learned':
                potential = self._learned_potential(features)

            else:
                potential = Decimal('0')

            # Update min/max for normalization
            self.max_potential = max(self.max_potential, potential)
            self.min_potential = min(self.min_potential, potential)

            # Normalize if configured
            if self.normalize_potential:
                potential = self._normalize_potential(potential)

            logger.debug(
                "calculated_potential",
                potential=str(potential),
                type=self.potential_type
            )

            return potential

        except Exception as e:
            logger.error("failed_to_calculate_potential", error=str(e))
            raise

    def _distance_to_goal_potential(
        self,
        features: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate potential based on distance to goal.

        Closer to goal = higher potential.

        Args:
            features: Current state features

        Returns:
            Potential value
        """
        if not self.goal_features:
            return Decimal('0')

        # Calculate normalized distance to goal
        total_distance = Decimal('0')
        feature_count = 0

        for feature_name, goal_value in self.goal_features.items():
            if feature_name in features:
                current_value = features[feature_name]

                # Normalized distance
                if goal_value != 0:
                    distance = abs(current_value - goal_value) / abs(goal_value)
                else:
                    distance = abs(current_value)

                total_distance += distance
                feature_count += 1

        if feature_count == 0:
            return Decimal('0')

        avg_distance = total_distance / Decimal(str(feature_count))

        # Potential is negative distance (closer = higher potential)
        potential = -avg_distance

        return potential

    def _heuristic_potential(
        self,
        features: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate potential using trading heuristics.

        Args:
            features: Current state features

        Returns:
            Potential value
        """
        potential = Decimal('0')

        # Positive portfolio value growth
        if 'portfolio_value' in features:
            portfolio_value = features['portfolio_value']
            initial_value = self.config.get('initial_portfolio_value', Decimal('100000'))

            if initial_value > 0:
                growth = (portfolio_value - initial_value) / initial_value
                potential += growth

        # Reward positive positions in uptrend
        if 'position' in features and 'trend' in features:
            position = features['position']
            trend = features['trend']

            # Aligned position with trend
            if (position > 0 and trend > 0) or (position < 0 and trend < 0):
                alignment = min(abs(position), abs(trend))
                potential += alignment * Decimal('0.1')

        # Penalty for excessive drawdown
        if 'drawdown' in features:
            drawdown = features['drawdown']
            if drawdown > 0:
                potential -= drawdown * Decimal('0.5')

        # Reward good risk-adjusted returns
        if 'sharpe_ratio' in features:
            sharpe = features['sharpe_ratio']
            potential += sharpe * Decimal('0.2')

        return potential

    def _learned_potential(
        self,
        features: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate potential using learned value function.

        Args:
            features: Current state features

        Returns:
            Potential value
        """
        # Placeholder for learned potential function
        # In practice, this would use a trained neural network or value function

        logger.warning(
            "learned_potential_not_implemented",
            message="Using zero potential"
        )

        return Decimal('0')

    def _normalize_potential(self, potential: Decimal) -> Decimal:
        """Normalize potential to [-1, 1] range.

        Args:
            potential: Raw potential value

        Returns:
            Normalized potential
        """
        if self.max_potential == self.min_potential:
            return Decimal('0')

        normalized = Decimal('2') * (potential - self.min_potential) / \
                    (self.max_potential - self.min_potential) - Decimal('1')

        return normalized

    def shape_reward(
        self,
        state_features: Dict[str, Decimal],
        next_state_features: Dict[str, Decimal],
        original_reward: Decimal,
        state_id: Optional[str] = None,
        next_state_id: Optional[str] = None
    ) -> Decimal:
        """Shape reward using potential-based reward shaping.

        Args:
            state_features: Current state features
            next_state_features: Next state features
            original_reward: Original reward signal
            state_id: Optional state identifier
            next_state_id: Optional next state identifier

        Returns:
            Shaped reward (Decimal)
        """
        try:
            # Calculate potentials
            phi_s = self.calculate_potential(state_features)
            phi_s_prime = self.calculate_potential(next_state_features)

            # Calculate shaping bonus: F = γ * Φ(s') - Φ(s)
            shaping_bonus = self.gamma * phi_s_prime - phi_s

            # Apply weight
            weighted_bonus = shaping_bonus * self.shaping_weight

            # Clip if configured
            if self.clip_shaping:
                weighted_bonus = max(
                    self.clip_min,
                    min(self.clip_max, weighted_bonus)
                )

            # Shaped reward
            shaped_reward = original_reward + weighted_bonus

            # Store potential history
            if state_id:
                self.potential_history.append(
                    ShapingPotential(
                        state_id=state_id,
                        potential=phi_s,
                        features=state_features
                    )
                )

            # Limit history size
            max_history = self.config.get('max_history_size', 10000)
            if len(self.potential_history) > max_history:
                self.potential_history = self.potential_history[-max_history:]

            logger.debug(
                "shaped_reward",
                original=str(original_reward),
                shaping_bonus=str(weighted_bonus),
                shaped=str(shaped_reward),
                phi_s=str(phi_s),
                phi_s_prime=str(phi_s_prime)
            )

            return shaped_reward

        except Exception as e:
            logger.error("failed_to_shape_reward", error=str(e))
            raise

    def shape_reward_batch(
        self,
        state_features_batch: List[Dict[str, Decimal]],
        next_state_features_batch: List[Dict[str, Decimal]],
        original_rewards_batch: List[Decimal]
    ) -> List[Decimal]:
        """Shape batch of rewards.

        Args:
            state_features_batch: Batch of state features
            next_state_features_batch: Batch of next state features
            original_rewards_batch: Batch of original rewards

        Returns:
            List of shaped rewards
        """
        try:
            shaped_rewards = []

            for state_feat, next_state_feat, original_reward in zip(
                state_features_batch,
                next_state_features_batch,
                original_rewards_batch
            ):
                shaped_reward = self.shape_reward(
                    state_feat,
                    next_state_feat,
                    original_reward
                )
                shaped_rewards.append(shaped_reward)

            logger.debug(
                "shaped_reward_batch",
                batch_size=len(shaped_rewards)
            )

            return shaped_rewards

        except Exception as e:
            logger.error("failed_to_shape_reward_batch", error=str(e))
            raise

    def update_goal(self, goal_features: Dict[str, Decimal]) -> None:
        """Update goal features for potential function.

        Args:
            goal_features: New goal feature values
        """
        self.goal_features = goal_features

        logger.info(
            "updated_goal_features",
            goal_features={k: str(v) for k, v in goal_features.items()}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get reward shaping statistics.

        Returns:
            Dictionary of statistics
        """
        if len(self.potential_history) > 0:
            potentials = [float(p.potential) for p in self.potential_history]
            mean_potential = Decimal(str(np.mean(potentials)))
            std_potential = Decimal(str(np.std(potentials)))
        else:
            mean_potential = Decimal('0')
            std_potential = Decimal('0')

        return {
            'potential_type': self.potential_type,
            'shaping_weight': str(self.shaping_weight),
            'gamma': str(self.gamma),
            'history_size': len(self.potential_history),
            'mean_potential': str(mean_potential),
            'std_potential': str(std_potential),
            'max_potential': str(self.max_potential),
            'min_potential': str(self.min_potential),
            'goal_features': {k: str(v) for k, v in self.goal_features.items()}
        }

    def reset(self) -> None:
        """Reset shaping state."""
        self.potential_history.clear()
        self.max_potential = Decimal('1.0')
        self.min_potential = Decimal('-1.0')

        logger.info("reset_reward_shaping")
