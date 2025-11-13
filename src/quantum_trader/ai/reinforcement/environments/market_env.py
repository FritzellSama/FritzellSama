"""
Market Environment for Reinforcement Learning Trading.

This module provides a gym-like environment for training RL agents
on market data with realistic trading dynamics and constraints.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import polars as pl
from structlog import get_logger

from quantum_trader.exceptions import EnvironmentError, ValidationError

logger = get_logger(__name__)


class ActionType(Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketState:
    """Current market state.

    Attributes:
        timestamp: Current timestamp
        prices: Price data [num_assets,]
        volumes: Volume data [num_assets,]
        indicators: Technical indicators
        portfolio_value: Current portfolio value
        positions: Current positions [num_assets,]
        cash: Available cash
        metadata: Additional state metadata
    """
    timestamp: datetime
    prices: np.ndarray
    volumes: np.ndarray
    indicators: Dict[str, np.ndarray]
    portfolio_value: Decimal
    positions: np.ndarray
    cash: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConfig:
    """Configuration for market environment.

    Attributes:
        data_path: Path to market data
        assets: List of asset symbols
        initial_cash: Initial cash balance
        transaction_cost: Transaction cost rate
        slippage: Slippage rate
        max_position_size: Maximum position size per asset
        lookback_window: Number of historical steps for state
        reward_scaling: Scaling factor for rewards
        normalize_obs: Whether to normalize observations
    """
    data_path: str
    assets: List[str]
    initial_cash: Decimal
    transaction_cost: Decimal = Decimal("0.001")
    slippage: Decimal = Decimal("0.0005")
    max_position_size: Decimal = Decimal("0.3")
    lookback_window: int = 20
    reward_scaling: Decimal = Decimal("1.0")
    normalize_obs: bool = True


class MarketEnvironment:
    """Trading environment for reinforcement learning.

    Provides a gym-like interface for training RL agents with
    realistic market dynamics, transaction costs, and constraints.

    Attributes:
        config: Environment configuration
        data: Market data
        current_step: Current time step
        state: Current market state
        done: Whether episode is done
        observation_space_dim: Observation space dimension
        action_space_dim: Action space dimension

    Example:
        >>> config = MarketConfig(
        ...     data_path="/data/market.csv",
        ...     assets=["BTCUSDT", "ETHUSDT"],
        ...     initial_cash=Decimal("100000"),
        ...     transaction_cost=Decimal("0.001"),
        ...     slippage=Decimal("0.0005")
        ... )
        >>> env = MarketEnvironment(config)
        >>> state = env.reset()
        >>> action = agent.select_action(state)
        >>> next_state, reward, done, info = env.step(action)
    """

    def __init__(self, config: MarketConfig) -> None:
        """Initialize market environment.

        Args:
            config: Environment configuration

        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self._validate_config()

        self.num_assets = len(config.assets)
        self.current_step = 0
        self.done = False

        # Load data
        self.data = self._load_data()
        self.data_length = len(self.data)

        # Initialize state
        self.state: Optional[MarketState] = None
        self.initial_cash = config.initial_cash
        self.cash = config.initial_cash
        self.positions = np.zeros(self.num_assets)
        self.portfolio_value = config.initial_cash

        # Transaction tracking
        self.total_trades = 0
        self.total_profit = Decimal("0")
        self.episode_return = Decimal("0")

        # Observation and action spaces
        self.observation_space_dim = self._calculate_obs_dim()
        self.action_space_dim = self.num_assets  # Continuous action per asset

        # Normalization parameters
        if config.normalize_obs:
            self._compute_normalization_params()

        logger.info(
            "Market environment initialized",
            num_assets=self.num_assets,
            data_length=self.data_length,
            obs_dim=self.observation_space_dim,
            action_dim=self.action_space_dim
        )

    def _validate_config(self) -> None:
        """Validate environment configuration.

        Raises:
            ValidationError: If configuration is invalid
        """
        if not self.config.assets:
            raise ValidationError("assets cannot be empty")

        if self.config.initial_cash <= 0:
            raise ValidationError("initial_cash must be > 0")

        if self.config.transaction_cost < 0:
            raise ValidationError("transaction_cost must be >= 0")

        if self.config.slippage < 0:
            raise ValidationError("slippage must be >= 0")

        if self.config.lookback_window < 1:
            raise ValidationError("lookback_window must be >= 1")

    def _load_data(self) -> pl.DataFrame:
        """Load market data.

        Returns:
            Market data as Polars DataFrame

        Raises:
            EnvironmentError: If data loading fails
        """
        try:
            # Load from config path
            data = pl.read_csv(self.config.data_path)

            # Validate required columns
            required_cols = ["timestamp"]
            for asset in self.config.assets:
                required_cols.extend([
                    f"{asset}_price",
                    f"{asset}_volume"
                ])

            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")

            logger.info("Market data loaded", num_rows=len(data), num_cols=len(data.columns))

            return data

        except Exception as e:
            logger.error("Failed to load market data", error=str(e))
            raise EnvironmentError(f"Failed to load market data: {e}") from e

    def _calculate_obs_dim(self) -> int:
        """Calculate observation space dimension.

        Returns:
            Observation dimension
        """
        # Base: prices, volumes, positions, cash for each asset
        base_dim = self.num_assets * 2  # prices + volumes

        # Portfolio state
        portfolio_dim = self.num_assets + 2  # positions + cash + portfolio_value

        # Historical window
        window_dim = base_dim * self.config.lookback_window

        total_dim = base_dim + portfolio_dim + window_dim

        return total_dim

    def _compute_normalization_params(self) -> None:
        """Compute normalization parameters from data."""
        try:
            # Compute mean and std for each feature
            self.norm_mean = {}
            self.norm_std = {}

            for asset in self.config.assets:
                price_col = f"{asset}_price"
                volume_col = f"{asset}_volume"

                if price_col in self.data.columns:
                    self.norm_mean[price_col] = self.data[price_col].mean()
                    self.norm_std[price_col] = self.data[price_col].std()

                if volume_col in self.data.columns:
                    self.norm_mean[volume_col] = self.data[volume_col].mean()
                    self.norm_std[volume_col] = self.data[volume_col].std()

            logger.debug("Normalization parameters computed")

        except Exception as e:
            logger.warning("Failed to compute normalization params", error=str(e))
            self.config.normalize_obs = False

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial observation

        Raises:
            EnvironmentError: If reset fails
        """
        try:
            self.current_step = self.config.lookback_window
            self.done = False

            self.cash = self.initial_cash
            self.positions = np.zeros(self.num_assets)
            self.portfolio_value = self.initial_cash

            self.total_trades = 0
            self.total_profit = Decimal("0")
            self.episode_return = Decimal("0")

            # Get initial state
            self.state = self._get_state()

            observation = self._get_observation()

            logger.debug("Environment reset", step=self.current_step)

            return observation

        except Exception as e:
            logger.error("Reset failed", error=str(e))
            raise EnvironmentError(f"Reset failed: {e}") from e

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, Decimal, bool, Dict[str, Any]]:
        """Execute one time step.

        Args:
            action: Trading action [num_assets,] in range [-1, 1]
                   Negative = sell, Positive = buy, 0 = hold

        Returns:
            Tuple of (observation, reward, done, info)

        Raises:
            EnvironmentError: If step fails
        """
        try:
            if self.done:
                raise EnvironmentError("Episode is done, call reset()")

            # Validate action
            self._validate_action(action)

            # Execute trades
            self._execute_trades(action)

            # Move to next step
            self.current_step += 1

            # Check if episode is done
            if self.current_step >= self.data_length - 1:
                self.done = True

            # Update state
            self.state = self._get_state()

            # Calculate reward
            reward = self._calculate_reward()
            self.episode_return += reward

            # Get observation
            observation = self._get_observation()

            # Compile info
            info = {
                "portfolio_value": float(self.portfolio_value),
                "cash": float(self.cash),
                "positions": self.positions.tolist(),
                "total_trades": self.total_trades,
                "episode_return": float(self.episode_return)
            }

            return observation, reward, self.done, info

        except Exception as e:
            logger.error("Step failed", error=str(e))
            raise EnvironmentError(f"Step failed: {e}") from e

    def _execute_trades(self, action: np.ndarray) -> None:
        """Execute trading actions.

        Args:
            action: Trading action array
        """
        for i, asset in enumerate(self.config.assets):
            action_value = action[i]

            # Get current price
            price_col = f"{asset}_price"
            current_price = Decimal(str(self.data[price_col][self.current_step]))

            # Apply slippage
            if action_value > 0:  # Buy
                execution_price = current_price * (Decimal("1") + self.config.slippage)
            elif action_value < 0:  # Sell
                execution_price = current_price * (Decimal("1") - self.config.slippage)
            else:
                continue  # Hold

            # Calculate trade size
            max_position = float(self.config.max_position_size) * float(self.portfolio_value)
            trade_value = abs(action_value) * max_position

            if action_value > 0:  # Buy
                # Check if we have enough cash
                cost = Decimal(str(trade_value)) * (Decimal("1") + self.config.transaction_cost)
                if cost <= self.cash:
                    shares = Decimal(str(trade_value)) / execution_price
                    self.positions[i] += float(shares)
                    self.cash -= cost
                    self.total_trades += 1

            elif action_value < 0:  # Sell
                # Check if we have enough shares
                max_sell = abs(self.positions[i])
                if max_sell > 0:
                    shares = min(
                        Decimal(str(trade_value)) / execution_price,
                        Decimal(str(max_sell))
                    )
                    proceeds = shares * execution_price * (Decimal("1") - self.config.transaction_cost)
                    self.positions[i] -= float(shares)
                    self.cash += proceeds
                    self.total_trades += 1

    def _calculate_reward(self) -> Decimal:
        """Calculate reward for current step.

        Returns:
            Reward value
        """
        # Calculate new portfolio value
        new_portfolio_value = self.cash

        for i, asset in enumerate(self.config.assets):
            price_col = f"{asset}_price"
            current_price = Decimal(str(self.data[price_col][self.current_step]))
            position_value = Decimal(str(self.positions[i])) * current_price
            new_portfolio_value += position_value

        # Reward is change in portfolio value
        reward = (new_portfolio_value - self.portfolio_value) / self.initial_cash
        reward = reward * self.config.reward_scaling

        self.portfolio_value = new_portfolio_value

        return reward

    def _get_state(self) -> MarketState:
        """Get current market state.

        Returns:
            Market state object
        """
        # Get current prices and volumes
        prices = np.array([
            self.data[f"{asset}_price"][self.current_step]
            for asset in self.config.assets
        ])

        volumes = np.array([
            self.data[f"{asset}_volume"][self.current_step]
            for asset in self.config.assets
        ])

        # Get timestamp
        timestamp = self.data["timestamp"][self.current_step]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        state = MarketState(
            timestamp=timestamp,
            prices=prices,
            volumes=volumes,
            indicators={},
            portfolio_value=self.portfolio_value,
            positions=self.positions.copy(),
            cash=self.cash
        )

        return state

    def _get_observation(self) -> np.ndarray:
        """Get observation from current state.

        Returns:
            Observation array
        """
        obs_list = []

        # Current prices and volumes
        obs_list.append(self.state.prices)
        obs_list.append(self.state.volumes)

        # Historical window
        start_idx = max(0, self.current_step - self.config.lookback_window)
        for step in range(start_idx, self.current_step):
            prices = np.array([
                self.data[f"{asset}_price"][step]
                for asset in self.config.assets
            ])
            volumes = np.array([
                self.data[f"{asset}_volume"][step]
                for asset in self.config.assets
            ])
            obs_list.append(prices)
            obs_list.append(volumes)

        # Pad if necessary
        if self.current_step < self.config.lookback_window:
            padding_needed = self.config.lookback_window - self.current_step
            for _ in range(padding_needed):
                obs_list.append(np.zeros(self.num_assets))
                obs_list.append(np.zeros(self.num_assets))

        # Portfolio state
        obs_list.append(self.state.positions)
        obs_list.append(np.array([float(self.state.cash)]))
        obs_list.append(np.array([float(self.state.portfolio_value)]))

        observation = np.concatenate(obs_list)

        # Normalize if configured
        if self.config.normalize_obs:
            observation = self._normalize_observation(observation)

        return observation

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation.

        Args:
            obs: Raw observation

        Returns:
            Normalized observation
        """
        # Simple standardization
        mean = np.mean(obs)
        std = np.std(obs)
        if std > 0:
            obs = (obs - mean) / std

        return obs

    def _validate_action(self, action: np.ndarray) -> None:
        """Validate trading action.

        Args:
            action: Action array

        Raises:
            ValidationError: If action is invalid
        """
        if action is None or len(action) == 0:
            raise ValidationError("Action cannot be empty")

        if len(action) != self.num_assets:
            raise ValidationError(
                f"Action dimension mismatch: expected {self.num_assets}, got {len(action)}"
            )

        if np.any(np.isnan(action)):
            raise ValidationError("Action contains NaN values")

        if np.any(np.isinf(action)):
            raise ValidationError("Action contains Inf values")

    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "current_step": self.current_step,
            "done": self.done,
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "total_trades": self.total_trades,
            "total_profit": float(self.total_profit),
            "episode_return": float(self.episode_return),
            "positions": self.positions.tolist()
        }

    def render(self) -> str:
        """Render current environment state.

        Returns:
            State description string
        """
        if self.state is None:
            return "Environment not initialized"

        output = f"Step: {self.current_step}\n"
        output += f"Portfolio Value: ${float(self.portfolio_value):.2f}\n"
        output += f"Cash: ${float(self.cash):.2f}\n"
        output += f"Positions: {self.positions}\n"
        output += f"Total Trades: {self.total_trades}\n"
        output += f"Episode Return: {float(self.episode_return):.4f}\n"

        return output
