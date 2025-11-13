"""
Multi-Asset Trading Environment for Reinforcement Learning.

This module provides a gymnasium-compatible environment for training
RL agents on multi-asset portfolio management.
"""

import asyncio
from decimal import Decimal
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import polars as pl
import gymnasium as gym
from gymnasium import spaces
from structlog import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """Trading action types."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class PortfolioState:
    """Current portfolio state.

    Attributes:
        cash: Available cash in base currency
        positions: Dict of symbol -> quantity
        total_value: Total portfolio value
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss
        num_trades: Number of trades executed
    """
    cash: Decimal
    positions: Dict[str, Decimal]
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    num_trades: int


class MultiAssetEnv(gym.Env):
    """Multi-asset trading environment for RL.

    A production-ready trading environment that supports multiple assets,
    realistic transaction costs, and comprehensive state representation.

    Attributes:
        config: Configuration dictionary
        symbols: List of trading symbols
        initial_balance: Starting portfolio balance
        transaction_cost: Transaction cost rate
        max_position_size: Maximum position size per asset
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        config: Dict[str, Any],
        market_data: pl.DataFrame,
        symbols: List[str]
    ) -> None:
        """Initialize multi-asset trading environment.

        Args:
            config: Configuration dictionary containing:
                - rl.env.initial_balance
                - rl.env.transaction_cost_rate
                - rl.env.max_position_size_pct
                - rl.env.lookback_window
                - rl.env.features_per_asset
            market_data: Historical market data (Polars DataFrame)
            symbols: List of tradeable symbols

        Raises:
            ValueError: If configuration or data is invalid
        """
        super().__init__()

        self.config = config
        self.symbols = symbols
        self.market_data = market_data
        self._validate_config()
        self._validate_data()

        env_config = self.config["rl"]["env"]
        self.initial_balance = Decimal(str(env_config["initial_balance"]))
        self.transaction_cost = Decimal(str(env_config["transaction_cost_rate"]))
        self.max_position_pct = Decimal(str(env_config["max_position_size_pct"]))
        self.lookback_window = env_config["lookback_window"]
        self.features_per_asset = env_config["features_per_asset"]

        # State variables
        self.current_step = 0
        self.portfolio_state: Optional[PortfolioState] = None
        self.price_history: Dict[str, List[Decimal]] = {s: [] for s in symbols}
        self.episode_trades: List[Dict[str, Any]] = []

        # Define observation and action spaces
        num_assets = len(self.symbols)
        obs_dim = (
            self.lookback_window * self.features_per_asset * num_assets +  # Market features
            num_assets +  # Current positions
            1  # Cash ratio
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: [action_type, symbol_index, position_size] for each asset
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # position_size: 0.0 to 1.0 (fraction of available capital)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0] * num_assets),
            high=np.array([2, num_assets - 1, 1] * num_assets),
            dtype=np.float32
        )

        logger.info("multi_asset_env_initialized",
                   symbols=self.symbols,
                   initial_balance=str(self.initial_balance),
                   observation_dim=obs_dim,
                   action_dim=self.action_space.shape[0])

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = [
            "rl.env.initial_balance",
            "rl.env.transaction_cost_rate",
            "rl.env.max_position_size_pct",
            "rl.env.lookback_window",
            "rl.env.features_per_asset"
        ]

        for key in required_keys:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if part not in current:
                    raise ValueError(f"Missing required config key: {key}")
                current = current[part]

    def _validate_data(self) -> None:
        """Validate market data.

        Raises:
            ValueError: If market data is invalid
        """
        required_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in self.market_data.columns:
                raise ValueError(f"Required column {col} not found in market data")

        for symbol in self.symbols:
            symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
            if len(symbol_data) == 0:
                raise ValueError(f"No data found for symbol: {symbol}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.episode_trades = []

        # Initialize portfolio
        self.portfolio_state = PortfolioState(
            cash=self.initial_balance,
            positions={symbol: Decimal("0") for symbol in self.symbols},
            total_value=self.initial_balance,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            num_trades=0
        )

        # Initialize price history
        for symbol in self.symbols:
            symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
            prices = symbol_data.select("close").to_series().to_list()[:self.current_step]
            self.price_history[symbol] = [Decimal(str(p)) for p in prices]

        observation = self._get_observation()
        info = self._get_info()

        logger.debug("environment_reset", step=self.current_step)

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action array from agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.portfolio_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Parse and execute actions
        rewards = []
        for i in range(len(self.symbols)):
            action_idx = i * 3
            action_type = int(action[action_idx])
            symbol_idx = int(action[action_idx + 1])
            position_size = float(action[action_idx + 2])

            if symbol_idx < len(self.symbols):
                symbol = self.symbols[symbol_idx]
                reward = self._execute_action(symbol, action_type, position_size)
                rewards.append(reward)

        # Update state
        self.current_step += 1
        self._update_portfolio_value()

        # Calculate total reward
        total_reward = float(sum(rewards))

        # Get new observation
        observation = self._get_observation()

        # Check if episode is done
        max_steps = len(self.market_data.filter(pl.col("symbol") == self.symbols[0]))
        terminated = self.current_step >= max_steps - 1
        truncated = self.portfolio_state.total_value <= self.initial_balance * Decimal("0.5")

        info = self._get_info()

        if terminated or truncated:
            logger.info("episode_finished",
                       total_return=str(self._calculate_return()),
                       num_trades=self.portfolio_state.num_trades)

        return observation, total_reward, terminated, truncated, info

    def _execute_action(
        self,
        symbol: str,
        action_type: int,
        position_size: float
    ) -> Decimal:
        """Execute trading action.

        Args:
            symbol: Trading symbol
            action_type: Action type (0=HOLD, 1=BUY, 2=SELL)
            position_size: Position size (0.0 to 1.0)

        Returns:
            Immediate reward for this action
        """
        if self.portfolio_state is None:
            return Decimal("0")

        # Get current price
        current_price = self._get_current_price(symbol)

        reward = Decimal("0")
        position_size_decimal = Decimal(str(position_size))

        try:
            if action_type == ActionType.BUY.value:
                # Calculate maximum buyable quantity
                max_value = self.portfolio_state.cash * self.max_position_pct
                max_quantity = (max_value / current_price) * position_size_decimal

                if max_quantity > 0:
                    cost = max_quantity * current_price
                    transaction_fee = cost * self.transaction_cost

                    if self.portfolio_state.cash >= (cost + transaction_fee):
                        # Execute buy
                        self.portfolio_state.cash -= (cost + transaction_fee)
                        self.portfolio_state.positions[symbol] += max_quantity
                        self.portfolio_state.num_trades += 1

                        self.episode_trades.append({
                            "step": self.current_step,
                            "symbol": symbol,
                            "action": "BUY",
                            "quantity": str(max_quantity),
                            "price": str(current_price),
                            "cost": str(cost),
                            "fee": str(transaction_fee)
                        })

                        # Reward for successful trade (will be refined by portfolio value change)
                        reward = Decimal("0.01")

            elif action_type == ActionType.SELL.value:
                current_position = self.portfolio_state.positions.get(symbol, Decimal("0"))

                if current_position > 0:
                    sell_quantity = current_position * position_size_decimal

                    if sell_quantity > 0:
                        proceeds = sell_quantity * current_price
                        transaction_fee = proceeds * self.transaction_cost

                        # Execute sell
                        self.portfolio_state.cash += (proceeds - transaction_fee)
                        self.portfolio_state.positions[symbol] -= sell_quantity
                        self.portfolio_state.num_trades += 1

                        self.episode_trades.append({
                            "step": self.current_step,
                            "symbol": symbol,
                            "action": "SELL",
                            "quantity": str(sell_quantity),
                            "price": str(current_price),
                            "proceeds": str(proceeds),
                            "fee": str(transaction_fee)
                        })

                        reward = Decimal("0.01")

        except Exception as e:
            logger.error("action_execution_failed",
                       symbol=symbol,
                       action_type=action_type,
                       error=str(e))
            reward = Decimal("-0.1")  # Penalty for failed action

        return reward

    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price as Decimal
        """
        symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
        price_row = symbol_data.slice(self.current_step, 1).select("close")

        if len(price_row) == 0:
            # Return last known price
            return self.price_history[symbol][-1] if self.price_history[symbol] else Decimal("0")

        return Decimal(str(price_row.item()))

    def _update_portfolio_value(self) -> None:
        """Update current portfolio value and P&L."""
        if self.portfolio_state is None:
            return

        total_value = self.portfolio_state.cash

        for symbol, quantity in self.portfolio_state.positions.items():
            if quantity > 0:
                current_price = self._get_current_price(symbol)
                total_value += quantity * current_price

        self.portfolio_state.total_value = total_value
        self.portfolio_state.unrealized_pnl = total_value - self.initial_balance

    def _calculate_return(self) -> Decimal:
        """Calculate portfolio return.

        Returns:
            Portfolio return as percentage
        """
        if self.portfolio_state is None:
            return Decimal("0")

        return (
            (self.portfolio_state.total_value - self.initial_balance) /
            self.initial_balance * Decimal("100")
        )

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array
        """
        if self.portfolio_state is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        observation = []

        # Market features for each asset
        for symbol in self.symbols:
            symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
            window_data = symbol_data.slice(
                self.current_step - self.lookback_window,
                self.lookback_window
            )

            # Extract features (normalized)
            if len(window_data) > 0:
                features = window_data.select([
                    "open", "high", "low", "close", "volume"
                ]).to_numpy()

                # Normalize features
                features_flat = features.flatten()
                if features_flat.max() > 0:
                    features_flat = features_flat / features_flat.max()

                observation.extend(features_flat.tolist())
            else:
                observation.extend([0.0] * (self.lookback_window * self.features_per_asset))

        # Portfolio state features
        for symbol in self.symbols:
            position = self.portfolio_state.positions.get(symbol, Decimal("0"))
            position_value = position * self._get_current_price(symbol)
            position_ratio = float(position_value / self.portfolio_state.total_value) if self.portfolio_state.total_value > 0 else 0.0
            observation.append(position_ratio)

        # Cash ratio
        cash_ratio = float(self.portfolio_state.cash / self.portfolio_state.total_value) if self.portfolio_state.total_value > 0 else 0.0
        observation.append(cash_ratio)

        return np.array(observation, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info.

        Returns:
            Info dictionary
        """
        if self.portfolio_state is None:
            return {}

        return {
            "step": self.current_step,
            "portfolio_value": str(self.portfolio_state.total_value),
            "cash": str(self.portfolio_state.cash),
            "positions": {k: str(v) for k, v in self.portfolio_state.positions.items()},
            "return_pct": str(self._calculate_return()),
            "num_trades": self.portfolio_state.num_trades,
            "unrealized_pnl": str(self.portfolio_state.unrealized_pnl)
        }

    def render(self) -> Optional[np.ndarray]:
        """Render environment state.

        Returns:
            Rendered frame (for rgb_array mode) or None
        """
        if self.portfolio_state is None:
            return None

        logger.info("portfolio_state",
                   value=str(self.portfolio_state.total_value),
                   return_pct=str(self._calculate_return()),
                   num_trades=self.portfolio_state.num_trades)

        return None

    def close(self) -> None:
        """Close environment and clean up resources."""
        logger.info("environment_closed")
