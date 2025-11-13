"""
Portfolio Management Environment for Reinforcement Learning.

This module provides a gymnasium-compatible environment for training
RL agents on portfolio management and asset allocation.
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


class AllocationAction(Enum):
    """Portfolio allocation action types."""
    REBALANCE = "rebalance"
    HOLD = "hold"
    INCREASE = "increase"
    DECREASE = "decrease"


@dataclass
class PortfolioState:
    """Current portfolio state.

    Attributes:
        timestamp: Current timestamp
        total_value: Total portfolio value
        cash: Available cash
        allocations: Dict of symbol -> allocation percentage
        positions: Dict of symbol -> quantity
        returns: Historical returns
        sharpe_ratio: Current Sharpe ratio
        max_drawdown: Maximum drawdown experienced
    """
    timestamp: datetime
    total_value: Decimal
    cash: Decimal
    allocations: Dict[str, Decimal]
    positions: Dict[str, Decimal]
    returns: List[Decimal]
    sharpe_ratio: Decimal
    max_drawdown: Decimal


class PortfolioEnv(gym.Env):
    """Portfolio management environment for RL.

    A production-ready environment for training portfolio management
    agents with realistic constraints and risk metrics.

    Attributes:
        config: Configuration dictionary
        symbols: List of tradeable symbols
        initial_balance: Starting portfolio balance
        rebalance_cost: Cost of rebalancing (as percentage)
        risk_free_rate: Risk-free rate for Sharpe calculation
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config: Dict[str, Any],
        market_data: pl.DataFrame,
        symbols: List[str]
    ) -> None:
        """Initialize portfolio management environment.

        Args:
            config: Configuration dictionary containing:
                - rl.portfolio_env.initial_balance
                - rl.portfolio_env.rebalance_cost_pct
                - rl.portfolio_env.risk_free_rate
                - rl.portfolio_env.lookback_window
                - rl.portfolio_env.max_position_pct
                - rl.portfolio_env.min_position_pct
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

        env_config = self.config["rl"]["portfolio_env"]
        self.initial_balance = Decimal(str(env_config["initial_balance"]))
        self.rebalance_cost = Decimal(str(env_config["rebalance_cost_pct"])) / Decimal("100")
        self.risk_free_rate = Decimal(str(env_config["risk_free_rate"]))
        self.lookback_window = env_config["lookback_window"]
        self.max_position = Decimal(str(env_config["max_position_pct"])) / Decimal("100")
        self.min_position = Decimal(str(env_config["min_position_pct"])) / Decimal("100")

        # State variables
        self.current_step = 0
        self.portfolio_state: Optional[PortfolioState] = None
        self.price_history: Dict[str, List[Decimal]] = {s: [] for s in symbols}
        self.portfolio_values: List[Decimal] = []
        self.peak_value = self.initial_balance

        # Define observation space
        # Features: price returns (lookback * num_assets), current allocations (num_assets),
        # portfolio metrics (sharpe, drawdown, total_return)
        num_assets = len(self.symbols)
        obs_dim = (
            self.lookback_window * num_assets +  # Price returns
            num_assets +  # Current allocations
            3  # Portfolio metrics (sharpe, drawdown, total_return)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: target allocation for each asset (0 to 1, sum to 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_assets,),
            dtype=np.float32
        )

        logger.info("portfolio_env_initialized",
                   symbols=self.symbols,
                   initial_balance=str(self.initial_balance),
                   observation_dim=obs_dim,
                   action_dim=num_assets)

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing
        """
        required_keys = [
            "rl.portfolio_env.initial_balance",
            "rl.portfolio_env.rebalance_cost_pct",
            "rl.portfolio_env.risk_free_rate",
            "rl.portfolio_env.lookback_window",
            "rl.portfolio_env.max_position_pct",
            "rl.portfolio_env.min_position_pct"
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
        required_cols = ["symbol", "timestamp", "close", "returns"]
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
        self.portfolio_values = []
        self.peak_value = self.initial_balance

        # Initialize portfolio with equal weights
        num_assets = len(self.symbols)
        initial_allocation = Decimal("1") / Decimal(str(num_assets))

        allocations = {symbol: initial_allocation for symbol in self.symbols}

        # Initialize portfolio state
        self.portfolio_state = PortfolioState(
            timestamp=datetime.utcnow(),
            total_value=self.initial_balance,
            cash=Decimal("0"),  # All invested initially
            allocations=allocations,
            positions={},
            returns=[],
            sharpe_ratio=Decimal("0"),
            max_drawdown=Decimal("0")
        )

        # Initialize positions based on allocations
        for symbol in self.symbols:
            price = self._get_current_price(symbol)
            position_value = self.initial_balance * allocations[symbol]
            self.portfolio_state.positions[symbol] = position_value / price if price > 0 else Decimal("0")

        # Initialize price history
        for symbol in self.symbols:
            symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
            prices = symbol_data.select("close").to_series().to_list()[:self.current_step]
            self.price_history[symbol] = [Decimal(str(p)) for p in prices]

        observation = self._get_observation()
        info = self._get_info()

        logger.debug("portfolio_env_reset", step=self.current_step)

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action array (target allocations for each asset)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.portfolio_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Normalize action to ensure sum = 1
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones_like(action) / len(action)

        # Apply position limits
        action = np.clip(action, float(self.min_position), float(self.max_position))

        # Renormalize after clipping
        action = action / np.sum(action)

        # Update prices
        self.current_step += 1
        self._update_prices()

        # Calculate current portfolio value before rebalancing
        current_value = self._calculate_portfolio_value()

        # Execute rebalancing
        target_allocations = {
            self.symbols[i]: Decimal(str(action[i]))
            for i in range(len(self.symbols))
        }

        rebalance_cost = self._rebalance_portfolio(target_allocations)

        # Calculate new portfolio value after rebalancing and costs
        new_value = self._calculate_portfolio_value() - rebalance_cost

        # Update portfolio state
        self.portfolio_state.total_value = new_value
        self.portfolio_state.allocations = target_allocations
        self.portfolio_values.append(new_value)

        # Calculate return
        if current_value > 0:
            period_return = (new_value - current_value) / current_value
        else:
            period_return = Decimal("0")

        self.portfolio_state.returns.append(period_return)

        # Update risk metrics
        self._update_risk_metrics()

        # Calculate reward
        reward = float(self._calculate_reward(period_return, rebalance_cost))

        # Get new observation
        observation = self._get_observation()

        # Check if episode is done
        max_steps = min(
            len(self.market_data.filter(pl.col("symbol") == s)) for s in self.symbols
        )
        terminated = self.current_step >= max_steps - 1

        # Check for truncation (excessive losses)
        truncated = self.portfolio_state.total_value <= self.initial_balance * Decimal("0.5")

        info = self._get_info()

        if terminated or truncated:
            final_return = (
                (self.portfolio_state.total_value - self.initial_balance) /
                self.initial_balance * Decimal("100")
            )
            logger.info("episode_finished",
                       final_return=str(final_return),
                       sharpe_ratio=str(self.portfolio_state.sharpe_ratio),
                       max_drawdown=str(self.portfolio_state.max_drawdown))

        return observation, reward, terminated, truncated, info

    def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        symbol_data = self.market_data.filter(pl.col("symbol") == symbol)
        price_row = symbol_data.slice(self.current_step, 1).select("close")

        if len(price_row) == 0:
            return self.price_history[symbol][-1] if self.price_history[symbol] else Decimal("0")

        return Decimal(str(price_row.item()))

    def _update_prices(self) -> None:
        """Update price history with current prices."""
        for symbol in self.symbols:
            price = self._get_current_price(symbol)
            self.price_history[symbol].append(price)

    def _calculate_portfolio_value(self) -> Decimal:
        """Calculate current portfolio value.

        Returns:
            Total portfolio value
        """
        if self.portfolio_state is None:
            return Decimal("0")

        total_value = self.portfolio_state.cash

        for symbol, quantity in self.portfolio_state.positions.items():
            if quantity > 0:
                price = self._get_current_price(symbol)
                total_value += quantity * price

        return total_value

    def _rebalance_portfolio(
        self,
        target_allocations: Dict[str, Decimal]
    ) -> Decimal:
        """Rebalance portfolio to target allocations.

        Args:
            target_allocations: Target allocation percentages

        Returns:
            Total rebalancing cost
        """
        if self.portfolio_state is None:
            return Decimal("0")

        current_value = self._calculate_portfolio_value()
        total_cost = Decimal("0")

        # Calculate target position values
        target_values = {
            symbol: current_value * allocation
            for symbol, allocation in target_allocations.items()
        }

        # Rebalance each position
        for symbol in self.symbols:
            current_position = self.portfolio_state.positions.get(symbol, Decimal("0"))
            current_price = self._get_current_price(symbol)
            current_position_value = current_position * current_price

            target_value = target_values[symbol]

            # Calculate trade size
            trade_value = abs(target_value - current_position_value)

            # Calculate transaction cost
            cost = trade_value * self.rebalance_cost
            total_cost += cost

            # Update position
            if current_price > 0:
                new_quantity = target_value / current_price
                self.portfolio_state.positions[symbol] = new_quantity

        return total_cost

    def _update_risk_metrics(self) -> None:
        """Update portfolio risk metrics."""
        if self.portfolio_state is None or len(self.portfolio_state.returns) < 2:
            return

        # Calculate Sharpe ratio
        returns_array = np.array([float(r) for r in self.portfolio_state.returns])
        avg_return = Decimal(str(np.mean(returns_array)))
        std_return = Decimal(str(np.std(returns_array)))

        if std_return > 0:
            self.portfolio_state.sharpe_ratio = (
                (avg_return - self.risk_free_rate) / std_return
            )
        else:
            self.portfolio_state.sharpe_ratio = Decimal("0")

        # Calculate maximum drawdown
        if self.portfolio_state.total_value > self.peak_value:
            self.peak_value = self.portfolio_state.total_value

        drawdown = (self.peak_value - self.portfolio_state.total_value) / self.peak_value

        if drawdown > self.portfolio_state.max_drawdown:
            self.portfolio_state.max_drawdown = drawdown

    def _calculate_reward(
        self,
        period_return: Decimal,
        rebalance_cost: Decimal
    ) -> Decimal:
        """Calculate reward for the step.

        Args:
            period_return: Return for this period
            rebalance_cost: Cost of rebalancing

        Returns:
            Reward value
        """
        # Base reward: period return
        reward = period_return * Decimal("100")

        # Penalty for rebalancing costs
        cost_penalty = rebalance_cost / self.portfolio_state.total_value * Decimal("100")
        reward -= cost_penalty

        # Bonus for good Sharpe ratio
        if len(self.portfolio_state.returns) >= 10:
            sharpe_bonus = self.portfolio_state.sharpe_ratio * Decimal("0.1")
            reward += sharpe_bonus

        # Penalty for high drawdown
        drawdown_penalty = self.portfolio_state.max_drawdown * Decimal("10")
        reward -= drawdown_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array
        """
        if self.portfolio_state is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        observation = []

        # Price returns for each asset
        for symbol in self.symbols:
            if len(self.price_history[symbol]) >= self.lookback_window + 1:
                prices = self.price_history[symbol][-self.lookback_window-1:]
                returns = [
                    float((prices[i+1] - prices[i]) / prices[i]) if prices[i] > 0 else 0.0
                    for i in range(len(prices) - 1)
                ]
                observation.extend(returns)
            else:
                observation.extend([0.0] * self.lookback_window)

        # Current allocations
        for symbol in self.symbols:
            allocation = self.portfolio_state.allocations.get(symbol, Decimal("0"))
            observation.append(float(allocation))

        # Portfolio metrics
        observation.append(float(self.portfolio_state.sharpe_ratio))
        observation.append(float(self.portfolio_state.max_drawdown))

        # Total return
        total_return = (
            (self.portfolio_state.total_value - self.initial_balance) /
            self.initial_balance
        )
        observation.append(float(total_return))

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
            "allocations": {k: str(v) for k, v in self.portfolio_state.allocations.items()},
            "sharpe_ratio": str(self.portfolio_state.sharpe_ratio),
            "max_drawdown": str(self.portfolio_state.max_drawdown),
            "total_return_pct": str(
                (self.portfolio_state.total_value - self.initial_balance) /
                self.initial_balance * Decimal("100")
            )
        }

    def render(self) -> None:
        """Render environment state."""
        if self.portfolio_state is None:
            return

        logger.info("portfolio_state",
                   value=str(self.portfolio_state.total_value),
                   sharpe=str(self.portfolio_state.sharpe_ratio),
                   drawdown=str(self.portfolio_state.max_drawdown))

    def close(self) -> None:
        """Close environment and clean up resources."""
        logger.info("portfolio_env_closed")
