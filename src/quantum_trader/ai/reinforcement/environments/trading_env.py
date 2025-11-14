"""
Trading Environment for Reinforcement Learning

Production-ready gym-style trading environment for RL agent training.
Simulates realistic market conditions with proper execution, fees, and slippage.
"""

import asyncio
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import deque
from enum import Enum
import os

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

# Set high precision
getcontext().prec = 28


class SignalAction(Enum):
    """Trading actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class TradingEnvironment:
    """
    Gym-style trading environment for RL.

    Simulates realistic trading with proper accounting for fees,
    slippage, and market dynamics.

    Attributes:
        config: Configuration dictionary
        initial_capital: Starting capital
        current_capital: Current available capital
        position: Current position (Decimal)
        position_price: Entry price of current position
        transaction_cost: Fee per trade (fraction)

    Example:
        >>> config = {"environment": {"initial_capital": "10000", "fee": "0.001"}}
        >>> env = TradingEnvironment(config, market_data_df)
        >>> state = env.reset()
        >>> next_state, reward, done, info = await env.step(action)
    """

    def __init__(self, config: Dict[str, Any], market_data: pl.DataFrame) -> None:
        """
        Initialize trading environment.

        Args:
            config: Configuration dictionary
            market_data: Historical market data DataFrame
                Required columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If configuration or data is invalid
        """
        self.config = config
        self.market_data = market_data
        self._validate_config()
        self._validate_data()

        env_config = self.config.get("environment", {})

        # Capital and risk parameters
        self.initial_capital: Decimal = Decimal(
            str(env_config.get("initial_capital", os.getenv("ENV_INITIAL_CAPITAL", "10000")))
        )
        self.transaction_cost: Decimal = Decimal(
            str(env_config.get("fee", os.getenv("ENV_TRANSACTION_FEE", "0.001")))
        )
        self.slippage: Decimal = Decimal(
            str(env_config.get("slippage", os.getenv("ENV_SLIPPAGE", "0.0005")))
        )

        # Position sizing
        self.max_position_size: Decimal = Decimal(
            str(env_config.get("max_position_size", os.getenv("ENV_MAX_POSITION_SIZE", "1.0")))
        )
        self.position_fraction: Decimal = Decimal(
            str(env_config.get("position_fraction", os.getenv("ENV_POSITION_FRACTION", "0.95")))
        )

        # State configuration
        self.lookback_window: int = env_config.get("lookback_window", int(os.getenv("ENV_LOOKBACK_WINDOW", "20")))
        self.reward_scaling: Decimal = Decimal(
            str(env_config.get("reward_scaling", os.getenv("ENV_REWARD_SCALING", "1.0")))
        )

        # Environment state
        self.current_step: int = 0
        self.current_capital: Decimal = self.initial_capital
        self.position: Decimal = Decimal("0")
        self.position_price: Decimal = Decimal("0")
        self.total_trades: int = 0
        self.winning_trades: int = 0

        # History tracking
        self.equity_history: List[Decimal] = []
        self.action_history: List[SignalAction] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Episode management
        self.done: bool = False
        self.max_steps: int = len(self.market_data) - self.lookback_window - 1

        logger.info(
            "trading_environment_initialized",
            initial_capital=str(self.initial_capital),
            max_steps=self.max_steps,
            transaction_cost=str(self.transaction_cost)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        env_config = self.config.get("environment", {})

        if env_config:
            initial_capital = env_config.get("initial_capital", "10000")
            if Decimal(str(initial_capital)) <= 0:
                raise ValueError("initial_capital must be positive")

    def _validate_data(self) -> None:
        """Validate market data."""
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in self.market_data.columns for col in required_cols):
            raise ValueError(f"market_data missing required columns: {required_cols}")

        if len(self.market_data) == 0:
            raise ValueError("market_data cannot be empty")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state observation

        Example:
            >>> state = env.reset()
        """
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.position = Decimal("0")
        self.position_price = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0
        self.done = False

        self.equity_history = [self.initial_capital]
        self.action_history = []
        self.trade_history = []

        logger.info("environment_reset", initial_capital=str(self.initial_capital))

        return self._get_observation()

    async def step(self, action: int) -> Tuple[np.ndarray, Decimal, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL, 3=CLOSE)

        Returns:
            Tuple of (next_state, reward, done, info)

        Raises:
            ValueError: If environment is done or action is invalid

        Example:
            >>> next_state, reward, done, info = await env.step(action=1)
        """
        try:
            if self.done:
                raise ValueError("Environment is done. Call reset() first.")

            if action not in [0, 1, 2, 3]:
                raise ValueError("Action must be 0 (HOLD), 1 (BUY), 2 (SELL), or 3 (CLOSE)")

            # Map action to enum
            action_enum = [SignalAction.HOLD, SignalAction.BUY, SignalAction.SELL, SignalAction.CLOSE][action]

            # Get current price
            current_price = self._get_current_price()

            # Execute action
            reward = await self._execute_action(action_enum, current_price)

            # Update state
            self.current_step += 1
            self.action_history.append(action_enum)

            # Calculate current equity
            equity = self._calculate_equity(current_price)
            self.equity_history.append(equity)

            # Check if done
            if self.current_step >= self.max_steps:
                self.done = True
                # Close any open position at end
                if self.position != Decimal("0"):
                    await self._close_position(current_price)

            # Get next observation
            next_state = self._get_observation()

            # Prepare info dict
            info = self._get_info()

            logger.debug(
                "environment_step",
                step=self.current_step,
                action=action_enum.value,
                reward=str(reward),
                equity=str(equity),
                done=self.done
            )

            return next_state, reward, self.done, info

        except Exception as e:
            logger.error("environment_step_failed", error=str(e))
            raise

    async def _execute_action(self, action: SignalAction, price: Decimal) -> Decimal:
        """Execute trading action and return reward."""
        reward = Decimal("0")

        if action == SignalAction.BUY:
            if self.position <= Decimal("0"):  # Can only buy if not long
                reward = await self._open_long_position(price)
        elif action == SignalAction.SELL:
            if self.position >= Decimal("0"):  # Can only sell if not short
                reward = await self._open_short_position(price)
        elif action == SignalAction.CLOSE:
            if self.position != Decimal("0"):
                reward = await self._close_position(price)
        else:  # HOLD
            # Small penalty for holding to encourage action
            reward = Decimal("-0.001") * self.reward_scaling

        return reward

    async def _open_long_position(self, price: Decimal) -> Decimal:
        """Open a long position."""
        # Close any existing short position first
        if self.position < Decimal("0"):
            await self._close_position(price)

        # Calculate position size
        position_value = self.current_capital * self.position_fraction
        position_size = position_value / price

        # Apply slippage and fees
        execution_price = price * (Decimal("1") + self.slippage)
        fee = position_value * self.transaction_cost
        total_cost = position_value + fee

        if total_cost > self.current_capital:
            # Insufficient capital
            return Decimal("-0.01") * self.reward_scaling

        # Execute trade
        self.position = position_size
        self.position_price = execution_price
        self.current_capital -= total_cost
        self.total_trades += 1

        self.trade_history.append({
            "step": self.current_step,
            "action": "OPEN_LONG",
            "price": execution_price,
            "size": position_size,
            "fee": fee
        })

        logger.debug(
            "long_position_opened",
            price=str(execution_price),
            size=str(position_size),
            capital_remaining=str(self.current_capital)
        )

        return Decimal("0")  # Reward comes from closing position

    async def _open_short_position(self, price: Decimal) -> Decimal:
        """Open a short position."""
        # Close any existing long position first
        if self.position > Decimal("0"):
            await self._close_position(price)

        # Calculate position size
        position_value = self.current_capital * self.position_fraction
        position_size = position_value / price

        # Apply slippage and fees
        execution_price = price * (Decimal("1") - self.slippage)
        fee = position_value * self.transaction_cost

        # For short, we receive capital
        self.position = -position_size
        self.position_price = execution_price
        self.current_capital += position_value - fee
        self.total_trades += 1

        self.trade_history.append({
            "step": self.current_step,
            "action": "OPEN_SHORT",
            "price": execution_price,
            "size": position_size,
            "fee": fee
        })

        logger.debug(
            "short_position_opened",
            price=str(execution_price),
            size=str(position_size),
            capital_after=str(self.current_capital)
        )

        return Decimal("0")

    async def _close_position(self, price: Decimal) -> Decimal:
        """Close current position and return reward."""
        if self.position == Decimal("0"):
            return Decimal("0")

        # Calculate P&L
        if self.position > Decimal("0"):  # Long position
            execution_price = price * (Decimal("1") - self.slippage)
            pnl = (execution_price - self.position_price) * self.position
        else:  # Short position
            execution_price = price * (Decimal("1") + self.slippage)
            pnl = (self.position_price - execution_price) * abs(self.position)

        # Calculate fees
        position_value = abs(self.position) * execution_price
        fee = position_value * self.transaction_cost

        # Update capital
        self.current_capital += position_value - fee

        # Track winning trades
        if pnl > Decimal("0"):
            self.winning_trades += 1

        # Record trade
        self.trade_history.append({
            "step": self.current_step,
            "action": "CLOSE",
            "price": execution_price,
            "pnl": pnl,
            "fee": fee
        })

        logger.debug(
            "position_closed",
            price=str(execution_price),
            pnl=str(pnl),
            fee=str(fee),
            capital=str(self.current_capital)
        )

        # Reset position
        self.position = Decimal("0")
        self.position_price = Decimal("0")

        # Return reward (scaled P&L percentage)
        reward = (pnl / position_value) * self.reward_scaling if position_value > 0 else Decimal("0")

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Get market data window
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window

        if end_idx > len(self.market_data):
            end_idx = len(self.market_data)
            start_idx = end_idx - self.lookback_window

        window_data = self.market_data[start_idx:end_idx]

        # Extract OHLCV data
        ohlcv = window_data.select(["open", "high", "low", "close", "volume"]).to_numpy()

        # Normalize prices by current close
        current_close = float(window_data["close"][-1])
        if current_close > 0:
            ohlcv[:, :4] = ohlcv[:, :4] / current_close

        # Normalize volume
        max_volume = np.max(ohlcv[:, 4])
        if max_volume > 0:
            ohlcv[:, 4] = ohlcv[:, 4] / max_volume

        # Flatten OHLCV
        market_features = ohlcv.flatten()

        # Add account state features
        current_price = self._get_current_price()
        equity = self._calculate_equity(current_price)

        account_features = np.array([
            float(self.current_capital / self.initial_capital),  # Capital ratio
            float(self.position),  # Position size
            float(equity / self.initial_capital),  # Equity ratio
            float(self.position_price / current_price) if current_price > 0 else 0,  # Position price ratio
        ])

        # Concatenate features
        observation = np.concatenate([market_features, account_features])

        return observation.astype(np.float32)

    def _get_current_price(self) -> Decimal:
        """Get current market price."""
        idx = self.current_step + self.lookback_window
        if idx >= len(self.market_data):
            idx = len(self.market_data) - 1

        close_price = self.market_data[idx]["close"]
        return Decimal(str(close_price))

    def _calculate_equity(self, current_price: Decimal) -> Decimal:
        """Calculate current equity."""
        if self.position == Decimal("0"):
            return self.current_capital

        # Calculate unrealized P&L
        if self.position > Decimal("0"):  # Long
            unrealized_pnl = (current_price - self.position_price) * self.position
        else:  # Short
            unrealized_pnl = (self.position_price - current_price) * abs(self.position)

        position_value = abs(self.position) * current_price
        equity = self.current_capital + position_value + unrealized_pnl

        return equity

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        current_price = self._get_current_price()
        equity = self._calculate_equity(current_price)

        # Calculate metrics
        total_return = (equity - self.initial_capital) / self.initial_capital
        win_rate = (
            Decimal(str(self.winning_trades)) / Decimal(str(self.total_trades))
            if self.total_trades > 0
            else Decimal("0")
        )

        # Calculate max drawdown
        if len(self.equity_history) > 0:
            peak = max(self.equity_history)
            drawdown = (peak - equity) / peak if peak > 0 else Decimal("0")
        else:
            drawdown = Decimal("0")

        return {
            "step": self.current_step,
            "equity": str(equity),
            "total_return": str(total_return),
            "position": str(self.position),
            "capital": str(self.current_capital),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": str(win_rate),
            "max_drawdown": str(drawdown)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get episode statistics."""
        if len(self.equity_history) == 0:
            return {}

        equity_array = np.array([float(e) for e in self.equity_history])

        # Returns calculation
        returns = np.diff(equity_array) / equity_array[:-1]
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]

        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peaks = np.maximum.accumulate(equity_array)
        drawdowns = (peaks - equity_array) / peaks
        max_drawdown = np.max(drawdowns)

        return {
            "total_return": str(Decimal(str(total_return))),
            "sharpe_ratio": str(Decimal(str(sharpe))),
            "max_drawdown": str(Decimal(str(max_drawdown))),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": str(Decimal(str(self.winning_trades / max(self.total_trades, 1)))),
            "final_equity": str(self.equity_history[-1])
        }
