"""Trading environment for reinforcement learning agents.

This module implements a comprehensive trading environment compatible with
OpenAI Gym interface for training RL agents on market data.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class TradingAction(Enum):
    """Trading actions available to the agent."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


class TradingEnv:
    """Gym-style trading environment for RL agents.

    Simulates a trading environment where an agent can take actions
    (buy, sell, hold) and receive rewards based on trading performance.

    Attributes:
        config: Environment configuration
        data: Market data (OHLCV + features)
        initial_balance: Starting capital
        current_step: Current time step
        position: Current position (-1, 0, 1)
        balance: Current cash balance
        portfolio_value: Total portfolio value
        trades: List of executed trades
        done: Whether episode is finished

    Examples:
        >>> config = {
        ...     "initial_balance": "10000",
        ...     "transaction_fee": "0.001",
        ...     "max_position": "1.0"
        ... }
        >>> env = TradingEnv(config)
        >>> state = await env.reset(market_data)
        >>> next_state, reward, done, info = await env.step(TradingAction.BUY)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize trading environment.

        Args:
            config: Configuration with balance, fees, limits, etc.

        Raises:
            ValueError: If required config parameters missing
        """
        self.config = config
        self._validate_config()

        # Environment parameters from config
        self.initial_balance: Decimal = Decimal(str(config.get("initial_balance", os.getenv("ENV_INITIAL_BALANCE", "10000"))))
        self.transaction_fee: Decimal = Decimal(str(config.get("transaction_fee", os.getenv("ENV_TRANSACTION_FEE", "0.001"))))
        self.max_position: Decimal = Decimal(str(config.get("max_position", os.getenv("ENV_MAX_POSITION", "1.0"))))
        self.leverage: Decimal = Decimal(str(config.get("leverage", os.getenv("ENV_LEVERAGE", "1"))))
        self.slippage: Decimal = Decimal(str(config.get("slippage", os.getenv("ENV_SLIPPAGE", "0.0001"))))

        # Episode parameters
        self.max_steps: int = int(config.get("max_steps", os.getenv("ENV_MAX_STEPS", "1000")))
        self.reward_scaling: Decimal = Decimal(str(config.get("reward_scaling", os.getenv("ENV_REWARD_SCALING", "1.0"))))

        # State
        self.data: Optional[pl.DataFrame] = None
        self.current_step: int = 0
        self.position: Decimal = Decimal("0")  # Current position size
        self.entry_price: Decimal = Decimal("0")  # Entry price for current position
        self.balance: Decimal = self.initial_balance
        self.equity: Decimal = self.initial_balance
        self.trades: List[Dict[str, Any]] = []
        self.done: bool = False

        logger.info(
            "Trading environment initialized",
            initial_balance=str(self.initial_balance),
            transaction_fee=str(self.transaction_fee),
            max_position=str(self.max_position)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If config validation fails
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")

        if "initial_balance" in self.config:
            balance = Decimal(str(self.config["initial_balance"]))
            if balance <= 0:
                raise ValueError(f"initial_balance must be positive, got {balance}")

        if "leverage" in self.config:
            leverage = Decimal(str(self.config["leverage"]))
            if leverage < 1:
                raise ValueError(f"leverage must be >= 1, got {leverage}")

    async def reset(
        self,
        data: pl.DataFrame,
        start_step: int = 0
    ) -> np.ndarray:
        """Reset environment for new episode.

        Args:
            data: Market data DataFrame with OHLCV and features
            start_step: Starting step in the data

        Returns:
            Initial state observation

        Raises:
            ValueError: If data invalid
        """
        try:
            if data.is_empty():
                raise ValueError("Cannot reset with empty data")

            required_cols = ["open", "high", "low", "close", "volume"]
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            self.data = data
            self.current_step = start_step
            self.position = Decimal("0")
            self.entry_price = Decimal("0")
            self.balance = self.initial_balance
            self.equity = self.initial_balance
            self.trades = []
            self.done = False

            logger.debug(
                "Environment reset",
                data_length=len(data),
                start_step=start_step
            )

            return self._get_observation()

        except Exception as e:
            logger.error(
                "Environment reset failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def step(self, action: TradingAction) -> Tuple[np.ndarray, Decimal, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Trading action to execute

        Returns:
            Tuple of (next_state, reward, done, info)

        Raises:
            RuntimeError: If environment not reset
        """
        try:
            if self.data is None:
                raise RuntimeError("Environment must be reset before step")

            if self.done:
                raise RuntimeError("Episode already finished, call reset()")

            # Get current price
            current_price = Decimal(str(self.data["close"][self.current_step]))

            # Execute action
            reward = await self._execute_action(action, current_price)

            # Update step
            self.current_step += 1

            # Check if episode is done
            self.done = (
                self.current_step >= len(self.data) - 1 or
                self.current_step >= self.max_steps or
                self.equity <= Decimal("0")
            )

            # Get next observation
            next_state = self._get_observation()

            # Prepare info dict
            info = {
                "step": self.current_step,
                "balance": str(self.balance),
                "equity": str(self.equity),
                "position": str(self.position),
                "num_trades": len(self.trades),
                "portfolio_value": str(self.equity)
            }

            logger.debug(
                "Environment step",
                action=action.name,
                reward=str(reward),
                done=self.done
            )

            return next_state, reward, self.done, info

        except Exception as e:
            logger.error(
                "Environment step failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise

    async def _execute_action(self, action: TradingAction, price: Decimal) -> Decimal:
        """Execute trading action and calculate reward.

        Args:
            action: Trading action
            price: Current market price

        Returns:
            Reward value
        """
        old_equity = self.equity
        price_with_slippage = price * (Decimal("1") + self.slippage)

        if action == TradingAction.BUY:
            if self.position <= Decimal("0"):
                # Open long position
                position_size = self.max_position
                cost = price_with_slippage * position_size
                fee = cost * self.transaction_fee

                if self.balance >= cost + fee:
                    self.balance -= (cost + fee)
                    self.position = position_size
                    self.entry_price = price_with_slippage

                    self.trades.append({
                        "step": self.current_step,
                        "action": "BUY",
                        "price": str(price_with_slippage),
                        "size": str(position_size),
                        "fee": str(fee),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

        elif action == TradingAction.SELL:
            if self.position >= Decimal("0"):
                # Open short position
                position_size = self.max_position
                proceeds = price_with_slippage * position_size
                fee = proceeds * self.transaction_fee

                self.balance += (proceeds - fee)
                self.position = -position_size
                self.entry_price = price_with_slippage

                self.trades.append({
                    "step": self.current_step,
                    "action": "SELL",
                    "price": str(price_with_slippage),
                    "size": str(position_size),
                    "fee": str(fee),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        elif action == TradingAction.CLOSE:
            if self.position != Decimal("0"):
                # Close position
                proceeds = price_with_slippage * abs(self.position)
                fee = proceeds * self.transaction_fee

                if self.position > Decimal("0"):
                    # Close long
                    self.balance += (proceeds - fee)
                    pnl = (price_with_slippage - self.entry_price) * self.position - fee
                else:
                    # Close short
                    cost = proceeds
                    self.balance -= (cost + fee)
                    pnl = (self.entry_price - price_with_slippage) * abs(self.position) - fee

                self.trades.append({
                    "step": self.current_step,
                    "action": "CLOSE",
                    "price": str(price_with_slippage),
                    "size": str(abs(self.position)),
                    "fee": str(fee),
                    "pnl": str(pnl),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                self.position = Decimal("0")
                self.entry_price = Decimal("0")

        # Update equity
        position_value = Decimal("0")
        if self.position > Decimal("0"):
            position_value = price * self.position
        elif self.position < Decimal("0"):
            position_value = (self.entry_price - price) * abs(self.position)

        self.equity = self.balance + position_value

        # Calculate reward
        reward = (self.equity - old_equity) * self.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current state observation.

        Returns:
            Observation array with market data and portfolio state
        """
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1

        # Get current market data
        row = self.data[self.current_step]

        # Market features
        market_features = []
        for col in ["open", "high", "low", "close", "volume"]:
            if col in row:
                market_features.append(float(row[col][0]))

        # Portfolio features
        portfolio_features = [
            float(self.position),
            float(self.balance) / float(self.initial_balance),
            float(self.equity) / float(self.initial_balance)
        ]

        # Combine features
        observation = np.array(market_features + portfolio_features, dtype=np.float32)

        return observation

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate episode performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                "total_return": "0",
                "num_trades": 0,
                "win_rate": "0",
                "sharpe_ratio": "0"
            }

        # Calculate returns
        total_return = (self.equity - self.initial_balance) / self.initial_balance

        # Calculate win rate
        profitable_trades = sum(
            1 for trade in self.trades
            if "pnl" in trade and Decimal(trade["pnl"]) > 0
        )
        win_rate = Decimal(str(profitable_trades)) / Decimal(str(len(self.trades))) if self.trades else Decimal("0")

        # Calculate Sharpe ratio (simplified)
        if self.trades:
            returns = [
                Decimal(trade.get("pnl", "0")) / self.initial_balance
                for trade in self.trades
                if "pnl" in trade
            ]
            if returns:
                returns_array = np.array([float(r) for r in returns])
                sharpe_ratio = Decimal(str(np.mean(returns_array) / (np.std(returns_array) + 1e-10)))
            else:
                sharpe_ratio = Decimal("0")
        else:
            sharpe_ratio = Decimal("0")

        return {
            "total_return": str(total_return),
            "num_trades": len(self.trades),
            "win_rate": str(win_rate),
            "sharpe_ratio": str(sharpe_ratio),
            "final_equity": str(self.equity),
            "max_position": str(self.max_position)
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """Render environment state.

        Args:
            mode: Rendering mode

        Returns:
            String representation if mode is "ansi"
        """
        if mode == "ansi":
            return (
                f"Step: {self.current_step}\n"
                f"Balance: {self.balance}\n"
                f"Equity: {self.equity}\n"
                f"Position: {self.position}\n"
                f"Trades: {len(self.trades)}"
            )
        else:
            logger.info(
                "Environment state",
                step=self.current_step,
                balance=str(self.balance),
                equity=str(self.equity),
                position=str(self.position)
            )
            return None
