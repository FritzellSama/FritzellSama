"""
Market Environment for Reinforcement Learning.

Production-ready trading environment compatible with OpenAI Gym interface.
Supports multi-asset trading with realistic market simulation.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class EnvironmentState:
    """Current state of the trading environment."""
    positions: Dict[str, Decimal]
    cash: Decimal
    portfolio_value: Decimal
    timestamp: datetime
    market_data: pl.DataFrame


class MarketEnvironment:
    """
    Trading environment for reinforcement learning agents.

    Simulates realistic market conditions including:
    - Transaction costs
    - Slippage
    - Market impact
    - Position limits

    Attributes:
        config: Environment configuration from config files
        data: Historical market data (Polars DataFrame)
        current_step: Current time step in episode
        state: Current environment state
        done: Episode completion flag

    Example:
        >>> config = load_config('config/ai/market_env.yaml')
        >>> env = MarketEnvironment(config, market_data)
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step(action)
    """

    def __init__(self, config: Dict[str, Any], data: pl.DataFrame) -> None:
        """
        Initialize market environment.

        Args:
            config: Configuration dictionary from YAML
            data: Historical market data with OHLCV columns

        Raises:
            ValueError: If configuration or data is invalid
        """
        self.config = config
        self.data = data
        self._validate_config()
        self._validate_data()

        # Extract configuration
        self.initial_cash = Decimal(str(config['initial_cash']))
        self.commission_rate = Decimal(str(config['commission_rate']))
        self.slippage_rate = Decimal(str(config['slippage_rate']))
        self.max_position_size = Decimal(str(config['max_position_size']))
        self.window_size = config['window_size']

        # Get symbols from data
        if 'symbol' in data.columns:
            self.symbols = data['symbol'].unique().to_list()
        else:
            self.symbols = config['symbols']

        # Initialize state
        self.current_step = 0
        self.done = False
        self.state: Optional[EnvironmentState] = None
        self.positions: Dict[str, Decimal] = {symbol: Decimal('0') for symbol in self.symbols}
        self.cash = self.initial_cash
        self.portfolio_values: List[Decimal] = []
        self.trades: List[Dict] = []

        logger.info(
            "Market environment initialized",
            symbols=self.symbols,
            initial_cash=str(self.initial_cash),
            data_length=len(data)
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_keys = [
            'initial_cash', 'commission_rate', 'slippage_rate',
            'max_position_size', 'window_size'
        ]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate ranges
        if Decimal(str(self.config['commission_rate'])) < 0:
            raise ValueError("commission_rate must be non-negative")

        if Decimal(str(self.config['slippage_rate'])) < 0:
            raise ValueError("slippage_rate must be non-negative")

    def _validate_data(self) -> None:
        """Validate market data."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required data column: {col}")

        if len(self.data) == 0:
            raise ValueError("Empty market data")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (state)
        """
        self.current_step = self.window_size
        self.done = False
        self.positions = {symbol: Decimal('0') for symbol in self.symbols}
        self.cash = self.initial_cash
        self.portfolio_values = []
        self.trades = []

        logger.info("Environment reset", step=self.current_step)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).

        Returns:
            State vector as numpy array
        """
        # Get recent market data window
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step

        recent_data = self.data[start_idx:end_idx]

        # Build observation features
        features = []

        # Price features (normalized)
        for symbol in self.symbols:
            if 'symbol' in self.data.columns:
                symbol_data = recent_data.filter(pl.col('symbol') == symbol)
            else:
                symbol_data = recent_data

            if len(symbol_data) > 0:
                close_prices = symbol_data['close'].to_numpy()
                # Normalize by first price in window
                if close_prices[0] != 0:
                    normalized_prices = close_prices / close_prices[0] - 1.0
                else:
                    normalized_prices = np.zeros_like(close_prices)
                features.extend(normalized_prices)

                # Volume (normalized)
                volumes = symbol_data['volume'].to_numpy()
                if volumes.max() > 0:
                    normalized_volumes = volumes / volumes.max()
                else:
                    normalized_volumes = np.zeros_like(volumes)
                features.extend(normalized_volumes)

        # Portfolio features
        portfolio_value = self._calculate_portfolio_value()
        for symbol in self.symbols:
            position_ratio = float(self.positions[symbol]) / float(portfolio_value) \
                if portfolio_value > 0 else 0.0
            features.append(position_ratio)

        # Cash ratio
        cash_ratio = float(self.cash) / float(portfolio_value) if portfolio_value > 0 else 1.0
        features.append(cash_ratio)

        return np.array(features, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Decimal, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Action vector (target positions for each symbol)

        Returns:
            Tuple of (next_state, reward, done, info)

        Raises:
            RuntimeError: If step execution fails
        """
        try:
            # Store previous portfolio value
            prev_portfolio_value = self._calculate_portfolio_value()

            # Execute trades based on action
            self._execute_trades(action)

            # Move to next step
            self.current_step += 1

            # Check if episode is done
            if self.current_step >= len(self.data) - 1:
                self.done = True

            # Calculate reward
            current_portfolio_value = self._calculate_portfolio_value()
            self.portfolio_values.append(current_portfolio_value)

            reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)

            # Get next observation
            next_state = self._get_observation()

            # Build info dictionary
            info = {
                'portfolio_value': current_portfolio_value,
                'cash': self.cash,
                'positions': self.positions.copy(),
                'step': self.current_step,
                'num_trades': len(self.trades)
            }

            return next_state, reward, self.done, info

        except Exception as e:
            logger.error("Environment step failed", error=str(e), step=self.current_step)
            raise RuntimeError(f"Step execution failed: {e}")

    def _execute_trades(self, action: np.ndarray) -> None:
        """
        Execute trades to reach target positions.

        Args:
            action: Target position weights for each symbol
        """
        # Get current prices
        current_data = self.data[self.current_step]
        prices = {}

        for symbol in self.symbols:
            if 'symbol' in self.data.columns:
                symbol_data = current_data.filter(pl.col('symbol') == symbol)
                if len(symbol_data) > 0:
                    prices[symbol] = Decimal(str(symbol_data['close'][0]))
                else:
                    prices[symbol] = Decimal('0')
            else:
                prices[symbol] = Decimal(str(current_data['close'][0]))

        # Calculate target positions
        portfolio_value = self._calculate_portfolio_value()

        for idx, symbol in enumerate(self.symbols):
            if idx >= len(action):
                continue

            # Clip action to [-1, 1] and scale to position size
            target_weight = np.clip(action[idx], -1.0, 1.0)
            target_value = Decimal(str(target_weight)) * portfolio_value * self.max_position_size

            if prices[symbol] > 0:
                target_quantity = target_value / prices[symbol]
            else:
                target_quantity = Decimal('0')

            # Calculate quantity to trade
            current_quantity = self.positions[symbol]
            trade_quantity = target_quantity - current_quantity

            if abs(trade_quantity) > Decimal('0.0001'):  # Min trade size
                # Apply slippage
                slippage_factor = Decimal('1') + self.slippage_rate * \
                                (Decimal('1') if trade_quantity > 0 else Decimal('-1'))
                execution_price = prices[symbol] * slippage_factor

                # Calculate trade value
                trade_value = abs(trade_quantity) * execution_price

                # Calculate commission
                commission = trade_value * self.commission_rate

                # Check if sufficient cash for buy
                if trade_quantity > 0:
                    total_cost = trade_value + commission
                    if total_cost <= self.cash:
                        self.positions[symbol] += trade_quantity
                        self.cash -= total_cost

                        self.trades.append({
                            'timestamp': current_data['timestamp'][0],
                            'symbol': symbol,
                            'quantity': str(trade_quantity),
                            'price': str(execution_price),
                            'commission': str(commission)
                        })
                else:
                    # Sell
                    self.positions[symbol] += trade_quantity
                    proceeds = trade_value - commission
                    self.cash += proceeds

                    self.trades.append({
                        'timestamp': current_data['timestamp'][0],
                        'symbol': symbol,
                        'quantity': str(trade_quantity),
                        'price': str(execution_price),
                        'commission': str(commission)
                    })

    def _calculate_portfolio_value(self) -> Decimal:
        """
        Calculate current portfolio value.

        Returns:
            Total portfolio value including cash and positions
        """
        if self.current_step >= len(self.data):
            return self.cash

        current_data = self.data[self.current_step]
        total_value = self.cash

        for symbol in self.symbols:
            if 'symbol' in self.data.columns:
                symbol_data = current_data.filter(pl.col('symbol') == symbol)
                if len(symbol_data) > 0:
                    price = Decimal(str(symbol_data['close'][0]))
                else:
                    price = Decimal('0')
            else:
                price = Decimal(str(current_data['close'][0]))

            position_value = self.positions[symbol] * price
            total_value += position_value

        return total_value

    def _calculate_reward(self, prev_value: Decimal, current_value: Decimal) -> Decimal:
        """
        Calculate reward for the step.

        Args:
            prev_value: Previous portfolio value
            current_value: Current portfolio value

        Returns:
            Reward value
        """
        # Simple return-based reward
        if prev_value > 0:
            simple_return = (current_value - prev_value) / prev_value
        else:
            simple_return = Decimal('0')

        # Apply reward shaping from config
        reward_scale = Decimal(str(self.config.get('reward_scale', 1.0)))

        return simple_return * reward_scale

    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render environment state.

        Args:
            mode: Rendering mode ('human' or 'ansi')

        Returns:
            String representation if mode='ansi', None otherwise
        """
        portfolio_value = self._calculate_portfolio_value()

        output = f"\n=== Market Environment (Step {self.current_step}) ===\n"
        output += f"Portfolio Value: {portfolio_value:.2f}\n"
        output += f"Cash: {self.cash:.2f}\n"
        output += f"Positions:\n"

        for symbol in self.symbols:
            output += f"  {symbol}: {self.positions[symbol]:.4f}\n"

        output += f"Total Trades: {len(self.trades)}\n"

        if mode == 'ansi':
            return output
        else:
            print(output)
            return None

    def get_performance_metrics(self) -> Dict[str, Decimal]:
        """
        Calculate performance metrics for the episode.

        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) == 0:
            return {}

        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # Calculate Sharpe ratio
        if len(self.portfolio_values) > 1:
            returns = []
            for i in range(1, len(self.portfolio_values)):
                ret = (self.portfolio_values[i] - self.portfolio_values[i-1]) / \
                      self.portfolio_values[i-1]
                returns.append(float(ret))

            returns_array = np.array(returns)
            sharpe_ratio = Decimal(str(
                np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
                if np.std(returns_array) > 0 else 0.0
            ))
        else:
            sharpe_ratio = Decimal('0')

        # Calculate max drawdown
        max_value = self.portfolio_values[0]
        max_drawdown = Decimal('0')

        for value in self.portfolio_values:
            if value > max_value:
                max_value = value
            drawdown = (max_value - value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': Decimal(str(len(self.trades)))
        }
