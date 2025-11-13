"""Reinforcement learning environment for trade execution.

This module implements a gym-like environment for training RL agents to execute
large orders with minimal market impact and optimal timing.
"""

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionState:
    """State representation for execution environment.

    Attributes:
        timestamp: Current timestamp
        position: Current position size
        remaining_quantity: Remaining quantity to execute
        avg_execution_price: Average execution price so far
        market_price: Current market price
        bid_ask_spread: Current bid-ask spread
        volume: Current market volume
        volatility: Recent volatility measure
        time_remaining: Time remaining in execution window
        orderbook_imbalance: Order book imbalance
        vwap: Volume-weighted average price
        participation_rate: Current participation rate
    """
    timestamp: datetime
    position: Decimal
    remaining_quantity: Decimal
    avg_execution_price: Decimal
    market_price: Decimal
    bid_ask_spread: Decimal
    volume: Decimal
    volatility: Decimal
    time_remaining: Decimal
    orderbook_imbalance: Decimal
    vwap: Decimal
    participation_rate: Decimal


class ExecutionEnvironment:
    """RL environment for optimal trade execution.

    Simulates the execution of a large order over time, allowing an RL agent
    to learn optimal execution strategies considering market impact, timing,
    and transaction costs.

    Attributes:
        config: Configuration dictionary
        total_quantity: Total quantity to execute
        execution_horizon: Time horizon for execution
        market_data: Historical market data for simulation
        current_step: Current step in episode
        state: Current execution state
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize execution environment.

        Args:
            config: Configuration dictionary containing:
                - total_quantity: Total order quantity
                - execution_horizon_seconds: Execution time window
                - step_size_seconds: Time between actions
                - market_impact_coef: Market impact coefficient
                - transaction_cost_bps: Transaction cost in bps
                - max_participation_rate: Maximum participation rate
                - risk_aversion: Risk aversion parameter
                - slippage_model: Slippage model type
                - reward_type: Reward function type

        Raises:
            ValueError: If config is invalid
        """
        self.config = config
        self._validate_config()

        self.total_quantity = Decimal(str(config['total_quantity']))
        self.execution_horizon = config['execution_horizon_seconds']
        self.step_size = config.get('step_size_seconds', 60)
        self.market_impact_coef = Decimal(str(config.get('market_impact_coef', '0.1')))
        self.transaction_cost_bps = Decimal(str(config.get('transaction_cost_bps', '5')))
        self.max_participation_rate = Decimal(str(config.get('max_participation_rate', '0.2')))
        self.risk_aversion = Decimal(str(config.get('risk_aversion', '0.5')))
        self.slippage_model = config.get('slippage_model', 'sqrt')
        self.reward_type = config.get('reward_type', 'implementation_shortfall')

        # Episode state
        self.market_data: Optional[pl.DataFrame] = None
        self.current_step = 0
        self.state: Optional[ExecutionState] = None
        self.episode_history: List[Dict[str, Any]] = []

        # Execution tracking
        self.executed_quantity = Decimal('0')
        self.total_cost = Decimal('0')
        self.execution_prices: List[Decimal] = []
        self.arrival_price: Optional[Decimal] = None

        # Action space: percentage of remaining quantity to execute (0 to 1)
        self.action_space_low = Decimal('0')
        self.action_space_high = Decimal('1')

        # Observation space dimension
        self.observation_dim = 12

        logger.info(
            "execution_environment_initialized",
            total_quantity=str(self.total_quantity),
            execution_horizon=self.execution_horizon,
            step_size=self.step_size
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        required_keys = [
            'total_quantity',
            'execution_horizon_seconds'
        ]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            error_msg = f"Missing required config keys: {missing_keys}"
            logger.error("config_validation_failed", error=error_msg)
            raise ValueError(error_msg)

        if Decimal(str(self.config['total_quantity'])) <= 0:
            raise ValueError("total_quantity must be positive")

        if self.config['execution_horizon_seconds'] <= 0:
            raise ValueError("execution_horizon_seconds must be positive")

    def reset(
        self,
        market_data: pl.DataFrame,
        initial_timestamp: Optional[datetime] = None
    ) -> np.ndarray:
        """Reset environment for new episode.

        Args:
            market_data: Market data for simulation
            initial_timestamp: Starting timestamp (uses first in data if None)

        Returns:
            Initial observation

        Raises:
            ValueError: If market data is invalid
        """
        try:
            self._validate_market_data(market_data)

            self.market_data = market_data
            self.current_step = 0
            self.executed_quantity = Decimal('0')
            self.total_cost = Decimal('0')
            self.execution_prices = []
            self.episode_history = []

            # Set initial timestamp
            if initial_timestamp is None:
                initial_timestamp = market_data['timestamp'][0]

            # Get initial market state
            initial_idx = 0
            initial_row = market_data.row(initial_idx, named=True)

            self.arrival_price = Decimal(str(initial_row['close']))

            # Initialize state
            self.state = ExecutionState(
                timestamp=initial_timestamp,
                position=Decimal('0'),
                remaining_quantity=self.total_quantity,
                avg_execution_price=Decimal('0'),
                market_price=self.arrival_price,
                bid_ask_spread=self._calculate_spread(initial_row),
                volume=Decimal(str(initial_row['volume'])),
                volatility=self._calculate_volatility(initial_idx),
                time_remaining=Decimal(str(self.execution_horizon)),
                orderbook_imbalance=Decimal('0'),
                vwap=self.arrival_price,
                participation_rate=Decimal('0')
            )

            observation = self._get_observation()

            logger.info(
                "environment_reset",
                arrival_price=str(self.arrival_price),
                total_quantity=str(self.total_quantity)
            )

            return observation

        except Exception as e:
            logger.error("environment_reset_failed", error=str(e))
            raise

    def _validate_market_data(self, df: pl.DataFrame) -> None:
        """Validate market data format.

        Args:
            df: Market data dataframe

        Raises:
            ValueError: If data format is invalid
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if df.height == 0:
            raise ValueError("Empty market data")

    def step(
        self,
        action: float
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (fraction of remaining quantity to execute)

        Returns:
            Tuple of (observation, reward, done, info)

        Raises:
            ValueError: If action is invalid
            RuntimeError: If environment not reset
        """
        try:
            if self.state is None or self.market_data is None:
                raise RuntimeError("Environment must be reset before stepping")

            # Validate and clip action
            action_decimal = Decimal(str(np.clip(action, 0.0, 1.0)))

            # Calculate quantity to execute
            quantity_to_execute = self.state.remaining_quantity * action_decimal

            # Apply participation rate limit
            max_quantity = self._calculate_max_quantity()
            quantity_to_execute = min(quantity_to_execute, max_quantity)

            # Execute trade and calculate cost
            execution_price, execution_cost = self._execute_trade(
                quantity_to_execute
            )

            # Update state
            self.executed_quantity += quantity_to_execute
            self.total_cost += execution_cost

            if quantity_to_execute > 0:
                self.execution_prices.append(execution_price)

            # Move to next time step
            self.current_step += 1
            next_idx = min(self.current_step, self.market_data.height - 1)
            next_row = self.market_data.row(next_idx, named=True)

            # Update execution state
            remaining = self.total_quantity - self.executed_quantity
            time_remaining = Decimal(str(
                max(0, self.execution_horizon - (self.current_step * self.step_size))
            ))

            avg_price = (
                self.total_cost / self.executed_quantity
                if self.executed_quantity > 0
                else Decimal('0')
            )

            self.state = ExecutionState(
                timestamp=next_row['timestamp'],
                position=self.executed_quantity,
                remaining_quantity=remaining,
                avg_execution_price=avg_price,
                market_price=Decimal(str(next_row['close'])),
                bid_ask_spread=self._calculate_spread(next_row),
                volume=Decimal(str(next_row['volume'])),
                volatility=self._calculate_volatility(next_idx),
                time_remaining=time_remaining,
                orderbook_imbalance=self._calculate_imbalance(next_row),
                vwap=self._calculate_vwap(next_idx),
                participation_rate=action_decimal
            )

            # Check if episode is done
            done = (
                remaining <= Decimal('0.001') or
                time_remaining <= Decimal('0') or
                self.current_step >= self.market_data.height - 1
            )

            # Calculate reward
            reward = self._calculate_reward(
                quantity_to_execute,
                execution_price,
                done
            )

            # Get next observation
            observation = self._get_observation()

            # Info dict
            info = {
                'step': self.current_step,
                'executed_quantity': float(quantity_to_execute),
                'execution_price': float(execution_price),
                'remaining_quantity': float(remaining),
                'total_executed': float(self.executed_quantity),
                'avg_execution_price': float(avg_price),
                'market_price': float(self.state.market_price),
                'done': done
            }

            self.episode_history.append(info)

            logger.debug(
                "environment_step",
                step=self.current_step,
                action=float(action_decimal),
                reward=float(reward),
                done=done
            )

            return observation, float(reward), done, info

        except Exception as e:
            logger.error("environment_step_failed", error=str(e))
            raise

    def _calculate_spread(self, market_row: Dict[str, Any]) -> Decimal:
        """Calculate bid-ask spread from market data.

        Args:
            market_row: Market data row

        Returns:
            Bid-ask spread
        """
        # Use high-low as proxy for spread
        high = Decimal(str(market_row['high']))
        low = Decimal(str(market_row['low']))
        return high - low

    def _calculate_volatility(self, idx: int, lookback: int = 20) -> Decimal:
        """Calculate recent volatility.

        Args:
            idx: Current index
            lookback: Lookback period

        Returns:
            Volatility measure
        """
        if self.market_data is None:
            return Decimal('0')

        start_idx = max(0, idx - lookback)
        window = self.market_data[start_idx:idx + 1]

        if window.height < 2:
            return Decimal('0')

        returns = (
            (window['close'] - window['close'].shift(1)) / window['close'].shift(1)
        ).drop_nulls()

        if returns.height == 0:
            return Decimal('0')

        volatility = returns.std()
        return Decimal(str(volatility)) if volatility is not None else Decimal('0')

    def _calculate_imbalance(self, market_row: Dict[str, Any]) -> Decimal:
        """Calculate order book imbalance proxy.

        Args:
            market_row: Market data row

        Returns:
            Imbalance measure
        """
        # Use close relative to high-low range as proxy
        close = Decimal(str(market_row['close']))
        high = Decimal(str(market_row['high']))
        low = Decimal(str(market_row['low']))

        if high == low:
            return Decimal('0')

        # -1 to 1, where 1 means close at high
        imbalance = (close - low) / (high - low) * Decimal('2') - Decimal('1')
        return imbalance

    def _calculate_vwap(self, idx: int, lookback: int = 20) -> Decimal:
        """Calculate volume-weighted average price.

        Args:
            idx: Current index
            lookback: Lookback period

        Returns:
            VWAP
        """
        if self.market_data is None:
            return Decimal('0')

        start_idx = max(0, idx - lookback)
        window = self.market_data[start_idx:idx + 1]

        if window.height == 0:
            return Decimal('0')

        total_volume = window['volume'].sum()
        if total_volume == 0:
            return Decimal(str(window['close'].mean()))

        vwap = (window['close'] * window['volume']).sum() / total_volume
        return Decimal(str(vwap))

    def _calculate_max_quantity(self) -> Decimal:
        """Calculate maximum quantity based on participation rate.

        Returns:
            Maximum quantity to execute this step
        """
        if self.state is None:
            return Decimal('0')

        # Max quantity is participation rate * market volume
        max_qty = self.state.volume * self.max_participation_rate

        return min(max_qty, self.state.remaining_quantity)

    def _execute_trade(
        self,
        quantity: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Execute trade and calculate execution price with market impact.

        Args:
            quantity: Quantity to execute

        Returns:
            Tuple of (execution_price, total_cost)
        """
        if self.state is None or quantity == 0:
            return Decimal('0'), Decimal('0')

        # Base price
        base_price = self.state.market_price

        # Market impact (depends on quantity and volume)
        if self.state.volume > 0:
            participation = quantity / self.state.volume
        else:
            participation = Decimal('1')

        # Market impact using square root model or linear model
        if self.slippage_model == 'sqrt':
            impact_factor = participation.sqrt()
        else:  # linear
            impact_factor = participation

        market_impact = base_price * self.market_impact_coef * impact_factor

        # Transaction costs
        transaction_cost = base_price * quantity * self.transaction_cost_bps / Decimal('10000')

        # Total execution price (assume buying, so impact increases price)
        execution_price = base_price + market_impact

        # Total cost
        total_cost = execution_price * quantity + transaction_cost

        return execution_price, total_cost

    def _calculate_reward(
        self,
        quantity: Decimal,
        execution_price: Decimal,
        done: bool
    ) -> Decimal:
        """Calculate reward for current step.

        Args:
            quantity: Executed quantity
            execution_price: Execution price
            done: Whether episode is done

        Returns:
            Reward value
        """
        if self.state is None or self.arrival_price is None:
            return Decimal('0')

        if self.reward_type == 'implementation_shortfall':
            # Implementation shortfall: difference from arrival price
            if quantity > 0:
                shortfall = (execution_price - self.arrival_price) * quantity
                reward = -shortfall  # Negative because we want to minimize
            else:
                reward = Decimal('0')

            # Penalty for unexecuted quantity at end
            if done and self.state.remaining_quantity > Decimal('0.001'):
                penalty = self.state.remaining_quantity * self.state.market_price * Decimal('0.1')
                reward -= penalty

        elif self.reward_type == 'vwap':
            # Reward for beating VWAP
            if quantity > 0:
                vwap_diff = self.state.vwap - execution_price
                reward = vwap_diff * quantity
            else:
                reward = Decimal('0')

        else:  # 'pnl'
            # Simple PnL-based reward
            if quantity > 0:
                pnl = (self.state.market_price - execution_price) * quantity
                reward = pnl
            else:
                reward = Decimal('0')

        # Risk penalty (penalize high volatility trades)
        if quantity > 0:
            risk_penalty = self.state.volatility * quantity * self.risk_aversion
            reward -= risk_penalty

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector.

        Returns:
            Observation array
        """
        if self.state is None:
            return np.zeros(self.observation_dim)

        observation = np.array([
            float(self.state.remaining_quantity / self.total_quantity),  # Normalized remaining
            float(self.state.position / self.total_quantity),  # Normalized position
            float(self.state.time_remaining / Decimal(str(self.execution_horizon))),  # Normalized time
            float(self.state.bid_ask_spread / self.state.market_price),  # Relative spread
            float(self.state.volume / Decimal('1000000')),  # Normalized volume
            float(self.state.volatility),
            float(self.state.orderbook_imbalance),
            float(self.state.participation_rate),
            float((self.state.market_price - self.arrival_price) / self.arrival_price)
            if self.arrival_price > 0 else 0.0,  # Price change
            float((self.state.vwap - self.arrival_price) / self.arrival_price)
            if self.arrival_price > 0 else 0.0,  # VWAP deviation
            float(self.state.avg_execution_price / self.arrival_price - Decimal('1'))
            if self.arrival_price > 0 and self.state.avg_execution_price > 0 else 0.0,  # Execution performance
            float(self.current_step / (self.execution_horizon / self.step_size)),  # Normalized step
        ])

        return observation

    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get metrics for completed episode.

        Returns:
            Dictionary with episode metrics

        Raises:
            RuntimeError: If episode not completed
        """
        if self.arrival_price is None or self.executed_quantity == 0:
            raise RuntimeError("No episode data available")

        avg_execution_price = self.total_cost / self.executed_quantity
        implementation_shortfall = avg_execution_price - self.arrival_price
        is_bps = (implementation_shortfall / self.arrival_price) * Decimal('10000')

        metrics = {
            'total_executed': float(self.executed_quantity),
            'execution_rate': float(self.executed_quantity / self.total_quantity),
            'avg_execution_price': float(avg_execution_price),
            'arrival_price': float(self.arrival_price),
            'implementation_shortfall': float(implementation_shortfall),
            'implementation_shortfall_bps': float(is_bps),
            'total_cost': float(self.total_cost),
            'num_steps': self.current_step,
            'episode_history': self.episode_history
        }

        return metrics
