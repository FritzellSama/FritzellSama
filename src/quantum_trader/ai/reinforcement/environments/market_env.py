"""Market environment for reinforcement learning agents.

This module provides a realistic trading environment for training RL agents
with proper handling of market dynamics, transaction costs, and risk constraints.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from structlog import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime states."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


@dataclass
class EnvironmentConfig:
    """Configuration for market environment.

    Attributes:
        initial_balance: Starting balance in quote currency
        max_position_size: Maximum position size as fraction of balance
        transaction_fee: Transaction fee as percentage
        slippage_factor: Slippage as percentage
        max_leverage: Maximum allowed leverage
        liquidation_threshold: Liquidation threshold as percentage loss
        reward_scaling: Scaling factor for rewards
        lookback_window: Number of historical bars to include in state
        normalization_window: Window for price normalization
        include_regime: Whether to include market regime in state
        include_microstructure: Whether to include microstructure features
        data_config_path: Path to data configuration file
    """

    initial_balance: Decimal
    max_position_size: Decimal
    transaction_fee: Decimal
    slippage_factor: Decimal
    max_leverage: Decimal
    liquidation_threshold: Decimal
    reward_scaling: Decimal
    lookback_window: int
    normalization_window: int
    include_regime: bool
    include_microstructure: bool
    data_config_path: str


@dataclass
class Position:
    """Trading position information.

    Attributes:
        symbol: Trading symbol
        size: Position size (positive for long, negative for short)
        entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized profit/loss
        timestamp: Position timestamp
    """

    symbol: str
    size: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_pnl(self, current_price: Decimal) -> None:
        """Update unrealized PnL.

        Args:
            current_price: Current market price
        """
        self.current_price = current_price
        self.unrealized_pnl = self.size * (current_price - self.entry_price)


class MarketEnvironment:
    """Realistic market environment for RL training.

    Simulates trading with proper market mechanics including transaction costs,
    slippage, leverage constraints, and risk management.

    Example:
        >>> config = EnvironmentConfig(
        ...     initial_balance=Decimal("10000"),
        ...     max_position_size=Decimal("0.1"),
        ...     transaction_fee=Decimal("0.001"),
        ...     slippage_factor=Decimal("0.0005"),
        ...     max_leverage=Decimal("3.0"),
        ...     liquidation_threshold=Decimal("0.8"),
        ...     reward_scaling=Decimal("1.0"),
        ...     lookback_window=100,
        ...     normalization_window=1000,
        ...     include_regime=True,
        ...     include_microstructure=True,
        ...     data_config_path="config/data/market_data.yaml"
        ... )
        >>> env = MarketEnvironment(
        ...     symbols=["BTCUSDT", "ETHUSDT"],
        ...     config=config
        ... )
        >>> await env.initialize(market_data)
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step(actions)
    """

    def __init__(
        self,
        symbols: List[str],
        config: EnvironmentConfig
    ) -> None:
        """Initialize market environment.

        Args:
            symbols: List of trading symbols
            config: Environment configuration
        """
        self.symbols = symbols
        self.config = config

        # Environment state
        self.current_step = 0
        self.max_steps = 0

        # Account state
        self.balance = config.initial_balance
        self.initial_balance = config.initial_balance
        self.equity = config.initial_balance
        self.positions: Dict[str, Position] = {}

        # Historical data
        self.market_data: Optional[pl.DataFrame] = None
        self.price_history: Dict[str, List[Decimal]] = {
            symbol: [] for symbol in symbols
        }

        # Performance tracking
        self.total_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0

        # Episode history
        self.episode_rewards: List[Decimal] = []
        self.episode_actions: List[np.ndarray] = []
        self.episode_states: List[np.ndarray] = []

        # Market regime
        self.current_regime = MarketRegime.RANGING

        logger.info(
            "market_environment_initialized",
            symbols=symbols,
            initial_balance=str(config.initial_balance)
        )

    async def initialize(self, market_data: pl.DataFrame) -> None:
        """Initialize environment with market data.

        Args:
            market_data: Historical market data as Polars DataFrame
                        Expected columns: timestamp, symbol, open, high, low, close, volume
        """
        try:
            self.market_data = market_data
            self.max_steps = len(market_data.select('timestamp').unique()) - self.config.lookback_window

            # Validate data
            required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(market_data.columns)

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(
                "environment_initialized",
                max_steps=self.max_steps,
                data_rows=len(market_data)
            )

        except Exception as e:
            logger.error("initialization_failed", error=str(e))
            raise

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        Returns:
            Initial state observation
        """
        try:
            # Reset account state
            self.balance = self.config.initial_balance
            self.equity = self.config.initial_balance
            self.positions = {}

            # Reset performance metrics
            self.total_pnl = Decimal("0")
            self.realized_pnl = Decimal("0")
            self.unrealized_pnl = Decimal("0")
            self.trade_count = 0
            self.win_count = 0
            self.loss_count = 0

            # Reset episode tracking
            self.episode_rewards = []
            self.episode_actions = []
            self.episode_states = []

            # Reset step counter
            self.current_step = self.config.lookback_window

            # Reset price history
            self.price_history = {symbol: [] for symbol in self.symbols}

            return self._get_state()

        except Exception as e:
            logger.error("reset_failed", error=str(e))
            return np.zeros(self._get_state_dim())

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            actions: Action array with shape (num_symbols,)
                    Values in [-1, 1] representing position size

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        try:
            # Validate actions
            if len(actions) != len(self.symbols):
                raise ValueError(
                    f"Expected {len(self.symbols)} actions, got {len(actions)}"
                )

            # Store action
            self.episode_actions.append(actions.copy())

            # Get current prices
            current_prices = self._get_current_prices()

            # Execute trades
            trade_info = self._execute_trades(actions, current_prices)

            # Update positions
            self._update_positions(current_prices)

            # Calculate reward
            reward = self._calculate_reward()

            # Update step
            self.current_step += 1

            # Check if done
            done = self._check_done()

            # Get next state
            next_state = self._get_state()

            # Compile info
            info = {
                'balance': float(self.balance),
                'equity': float(self.equity),
                'total_pnl': float(self.total_pnl),
                'realized_pnl': float(self.realized_pnl),
                'unrealized_pnl': float(self.unrealized_pnl),
                'trade_count': self.trade_count,
                'win_rate': float(self.win_count / max(self.trade_count, 1)),
                'positions': {
                    symbol: {
                        'size': float(pos.size),
                        'entry_price': float(pos.entry_price),
                        'current_price': float(pos.current_price),
                        'pnl': float(pos.unrealized_pnl)
                    }
                    for symbol, pos in self.positions.items()
                },
                'trades': trade_info,
                'regime': self.current_regime.value
            }

            # Store state and reward
            self.episode_states.append(next_state.copy())
            self.episode_rewards.append(Decimal(str(reward)))

            return next_state, reward, done, info

        except Exception as e:
            logger.error("step_failed", error=str(e), step=self.current_step)
            # Return safe defaults
            return (
                self._get_state(),
                0.0,
                True,
                {'error': str(e)}
            )

    def _get_state(self) -> np.ndarray:
        """Get current state observation.

        Returns:
            State array with market features
        """
        try:
            if self.market_data is None:
                return np.zeros(self._get_state_dim())

            features = []

            # Get data for current window
            end_idx = self.current_step
            start_idx = max(0, end_idx - self.config.lookback_window)

            for symbol in self.symbols:
                # Filter data for symbol
                symbol_data = self.market_data.filter(
                    pl.col('symbol') == symbol
                ).sort('timestamp')[start_idx:end_idx]

                if len(symbol_data) == 0:
                    continue

                # Price features (normalized)
                closes = symbol_data.select('close').to_numpy().flatten()

                if len(closes) > 0:
                    normalized_prices = self._normalize_prices(closes)
                    features.extend(normalized_prices[-self.config.lookback_window:])

                # Volume features (normalized)
                volumes = symbol_data.select('volume').to_numpy().flatten()

                if len(volumes) > 0:
                    max_vol = volumes.max()
                    if max_vol > 0:
                        normalized_volumes = volumes / max_vol
                        features.extend(normalized_volumes[-self.config.lookback_window:])

            # Position features
            for symbol in self.symbols:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    features.extend([
                        float(pos.size / self.balance),  # Normalized position size
                        float(pos.unrealized_pnl / self.balance)  # Normalized PnL
                    ])
                else:
                    features.extend([0.0, 0.0])

            # Account features
            features.extend([
                float(self.balance / self.initial_balance),  # Normalized balance
                float(self.equity / self.initial_balance),  # Normalized equity
                float(self.total_pnl / self.initial_balance)  # Normalized total PnL
            ])

            # Market regime (one-hot encoded)
            if self.config.include_regime:
                regime_features = [0.0] * len(MarketRegime)
                regime_idx = list(MarketRegime).index(self.current_regime)
                regime_features[regime_idx] = 1.0
                features.extend(regime_features)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.error("state_generation_failed", error=str(e))
            return np.zeros(self._get_state_dim())

    def _get_state_dim(self) -> int:
        """Calculate state dimension.

        Returns:
            State dimension
        """
        dim = 0

        # Price and volume features per symbol
        dim += len(self.symbols) * self.config.lookback_window * 2

        # Position features per symbol (size, pnl)
        dim += len(self.symbols) * 2

        # Account features (balance, equity, total_pnl)
        dim += 3

        # Market regime
        if self.config.include_regime:
            dim += len(MarketRegime)

        return dim

    def _get_current_prices(self) -> Dict[str, Decimal]:
        """Get current market prices.

        Returns:
            Dictionary mapping symbols to current prices
        """
        prices = {}

        if self.market_data is None:
            return prices

        for symbol in self.symbols:
            symbol_data = self.market_data.filter(
                pl.col('symbol') == symbol
            ).sort('timestamp')

            if self.current_step < len(symbol_data):
                price = symbol_data[self.current_step].select('close')[0, 0]
                prices[symbol] = Decimal(str(price))

        return prices

    def _execute_trades(
        self,
        actions: np.ndarray,
        current_prices: Dict[str, Decimal]
    ) -> List[Dict[str, Any]]:
        """Execute trades based on actions.

        Args:
            actions: Action array
            current_prices: Current market prices

        Returns:
            List of trade execution info
        """
        trades = []

        for idx, symbol in enumerate(self.symbols):
            if symbol not in current_prices:
                continue

            action = actions[idx]
            target_size = Decimal(str(action)) * self.config.max_position_size * self.balance

            # Get current position
            current_size = self.positions[symbol].size if symbol in self.positions else Decimal("0")

            # Calculate trade size
            trade_size = target_size - current_size

            if abs(trade_size) < Decimal("0.01"):  # Minimum trade size
                continue

            # Execute trade
            trade_info = self._execute_trade(
                symbol,
                trade_size,
                current_prices[symbol]
            )

            if trade_info:
                trades.append(trade_info)

        return trades

    def _execute_trade(
        self,
        symbol: str,
        size: Decimal,
        price: Decimal
    ) -> Optional[Dict[str, Any]]:
        """Execute a single trade.

        Args:
            symbol: Trading symbol
            size: Trade size (positive for buy, negative for sell)
            price: Execution price

        Returns:
            Trade execution info or None if trade failed
        """
        try:
            # Apply slippage
            slippage = price * self.config.slippage_factor
            execution_price = price + slippage if size > 0 else price - slippage

            # Calculate fees
            notional = abs(size * execution_price)
            fee = notional * self.config.transaction_fee

            # Check balance
            if size > 0 and notional + fee > self.balance:
                logger.warning(
                    "insufficient_balance",
                    symbol=symbol,
                    required=float(notional + fee),
                    available=float(self.balance)
                )
                return None

            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]

                # Closing or reducing position
                if (pos.size > 0 and size < 0) or (pos.size < 0 and size > 0):
                    close_size = min(abs(size), abs(pos.size))

                    # Calculate realized PnL
                    pnl = close_size * (execution_price - pos.entry_price)
                    if pos.size < 0:
                        pnl = -pnl

                    pnl -= fee

                    self.realized_pnl += pnl
                    self.total_pnl += pnl
                    self.balance += pnl

                    # Update position
                    pos.size += size

                    # Track win/loss
                    if pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1

                    self.trade_count += 1

                    # Remove position if closed
                    if abs(pos.size) < Decimal("0.01"):
                        del self.positions[symbol]

                else:
                    # Increasing position
                    total_size = pos.size + size
                    pos.entry_price = (
                        (pos.entry_price * pos.size + execution_price * size) / total_size
                    )
                    pos.size = total_size
                    self.balance -= fee

            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    size=size,
                    entry_price=execution_price,
                    current_price=execution_price
                )
                self.balance -= fee
                self.trade_count += 1

            return {
                'symbol': symbol,
                'size': float(size),
                'price': float(execution_price),
                'fee': float(fee),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(
                "trade_execution_failed",
                symbol=symbol,
                error=str(e)
            )
            return None

    def _update_positions(self, current_prices: Dict[str, Decimal]) -> None:
        """Update position values and unrealized PnL.

        Args:
            current_prices: Current market prices
        """
        self.unrealized_pnl = Decimal("0")

        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                pos.update_pnl(current_prices[symbol])
                self.unrealized_pnl += pos.unrealized_pnl

        self.equity = self.balance + self.unrealized_pnl

    def _calculate_reward(self) -> float:
        """Calculate reward for current step.

        Returns:
            Reward value
        """
        # Use change in equity as base reward
        equity_change = self.equity - self.initial_balance

        # Normalize by initial balance
        reward = float(equity_change / self.initial_balance)

        # Apply scaling
        reward *= float(self.config.reward_scaling)

        # Penalty for excessive trading
        trade_penalty = -0.001 * self.trade_count / max(self.current_step, 1)

        # Penalty for large drawdown
        drawdown = (self.initial_balance - self.equity) / self.initial_balance
        drawdown_penalty = -float(max(Decimal("0"), drawdown * Decimal("10")))

        total_reward = reward + trade_penalty + drawdown_penalty

        return float(total_reward)

    def _check_done(self) -> bool:
        """Check if episode is done.

        Returns:
            True if episode should end
        """
        # Episode length limit
        if self.current_step >= self.max_steps - 1:
            return True

        # Liquidation check
        if self.equity < self.initial_balance * self.config.liquidation_threshold:
            logger.warning(
                "liquidation_triggered",
                equity=float(self.equity),
                threshold=float(self.initial_balance * self.config.liquidation_threshold)
            )
            return True

        return False

    def _normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize price array.

        Args:
            prices: Raw prices

        Returns:
            Normalized prices
        """
        if len(prices) == 0:
            return prices

        # Use percentage change from first price
        first_price = prices[0]
        if first_price > 0:
            return (prices - first_price) / first_price
        else:
            return np.zeros_like(prices)

    def _detect_market_regime(self) -> MarketRegime:
        """Detect current market regime.

        Returns:
            Market regime
        """
        # Simplified regime detection
        # In production, use more sophisticated methods

        try:
            if self.market_data is None or self.current_step < 50:
                return MarketRegime.RANGING

            # Get recent prices
            symbol_data = self.market_data.filter(
                pl.col('symbol') == self.symbols[0]
            ).sort('timestamp')

            recent_data = symbol_data[max(0, self.current_step - 50):self.current_step]
            closes = recent_data.select('close').to_numpy().flatten()

            if len(closes) < 20:
                return MarketRegime.RANGING

            # Calculate volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns)

            # Calculate trend
            trend = (closes[-1] - closes[0]) / closes[0]

            # Classify regime
            if volatility > 0.03:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.01:
                return MarketRegime.LOW_VOLATILITY
            elif trend > 0.05:
                return MarketRegime.TRENDING_UP
            elif trend < -0.05:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING

        except Exception as e:
            logger.error("regime_detection_failed", error=str(e))
            return MarketRegime.RANGING

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of current episode.

        Returns:
            Episode summary statistics
        """
        return {
            'initial_balance': float(self.initial_balance),
            'final_balance': float(self.balance),
            'final_equity': float(self.equity),
            'total_pnl': float(self.total_pnl),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'return_pct': float((self.equity - self.initial_balance) / self.initial_balance * 100),
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': float(self.win_count / max(self.trade_count, 1)),
            'steps': self.current_step,
            'total_reward': float(sum(self.episode_rewards))
        }

    def get_metrics_dataframe(self) -> pl.DataFrame:
        """Get episode metrics as Polars DataFrame.

        Returns:
            DataFrame with episode metrics
        """
        try:
            data = {
                'step': list(range(len(self.episode_rewards))),
                'reward': [float(r) for r in self.episode_rewards]
            }

            return pl.DataFrame(data)

        except Exception as e:
            logger.error("metrics_dataframe_creation_failed", error=str(e))
            return pl.DataFrame()
