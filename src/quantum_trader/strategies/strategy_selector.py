"""Strategy Selector for Adaptive Strategy Selection.

Dynamically selects optimal trading strategies based on real-time market
conditions, performance metrics, and risk parameters.

Performance Target: <100ms selection decision
Capital Allocation: N/A (selection module)
Risk: Suboptimal strategy selection, market regime misclassification
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field

import polars as pl
import numpy as np
from structlog import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum"


@dataclass
class MarketConditions:
    """Current market conditions analysis."""
    regime: MarketRegime
    volatility: Decimal
    trend_strength: Decimal
    volume_profile: str  # 'low', 'medium', 'high'
    spread_environment: str  # 'tight', 'normal', 'wide'
    liquidity_score: Decimal
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyScore:
    """Strategy suitability score."""
    strategy_type: str
    suitability_score: Decimal
    performance_score: Decimal
    risk_score: Decimal
    combined_score: Decimal
    reasons: List[str]
    timestamp: datetime


class StrategySelector:
    """Adaptive strategy selector.

    Selects optimal trading strategies based on:
    - Current market regime
    - Strategy historical performance
    - Risk-adjusted returns
    - Market conditions
    - Resource constraints

    Key Features:
    - Market regime detection
    - Performance-based selection
    - Risk-adjusted scoring
    - Multi-criteria optimization
    - Dynamic rebalancing

    Attributes:
        config: Selector configuration
        strategy_manager: Strategy manager instance
        performance_tracker: Performance tracking
        selected_strategies: Currently selected strategies

    Example:
        >>> selector = StrategySelector(config, strategy_manager)
        >>> market_conditions = await selector.analyze_market(market_data)
        >>> selected = await selector.select_strategies(market_conditions)
        >>> allocations = selector.calculate_allocations(selected)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        strategy_manager: Any
    ) -> None:
        """Initialize strategy selector.

        Args:
            config: Selector configuration
            strategy_manager: Strategy manager instance

        Raises:
            ValueError: If configuration invalid
        """
        self.config = config
        self.strategy_manager = strategy_manager

        # Selection parameters
        self.max_active_strategies = config.get('max_active_strategies', 5)
        self.min_suitability_score = Decimal(str(config.get('min_suitability_score', 0.6)))
        self.rebalance_frequency_hours = config.get('rebalance_frequency_hours', 24)

        # Performance weighting
        self.performance_weight = Decimal(str(config.get('performance_weight', 0.4)))
        self.suitability_weight = Decimal(str(config.get('suitability_weight', 0.4)))
        self.risk_weight = Decimal(str(config.get('risk_weight', 0.2)))

        # Strategy-regime mapping
        self.strategy_regime_fit = self._initialize_strategy_regime_fit()

        # State tracking
        self.selected_strategies: Set[str] = set()
        self.last_selection_time: Optional[datetime] = None
        self.current_regime: Optional[MarketRegime] = None

        # Performance history
        self.performance_history: Dict[str, List[Decimal]] = {}
        self.selection_history: List[Dict[str, Any]] = []

        logger.info(
            "strategy_selector_initialized",
            max_active_strategies=self.max_active_strategies,
            min_suitability_score=float(self.min_suitability_score)
        )

    def _initialize_strategy_regime_fit(self) -> Dict[str, Dict[MarketRegime, Decimal]]:
        """Initialize strategy-regime suitability mapping.

        Returns:
            Dictionary mapping strategy types to regime suitability scores
        """
        return {
            'queue_racing': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.9'),
                MarketRegime.RANGE_BOUND: Decimal('0.8'),
                MarketRegime.TRENDING_UP: Decimal('0.5'),
                MarketRegime.TRENDING_DOWN: Decimal('0.5'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.3'),
                MarketRegime.MEAN_REVERTING: Decimal('0.6'),
                MarketRegime.MOMENTUM: Decimal('0.4')
            },
            'scalping': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.8'),
                MarketRegime.RANGE_BOUND: Decimal('0.9'),
                MarketRegime.TRENDING_UP: Decimal('0.7'),
                MarketRegime.TRENDING_DOWN: Decimal('0.7'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.4'),
                MarketRegime.MEAN_REVERTING: Decimal('0.8'),
                MarketRegime.MOMENTUM: Decimal('0.6')
            },
            'rsi_reversion': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.7'),
                MarketRegime.RANGE_BOUND: Decimal('0.9'),
                MarketRegime.TRENDING_UP: Decimal('0.4'),
                MarketRegime.TRENDING_DOWN: Decimal('0.4'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.6'),
                MarketRegime.MEAN_REVERTING: Decimal('0.95'),
                MarketRegime.MOMENTUM: Decimal('0.3')
            },
            'statistical_arb': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.8'),
                MarketRegime.RANGE_BOUND: Decimal('0.85'),
                MarketRegime.TRENDING_UP: Decimal('0.6'),
                MarketRegime.TRENDING_DOWN: Decimal('0.6'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.5'),
                MarketRegime.MEAN_REVERTING: Decimal('0.9'),
                MarketRegime.MOMENTUM: Decimal('0.5')
            },
            'rebate_capture': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.95'),
                MarketRegime.RANGE_BOUND: Decimal('0.9'),
                MarketRegime.TRENDING_UP: Decimal('0.6'),
                MarketRegime.TRENDING_DOWN: Decimal('0.6'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.3'),
                MarketRegime.MEAN_REVERTING: Decimal('0.7'),
                MarketRegime.MOMENTUM: Decimal('0.5')
            },
            'options_spreads': {
                MarketRegime.LOW_VOLATILITY: Decimal('0.6'),
                MarketRegime.RANGE_BOUND: Decimal('0.85'),
                MarketRegime.TRENDING_UP: Decimal('0.7'),
                MarketRegime.TRENDING_DOWN: Decimal('0.7'),
                MarketRegime.HIGH_VOLATILITY: Decimal('0.9'),
                MarketRegime.MEAN_REVERTING: Decimal('0.7'),
                MarketRegime.MOMENTUM: Decimal('0.6')
            }
        }

    async def analyze_market(self, market_data: pl.DataFrame) -> MarketConditions:
        """Analyze current market conditions and regime.

        Args:
            market_data: Recent market data

        Returns:
            MarketConditions analysis

        Raises:
            ValueError: If market data insufficient
        """
        if market_data.is_empty():
            raise ValueError("Empty market data provided")

        try:
            # Calculate volatility
            volatility = self._calculate_volatility(market_data)

            # Determine trend strength
            trend_strength = self._calculate_trend_strength(market_data)

            # Classify market regime
            regime = self._classify_regime(volatility, trend_strength, market_data)

            # Analyze volume
            volume_profile = self._analyze_volume(market_data)

            # Analyze spread environment
            spread_environment = self._analyze_spreads(market_data)

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(market_data)

            conditions = MarketConditions(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                spread_environment=spread_environment,
                liquidity_score=liquidity_score,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'data_points': market_data.height,
                    'symbols_analyzed': market_data['symbol'].n_unique() if 'symbol' in market_data.columns else 1
                }
            )

            self.current_regime = regime

            logger.info(
                "market_conditions_analyzed",
                regime=regime.value,
                volatility=float(volatility),
                trend_strength=float(trend_strength)
            )

            return conditions

        except Exception as e:
            logger.error("market_analysis_failed", error=str(e))
            raise

    def _calculate_volatility(self, data: pl.DataFrame) -> Decimal:
        """Calculate market volatility.

        Args:
            data: Market data

        Returns:
            Volatility measure (annualized standard deviation)
        """
        if 'close' not in data.columns or data.height < 10:
            return Decimal('0.01')

        prices = data['close'].to_numpy().astype(float)
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        return Decimal(str(volatility))

    def _calculate_trend_strength(self, data: pl.DataFrame) -> Decimal:
        """Calculate trend strength.

        Args:
            data: Market data

        Returns:
            Trend strength (0-1)
        """
        if 'close' not in data.columns or data.height < 20:
            return Decimal('0.5')

        prices = data['close'].to_numpy().astype(float)

        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize slope to 0-1 range
        # Assume 5% move over period is strong trend
        trend_strength = min(abs(slope) / (prices[-1] * 0.05 / len(prices)), 1.0)

        return Decimal(str(trend_strength))

    def _classify_regime(
        self,
        volatility: Decimal,
        trend_strength: Decimal,
        data: pl.DataFrame
    ) -> MarketRegime:
        """Classify market regime.

        Args:
            volatility: Current volatility
            trend_strength: Trend strength
            data: Market data

        Returns:
            Classified market regime
        """
        # High/low volatility threshold
        if volatility > Decimal('0.3'):
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < Decimal('0.1'):
            if trend_strength > Decimal('0.6'):
                # Low vol with trend
                return MarketRegime.TRENDING_UP if self._is_uptrend(data) else MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.LOW_VOLATILITY

        # Medium volatility - check trend
        if trend_strength > Decimal('0.6'):
            return MarketRegime.TRENDING_UP if self._is_uptrend(data) else MarketRegime.TRENDING_DOWN
        elif trend_strength < Decimal('0.3'):
            return MarketRegime.RANGE_BOUND
        else:
            # Check for mean reversion
            if self._detect_mean_reversion(data):
                return MarketRegime.MEAN_REVERTING
            else:
                return MarketRegime.MOMENTUM

    def _is_uptrend(self, data: pl.DataFrame) -> bool:
        """Check if market is in uptrend.

        Args:
            data: Market data

        Returns:
            True if uptrending
        """
        if 'close' not in data.columns or data.height < 2:
            return True

        prices = data['close'].to_numpy().astype(float)
        return prices[-1] > prices[0]

    def _detect_mean_reversion(self, data: pl.DataFrame) -> bool:
        """Detect if market exhibits mean reversion.

        Args:
            data: Market data

        Returns:
            True if mean reverting
        """
        # Simplified: check if price oscillates around mean
        if 'close' not in data.columns or data.height < 10:
            return False

        prices = data['close'].to_numpy().astype(float)
        mean_price = np.mean(prices)

        # Count crosses of mean
        crosses = np.sum(np.diff(np.sign(prices - mean_price)) != 0)

        # Mean reverting if crosses frequently
        return crosses > len(prices) * 0.3

    def _analyze_volume(self, data: pl.DataFrame) -> str:
        """Analyze volume profile.

        Args:
            data: Market data

        Returns:
            Volume profile classification
        """
        if 'volume' not in data.columns or data.height < 10:
            return 'medium'

        volumes = data['volume'].to_numpy().astype(float)
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-5:])

        if recent_volume > avg_volume * 1.5:
            return 'high'
        elif recent_volume < avg_volume * 0.5:
            return 'low'
        else:
            return 'medium'

    def _analyze_spreads(self, data: pl.DataFrame) -> str:
        """Analyze spread environment.

        Args:
            data: Market data

        Returns:
            Spread environment classification
        """
        # Simplified analysis
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            bids = data['bid_price'].to_numpy().astype(float)
            asks = data['ask_price'].to_numpy().astype(float)
            spreads = (asks - bids) / ((asks + bids) / 2) * 10000  # bps

            avg_spread = np.mean(spreads)

            if avg_spread < 5:
                return 'tight'
            elif avg_spread > 15:
                return 'wide'
            else:
                return 'normal'

        return 'normal'

    def _calculate_liquidity_score(self, data: pl.DataFrame) -> Decimal:
        """Calculate liquidity score.

        Args:
            data: Market data

        Returns:
            Liquidity score (0-1)
        """
        # Simplified liquidity score based on volume and spread
        if 'volume' in data.columns:
            volumes = data['volume'].to_numpy().astype(float)
            avg_volume = np.mean(volumes)

            # Normalize to 0-1 (assume 1M is high liquidity)
            liquidity = min(avg_volume / 1000000, 1.0)

            return Decimal(str(liquidity))

        return Decimal('0.5')

    async def select_strategies(
        self,
        market_conditions: MarketConditions
    ) -> List[str]:
        """Select optimal strategies for current market conditions.

        Args:
            market_conditions: Current market conditions

        Returns:
            List of selected strategy types

        Raises:
            RuntimeError: If selection fails
        """
        try:
            # Get all available strategies
            available_strategies = self.strategy_manager.strategies.keys()

            # Score each strategy
            scores = []
            for strategy_id in available_strategies:
                score = await self._score_strategy(strategy_id, market_conditions)
                if score and score.combined_score >= self.min_suitability_score:
                    scores.append(score)

            # Sort by combined score
            scores.sort(key=lambda s: s.combined_score, reverse=True)

            # Select top N strategies
            selected = [s.strategy_type for s in scores[:self.max_active_strategies]]

            # Update state
            self.selected_strategies = set(selected)
            self.last_selection_time = datetime.now(timezone.utc)

            # Store in history
            self.selection_history.append({
                'timestamp': datetime.now(timezone.utc),
                'selected': selected,
                'market_regime': market_conditions.regime.value,
                'scores': [{'strategy': s.strategy_type, 'score': float(s.combined_score)} for s in scores]
            })

            logger.info(
                "strategies_selected",
                count=len(selected),
                strategies=selected,
                regime=market_conditions.regime.value
            )

            return selected

        except Exception as e:
            logger.error("strategy_selection_failed", error=str(e))
            raise

    async def _score_strategy(
        self,
        strategy_id: str,
        market_conditions: MarketConditions
    ) -> Optional[StrategyScore]:
        """Score a strategy's suitability.

        Args:
            strategy_id: Strategy identifier
            market_conditions: Market conditions

        Returns:
            StrategyScore or None
        """
        # Get strategy status
        status = self.strategy_manager.get_strategy_status(strategy_id)
        if not status:
            return None

        strategy_type = status.strategy_type

        # Calculate suitability score based on regime fit
        regime_fit = self.strategy_regime_fit.get(strategy_type, {})
        suitability_score = regime_fit.get(market_conditions.regime, Decimal('0.5'))

        # Calculate performance score (simplified)
        performance_score = self._calculate_performance_score(strategy_id)

        # Calculate risk score
        risk_score = self._calculate_risk_score(strategy_id)

        # Combined score (weighted average)
        combined_score = (
            suitability_score * self.suitability_weight +
            performance_score * self.performance_weight +
            risk_score * self.risk_weight
        )

        reasons = []
        if suitability_score > Decimal('0.8'):
            reasons.append(f"High fit for {market_conditions.regime.value}")
        if performance_score > Decimal('0.7'):
            reasons.append("Strong recent performance")
        if risk_score > Decimal('0.8'):
            reasons.append("Low risk profile")

        return StrategyScore(
            strategy_type=strategy_type,
            suitability_score=suitability_score,
            performance_score=performance_score,
            risk_score=risk_score,
            combined_score=combined_score,
            reasons=reasons,
            timestamp=datetime.now(timezone.utc)
        )

    def _calculate_performance_score(self, strategy_id: str) -> Decimal:
        """Calculate strategy performance score.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance score (0-1)
        """
        # Simplified: use signal generation rate as proxy
        status = self.strategy_manager.get_strategy_status(strategy_id)
        if not status:
            return Decimal('0.5')

        # Normalize signals generated to 0-1
        # Assume 100 signals is good performance
        score = min(Decimal(str(status.signals_generated)) / Decimal('100'), Decimal('1'))

        return score

    def _calculate_risk_score(self, strategy_id: str) -> Decimal:
        """Calculate strategy risk score.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Risk score (0-1, higher is lower risk)
        """
        status = self.strategy_manager.get_strategy_status(strategy_id)
        if not status:
            return Decimal('0.5')

        # Lower error count = higher score
        if status.error_count == 0:
            return Decimal('1.0')
        elif status.error_count < 5:
            return Decimal('0.8')
        elif status.error_count < 10:
            return Decimal('0.5')
        else:
            return Decimal('0.2')

    def calculate_allocations(self, selected_strategies: List[str]) -> Dict[str, Decimal]:
        """Calculate capital allocations for selected strategies.

        Args:
            selected_strategies: List of selected strategy types

        Returns:
            Dictionary mapping strategy types to allocation percentages
        """
        if not selected_strategies:
            return {}

        # Equal weight allocation (simplified)
        equal_weight = Decimal('100') / Decimal(str(len(selected_strategies)))

        allocations = {strategy: equal_weight for strategy in selected_strategies}

        logger.debug(
            "allocations_calculated",
            strategies=len(selected_strategies),
            allocation_per_strategy=float(equal_weight)
        )

        return allocations

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get selector performance metrics.

        Returns:
            Performance metrics dictionary
        """
        return {
            'selected_strategies_count': len(self.selected_strategies),
            'selected_strategies': list(self.selected_strategies),
            'current_regime': self.current_regime.value if self.current_regime else None,
            'last_selection_time': self.last_selection_time,
            'selection_history_length': len(self.selection_history)
        }
