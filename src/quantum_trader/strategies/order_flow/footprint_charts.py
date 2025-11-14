"""
Footprint Charts Order Flow Strategy.

Implements footprint chart analysis for order flow trading, visualizing
buying and selling activity at each price level within a bar.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict

import polars as pl
from structlog import get_logger

from quantum_trader.models import Signal, SignalAction
from quantum_trader.strategies.base_strategy import BaseStrategy
from quantum_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class FootprintChartsStrategy(BaseStrategy):
    """Footprint charts order flow strategy.

    This strategy analyzes price-level volume distribution within bars
    to identify:
    - Point of Control (POC) - price level with highest volume
    - High Volume Nodes (HVN) - areas of high acceptance
    - Low Volume Nodes (LVN) - areas of low acceptance/rejection
    - Imbalances at specific price levels
    - Absorption and exhaustion patterns

    Features:
    - Price-level volume profiling
    - Bid/ask volume tracking per price
    - Delta analysis at each price level
    - Support/resistance from volume clusters

    Attributes:
        min_imbalance_ratio: Minimum bid/ask ratio for imbalance
        poc_threshold: Threshold for POC identification
        hvn_multiplier: Volume multiplier for HVN identification

    Example:
        >>> config = {
        ...     "min_imbalance_ratio": "2.0",
        ...     "poc_threshold": "0.2",
        ...     "hvn_multiplier": "1.5"
        ... }
        >>> strategy = FootprintChartsStrategy(config, risk_manager)
    """

    def __init__(self, config: Dict[str, Any], risk_manager: RiskManager) -> None:
        """Initialize footprint charts strategy.

        Args:
            config: Strategy configuration
            risk_manager: Risk manager instance

        Raises:
            ValueError: If required parameters missing
        """
        super().__init__(config, risk_manager)

        # Footprint parameters
        self.min_imbalance_ratio = Decimal(str(config.get("min_imbalance_ratio", "2.0")))
        self.poc_threshold = Decimal(str(config.get("poc_threshold", "0.2")))  # 20% of total volume
        self.hvn_multiplier = Decimal(str(config.get("hvn_multiplier", "1.5")))
        self.tick_size = Decimal(str(config.get("tick_size", "0.01")))

        # Volume profile tracking
        self.price_volume_profile: Dict[str, Dict[Decimal, Dict]] = {}  # {symbol: {price: {bid_vol, ask_vol, delta}}}
        self.poc_levels: Dict[str, Decimal] = {}  # Point of Control by symbol
        self.hvn_levels: Dict[str, List[Decimal]] = {}  # High Volume Nodes
        self.lvn_levels: Dict[str, List[Decimal]] = {}  # Low Volume Nodes

        # Imbalance tracking
        self.price_imbalances: Dict[str, List[Dict]] = {}  # List of imbalances by symbol
        self.last_footprint: Dict[str, Dict] = {}

        logger.info(
            "Footprint charts strategy initialized",
            strategy=self.name,
            min_imbalance_ratio=self.min_imbalance_ratio,
            poc_threshold=self.poc_threshold
        )

    def _validate_config(self) -> None:
        """Validate strategy configuration.

        Raises:
            ValueError: If parameters invalid
        """
        super()._validate_config()

        imbalance = Decimal(str(self.config.get("min_imbalance_ratio", "2.0")))
        if imbalance <= Decimal("1"):
            raise ValueError(f"min_imbalance_ratio must be > 1: {imbalance}")

        poc = Decimal(str(self.config.get("poc_threshold", "0.2")))
        if poc <= Decimal("0") or poc > Decimal("1"):
            raise ValueError(f"poc_threshold must be between 0 and 1: {poc}")

    def build_volume_profile(
        self,
        data: pl.DataFrame,
        symbol: str
    ) -> Dict[Decimal, Dict[str, Decimal]]:
        """Build volume profile from bar data.

        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            Dictionary mapping price to volume statistics
        """
        try:
            profile: Dict[Decimal, Dict[str, Decimal]] = defaultdict(
                lambda: {"bid_volume": Decimal("0"), "ask_volume": Decimal("0"), "total_volume": Decimal("0"), "delta": Decimal("0")}
            )

            rows = data.to_dicts()

            for row in rows:
                open_price = Decimal(str(row["open"]))
                close_price = Decimal(str(row["close"]))
                high_price = Decimal(str(row["high"]))
                low_price = Decimal(str(row["low"]))
                volume = Decimal(str(row["volume"]))

                # Distribute volume across price levels within the bar
                # Using tick size to create price levels
                price_range = high_price - low_price

                if price_range == Decimal("0"):
                    # All volume at single price
                    price_level = close_price.quantize(self.tick_size)
                    profile[price_level]["total_volume"] += volume

                    # Determine bid/ask based on close vs open
                    if close_price >= open_price:
                        profile[price_level]["bid_volume"] += volume * Decimal("0.6")
                        profile[price_level]["ask_volume"] += volume * Decimal("0.4")
                    else:
                        profile[price_level]["bid_volume"] += volume * Decimal("0.4")
                        profile[price_level]["ask_volume"] += volume * Decimal("0.6")

                else:
                    # Distribute volume across price levels
                    num_levels = int((price_range / self.tick_size)) + 1
                    num_levels = min(num_levels, 100)  # Cap at 100 levels

                    volume_per_level = volume / Decimal(str(num_levels))

                    for i in range(num_levels):
                        price_level = (low_price + (self.tick_size * Decimal(str(i)))).quantize(self.tick_size)

                        profile[price_level]["total_volume"] += volume_per_level

                        # Weight bid/ask based on position in range
                        if price_level < close_price:
                            # Below close = more support/buying
                            profile[price_level]["bid_volume"] += volume_per_level * Decimal("0.7")
                            profile[price_level]["ask_volume"] += volume_per_level * Decimal("0.3")
                        elif price_level > close_price:
                            # Above close = more resistance/selling
                            profile[price_level]["bid_volume"] += volume_per_level * Decimal("0.3")
                            profile[price_level]["ask_volume"] += volume_per_level * Decimal("0.7")
                        else:
                            # At close = balanced
                            profile[price_level]["bid_volume"] += volume_per_level * Decimal("0.5")
                            profile[price_level]["ask_volume"] += volume_per_level * Decimal("0.5")

            # Calculate deltas
            for price_level in profile:
                bid_vol = profile[price_level]["bid_volume"]
                ask_vol = profile[price_level]["ask_volume"]
                profile[price_level]["delta"] = bid_vol - ask_vol

            logger.debug(
                "Volume profile built",
                symbol=symbol,
                price_levels=len(profile)
            )

            return dict(profile)

        except Exception as e:
            logger.error("Error building volume profile", error=str(e))
            return {}

    def identify_poc(
        self,
        profile: Dict[Decimal, Dict[str, Decimal]]
    ) -> Optional[Decimal]:
        """Identify Point of Control (price with highest volume).

        Args:
            profile: Volume profile

        Returns:
            POC price level or None
        """
        try:
            if not profile:
                return None

            # Find price with highest total volume
            max_volume = Decimal("0")
            poc_price = None

            for price, stats in profile.items():
                total_vol = stats["total_volume"]
                if total_vol > max_volume:
                    max_volume = total_vol
                    poc_price = price

            logger.debug("POC identified", poc_price=poc_price, volume=max_volume)

            return poc_price

        except Exception as e:
            logger.error("Error identifying POC", error=str(e))
            return None

    def identify_hvn_lvn(
        self,
        profile: Dict[Decimal, Dict[str, Decimal]]
    ) -> Tuple[List[Decimal], List[Decimal]]:
        """Identify High Volume Nodes and Low Volume Nodes.

        Args:
            profile: Volume profile

        Returns:
            Tuple of (HVN list, LVN list)
        """
        try:
            if not profile:
                return [], []

            # Calculate average volume
            volumes = [stats["total_volume"] for stats in profile.values()]
            avg_volume = sum(volumes) / Decimal(str(len(volumes)))

            hvn_threshold = avg_volume * self.hvn_multiplier
            lvn_threshold = avg_volume * (Decimal("1") / self.hvn_multiplier)

            hvn_levels = []
            lvn_levels = []

            for price, stats in profile.items():
                total_vol = stats["total_volume"]

                if total_vol >= hvn_threshold:
                    hvn_levels.append(price)
                elif total_vol <= lvn_threshold:
                    lvn_levels.append(price)

            logger.debug(
                "HVN/LVN identified",
                hvn_count=len(hvn_levels),
                lvn_count=len(lvn_levels),
                avg_volume=avg_volume
            )

            return sorted(hvn_levels), sorted(lvn_levels)

        except Exception as e:
            logger.error("Error identifying HVN/LVN", error=str(e))
            return [], []

    def detect_price_imbalances(
        self,
        profile: Dict[Decimal, Dict[str, Decimal]]
    ) -> List[Dict[str, Any]]:
        """Detect significant bid/ask imbalances at price levels.

        Args:
            profile: Volume profile

        Returns:
            List of imbalance dictionaries
        """
        try:
            imbalances = []

            for price, stats in profile.items():
                bid_vol = stats["bid_volume"]
                ask_vol = stats["ask_volume"]

                if ask_vol == Decimal("0") and bid_vol > Decimal("0"):
                    # Pure bid imbalance
                    imbalances.append({
                        "price": price,
                        "type": "bid",
                        "ratio": float("inf"),
                        "volume": bid_vol
                    })
                elif bid_vol == Decimal("0") and ask_vol > Decimal("0"):
                    # Pure ask imbalance
                    imbalances.append({
                        "price": price,
                        "type": "ask",
                        "ratio": float("inf"),
                        "volume": ask_vol
                    })
                elif bid_vol > Decimal("0") and ask_vol > Decimal("0"):
                    # Calculate ratio
                    ratio = bid_vol / ask_vol

                    if ratio >= self.min_imbalance_ratio:
                        imbalances.append({
                            "price": price,
                            "type": "bid",
                            "ratio": float(ratio),
                            "volume": bid_vol
                        })
                    elif (Decimal("1") / ratio) >= self.min_imbalance_ratio:
                        imbalances.append({
                            "price": price,
                            "type": "ask",
                            "ratio": float(ask_vol / bid_vol),
                            "volume": ask_vol
                        })

            logger.debug("Price imbalances detected", count=len(imbalances))

            return imbalances

        except Exception as e:
            logger.error("Error detecting imbalances", error=str(e))
            return []

    def generate_signals(self, market_data: pl.DataFrame) -> List[Signal]:
        """Generate signals from footprint analysis.

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            List of Signal objects
        """
        signals: List[Signal] = []

        try:
            # Validate data
            is_valid, error = self.validate_market_data(market_data)
            if not is_valid:
                logger.warning("Invalid market data", error=error)
                return signals

            if not self.active:
                return signals

            # Get symbol
            symbols = self.config.get("symbols", [])
            if not symbols:
                logger.warning("No symbols configured")
                return signals

            symbol = symbols[0]

            # Build volume profile
            profile = self.build_volume_profile(market_data, symbol)
            self.price_volume_profile[symbol] = profile

            # Identify POC
            poc = self.identify_poc(profile)
            if poc:
                self.poc_levels[symbol] = poc

            # Identify HVN and LVN
            hvn, lvn = self.identify_hvn_lvn(profile)
            self.hvn_levels[symbol] = hvn
            self.lvn_levels[symbol] = lvn

            # Detect imbalances
            imbalances = self.detect_price_imbalances(profile)
            self.price_imbalances[symbol] = imbalances

            # Get current price
            current_price = Decimal(str(market_data["close"][-1]))

            # Generate signals based on footprint analysis
            action = SignalAction.HOLD
            strength = Decimal("0")
            confidence = Decimal("0.5")

            # Check for bullish imbalances near current price
            bullish_imbalances = [
                imp for imp in imbalances
                if imp["type"] == "bid" and abs(imp["price"] - current_price) / current_price < Decimal("0.02")
            ]

            # Check for bearish imbalances near current price
            bearish_imbalances = [
                imp for imp in imbalances
                if imp["type"] == "ask" and abs(imp["price"] - current_price) / current_price < Decimal("0.02")
            ]

            if bullish_imbalances and len(bullish_imbalances) > len(bearish_imbalances):
                action = SignalAction.BUY
                strength = min(Decimal("1"), Decimal(str(len(bullish_imbalances))) / Decimal("5"))
                confidence = Decimal("0.7")

                logger.info(
                    "Bullish footprint signal",
                    symbol=symbol,
                    current_price=current_price,
                    bullish_imbalances=len(bullish_imbalances)
                )

            elif bearish_imbalances and len(bearish_imbalances) > len(bullish_imbalances):
                action = SignalAction.SELL
                strength = min(Decimal("1"), Decimal(str(len(bearish_imbalances))) / Decimal("5"))
                confidence = Decimal("0.7")

                logger.info(
                    "Bearish footprint signal",
                    symbol=symbol,
                    current_price=current_price,
                    bearish_imbalances=len(bearish_imbalances)
                )

            # Create signal if action determined
            if action != SignalAction.HOLD:
                signal = Signal(
                    symbol=symbol,
                    action=action,
                    strength=strength,
                    confidence=confidence,
                    timestamp=datetime.now(timezone.utc),
                    strategy=self.name,
                    timeframe=self.config.get("timeframe", "5m"),
                    indicators={
                        "poc": poc if poc else Decimal("0"),
                        "current_price": current_price,
                        "hvn_count": Decimal(str(len(hvn))),
                        "lvn_count": Decimal(str(len(lvn))),
                        "total_imbalances": Decimal(str(len(imbalances)))
                    },
                    metadata={
                        "strategy_type": "order_flow",
                        "analysis_type": "footprint",
                        "imbalances": imbalances[:5]  # Top 5 imbalances
                    }
                )

                if self.validate_signal(signal):
                    signals.append(signal)

                    # Update state
                    self.state["signals_generated"] = self.state.get("signals_generated", 0) + 1
                    self.state["last_signal_time"] = datetime.now(timezone.utc)

            return signals

        except Exception as e:
            logger.error("Error generating signals", error=str(e))
            return signals

    def calculate_indicators(self, data: pl.DataFrame) -> Dict[str, Decimal]:
        """Calculate footprint indicators.

        Args:
            data: Market data

        Returns:
            Dictionary of indicators
        """
        try:
            indicators = {}

            symbols = self.config.get("symbols", [])
            for symbol in symbols:
                if symbol in self.poc_levels:
                    indicators[f"{symbol}_poc"] = self.poc_levels[symbol]

                if symbol in self.hvn_levels:
                    indicators[f"{symbol}_hvn_count"] = Decimal(str(len(self.hvn_levels[symbol])))

                if symbol in self.lvn_levels:
                    indicators[f"{symbol}_lvn_count"] = Decimal(str(len(self.lvn_levels[symbol])))

                if symbol in self.price_imbalances:
                    indicators[f"{symbol}_imbalance_count"] = Decimal(str(len(self.price_imbalances[symbol])))

            return indicators

        except Exception as e:
            logger.error("Error calculating indicators", error=str(e))
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        base_metrics = super().get_performance_metrics()

        footprint_metrics = {
            "poc_levels": {symbol: float(poc) for symbol, poc in self.poc_levels.items()},
            "hvn_levels": {symbol: [float(p) for p in levels] for symbol, levels in self.hvn_levels.items()},
            "lvn_levels": {symbol: [float(p) for p in levels] for symbol, levels in self.lvn_levels.items()},
            "price_imbalances": self.price_imbalances
        }

        return {**base_metrics, **footprint_metrics}
