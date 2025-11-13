"""
Quantum Trader Risk Management Module
CRITICAL: Comprehensive risk management, position sizing, and portfolio optimization
"""

# Core risk management
from quantum_trader.risk.risk_manager import RiskManager
from quantum_trader.risk.risk_calculator import RiskCalculator
from quantum_trader.risk.risk_limits import RiskLimitsManager, LimitType, LimitSeverity
from quantum_trader.risk.risk_monitor import RiskMonitor

# Risk metrics
from quantum_trader.risk.metrics.beta_calculator import BetaCalculator
from quantum_trader.risk.metrics.conditional_var import ConditionalVaRCalculator
from quantum_trader.risk.metrics.correlation_matrix import CorrelationMatrixCalculator
from quantum_trader.risk.metrics.max_drawdown import MaxDrawdownCalculator
from quantum_trader.risk.metrics.risk_adjusted_returns import RiskAdjustedReturnsCalculator
from quantum_trader.risk.metrics.sharpe_ratio import SharpeRatioCalculator
from quantum_trader.risk.metrics.value_at_risk import ValueAtRiskCalculator

# Portfolio management
from quantum_trader.risk.portfolio.allocation import PortfolioAllocator
from quantum_trader.risk.portfolio.diversification import DiversificationAnalyzer
from quantum_trader.risk.portfolio.hedging import HedgingManager
from quantum_trader.risk.portfolio.rebalancing import PortfolioRebalancer

# Position sizing
from quantum_trader.risk.position_sizing.dynamic_sizing import DynamicPositionSizer
from quantum_trader.risk.position_sizing.fixed_fractional import FixedFractionalSizer
from quantum_trader.risk.position_sizing.kelly_criterion import KellyCriterionSizer
from quantum_trader.risk.position_sizing.ml_position_sizer import MLPositionSizer
from quantum_trader.risk.position_sizing.risk_parity import RiskParitySizer
from quantum_trader.risk.position_sizing.volatility_based import VolatilityBasedSizer

# Circuit breakers
from quantum_trader.risk.circuit_breakers.cascade_protection import CascadeProtectionBreaker
from quantum_trader.risk.circuit_breakers.correlation_cb import CorrelationCircuitBreaker
from quantum_trader.risk.circuit_breakers.drawdown_cb import DrawdownCircuitBreaker
from quantum_trader.risk.circuit_breakers.volatility_cb import VolatilityCircuitBreaker

__all__ = [
    # Core risk management
    "RiskManager",
    "RiskCalculator",
    "RiskLimitsManager",
    "LimitType",
    "LimitSeverity",
    "RiskMonitor",

    # Risk metrics
    "BetaCalculator",
    "ConditionalVaRCalculator",
    "CorrelationMatrixCalculator",
    "MaxDrawdownCalculator",
    "RiskAdjustedReturnsCalculator",
    "SharpeRatioCalculator",
    "ValueAtRiskCalculator",

    # Portfolio management
    "PortfolioAllocator",
    "DiversificationAnalyzer",
    "HedgingManager",
    "PortfolioRebalancer",

    # Position sizing
    "DynamicPositionSizer",
    "FixedFractionalSizer",
    "KellyCriterionSizer",
    "MLPositionSizer",
    "RiskParitySizer",
    "VolatilityBasedSizer",

    # Circuit breakers
    "CascadeProtectionBreaker",
    "CorrelationCircuitBreaker",
    "DrawdownCircuitBreaker",
    "VolatilityCircuitBreaker",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader Development Team"
