"""
Pydantic schemas for risk management API endpoints.

Defines request and response models for risk checks,
limits, and portfolio risk metrics with validation.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, validator
from structlog import get_logger

logger = get_logger(__name__)


# Request Schemas

class RiskLimitUpdate(BaseModel):
    """Request to update risk limits."""

    max_position_size: Optional[str] = Field(None, description="Maximum position size")
    max_portfolio_exposure: Optional[str] = Field(None, description="Maximum portfolio exposure")
    max_daily_loss: Optional[str] = Field(None, description="Maximum daily loss")
    max_drawdown_percent: Optional[str] = Field(None, description="Maximum drawdown percentage")
    max_leverage: Optional[str] = Field(None, description="Maximum leverage allowed")

    @validator("max_position_size", "max_portfolio_exposure", "max_daily_loss", "max_drawdown_percent", "max_leverage")
    def validate_decimal(cls, v: Optional[str]) -> Optional[str]:
        """Validate decimal fields."""
        if v is None:
            return v

        try:
            decimal_val = Decimal(v)
            if decimal_val < 0:
                raise ValueError("Value must be non-negative")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "max_position_size": "10000.00",
                "max_portfolio_exposure": "50000.00",
                "max_daily_loss": "5000.00",
                "max_drawdown_percent": "10.0",
                "max_leverage": "3.0"
            }
        }


class RiskCheckRequest(BaseModel):
    """Request to perform risk check."""

    symbol: str = Field(..., description="Trading pair")
    side: str = Field(..., description="Order side (BUY/SELL)")
    quantity: str = Field(..., description="Order quantity")
    price: Optional[str] = Field(None, description="Order price")
    strategy: str = Field(..., description="Strategy identifier")

    @validator("quantity", "price")
    def validate_decimal(cls, v: Optional[str]) -> Optional[str]:
        """Validate decimal fields."""
        if v is None:
            return v

        try:
            decimal_val = Decimal(v)
            if decimal_val <= 0:
                raise ValueError("Value must be positive")
            return v
        except (ValueError, ArithmeticError) as e:
            raise ValueError(f"Invalid decimal value: {e}")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.1",
                "price": "50000.00",
                "strategy": "momentum_strategy"
            }
        }


# Response Schemas

class RiskLimits(BaseModel):
    """Current risk limits configuration."""

    max_position_size: str = Field(..., description="Maximum position size")
    max_portfolio_exposure: str = Field(..., description="Maximum portfolio exposure")
    max_daily_loss: str = Field(..., description="Maximum daily loss")
    max_drawdown_percent: str = Field(..., description="Maximum drawdown percentage")
    max_leverage: str = Field(..., description="Maximum leverage allowed")
    max_open_positions: int = Field(..., description="Maximum open positions")
    max_correlation: str = Field(..., description="Maximum correlation between positions")
    updated_at: str = Field(..., description="Last update timestamp (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "max_position_size": "10000.00",
                "max_portfolio_exposure": "50000.00",
                "max_daily_loss": "5000.00",
                "max_drawdown_percent": "10.0",
                "max_leverage": "3.0",
                "max_open_positions": 10,
                "max_correlation": "0.7",
                "updated_at": "2025-01-15T12:00:00Z"
            }
        }


class RiskCheckResult(BaseModel):
    """Result of risk check."""

    passed: bool = Field(..., description="Whether risk check passed")
    checks_performed: List[str] = Field(..., description="List of checks performed")
    violations: List[str] = Field(..., description="List of violations (if any)")
    warnings: List[str] = Field(..., description="List of warnings")
    risk_score: str = Field(..., description="Overall risk score (0-100)")
    exposure_impact: str = Field(..., description="Impact on portfolio exposure")
    leverage_impact: str = Field(..., description="Impact on leverage")
    correlation_impact: Optional[str] = Field(None, description="Impact on portfolio correlation")
    details: Dict[str, Any] = Field(..., description="Detailed check results")

    class Config:
        json_schema_extra = {
            "example": {
                "passed": True,
                "checks_performed": [
                    "position_size_check",
                    "portfolio_exposure_check",
                    "leverage_check",
                    "correlation_check"
                ],
                "violations": [],
                "warnings": [
                    "Portfolio exposure approaching limit (85%)"
                ],
                "risk_score": "65.5",
                "exposure_impact": "5000.00",
                "leverage_impact": "0.5",
                "correlation_impact": "0.05",
                "details": {
                    "current_exposure": "42500.00",
                    "new_exposure": "47500.00",
                    "exposure_limit": "50000.00"
                }
            }
        }


class PortfolioRisk(BaseModel):
    """Portfolio risk metrics."""

    total_exposure: str = Field(..., description="Total portfolio exposure")
    total_value: str = Field(..., description="Total portfolio value")
    leverage: str = Field(..., description="Current leverage")
    margin_used: str = Field(..., description="Margin used")
    margin_available: str = Field(..., description="Margin available")
    var_95: str = Field(..., description="Value at Risk (95% confidence)")
    var_99: str = Field(..., description="Value at Risk (99% confidence)")
    expected_shortfall: str = Field(..., description="Expected Shortfall (CVaR)")
    sharpe_ratio: Optional[str] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[str] = Field(None, description="Sortino ratio")
    max_drawdown: str = Field(..., description="Maximum drawdown")
    current_drawdown: str = Field(..., description="Current drawdown")
    beta: Optional[str] = Field(None, description="Portfolio beta")
    correlation_matrix: Optional[Dict[str, Dict[str, str]]] = Field(None, description="Asset correlation matrix")

    class Config:
        json_schema_extra = {
            "example": {
                "total_exposure": "47500.00",
                "total_value": "100000.00",
                "leverage": "1.5",
                "margin_used": "25000.00",
                "margin_available": "75000.00",
                "var_95": "2500.00",
                "var_99": "3500.00",
                "expected_shortfall": "4000.00",
                "sharpe_ratio": "1.85",
                "sortino_ratio": "2.15",
                "max_drawdown": "5000.00",
                "current_drawdown": "1200.00",
                "beta": "1.2"
            }
        }


class PositionRisk(BaseModel):
    """Individual position risk metrics."""

    position_id: str = Field(..., description="Position identifier")
    symbol: str = Field(..., description="Trading pair")
    exposure: str = Field(..., description="Position exposure")
    risk_amount: str = Field(..., description="Amount at risk")
    var_95: str = Field(..., description="Value at Risk (95%)")
    distance_to_stop_loss_percent: Optional[str] = Field(None, description="Distance to stop loss %")
    distance_to_liquidation_percent: Optional[str] = Field(None, description="Distance to liquidation %")
    contribution_to_portfolio_var: str = Field(..., description="Contribution to portfolio VaR")
    concentration_risk: str = Field(..., description="Concentration risk score")

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": "pos_1234567890",
                "symbol": "BTC/USDT",
                "exposure": "5000.00",
                "risk_amount": "250.00",
                "var_95": "200.00",
                "distance_to_stop_loss_percent": "4.0",
                "distance_to_liquidation_percent": None,
                "contribution_to_portfolio_var": "125.00",
                "concentration_risk": "15.5"
            }
        }


class RiskMetricsSummary(BaseModel):
    """Summary of risk metrics."""

    timestamp: str = Field(..., description="Metrics timestamp (ISO 8601)")
    portfolio_risk: PortfolioRisk = Field(..., description="Portfolio-level risk metrics")
    position_risks: List[PositionRisk] = Field(..., description="Per-position risk metrics")
    limit_utilization: Dict[str, str] = Field(..., description="Risk limit utilization percentages")
    risk_alerts: List[str] = Field(..., description="Active risk alerts")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-15T12:00:00Z",
                "portfolio_risk": {
                    "total_exposure": "47500.00",
                    "total_value": "100000.00",
                    "leverage": "1.5",
                    "var_95": "2500.00",
                    "max_drawdown": "5000.00",
                    "current_drawdown": "1200.00"
                },
                "position_risks": [],
                "limit_utilization": {
                    "exposure": "95.0",
                    "leverage": "50.0",
                    "drawdown": "24.0"
                },
                "risk_alerts": [
                    "Portfolio exposure approaching limit"
                ]
            }
        }


class StressTestResult(BaseModel):
    """Stress test result."""

    scenario_name: str = Field(..., description="Stress scenario name")
    description: str = Field(..., description="Scenario description")
    price_shocks: Dict[str, str] = Field(..., description="Price shocks applied")
    portfolio_impact: str = Field(..., description="Portfolio P&L impact")
    portfolio_impact_percent: str = Field(..., description="Portfolio P&L impact %")
    new_portfolio_value: str = Field(..., description="Portfolio value after shock")
    margin_call_triggered: bool = Field(..., description="Whether margin call triggered")
    liquidations: List[str] = Field(..., description="Positions that would be liquidated")

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_name": "market_crash_10pct",
                "description": "10% market crash across all positions",
                "price_shocks": {
                    "BTC/USDT": "-10.0",
                    "ETH/USDT": "-10.0"
                },
                "portfolio_impact": "-4750.00",
                "portfolio_impact_percent": "-4.75",
                "new_portfolio_value": "95250.00",
                "margin_call_triggered": False,
                "liquidations": []
            }
        }


class StressTestResponse(BaseModel):
    """Stress test response."""

    test_id: str = Field(..., description="Stress test ID")
    timestamp: str = Field(..., description="Test timestamp (ISO 8601)")
    scenarios_tested: int = Field(..., description="Number of scenarios tested")
    worst_case_scenario: str = Field(..., description="Worst case scenario name")
    worst_case_impact: str = Field(..., description="Worst case P&L impact")
    results: List[StressTestResult] = Field(..., description="Detailed scenario results")

    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "test_1234567890",
                "timestamp": "2025-01-15T12:00:00Z",
                "scenarios_tested": 5,
                "worst_case_scenario": "market_crash_20pct",
                "worst_case_impact": "-9500.00",
                "results": []
            }
        }


class RiskAlert(BaseModel):
    """Risk alert."""

    alert_id: str = Field(..., description="Alert identifier")
    severity: str = Field(..., description="Alert severity (INFO, WARNING, CRITICAL)")
    category: str = Field(..., description="Risk category")
    message: str = Field(..., description="Alert message")
    triggered_at: str = Field(..., description="Alert trigger timestamp (ISO 8601)")
    current_value: str = Field(..., description="Current metric value")
    threshold_value: str = Field(..., description="Threshold that was crossed")
    acknowledged: bool = Field(..., description="Whether alert acknowledged")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional alert data")

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_1234567890",
                "severity": "WARNING",
                "category": "exposure",
                "message": "Portfolio exposure exceeds 90% of limit",
                "triggered_at": "2025-01-15T12:00:00Z",
                "current_value": "45500.00",
                "threshold_value": "45000.00",
                "acknowledged": False,
                "metadata": {
                    "limit": "50000.00",
                    "utilization_percent": "91.0"
                }
            }
        }


class RiskAlertsResponse(BaseModel):
    """Risk alerts response."""

    alerts: List[RiskAlert] = Field(..., description="Active risk alerts")
    total: int = Field(..., description="Total number of alerts")
    critical: int = Field(..., description="Number of critical alerts")
    warning: int = Field(..., description="Number of warning alerts")
    info: int = Field(..., description="Number of info alerts")

    class Config:
        json_schema_extra = {
            "example": {
                "alerts": [],
                "total": 3,
                "critical": 0,
                "warning": 2,
                "info": 1
            }
        }


class CorrelationMatrix(BaseModel):
    """Asset correlation matrix."""

    symbols: List[str] = Field(..., description="List of symbols")
    correlations: Dict[str, Dict[str, str]] = Field(..., description="Correlation matrix")
    computed_at: str = Field(..., description="Computation timestamp (ISO 8601)")
    lookback_period_days: int = Field(..., description="Lookback period in days")

    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "correlations": {
                    "BTC/USDT": {
                        "BTC/USDT": "1.0",
                        "ETH/USDT": "0.85"
                    },
                    "ETH/USDT": {
                        "BTC/USDT": "0.85",
                        "ETH/USDT": "1.0"
                    }
                },
                "computed_at": "2025-01-15T12:00:00Z",
                "lookback_period_days": 30
            }
        }


class PositionSizingRecommendation(BaseModel):
    """Position sizing recommendation."""

    symbol: str = Field(..., description="Trading pair")
    recommended_quantity: str = Field(..., description="Recommended position size")
    max_safe_quantity: str = Field(..., description="Maximum safe position size")
    risk_per_trade_percent: str = Field(..., description="Risk per trade %")
    expected_loss_per_unit: str = Field(..., description="Expected loss per unit")
    justification: str = Field(..., description="Recommendation justification")
    warnings: List[str] = Field(..., description="Any warnings or caveats")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTC/USDT",
                "recommended_quantity": "0.08",
                "max_safe_quantity": "0.12",
                "risk_per_trade_percent": "2.0",
                "expected_loss_per_unit": "100.00",
                "justification": "Based on 2% portfolio risk and stop loss distance",
                "warnings": [
                    "High volatility may increase actual risk"
                ]
            }
        }
