"""Base Arbitrage Strategy - PRODUCTION"""
from decimal import Decimal
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod
import os, logging
import polars as pl

logger = logging.getLogger(__name__)

class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity"""

    def __init__(
        self,
        opportunity_id: str,
        strategy_type: str,
        expected_profit: Decimal,
        legs: List[Dict[str, Any]],
        timestamp: datetime
    ):
        self.opportunity_id = opportunity_id
        self.strategy_type = strategy_type
        self.expected_profit = expected_profit
        self.legs = legs
        self.timestamp = timestamp
        self.status = 'detected'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'opportunity_id': self.opportunity_id,
            'strategy_type': self.strategy_type,
            'expected_profit': self.expected_profit,
            'legs': self.legs,
            'timestamp': self.timestamp,
            'status': self.status
        }

class BaseArbitrageStrategy(ABC):
    """Abstract base for all arbitrage strategies"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Core parameters
        self.min_profit_threshold = Decimal(str(config.get(
            'min_profit_threshold',
            os.getenv('ARB_MIN_PROFIT', '0.001')
        )))
        self.max_position_size = Decimal(str(config.get(
            'max_position_size',
            os.getenv('ARB_MAX_POSITION', '100000')
        )))
        self.execution_timeout = int(config.get(
            'execution_timeout',
            os.getenv('ARB_TIMEOUT', '5')
        ))

        # Trading costs
        self.trading_fee = Decimal(str(config.get(
            'trading_fee',
            os.getenv('ARB_TRADING_FEE', '0.001')
        )))
        self.slippage_estimate = Decimal(str(config.get(
            'slippage_estimate',
            os.getenv('ARB_SLIPPAGE', '0.0005')
        )))

        # Tracking
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.total_profit = Decimal('0')

        self.logger.info(
            f"{self.__class__.__name__} initialized: "
            f"min_profit={self.min_profit_threshold}, "
            f"max_position={self.max_position_size}"
        )

    @abstractmethod
    def scan_opportunities(
        self,
        market_data: pl.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """Scan market for arbitrage opportunities"""
        pass

    @abstractmethod
    def validate_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> bool:
        """Validate if opportunity is still viable"""
        pass

    @abstractmethod
    def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity
    ) -> Dict[str, Any]:
        """Execute the arbitrage opportunity"""
        pass

    def calculate_net_profit(
        self,
        gross_profit: Decimal,
        total_volume: Decimal
    ) -> Decimal:
        """Calculate net profit after fees and slippage"""
        try:
            trading_cost = total_volume * self.trading_fee
            slippage_cost = total_volume * self.slippage_estimate
            total_cost = trading_cost + slippage_cost

            net_profit = gross_profit - total_cost

            self.logger.debug(
                f"Profit calculation: gross={gross_profit}, "
                f"fees={trading_cost}, slippage={slippage_cost}, "
                f"net={net_profit}"
            )

            return net_profit

        except Exception as e:
            self.logger.error(f"Profit calculation error: {e}")
            return Decimal('0')

    def is_profitable(self, net_profit: Decimal) -> bool:
        """Check if net profit exceeds minimum threshold"""
        return net_profit >= self.min_profit_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        try:
            success_rate = Decimal('0')
            if self.opportunities_detected > 0:
                success_rate = (
                    Decimal(str(self.opportunities_executed)) /
                    Decimal(str(self.opportunities_detected))
                )

            avg_profit = Decimal('0')
            if self.opportunities_executed > 0:
                avg_profit = (
                    self.total_profit /
                    Decimal(str(self.opportunities_executed))
                )

            return {
                'strategy_name': self.__class__.__name__,
                'opportunities_detected': self.opportunities_detected,
                'opportunities_executed': self.opportunities_executed,
                'success_rate': success_rate,
                'total_profit': self.total_profit,
                'average_profit': avg_profit,
                'timestamp': datetime.utcnow()
            }

        except Exception as e:
            self.logger.error(f"Statistics error: {e}")
            return {'error': str(e)}

    def reset_statistics(self) -> None:
        """Reset tracking statistics"""
        self.opportunities_detected = 0
        self.opportunities_executed = 0
        self.total_profit = Decimal('0')
        self.logger.info("Statistics reset")
