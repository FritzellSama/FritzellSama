"""
Strategies Router - FastAPI endpoints for strategy management.

This module provides REST API endpoints for managing trading strategies,
including listing, creating, updating, and deleting strategies.
"""

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import JSONResponse
from structlog import get_logger

from quantum_trader.api.schemas.strategy_schemas import (
    StrategyResponse,
    StrategyCreateRequest,
    StrategyUpdateRequest,
    StrategyListResponse,
    StrategyPerformanceResponse,
    StrategyConfigResponse
)
from quantum_trader.api.services.strategy_service import StrategyService
from quantum_trader.core.exceptions import (
    StrategyNotFoundError,
    ValidationError,
    ConfigurationError
)

logger = get_logger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


async def get_strategy_service() -> StrategyService:
    """Dependency injection for StrategyService.

    Returns:
        StrategyService: Initialized strategy service instance

    Raises:
        ConfigurationError: If service initialization fails
    """
    try:
        service = StrategyService()
        await service.initialize()
        return service
    except Exception as e:
        logger.error("Failed to initialize strategy service", error=str(e))
        raise ConfigurationError(f"Service initialization failed: {e}")


@router.get("/", response_model=StrategyListResponse)
async def list_strategies(
    active_only: bool = Query(False, description="Filter active strategies only"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyListResponse:
    """List all available trading strategies.

    Args:
        active_only: Return only active strategies
        exchange: Filter by specific exchange
        service: Injected strategy service

    Returns:
        StrategyListResponse: List of strategies with metadata

    Example:
        >>> GET /strategies?active_only=true&exchange=BINANCE
        {
            "strategies": [...],
            "total": 5,
            "active": 3
        }
    """
    try:
        logger.info(
            "Listing strategies",
            active_only=active_only,
            exchange=exchange
        )

        strategies = await service.list_strategies(
            active_only=active_only,
            exchange=exchange
        )

        return StrategyListResponse(
            strategies=strategies,
            total=len(strategies),
            active=sum(1 for s in strategies if s.is_active)
        )

    except Exception as e:
        logger.error("Failed to list strategies", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategies: {str(e)}"
        )


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """Get detailed information about a specific strategy.

    Args:
        strategy_id: Unique strategy identifier
        service: Injected strategy service

    Returns:
        StrategyResponse: Detailed strategy information

    Raises:
        HTTPException: 404 if strategy not found

    Example:
        >>> GET /strategies/momentum_strategy_v1
        {
            "strategy_id": "momentum_strategy_v1",
            "name": "Momentum Strategy",
            ...
        }
    """
    try:
        logger.info("Retrieving strategy", strategy_id=strategy_id)

        strategy = await service.get_strategy(strategy_id)

        if not strategy:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        return strategy

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to retrieve strategy", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve strategy: {str(e)}"
        )


@router.post("/", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    request: StrategyCreateRequest,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """Create a new trading strategy.

    Args:
        request: Strategy creation parameters
        service: Injected strategy service

    Returns:
        StrategyResponse: Created strategy details

    Raises:
        HTTPException: 400 if validation fails

    Example:
        >>> POST /strategies
        {
            "name": "New Strategy",
            "type": "momentum",
            "config": {...}
        }
    """
    try:
        logger.info("Creating strategy", name=request.name, type=request.strategy_type)

        strategy = await service.create_strategy(request)

        logger.info("Strategy created successfully", strategy_id=strategy.strategy_id)

        return strategy

    except ValidationError as e:
        logger.warning("Strategy validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create strategy", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create strategy: {str(e)}"
        )


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: str,
    request: StrategyUpdateRequest,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """Update an existing trading strategy.

    Args:
        strategy_id: Strategy identifier to update
        request: Updated strategy parameters
        service: Injected strategy service

    Returns:
        StrategyResponse: Updated strategy details

    Raises:
        HTTPException: 404 if strategy not found, 400 if validation fails
    """
    try:
        logger.info("Updating strategy", strategy_id=strategy_id)

        strategy = await service.update_strategy(strategy_id, request)

        logger.info("Strategy updated successfully", strategy_id=strategy_id)

        return strategy

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found for update", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        logger.warning("Strategy update validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update strategy", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update strategy: {str(e)}"
        )


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> None:
    """Delete a trading strategy.

    Args:
        strategy_id: Strategy identifier to delete
        service: Injected strategy service

    Raises:
        HTTPException: 404 if strategy not found

    Example:
        >>> DELETE /strategies/old_strategy_v1
        204 No Content
    """
    try:
        logger.info("Deleting strategy", strategy_id=strategy_id)

        await service.delete_strategy(strategy_id)

        logger.info("Strategy deleted successfully", strategy_id=strategy_id)

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found for deletion", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to delete strategy", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete strategy: {str(e)}"
        )


@router.post("/{strategy_id}/start", response_model=StrategyResponse)
async def start_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """Start a trading strategy.

    Args:
        strategy_id: Strategy identifier to start
        service: Injected strategy service

    Returns:
        StrategyResponse: Updated strategy with active status

    Raises:
        HTTPException: 404 if strategy not found
    """
    try:
        logger.info("Starting strategy", strategy_id=strategy_id)

        strategy = await service.start_strategy(strategy_id)

        logger.info("Strategy started successfully", strategy_id=strategy_id)

        return strategy

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to start strategy", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start strategy: {str(e)}"
        )


@router.post("/{strategy_id}/stop", response_model=StrategyResponse)
async def stop_strategy(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyResponse:
    """Stop a trading strategy.

    Args:
        strategy_id: Strategy identifier to stop
        service: Injected strategy service

    Returns:
        StrategyResponse: Updated strategy with inactive status

    Raises:
        HTTPException: 404 if strategy not found
    """
    try:
        logger.info("Stopping strategy", strategy_id=strategy_id)

        strategy = await service.stop_strategy(strategy_id)

        logger.info("Strategy stopped successfully", strategy_id=strategy_id)

        return strategy

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to stop strategy", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop strategy: {str(e)}"
        )


@router.get("/{strategy_id}/performance", response_model=StrategyPerformanceResponse)
async def get_strategy_performance(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyPerformanceResponse:
    """Get performance metrics for a strategy.

    Args:
        strategy_id: Strategy identifier
        service: Injected strategy service

    Returns:
        StrategyPerformanceResponse: Performance metrics and statistics

    Raises:
        HTTPException: 404 if strategy not found

    Example:
        >>> GET /strategies/momentum_v1/performance
        {
            "total_pnl": "15234.56",
            "sharpe_ratio": "2.34",
            "win_rate": "0.67",
            ...
        }
    """
    try:
        logger.info("Retrieving strategy performance", strategy_id=strategy_id)

        performance = await service.get_strategy_performance(strategy_id)

        return performance

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to retrieve performance", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance: {str(e)}"
        )


@router.get("/{strategy_id}/config", response_model=StrategyConfigResponse)
async def get_strategy_config(
    strategy_id: str,
    service: StrategyService = Depends(get_strategy_service)
) -> StrategyConfigResponse:
    """Get strategy configuration.

    Args:
        strategy_id: Strategy identifier
        service: Injected strategy service

    Returns:
        StrategyConfigResponse: Strategy configuration parameters

    Raises:
        HTTPException: 404 if strategy not found
    """
    try:
        logger.info("Retrieving strategy config", strategy_id=strategy_id)

        config = await service.get_strategy_config(strategy_id)

        return config

    except StrategyNotFoundError as e:
        logger.warning("Strategy not found", strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to retrieve config", error=str(e), strategy_id=strategy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve config: {str(e)}"
        )
