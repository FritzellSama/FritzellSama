"""
GraphQL schema definition for trading API.

Defines GraphQL types, queries, and mutations for flexible
querying of trading data with strong typing.
"""

from typing import Dict, Any, Optional
from structlog import get_logger

logger = get_logger(__name__)

# GraphQL Schema Definition Language (SDL)
TYPE_DEFS = """
# Scalars
scalar DateTime
scalar Decimal

# Enums
enum OrderSide {
    BUY
    SELL
}

enum OrderType {
    MARKET
    LIMIT
    STOP_LOSS
    TAKE_PROFIT
    STOP_LIMIT
}

enum OrderStatus {
    PENDING
    OPEN
    PARTIAL
    FILLED
    CANCELLED
    REJECTED
    EXPIRED
    FAILED
}

enum PositionSide {
    LONG
    SHORT
    FLAT
}

enum PositionStatus {
    OPEN
    CLOSING
    CLOSED
}

enum TimeFrame {
    M1
    M5
    M15
    M30
    H1
    H4
    D1
    W1
}

# Order Types
type Order {
    orderId: ID!
    clientOrderId: String
    symbol: String!
    side: OrderSide!
    quantity: Decimal!
    filledQuantity: Decimal!
    remainingQuantity: Decimal!
    orderType: OrderType!
    price: Decimal
    stopPrice: Decimal
    averageFillPrice: Decimal
    status: OrderStatus!
    exchange: String!
    strategy: String!
    timeInForce: String!
    createdAt: DateTime!
    updatedAt: DateTime!
    filledAt: DateTime
    executionDetails: ExecutionDetails
    metadata: JSON
}

type ExecutionDetails {
    totalFees: Decimal!
    feeCurrency: String!
    fills: [Fill!]!
    executionTimeMs: Float!
}

type Fill {
    price: Decimal!
    quantity: Decimal!
    timestamp: DateTime!
    feeAmount: Decimal
}

# Position Types
type Position {
    positionId: ID!
    symbol: String!
    side: PositionSide!
    quantity: Decimal!
    entryPrice: Decimal!
    currentPrice: Decimal!
    unrealizedPnl: Decimal!
    unrealizedPnlPercent: Decimal!
    realizedPnl: Decimal!
    totalPnl: Decimal!
    exchange: String!
    strategy: String!
    leverage: Decimal
    liquidationPrice: Decimal
    stopLoss: Decimal
    takeProfit: Decimal
    status: PositionStatus!
    openedAt: DateTime!
    updatedAt: DateTime!
    closedAt: DateTime
    holdingTimeSeconds: Float!
    riskMetrics: PositionRiskMetrics
    metadata: JSON
}

type PositionRiskMetrics {
    exposure: Decimal!
    exposurePercent: Decimal!
    riskAmount: Decimal!
    var95: Decimal!
    distanceToStopLossPercent: Decimal
    contributionToPortfolioVar: Decimal!
}

# Market Data Types
type MarketData {
    symbol: String!
    exchange: String!
    lastPrice: Decimal!
    volume24h: Decimal!
    high24h: Decimal!
    low24h: Decimal!
    change24h: Decimal!
    changePercent24h: Decimal!
    timestamp: DateTime!
    orderBook(depth: Int = 20): OrderBook
}

type OrderBook {
    symbol: String!
    exchange: String!
    bids: [PriceLevel!]!
    asks: [PriceLevel!]!
    spread: Decimal!
    midPrice: Decimal
    timestamp: DateTime!
}

type PriceLevel {
    price: Decimal!
    quantity: Decimal!
}

type OHLCV {
    timestamp: DateTime!
    open: Decimal!
    high: Decimal!
    low: Decimal!
    close: Decimal!
    volume: Decimal!
}

# Portfolio Types
type PortfolioSummary {
    totalValue: Decimal!
    availableBalance: Decimal!
    totalExposure: Decimal!
    totalUnrealizedPnl: Decimal!
    totalRealizedPnl: Decimal!
    totalPnl: Decimal!
    marginUsed: Decimal!
    marginAvailable: Decimal!
    openPositions: Int!
    totalPositions: Int!
    positionsByExchange: JSON!
    positionsByStrategy: JSON!
}

# Risk Types
type RiskMetrics {
    totalExposure: Decimal!
    totalValue: Decimal!
    leverage: Decimal!
    marginUsed: Decimal!
    marginAvailable: Decimal!
    var95: Decimal!
    var99: Decimal!
    expectedShortfall: Decimal!
    sharpeRatio: Decimal
    maxDrawdown: Decimal!
    currentDrawdown: Decimal!
}

type RiskLimits {
    maxPositionSize: Decimal!
    maxPortfolioExposure: Decimal!
    maxDailyLoss: Decimal!
    maxDrawdownPercent: Decimal!
    maxLeverage: Decimal!
    maxOpenPositions: Int!
    maxCorrelation: Decimal!
    updatedAt: DateTime!
}

type RiskCheckResult {
    passed: Boolean!
    checksPerformed: [String!]!
    violations: [String!]!
    warnings: [String!]!
    riskScore: Decimal!
    exposureImpact: Decimal!
    leverageImpact: Decimal!
    details: JSON!
}

# Pagination Types
type OrderConnection {
    items: [Order!]!
    total: Int!
    offset: Int!
    limit: Int!
}

type PositionConnection {
    items: [Position!]!
    total: Int!
    offset: Int!
    limit: Int!
}

# Input Types
input CreateOrderInput {
    symbol: String!
    side: OrderSide!
    quantity: Decimal!
    orderType: OrderType!
    price: Decimal
    stopPrice: Decimal
    exchange: String!
    strategy: String!
    timeInForce: String
    clientOrderId: String
    metadata: JSON
}

input UpdateRiskLimitsInput {
    maxPositionSize: Decimal
    maxPortfolioExposure: Decimal
    maxDailyLoss: Decimal
    maxDrawdownPercent: Decimal
    maxLeverage: Decimal
}

input RiskCheckInput {
    symbol: String!
    side: OrderSide!
    quantity: Decimal!
    price: Decimal
    strategy: String!
}

# Response Types
type CreateOrderResponse {
    orderId: ID!
    status: OrderStatus!
    message: String!
}

type CancelOrderResponse {
    orderId: ID!
    status: OrderStatus!
    message: String!
}

type ClosePositionResponse {
    positionId: ID!
    status: PositionStatus!
    realizedPnl: Decimal!
    message: String!
}

# Queries
type Query {
    # Orders
    order(orderId: ID!): Order
    orders(
        symbol: String
        exchange: String
        strategy: String
        status: OrderStatus
        side: OrderSide
        limit: Int = 100
        offset: Int = 0
    ): OrderConnection!

    # Positions
    position(positionId: ID!): Position
    positions(
        symbol: String
        exchange: String
        strategy: String
        status: PositionStatus
        side: PositionSide
        limit: Int = 100
        offset: Int = 0
    ): PositionConnection!

    # Market Data
    marketData(symbol: String!, exchange: String!): MarketData
    ohlcv(
        symbol: String!
        exchange: String!
        timeframe: TimeFrame!
        limit: Int = 100
    ): [OHLCV!]!

    # Portfolio
    portfolioSummary: PortfolioSummary!

    # Risk
    riskMetrics: RiskMetrics!
    riskLimits: RiskLimits!
    checkRisk(input: RiskCheckInput!): RiskCheckResult!
}

# Mutations
type Mutation {
    # Orders
    createOrder(input: CreateOrderInput!): CreateOrderResponse!
    cancelOrder(orderId: ID!, reason: String): CancelOrderResponse!
    cancelAllOrders(strategy: String!, symbol: String): Int!

    # Positions
    closePosition(
        positionId: ID!
        quantity: Decimal
        reason: String
    ): ClosePositionResponse!
    closeAllPositions(strategy: String!, symbol: String): Int!
    modifyPosition(
        positionId: ID!
        stopLoss: Decimal
        takeProfit: Decimal
    ): Position!

    # Risk
    updateRiskLimits(input: UpdateRiskLimitsInput!): RiskLimits!
}

# Subscriptions
type Subscription {
    # Real-time market data
    marketDataUpdates(symbols: [String!]!): MarketData!

    # Order updates
    orderUpdates(strategy: String): Order!

    # Position updates
    positionUpdates(strategy: String): Position!

    # Risk alerts
    riskAlerts: RiskAlert!
}

type RiskAlert {
    alertId: ID!
    severity: String!
    category: String!
    message: String!
    timestamp: DateTime!
}

# JSON scalar for flexible metadata
scalar JSON
"""


class GraphQLSchemaManager:
    """
    Manager for GraphQL schema configuration.

    Handles schema definition, resolver binding, and
    schema validation for the trading API.

    Attributes:
        config: Schema configuration
        type_defs: GraphQL schema definition
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize GraphQL schema manager.

        Args:
            config: Schema configuration
        """
        self.config = config
        self.type_defs = TYPE_DEFS

        logger.info("graphql_schema_manager_initialized")

    def get_type_defs(self) -> str:
        """
        Get GraphQL type definitions.

        Returns:
            GraphQL SDL string
        """
        return self.type_defs

    def validate_schema(self) -> bool:
        """
        Validate GraphQL schema definition.

        Returns:
            True if schema is valid

        Raises:
            ValueError: If schema is invalid
        """
        try:
            # In production, use graphql library to validate
            # For now, basic validation
            if not self.type_defs:
                raise ValueError("Schema definition is empty")

            if "type Query" not in self.type_defs:
                raise ValueError("Schema must define Query type")

            logger.info("graphql_schema_validated")
            return True

        except Exception as e:
            logger.error("schema_validation_error", error=str(e))
            raise

    def create_executable_schema(self, resolvers: Dict[str, Any]) -> Any:
        """
        Create executable GraphQL schema.

        Args:
            resolvers: Resolver map

        Returns:
            Executable schema object

        Example:
            >>> from quantum_trader.api.graphql.resolvers import create_resolvers
            >>> resolvers = create_resolvers(config)
            >>> schema = manager.create_executable_schema(resolvers)
        """
        try:
            # In production, use graphql library (e.g., ariadne, strawberry)
            # to create executable schema
            logger.info("creating_executable_schema")

            # Validate first
            self.validate_schema()

            # Placeholder - in production would use library
            # from ariadne import make_executable_schema
            # schema = make_executable_schema(self.type_defs, resolvers)

            return {
                "type_defs": self.type_defs,
                "resolvers": resolvers
            }

        except Exception as e:
            logger.error("create_schema_error", error=str(e))
            raise

    def get_schema_documentation(self) -> str:
        """
        Get schema documentation in markdown format.

        Returns:
            Markdown documentation string
        """
        doc = """
# Quantum Trader GraphQL API

## Overview
GraphQL API for querying and mutating trading data with flexible, type-safe queries.

## Authentication
Include API key in `X-API-Key` header or use JWT bearer token.

## Queries

### Orders
- `order(orderId)` - Get single order
- `orders(...)` - List orders with filtering

### Positions
- `position(positionId)` - Get single position
- `positions(...)` - List positions with filtering

### Market Data
- `marketData(symbol, exchange)` - Get real-time market data
- `ohlcv(...)` - Get candlestick data

### Portfolio
- `portfolioSummary` - Get portfolio overview

### Risk
- `riskMetrics` - Get risk metrics
- `riskLimits` - Get risk limits
- `checkRisk(input)` - Check if trade passes risk checks

## Mutations

### Orders
- `createOrder(input)` - Create new order
- `cancelOrder(orderId)` - Cancel order
- `cancelAllOrders(strategy)` - Cancel all orders for strategy

### Positions
- `closePosition(positionId)` - Close position
- `closeAllPositions(strategy)` - Close all positions for strategy
- `modifyPosition(positionId, ...)` - Modify position parameters

### Risk
- `updateRiskLimits(input)` - Update risk limits

## Subscriptions
- `marketDataUpdates(symbols)` - Subscribe to real-time market data
- `orderUpdates(strategy)` - Subscribe to order updates
- `positionUpdates(strategy)` - Subscribe to position updates
- `riskAlerts` - Subscribe to risk alerts

## Examples

### Query Orders
```graphql
query GetOrders {
    orders(symbol: "BTC/USDT", status: OPEN, limit: 10) {
        items {
            orderId
            symbol
            quantity
            price
            status
        }
        total
    }
}
```

### Create Order
```graphql
mutation CreateOrder {
    createOrder(input: {
        symbol: "BTC/USDT"
        side: BUY
        quantity: "0.1"
        orderType: LIMIT
        price: "50000.00"
        exchange: "binance"
        strategy: "momentum"
    }) {
        orderId
        status
        message
    }
}
```

### Get Portfolio Summary
```graphql
query Portfolio {
    portfolioSummary {
        totalValue
        totalPnl
        openPositions
        totalExposure
    }
}
```

### Subscribe to Order Updates
```graphql
subscription OrderUpdates {
    orderUpdates(strategy: "momentum") {
        orderId
        status
        filledQuantity
    }
}
```
"""
        return doc


def create_schema_manager(config: Optional[Dict[str, Any]] = None) -> GraphQLSchemaManager:
    """
    Create GraphQL schema manager.

    Args:
        config: Optional configuration

    Returns:
        GraphQLSchemaManager instance

    Example:
        >>> manager = create_schema_manager(config)
        >>> type_defs = manager.get_type_defs()
        >>> schema = manager.create_executable_schema(resolvers)
    """
    config = config or {}
    return GraphQLSchemaManager(config)
