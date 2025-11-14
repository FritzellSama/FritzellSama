/**
 * Trading Type Definitions
 *
 * TypeScript type definitions for the Quantum Trader AI trading system.
 * Mirrors the Python backend data models and enums.
 *
 * @module trading.types
 */

/**
 * Order side enumeration
 */
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL',
}

/**
 * Order type enumeration
 */
export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP_LOSS = 'STOP_LOSS',
  TAKE_PROFIT = 'TAKE_PROFIT',
  STOP_LIMIT = 'STOP_LIMIT',
}

/**
 * Order status enumeration
 */
export enum OrderStatus {
  PENDING = 'PENDING',
  OPEN = 'OPEN',
  PARTIAL = 'PARTIAL',
  FILLED = 'FILLED',
  CANCELLED = 'CANCELLED',
  REJECTED = 'REJECTED',
  EXPIRED = 'EXPIRED',
  FAILED = 'FAILED',
}

/**
 * Signal action enumeration
 */
export enum SignalAction {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD',
  CLOSE = 'CLOSE',
}

/**
 * Exchange enumeration
 */
export enum Exchange {
  BINANCE = 'BINANCE',
  BYBIT = 'BYBIT',
  OKX = 'OKX',
  KUCOIN = 'KUCOIN',
  BITGET = 'BITGET',
}

/**
 * Timeframe enumeration
 */
export enum Timeframe {
  ONE_MINUTE = '1m',
  FIVE_MINUTES = '5m',
  FIFTEEN_MINUTES = '15m',
  THIRTY_MINUTES = '30m',
  ONE_HOUR = '1h',
  FOUR_HOURS = '4h',
  ONE_DAY = '1d',
  ONE_WEEK = '1w',
  ONE_MONTH = '1M',
}

/**
 * Trading order interface
 */
export interface Order {
  symbol: string;
  side: OrderSide;
  quantity: string; // Decimal as string to avoid precision loss
  price?: string | null; // Decimal as string, null for market orders
  order_type: OrderType;
  exchange: string;
  strategy: string;
  timestamp: string; // ISO 8601 datetime string
  order_id?: string | null;
  metadata?: Record<string, any>;
}

/**
 * Order creation request
 */
export interface OrderCreateRequest {
  symbol: string;
  side: OrderSide;
  quantity: string;
  price?: string;
  order_type: OrderType;
  exchange: string;
  strategy: string;
}

/**
 * Trading position interface
 */
export interface Position {
  symbol: string;
  quantity: string; // Decimal as string (positive=long, negative=short)
  entry_price: string; // Decimal as string
  current_price: string; // Decimal as string
  exchange: string;
  strategy: string;
  opened_at: string; // ISO 8601 datetime string
  position_id: string;
  pnl?: string; // Calculated P&L (Decimal as string)
  pnl_percent?: string; // Calculated P&L percentage (Decimal as string)
}

/**
 * Trading signal interface
 */
export interface Signal {
  symbol: string;
  action: SignalAction;
  strength: string; // Decimal as string (0.0 to 1.0)
  confidence: string; // Decimal as string (0.0 to 1.0)
  timestamp: string; // ISO 8601 datetime string
  strategy: string;
  timeframe: string;
  indicators?: Record<string, string>; // Indicator values as Decimal strings
  metadata?: Record<string, any>;
}

/**
 * Execution result interface
 */
export interface ExecutionResult {
  success: boolean;
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: string; // Decimal as string
  price: string; // Decimal as string
  filled_quantity: string; // Decimal as string
  average_price: string; // Decimal as string
  status: OrderStatus;
  exchange: string;
  timestamp: string; // ISO 8601 datetime string
  fees?: string; // Decimal as string
  error_message?: string;
  metadata?: Record<string, any>;
}

/**
 * Portfolio balance interface
 */
export interface Balance {
  currency: string;
  free: string; // Decimal as string
  locked: string; // Decimal as string
  total: string; // Decimal as string
}

/**
 * Portfolio summary interface
 */
export interface PortfolioSummary {
  total_value: string; // Decimal as string
  total_pnl: string; // Decimal as string
  total_pnl_percent: string; // Decimal as string
  positions_count: number;
  open_orders_count: number;
  balances: Balance[];
  timestamp: string; // ISO 8601 datetime string
}

/**
 * Risk metrics interface
 */
export interface RiskMetrics {
  var_95: string; // Value at Risk 95% (Decimal as string)
  var_99: string; // Value at Risk 99% (Decimal as string)
  max_drawdown: string; // Decimal as string
  current_drawdown: string; // Decimal as string
  sharpe_ratio: string; // Decimal as string
  sortino_ratio: string; // Decimal as string
  exposure: string; // Total exposure (Decimal as string)
  leverage: string; // Current leverage (Decimal as string)
  timestamp: string; // ISO 8601 datetime string
}

/**
 * Strategy performance interface
 */
export interface StrategyPerformance {
  strategy_name: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: string; // Decimal as string
  total_pnl: string; // Decimal as string
  average_win: string; // Decimal as string
  average_loss: string; // Decimal as string
  profit_factor: string; // Decimal as string
  sharpe_ratio: string; // Decimal as string
  max_drawdown: string; // Decimal as string
  start_date: string; // ISO 8601 datetime string
  end_date: string; // ISO 8601 datetime string
}

/**
 * Trade history entry interface
 */
export interface TradeHistory {
  trade_id: string;
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: string; // Decimal as string
  price: string; // Decimal as string
  value: string; // Decimal as string
  fees: string; // Decimal as string
  exchange: string;
  strategy: string;
  timestamp: string; // ISO 8601 datetime string
  pnl?: string; // Decimal as string (for closing trades)
}

/**
 * Market data subscription interface
 */
export interface MarketDataSubscription {
  symbol: string;
  exchange: string;
  channels: ('orderbook' | 'ticker' | 'trades' | 'ohlcv')[];
}

/**
 * OHLCV (candlestick) data interface
 */
export interface OHLCV {
  timestamp: string; // ISO 8601 datetime string
  open: string; // Decimal as string
  high: string; // Decimal as string
  low: string; // Decimal as string
  close: string; // Decimal as string
  volume: string; // Decimal as string
}

/**
 * Orderbook level interface
 */
export interface OrderBookLevel {
  price: string; // Decimal as string
  quantity: string; // Decimal as string
  total?: string; // Cumulative total (Decimal as string)
}

/**
 * Orderbook interface
 */
export interface OrderBook {
  symbol: string;
  exchange: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: string; // ISO 8601 datetime string
  last_update_id?: number;
}

/**
 * Ticker interface
 */
export interface Ticker {
  symbol: string;
  exchange: string;
  last: string; // Decimal as string
  bid: string; // Decimal as string
  ask: string; // Decimal as string
  volume_24h: string; // Decimal as string
  high_24h: string; // Decimal as string
  low_24h: string; // Decimal as string
  change_24h: string; // Decimal as string
  change_percent_24h: string; // Decimal as string
  timestamp: string; // ISO 8601 datetime string
}

/**
 * Backtest configuration interface
 */
export interface BacktestConfig {
  strategy: string;
  symbol: string;
  exchange: string;
  timeframe: Timeframe;
  start_date: string; // ISO 8601 datetime string
  end_date: string; // ISO 8601 datetime string
  initial_balance: string; // Decimal as string
  commission: string; // Decimal as string (e.g., '0.001' for 0.1%)
  slippage: string; // Decimal as string
  parameters?: Record<string, any>;
}

/**
 * Backtest result interface
 */
export interface BacktestResult {
  config: BacktestConfig;
  performance: StrategyPerformance;
  trades: TradeHistory[];
  equity_curve: Array<{ timestamp: string; value: string }>;
  drawdown_curve: Array<{ timestamp: string; value: string }>;
  metrics: {
    total_return: string; // Decimal as string
    annual_return: string; // Decimal as string
    volatility: string; // Decimal as string
    sharpe_ratio: string; // Decimal as string
    sortino_ratio: string; // Decimal as string
    max_drawdown: string; // Decimal as string
    calmar_ratio: string; // Decimal as string
  };
  timestamp: string; // ISO 8601 datetime string
}

/**
 * Alert configuration interface
 */
export interface Alert {
  alert_id: string;
  type: 'price' | 'indicator' | 'risk' | 'system';
  symbol?: string;
  exchange?: string;
  condition: string;
  threshold: string; // Decimal as string
  enabled: boolean;
  notification_channels: ('email' | 'sms' | 'webhook' | 'desktop')[];
  created_at: string; // ISO 8601 datetime string
  triggered_at?: string; // ISO 8601 datetime string
}

/**
 * System status interface
 */
export interface SystemStatus {
  status: 'running' | 'stopped' | 'error' | 'starting' | 'stopping';
  uptime: number; // Seconds
  connected_exchanges: string[];
  active_strategies: string[];
  active_positions: number;
  open_orders: number;
  last_trade_time?: string; // ISO 8601 datetime string
  health_checks: {
    database: boolean;
    cache: boolean;
    exchanges: Record<string, boolean>;
  };
  timestamp: string; // ISO 8601 datetime string
}

/**
 * API error response interface
 */
export interface APIError {
  detail: string;
  code?: string;
  timestamp: string; // ISO 8601 datetime string
  path?: string;
}

/**
 * Pagination interface
 */
export interface Pagination {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

/**
 * Paginated response interface
 */
export interface PaginatedResponse<T> {
  data: T[];
  pagination: Pagination;
}

/**
 * WebSocket message types
 */
export type WSMessageType =
  | 'ping'
  | 'pong'
  | 'subscribe'
  | 'unsubscribe'
  | 'orderbook_update'
  | 'ticker_update'
  | 'trade_update'
  | 'order_update'
  | 'position_update'
  | 'error';

/**
 * WebSocket message interface
 */
export interface WSMessage<T = any> {
  type: WSMessageType;
  channel?: string;
  data?: T;
  timestamp: string; // ISO 8601 datetime string
  error?: string;
}

/**
 * Type guards
 */
export function isOrder(obj: any): obj is Order {
  return (
    obj &&
    typeof obj.symbol === 'string' &&
    typeof obj.side === 'string' &&
    typeof obj.quantity === 'string' &&
    typeof obj.order_type === 'string' &&
    typeof obj.exchange === 'string' &&
    typeof obj.strategy === 'string' &&
    typeof obj.timestamp === 'string'
  );
}

export function isPosition(obj: any): obj is Position {
  return (
    obj &&
    typeof obj.symbol === 'string' &&
    typeof obj.quantity === 'string' &&
    typeof obj.entry_price === 'string' &&
    typeof obj.current_price === 'string' &&
    typeof obj.exchange === 'string' &&
    typeof obj.strategy === 'string' &&
    typeof obj.opened_at === 'string' &&
    typeof obj.position_id === 'string'
  );
}

export function isSignal(obj: any): obj is Signal {
  return (
    obj &&
    typeof obj.symbol === 'string' &&
    typeof obj.action === 'string' &&
    typeof obj.strength === 'string' &&
    typeof obj.confidence === 'string' &&
    typeof obj.timestamp === 'string' &&
    typeof obj.strategy === 'string' &&
    typeof obj.timeframe === 'string'
  );
}

/**
 * Utility type for Decimal calculations
 * All monetary and numeric values should be handled as strings in the frontend
 * and converted to Decimal on the backend to avoid precision loss
 */
export type DecimalString = string;

/**
 * Timestamp utility type
 * All timestamps are ISO 8601 formatted strings in UTC
 */
export type Timestamp = string;
