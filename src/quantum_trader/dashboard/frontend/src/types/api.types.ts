/**
 * API Type Definitions
 *
 * TypeScript types for API requests, responses, and configurations.
 * Production-ready types for institutional trading platform.
 */

// ============================================================================
// Core API Types
// ============================================================================

/**
 * Standard API response wrapper
 */
export interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Record<string, string>;
}

/**
 * API error structure
 */
export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  details?: any;
}

/**
 * Request configuration
 */
export interface RequestConfig {
  body?: any;
  headers?: Record<string, string>;
  retries?: number;
  skipRateLimit?: boolean;
  skipAuth?: boolean;
  timeout?: number;
}

/**
 * Retry configuration
 */
export interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
}

/**
 * Rate limit configuration (token bucket)
 */
export interface RateLimitConfig {
  maxTokens: number;
  refillRate: number;
  refillInterval: number;
}

// ============================================================================
// Trading Types
// ============================================================================

/**
 * Order side enumeration
 */
export enum OrderSide {
  BUY = 'BUY',
  SELL = 'SELL'
}

/**
 * Order type enumeration
 */
export enum OrderType {
  MARKET = 'MARKET',
  LIMIT = 'LIMIT',
  STOP_LOSS = 'STOP_LOSS',
  TAKE_PROFIT = 'TAKE_PROFIT',
  STOP_LIMIT = 'STOP_LIMIT'
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
  FAILED = 'FAILED'
}

/**
 * Order request
 */
export interface OrderRequest {
  symbol: string;
  side: OrderSide;
  quantity: number;
  order_type: OrderType;
  price?: number;
  stop_price?: number;
  time_in_force?: 'GTC' | 'IOC' | 'FOK';
  strategy?: string;
  metadata?: Record<string, any>;
}

/**
 * Order response
 */
export interface OrderResponse {
  order_id: string;
  client_order_id?: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price?: number;
  order_type: OrderType;
  status: OrderStatus;
  filled_quantity: number;
  remaining_quantity: number;
  average_price?: number;
  fee: number;
  exchange: string;
  strategy: string;
  timestamp: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

/**
 * Position
 */
export interface Position {
  position_id: string;
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  pnl_percent: number;
  realized_pnl: number;
  exchange: string;
  strategy: string;
  opened_at: string;
  updated_at: string;
  leverage?: number;
  margin_used?: number;
}

/**
 * Trade execution result
 */
export interface Trade {
  id: string;
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number;
  total: number;
  fee: number;
  pnl?: number;
  strategy: string;
  exchange: string;
  order_type: OrderType;
  status: OrderStatus;
  timestamp: string;
  metadata?: Record<string, any>;
}

// ============================================================================
// Market Data Types
// ============================================================================

/**
 * Ticker data
 */
export interface Ticker {
  symbol: string;
  last_price: number;
  bid: number;
  ask: number;
  volume_24h: number;
  change_24h: number;
  change_24h_percent: number;
  high_24h: number;
  low_24h: number;
  timestamp: string;
}

/**
 * Order book entry
 */
export interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

/**
 * Order book
 */
export interface OrderBook {
  symbol: string;
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  timestamp: string;
}

/**
 * OHLCV candle
 */
export interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Volume data
 */
export interface VolumeData {
  timestamp: string;
  volume: number;
  buy_volume: number;
  sell_volume: number;
  price: number;
  num_trades: number;
}

// ============================================================================
// Strategy Types
// ============================================================================

/**
 * Signal action
 */
export enum SignalAction {
  BUY = 'BUY',
  SELL = 'SELL',
  HOLD = 'HOLD',
  CLOSE = 'CLOSE'
}

/**
 * Trading signal
 */
export interface Signal {
  signal_id: string;
  symbol: string;
  action: SignalAction;
  strength: number;
  confidence: number;
  timestamp: string;
  strategy: string;
  timeframe: string;
  indicators: Record<string, number>;
  metadata?: Record<string, any>;
  executed: boolean;
}

/**
 * Strategy status
 */
export enum StrategyStatus {
  ACTIVE = 'active',
  PAUSED = 'paused',
  STOPPED = 'stopped'
}

/**
 * Strategy metrics
 */
export interface StrategyMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  sortino_ratio?: number;
  max_drawdown: number;
  total_pnl: number;
  avg_trade_duration: number;
  daily_pnl: number;
  weekly_pnl?: number;
  monthly_pnl?: number;
}

/**
 * Strategy configuration
 */
export interface Strategy {
  id: string;
  name: string;
  type: string;
  status: StrategyStatus;
  description: string;
  timeframe: string;
  symbols: string[];
  capital_allocated: number;
  metrics: StrategyMetrics;
  parameters?: Record<string, any>;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// Risk Management Types
// ============================================================================

/**
 * Risk metrics
 */
export interface RiskMetrics {
  portfolio_value: number;
  total_exposure: number;
  net_exposure: number;
  leverage: number;
  margin_usage: number;
  var_95: number;
  var_99: number;
  cvar_95?: number;
  cvar_99?: number;
  max_drawdown: number;
  current_drawdown: number;
  sharpe_ratio?: number;
  sortino_ratio?: number;
  timestamp: string;
}

/**
 * Risk limit
 */
export interface RiskLimit {
  metric: string;
  current: number;
  threshold: number;
  limit: number;
  status: 'safe' | 'warning' | 'critical';
}

/**
 * Position risk
 */
export interface PositionRisk {
  symbol: string;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  var_95: number;
  cvar_95?: number;
  delta: number;
  gamma?: number;
  vega?: number;
  theta?: number;
  strategy: string;
  exchange: string;
}

// ============================================================================
// Analytics Types
// ============================================================================

/**
 * Return data point
 */
export interface ReturnData {
  timestamp: string;
  return_pct: number;
  cumulative_return_pct: number;
  strategy: string;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  total_return: number;
  total_return_pct: number;
  annualized_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  avg_win: number;
  avg_loss: number;
  largest_win: number;
  largest_loss: number;
  avg_trade_duration: number;
  period_start: string;
  period_end: string;
}

// ============================================================================
// Settings Types
// ============================================================================

/**
 * Risk settings
 */
export interface RiskSettings {
  max_position_size: number;
  max_leverage: number;
  max_drawdown: number;
  max_daily_loss: number;
  stop_loss_enabled: boolean;
  stop_loss_percentage: number;
  take_profit_enabled: boolean;
  take_profit_percentage: number;
}

/**
 * Trading settings
 */
export interface TradingSettings {
  auto_trading_enabled: boolean;
  max_open_positions: number;
  default_order_type: OrderType;
  slippage_tolerance: number;
  min_order_size: number;
  max_order_size: number;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  email_enabled: boolean;
  email_address: string;
  telegram_enabled: boolean;
  telegram_chat_id: string;
  webhook_enabled: boolean;
  webhook_url: string;
  alert_on_trade: boolean;
  alert_on_error: boolean;
  alert_on_risk_breach: boolean;
}

/**
 * Exchange configuration
 */
export interface ExchangeConfig {
  exchange: string;
  enabled: boolean;
  api_key: string;
  api_secret: string;
  testnet: boolean;
  rate_limit?: number;
}

// ============================================================================
// Pagination Types
// ============================================================================

/**
 * Pagination parameters
 */
export interface PaginationParams {
  page?: number;
  limit?: number;
  offset?: number;
}

/**
 * Paginated response
 */
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

// ============================================================================
// WebSocket Types
// ============================================================================

/**
 * WebSocket message types
 */
export enum WSMessageType {
  SUBSCRIBE = 'subscribe',
  UNSUBSCRIBE = 'unsubscribe',
  TICKER = 'ticker',
  ORDERBOOK = 'orderbook',
  TRADE = 'trade',
  ORDER_UPDATE = 'order_update',
  POSITION_UPDATE = 'position_update',
  SIGNAL = 'signal',
  ALERT = 'alert',
  ERROR = 'error'
}

/**
 * WebSocket message
 */
export interface WSMessage<T = any> {
  type: WSMessageType;
  channel?: string;
  data: T;
  timestamp: string;
}

/**
 * WebSocket subscription
 */
export interface WSSubscription {
  channel: string;
  symbols?: string[];
  callback: (data: any) => void;
}

// ============================================================================
// Authentication Types
// ============================================================================

/**
 * Login credentials
 */
export interface LoginCredentials {
  email: string;
  password: string;
  remember_me?: boolean;
}

/**
 * Auth token response
 */
export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  user: User;
}

/**
 * User information
 */
export interface User {
  id: string;
  email: string;
  username: string;
  role: 'admin' | 'trader' | 'viewer';
  created_at: string;
  last_login?: string;
  preferences?: Record<string, any>;
}

// ============================================================================
// System Types
// ============================================================================

/**
 * System health
 */
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'down';
  uptime: number;
  services: Record<string, ServiceHealth>;
  timestamp: string;
}

/**
 * Service health
 */
export interface ServiceHealth {
  name: string;
  status: 'up' | 'down';
  latency?: number;
  error?: string;
}

/**
 * System metrics
 */
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_in: number;
  network_out: number;
  active_connections: number;
  requests_per_second: number;
  timestamp: string;
}
