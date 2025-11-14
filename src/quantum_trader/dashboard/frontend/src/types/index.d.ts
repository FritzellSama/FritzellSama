/**
 * Global Type Definitions for Quantum Trader Dashboard
 *
 * This file contains all shared type definitions used across the application
 */

// Environment Variables
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_BASE_URL: string;
  readonly VITE_ENVIRONMENT: 'development' | 'staging' | 'production';
  readonly VITE_ENABLE_DEVTOOLS: string;
  readonly VITE_LOG_LEVEL: 'debug' | 'info' | 'warn' | 'error';
  readonly VITE_SENTRY_DSN?: string;
  readonly VITE_API_TIMEOUT: string;
  readonly MODE: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
  readonly hot?: {
    accept: (cb?: (mod: any) => void) => void;
    dispose: (cb: (data: any) => void) => void;
    data: any;
  };
}

// API Types

export type OrderSide = 'BUY' | 'SELL';
export type OrderType = 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'STOP_LIMIT';
export type OrderStatus = 'PENDING' | 'OPEN' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED' | 'EXPIRED';
export type TimeFrame = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';
export type StrategyType = 'ARBITRAGE' | 'MOMENTUM' | 'MEAN_REVERSION' | 'MARKET_MAKING' | 'HFT' | 'ML' | 'HYBRID' | 'OPTIONS' | 'ORDER_FLOW';
export type StrategyStatus = 'ACTIVE' | 'PAUSED' | 'STOPPED' | 'ERROR';
export type Exchange = 'binance' | 'bybit' | 'okx' | 'kucoin' | 'bitget';
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

// User & Authentication

export interface User {
  id: string;
  username: string;
  email: string;
  firstName?: string;
  lastName?: string;
  role: 'admin' | 'trader' | 'viewer';
  permissions: string[];
  createdAt: string;
  lastLogin?: string;
  mfaEnabled: boolean;
  preferences?: UserPreferences;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  defaultTimeframe: TimeFrame;
  defaultExchange: Exchange;
  notifications: NotificationPreferences;
  displayCurrency: string;
}

export interface NotificationPreferences {
  email: boolean;
  push: boolean;
  sms: boolean;
  tradingAlerts: boolean;
  riskAlerts: boolean;
  systemAlerts: boolean;
  priceAlerts: boolean;
}

export interface AuthToken {
  accessToken: string;
  tokenType: string;
  expiresIn: number;
  refreshToken?: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
  mfaCode?: string;
}

// Portfolio & Positions

export interface Portfolio {
  id: string;
  userId: string;
  totalValue: number;
  cashBalance: number;
  investedValue: number;
  unrealizedPnL: number;
  realizedPnL: number;
  dailyPnL: number;
  totalReturnPct: number;
  sharpeRatio: number | null;
  sortinoRatio: number | null;
  maxDrawdownPct: number;
  winRatePct: number;
  numPositions: number;
  lastUpdated: string;
}

export interface Position {
  id: string;
  portfolioId: string;
  symbol: string;
  exchange: Exchange;
  side: OrderSide;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPct: number;
  marketValue: number;
  costBasis: number;
  openedAt: string;
  strategy?: string;
  strategyId?: string;
  stopLoss?: number;
  takeProfit?: number;
  notes?: string;
}

// Trading

export interface Order {
  id: string;
  userId: string;
  symbol: string;
  exchange: Exchange;
  side: OrderSide;
  type: OrderType;
  quantity: number;
  price?: number;
  stopPrice?: number;
  status: OrderStatus;
  filledQuantity: number;
  averagePrice?: number;
  fee?: number;
  feeCurrency?: string;
  createdAt: string;
  updatedAt: string;
  filledAt?: string;
  cancelledAt?: string;
  strategy?: string;
  strategyId?: string;
  timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'GTX';
  reduceOnly?: boolean;
}

export interface Trade {
  id: string;
  orderId: string;
  userId: string;
  symbol: string;
  exchange: Exchange;
  side: OrderSide;
  quantity: number;
  price: number;
  fee: number;
  feeCurrency: string;
  realizedPnL?: number;
  executedAt: string;
  strategy?: string;
  strategyId?: string;
  commission?: number;
  notes?: string;
}

// Strategies

export interface Strategy {
  id: string;
  name: string;
  type: StrategyType;
  description?: string;
  status: StrategyStatus;
  parameters: Record<string, any>;
  positions: number;
  pnl: number;
  pnlPct: number;
  winRatePct: number;
  tradesToday: number;
  totalTrades: number;
  avgHoldingPeriod?: number;
  sharpeRatio?: number;
  maxDrawdownPct?: number;
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  stoppedAt?: string;
  symbols: string[];
  exchanges: Exchange[];
  riskLevel: RiskLevel;
}

// Risk Management

export interface RiskMetrics {
  portfolioVar95: number;
  portfolioCVar95: number;
  currentLeverage: number;
  maxLeverage: number;
  exposureBySector: Record<string, number>;
  exposureByExchange: Record<string, number>;
  concentrationRisk: number;
  liquidityScore: number;
  riskScore: number;
  timestamp: string;
}

export interface RiskLimit {
  id: string;
  name: string;
  type: 'POSITION_SIZE' | 'DAILY_LOSS' | 'DRAWDOWN' | 'LEVERAGE' | 'CONCENTRATION' | 'VOLATILITY';
  value: number;
  threshold: number;
  currentValue: number;
  breached: boolean;
  enabled: boolean;
  action: 'ALERT' | 'PAUSE' | 'LIQUIDATE';
}

export interface RiskEvent {
  id: string;
  type: 'LIMIT_BREACH' | 'CIRCUIT_BREAKER' | 'MARGIN_CALL' | 'LIQUIDATION' | 'VOLATILITY_SPIKE';
  severity: RiskLevel;
  message: string;
  details: Record<string, any>;
  timestamp: string;
  resolved: boolean;
  resolvedAt?: string;
  actionTaken?: string;
}

// Market Data

export interface MarketData {
  symbol: string;
  exchange: Exchange;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quoteVolume?: number;
  trades?: number;
  timeframe?: TimeFrame;
}

export interface Ticker {
  symbol: string;
  exchange: Exchange;
  lastPrice: number;
  bidPrice: number;
  askPrice: number;
  bidSize: number;
  askSize: number;
  volume24h: number;
  quoteVolume24h: number;
  priceChange24h: number;
  priceChangePct24h: number;
  high24h: number;
  low24h: number;
  timestamp: string;
}

export interface OrderBookSnapshot {
  symbol: string;
  exchange: Exchange;
  timestamp: string;
  bids: [number, number][];
  asks: [number, number][];
  sequence?: number;
}

// Analytics & Performance

export interface PerformanceMetrics {
  timestamp: string;
  totalValue: number;
  dailyReturnPct: number;
  cumulativeReturnPct: number;
  sharpeRatio: number | null;
  sortinoRatio: number | null;
  volatility: number;
  maxDrawdownPct: number;
  winRatePct: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  avgHoldingPeriod: number;
  totalTrades: number;
}

export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  drawdown: number;
  drawdownPct: number;
  benchmark?: number;
}

export interface PnLBreakdown {
  timestamp: string;
  realizedPnL: number;
  unrealizedPnL: number;
  totalPnL: number;
  cumulativePnL: number;
  byStrategy: Record<string, number>;
  bySymbol: Record<string, number>;
  byExchange: Record<string, number>;
}

// Backtesting

export interface BacktestConfig {
  strategyId: string;
  strategyType: StrategyType;
  symbols: string[];
  exchanges: Exchange[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  parameters: Record<string, any>;
}

export interface BacktestResult {
  id: string;
  configId: string;
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  totalReturnPct: number;
  annualizedReturnPct: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdownPct: number;
  winRatePct: number;
  profitFactor: number;
  numTrades: number;
  avgWin: number;
  avgLoss: number;
  avgHoldingPeriod: number;
  equityCurve: EquityCurvePoint[];
  trades: Trade[];
  metrics: PerformanceMetrics[];
  completedAt?: string;
  error?: string;
}

// Alerts & Notifications

export interface Alert {
  id: string;
  userId: string;
  type: 'PRICE' | 'INDICATOR' | 'TRADE' | 'RISK' | 'SYSTEM';
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  title: string;
  message: string;
  details?: Record<string, any>;
  read: boolean;
  createdAt: string;
  expiresAt?: string;
  actionUrl?: string;
  actionLabel?: string;
}

export interface PriceAlert {
  id: string;
  userId: string;
  symbol: string;
  exchange: Exchange;
  condition: 'ABOVE' | 'BELOW' | 'CROSSES_ABOVE' | 'CROSSES_BELOW';
  targetPrice: number;
  enabled: boolean;
  triggered: boolean;
  createdAt: string;
  triggeredAt?: string;
  notificationChannels: ('email' | 'push' | 'sms')[];
}

// WebSocket Events

export interface WSMessage<T = any> {
  type: string;
  channel: string;
  data: T;
  timestamp: string;
  sequence?: number;
}

export interface WSSubscription {
  channel: string;
  symbols?: string[];
  exchanges?: Exchange[];
  params?: Record<string, any>;
}

// API Response Types

export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: APIError;
  timestamp: string;
  requestId?: string;
}

export interface APIError {
  code: string;
  message: string;
  details?: Record<string, any>;
  stack?: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// UI State Types

export interface LoadingState {
  isLoading: boolean;
  progress?: number;
  message?: string;
}

export interface ErrorState {
  hasError: boolean;
  error?: Error | APIError;
  retryable?: boolean;
}

export interface DataState<T> extends LoadingState, ErrorState {
  data: T | null;
  lastUpdated?: string;
}

// Table & List Types

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

export interface FilterConfig {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'contains' | 'in';
  value: any;
}

export interface PaginationConfig {
  page: number;
  pageSize: number;
  total?: number;
}

// Dashboard Widgets

export interface Widget {
  id: string;
  type: string;
  title: string;
  config: Record<string, any>;
  position: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  refreshInterval?: number;
}

export interface Dashboard {
  id: string;
  userId: string;
  name: string;
  description?: string;
  widgets: Widget[];
  layout: 'grid' | 'flex';
  isDefault: boolean;
  createdAt: string;
  updatedAt: string;
}

// System Health

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'down';
  timestamp: string;
  services: ServiceHealth[];
  uptime: number;
  version: string;
}

export interface ServiceHealth {
  name: string;
  status: 'up' | 'down' | 'degraded';
  latency?: number;
  lastCheck: string;
  error?: string;
}

// Utility Types

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};
export type RequireAtLeastOne<T, Keys extends keyof T = keyof T> = Pick<T, Exclude<keyof T, Keys>> &
  {
    [K in Keys]-?: Required<Pick<T, K>> & Partial<Pick<T, Exclude<Keys, K>>>;
  }[Keys];

// Re-export common types
export * from './chart.types';
