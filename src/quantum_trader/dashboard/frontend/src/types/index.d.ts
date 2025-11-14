/**
 * Global type definitions for Quantum Trader AI Dashboard
 *
 * Centralized TypeScript definitions for:
 * - Trading data structures
 * - API responses
 * - Component props
 * - Utility types
 */

/**
 * Base types
 */
export type Decimal = string;
export type Timestamp = number;
export type UUID = string;

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
  quantity: Decimal;
  price?: Decimal;
  order_type: OrderType;
  exchange: string;
  strategy: string;
  timestamp: Timestamp;
  order_id?: string;
  metadata?: Record<string, any>;
}

/**
 * Trading position interface
 */
export interface Position {
  symbol: string;
  quantity: Decimal;
  entry_price: Decimal;
  current_price: Decimal;
  exchange: string;
  strategy: string;
  opened_at: Timestamp;
  position_id: string;
  pnl?: Decimal;
  pnl_percent?: Decimal;
}

/**
 * Trading signal interface
 */
export interface Signal {
  symbol: string;
  action: SignalAction;
  strength: Decimal;
  confidence: Decimal;
  timestamp: Timestamp;
  strategy: string;
  timeframe: string;
  indicators?: Record<string, Decimal>;
  metadata?: Record<string, any>;
}

/**
 * Execution result interface
 */
export interface ExecutionResult {
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: Decimal;
  filled_quantity: Decimal;
  price: Decimal;
  average_price: Decimal;
  status: OrderStatus;
  timestamp: Timestamp;
  fees: Decimal;
  metadata?: Record<string, any>;
}

/**
 * Portfolio summary interface
 */
export interface PortfolioSummary {
  total_value: Decimal;
  cash_balance: Decimal;
  positions_value: Decimal;
  total_pnl: Decimal;
  total_pnl_percent: Decimal;
  position_count: number;
  open_order_count: number;
  currency: string;
}

/**
 * Performance metrics interface
 */
export interface PerformanceMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: Decimal;
  profit_factor: Decimal;
  sharpe_ratio: Decimal;
  max_drawdown: Decimal;
  total_pnl: Decimal;
  average_win: Decimal;
  average_loss: Decimal;
  largest_win: Decimal;
  largest_loss: Decimal;
  average_trade_duration: number;
}

/**
 * Market data (OHLCV) interface
 */
export interface MarketData {
  symbol: string;
  timestamp: Timestamp;
  open: Decimal;
  high: Decimal;
  low: Decimal;
  close: Decimal;
  volume: Decimal;
  exchange: string;
}

/**
 * Ticker data interface
 */
export interface TickerData {
  symbol: string;
  last_price: Decimal;
  bid_price: Decimal;
  ask_price: Decimal;
  volume_24h: Decimal;
  price_change_24h: Decimal;
  price_change_percent_24h: Decimal;
  high_24h: Decimal;
  low_24h: Decimal;
  timestamp: Timestamp;
}

/**
 * Order book level interface
 */
export interface OrderBookLevel {
  price: Decimal;
  quantity: Decimal;
  total: Decimal;
}

/**
 * Order book interface
 */
export interface OrderBook {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: Timestamp;
}

/**
 * Trade history interface
 */
export interface Trade {
  trade_id: string;
  order_id: string;
  symbol: string;
  side: OrderSide;
  quantity: Decimal;
  price: Decimal;
  fees: Decimal;
  pnl: Decimal;
  timestamp: Timestamp;
  exchange: string;
  strategy: string;
}

/**
 * Strategy configuration interface
 */
export interface StrategyConfig {
  strategy_id: string;
  name: string;
  description: string;
  enabled: boolean;
  parameters: Record<string, any>;
  symbols: string[];
  timeframes: string[];
  risk_config: RiskConfig;
}

/**
 * Risk configuration interface
 */
export interface RiskConfig {
  max_position_size: Decimal;
  max_leverage: number;
  max_drawdown_percent: Decimal;
  max_daily_loss_percent: Decimal;
  position_size_method: 'fixed' | 'risk_based' | 'kelly';
  stop_loss_percent?: Decimal;
  take_profit_percent?: Decimal;
}

/**
 * WebSocket message interface
 */
export interface WebSocketMessage {
  type: string;
  channel: string;
  data: any;
  timestamp: Timestamp;
}

/**
 * API response wrapper interface
 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: Timestamp;
  request_id?: string;
}

/**
 * API error interface
 */
export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: Timestamp;
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
  items: T[];
  pagination: Pagination;
}

/**
 * Filter interface
 */
export interface Filter {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'like';
  value: any;
}

/**
 * Sort interface
 */
export interface Sort {
  field: string;
  direction: 'asc' | 'desc';
}

/**
 * Query parameters interface
 */
export interface QueryParams {
  page?: number;
  page_size?: number;
  filters?: Filter[];
  sort?: Sort[];
  search?: string;
}

/**
 * Notification interface
 */
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: Timestamp;
  read: boolean;
  action?: NotificationAction;
}

/**
 * Notification action interface
 */
export interface NotificationAction {
  label: string;
  url?: string;
  callback?: () => void;
}

/**
 * User preferences interface
 */
export interface UserPreferences {
  theme: 'light' | 'dark';
  language: string;
  timezone: string;
  currency: string;
  notifications_enabled: boolean;
  notification_types: string[];
  chart_preferences: ChartPreferences;
}

/**
 * Chart preferences interface
 */
export interface ChartPreferences {
  default_timeframe: string;
  default_indicators: string[];
  color_scheme: string;
  show_volume: boolean;
  show_grid: boolean;
}

/**
 * Component base props interface
 */
export interface BaseComponentProps {
  className?: string;
  style?: React.CSSProperties;
  children?: React.ReactNode;
}

/**
 * Loading state interface
 */
export interface LoadingState {
  isLoading: boolean;
  progress?: number;
  message?: string;
}

/**
 * Error state interface
 */
export interface ErrorState {
  hasError: boolean;
  error?: Error;
  errorMessage?: string;
}

/**
 * Form field interface
 */
export interface FormField<T = any> {
  name: string;
  value: T;
  error?: string;
  touched: boolean;
  dirty: boolean;
}

/**
 * Validation result interface
 */
export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
}

/**
 * Theme interface
 */
export interface Theme {
  mode: 'light' | 'dark';
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    danger: string;
    info: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
  };
}

/**
 * Route configuration interface
 */
export interface RouteConfig {
  path: string;
  component: React.ComponentType<any>;
  exact?: boolean;
  protected?: boolean;
  permissions?: string[];
  title?: string;
}

/**
 * Breadcrumb interface
 */
export interface Breadcrumb {
  label: string;
  path?: string;
  active?: boolean;
}

/**
 * Action button interface
 */
export interface ActionButton {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'danger';
}

/**
 * Table column interface
 */
export interface TableColumn<T = any> {
  key: string;
  header: string;
  accessor?: (row: T) => any;
  render?: (value: any, row: T) => React.ReactNode;
  sortable?: boolean;
  filterable?: boolean;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

/**
 * Modal props interface
 */
export interface ModalProps extends BaseComponentProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
}

/**
 * Dropdown option interface
 */
export interface DropdownOption<T = any> {
  label: string;
  value: T;
  disabled?: boolean;
  icon?: React.ReactNode;
}

/**
 * Tab interface
 */
export interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  content: React.ReactNode;
  disabled?: boolean;
}

/**
 * Alert interface
 */
export interface Alert {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  dismissible?: boolean;
  duration?: number;
}

/**
 * Utility type: Make all properties optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Utility type: Make all properties required recursively
 */
export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

/**
 * Utility type: Extract promise return type
 */
export type Awaited<T> = T extends Promise<infer U> ? U : T;

/**
 * Utility type: Function with typed parameters
 */
export type TypedFunction<TParams extends any[] = any[], TReturn = void> = (
  ...args: TParams
) => TReturn;

/**
 * Global window extensions
 */
declare global {
  interface Window {
    __REDUX_STORE__?: any;
    __REDUX_DEVTOOLS_EXTENSION__?: any;
  }
}
