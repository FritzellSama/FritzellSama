/**
 * Global constants for trading dashboard
 *
 * Centralized configuration values loaded from environment:
 * - API endpoints
 * - WebSocket URLs
 * - Trading parameters
 * - UI configuration
 *
 * CRITICAL: All values loaded from environment variables
 */

/**
 * Environment helper
 */
function getEnvVar(key: string, defaultValue?: string): string {
  const value = import.meta.env[key] || process.env[key];
  if (value === undefined && defaultValue === undefined) {
    throw new Error(`Environment variable ${key} is required but not set`);
  }
  return value || defaultValue!;
}

function getEnvNumber(key: string, defaultValue?: number): number {
  const value = getEnvVar(key, defaultValue?.toString());
  const parsed = parseFloat(value);
  if (isNaN(parsed)) {
    throw new Error(`Environment variable ${key} must be a valid number`);
  }
  return parsed;
}

function getEnvBoolean(key: string, defaultValue: boolean = false): boolean {
  const value = getEnvVar(key, defaultValue.toString());
  return value.toLowerCase() === 'true' || value === '1';
}

/**
 * API Configuration
 */
export const API_CONFIG = {
  BASE_URL: getEnvVar('VITE_API_BASE_URL'),
  WEBSOCKET_URL: getEnvVar('VITE_WEBSOCKET_URL'),
  TIMEOUT: getEnvNumber('VITE_API_TIMEOUT', 30000),
  MAX_RETRIES: getEnvNumber('VITE_API_MAX_RETRIES', 3),
  RETRY_DELAY: getEnvNumber('VITE_API_RETRY_DELAY', 1000),
} as const;

/**
 * Authentication Configuration
 */
export const AUTH_CONFIG = {
  TOKEN_STORAGE_KEY: getEnvVar('VITE_TOKEN_STORAGE_KEY', 'quantum_trader_token'),
  REFRESH_TOKEN_STORAGE_KEY: getEnvVar('VITE_REFRESH_TOKEN_STORAGE_KEY', 'quantum_trader_refresh_token'),
  TOKEN_EXPIRY_KEY: getEnvVar('VITE_TOKEN_EXPIRY_KEY', 'quantum_trader_token_expiry'),
  REFRESH_THRESHOLD_SECONDS: getEnvNumber('VITE_TOKEN_REFRESH_THRESHOLD', 300),
  SESSION_TIMEOUT_MINUTES: getEnvNumber('VITE_SESSION_TIMEOUT', 60),
} as const;

/**
 * WebSocket Configuration
 */
export const WS_CONFIG = {
  RECONNECT_INTERVAL: getEnvNumber('VITE_WS_RECONNECT_INTERVAL', 5000),
  MAX_RECONNECT_ATTEMPTS: getEnvNumber('VITE_WS_MAX_RECONNECT_ATTEMPTS', 10),
  HEARTBEAT_INTERVAL: getEnvNumber('VITE_WS_HEARTBEAT_INTERVAL', 30000),
  MESSAGE_QUEUE_SIZE: getEnvNumber('VITE_WS_MESSAGE_QUEUE_SIZE', 1000),
} as const;

/**
 * Trading Configuration
 */
export const TRADING_CONFIG = {
  DEFAULT_LEVERAGE: getEnvNumber('VITE_DEFAULT_LEVERAGE', 1),
  MAX_LEVERAGE: getEnvNumber('VITE_MAX_LEVERAGE', 20),
  DEFAULT_SLIPPAGE: getEnvNumber('VITE_DEFAULT_SLIPPAGE', 0.001),
  MIN_ORDER_SIZE: getEnvNumber('VITE_MIN_ORDER_SIZE', 10),
  MAX_POSITION_SIZE: getEnvNumber('VITE_MAX_POSITION_SIZE', 1000000),
  DEFAULT_RISK_PERCENT: getEnvNumber('VITE_DEFAULT_RISK_PERCENT', 1),
} as const;

/**
 * Chart Configuration
 */
export const CHART_CONFIG = {
  DEFAULT_TIMEFRAME: getEnvVar('VITE_DEFAULT_TIMEFRAME', '1h'),
  DEFAULT_CANDLE_COUNT: getEnvNumber('VITE_DEFAULT_CANDLE_COUNT', 100),
  MAX_CANDLE_COUNT: getEnvNumber('VITE_MAX_CANDLE_COUNT', 1000),
  UPDATE_INTERVAL: getEnvNumber('VITE_CHART_UPDATE_INTERVAL', 1000),
  PRICE_DECIMALS: getEnvNumber('VITE_CHART_PRICE_DECIMALS', 2),
  VOLUME_DECIMALS: getEnvNumber('VITE_CHART_VOLUME_DECIMALS', 4),
} as const;

/**
 * Cache Configuration
 */
export const CACHE_CONFIG = {
  MARKET_DATA_TTL: getEnvNumber('VITE_CACHE_MARKET_DATA_TTL', 5),
  POSITION_DATA_TTL: getEnvNumber('VITE_CACHE_POSITION_DATA_TTL', 10),
  ORDER_DATA_TTL: getEnvNumber('VITE_CACHE_ORDER_DATA_TTL', 30),
  STATIC_DATA_TTL: getEnvNumber('VITE_CACHE_STATIC_DATA_TTL', 3600),
} as const;

/**
 * UI Configuration
 */
export const UI_CONFIG = {
  THEME: getEnvVar('VITE_DEFAULT_THEME', 'dark') as 'light' | 'dark',
  REFRESH_INTERVAL: getEnvNumber('VITE_UI_REFRESH_INTERVAL', 1000),
  NOTIFICATION_DURATION: getEnvNumber('VITE_NOTIFICATION_DURATION', 5000),
  TOAST_POSITION: getEnvVar('VITE_TOAST_POSITION', 'top-right'),
  ITEMS_PER_PAGE: getEnvNumber('VITE_ITEMS_PER_PAGE', 20),
  MAX_ITEMS_PER_PAGE: getEnvNumber('VITE_MAX_ITEMS_PER_PAGE', 100),
} as const;

/**
 * Supported exchanges
 */
export const EXCHANGES = [
  { id: 'binance', name: 'Binance', enabled: getEnvBoolean('VITE_EXCHANGE_BINANCE_ENABLED', true) },
  { id: 'bybit', name: 'Bybit', enabled: getEnvBoolean('VITE_EXCHANGE_BYBIT_ENABLED', true) },
  { id: 'okx', name: 'OKX', enabled: getEnvBoolean('VITE_EXCHANGE_OKX_ENABLED', true) },
  { id: 'kucoin', name: 'KuCoin', enabled: getEnvBoolean('VITE_EXCHANGE_KUCOIN_ENABLED', false) },
  { id: 'bitget', name: 'Bitget', enabled: getEnvBoolean('VITE_EXCHANGE_BITGET_ENABLED', false) },
] as const;

/**
 * Supported timeframes
 */
export const TIMEFRAMES = [
  { value: '1m', label: '1 Minute', seconds: 60 },
  { value: '5m', label: '5 Minutes', seconds: 300 },
  { value: '15m', label: '15 Minutes', seconds: 900 },
  { value: '30m', label: '30 Minutes', seconds: 1800 },
  { value: '1h', label: '1 Hour', seconds: 3600 },
  { value: '4h', label: '4 Hours', seconds: 14400 },
  { value: '1d', label: '1 Day', seconds: 86400 },
  { value: '1w', label: '1 Week', seconds: 604800 },
  { value: '1M', label: '1 Month', seconds: 2592000 },
] as const;

/**
 * Order types
 */
export const ORDER_TYPES = [
  { value: 'MARKET', label: 'Market' },
  { value: 'LIMIT', label: 'Limit' },
  { value: 'STOP_LOSS', label: 'Stop Loss' },
  { value: 'TAKE_PROFIT', label: 'Take Profit' },
  { value: 'STOP_LIMIT', label: 'Stop Limit' },
] as const;

/**
 * Order sides
 */
export const ORDER_SIDES = [
  { value: 'BUY', label: 'Buy', color: '#10b981' },
  { value: 'SELL', label: 'Sell', color: '#ef4444' },
] as const;

/**
 * Order status
 */
export const ORDER_STATUS = [
  { value: 'PENDING', label: 'Pending', color: '#6b7280' },
  { value: 'OPEN', label: 'Open', color: '#3b82f6' },
  { value: 'PARTIAL', label: 'Partially Filled', color: '#f59e0b' },
  { value: 'FILLED', label: 'Filled', color: '#10b981' },
  { value: 'CANCELLED', label: 'Cancelled', color: '#6b7280' },
  { value: 'REJECTED', label: 'Rejected', color: '#ef4444' },
  { value: 'EXPIRED', label: 'Expired', color: '#6b7280' },
  { value: 'FAILED', label: 'Failed', color: '#ef4444' },
] as const;

/**
 * Signal actions
 */
export const SIGNAL_ACTIONS = [
  { value: 'BUY', label: 'Buy', color: '#10b981' },
  { value: 'SELL', label: 'Sell', color: '#ef4444' },
  { value: 'HOLD', label: 'Hold', color: '#6b7280' },
  { value: 'CLOSE', label: 'Close', color: '#f59e0b' },
] as const;

/**
 * Technical indicators
 */
export const INDICATORS = [
  { id: 'sma', name: 'Simple Moving Average', category: 'trend', overlay: true },
  { id: 'ema', name: 'Exponential Moving Average', category: 'trend', overlay: true },
  { id: 'bb', name: 'Bollinger Bands', category: 'volatility', overlay: true },
  { id: 'rsi', name: 'Relative Strength Index', category: 'momentum', overlay: false },
  { id: 'macd', name: 'MACD', category: 'momentum', overlay: false },
  { id: 'stoch', name: 'Stochastic', category: 'momentum', overlay: false },
  { id: 'atr', name: 'Average True Range', category: 'volatility', overlay: false },
  { id: 'volume', name: 'Volume', category: 'volume', overlay: false },
] as const;

/**
 * Date/time formats
 */
export const DATE_FORMATS = {
  FULL: getEnvVar('VITE_DATE_FORMAT_FULL', 'YYYY-MM-DD HH:mm:ss'),
  DATE: getEnvVar('VITE_DATE_FORMAT_DATE', 'YYYY-MM-DD'),
  TIME: getEnvVar('VITE_DATE_FORMAT_TIME', 'HH:mm:ss'),
  SHORT: getEnvVar('VITE_DATE_FORMAT_SHORT', 'MM/DD HH:mm'),
} as const;

/**
 * Number formats
 */
export const NUMBER_FORMATS = {
  PRICE_DECIMALS: getEnvNumber('VITE_NUMBER_PRICE_DECIMALS', 2),
  QUANTITY_DECIMALS: getEnvNumber('VITE_NUMBER_QUANTITY_DECIMALS', 8),
  PERCENTAGE_DECIMALS: getEnvNumber('VITE_NUMBER_PERCENTAGE_DECIMALS', 2),
  CURRENCY_DECIMALS: getEnvNumber('VITE_NUMBER_CURRENCY_DECIMALS', 2),
} as const;

/**
 * Performance metrics thresholds
 */
export const METRICS_THRESHOLDS = {
  WIN_RATE_GOOD: getEnvNumber('VITE_METRICS_WIN_RATE_GOOD', 60),
  WIN_RATE_WARNING: getEnvNumber('VITE_METRICS_WIN_RATE_WARNING', 50),
  SHARPE_RATIO_GOOD: getEnvNumber('VITE_METRICS_SHARPE_GOOD', 2),
  SHARPE_RATIO_WARNING: getEnvNumber('VITE_METRICS_SHARPE_WARNING', 1),
  MAX_DRAWDOWN_GOOD: getEnvNumber('VITE_METRICS_DRAWDOWN_GOOD', 10),
  MAX_DRAWDOWN_WARNING: getEnvNumber('VITE_METRICS_DRAWDOWN_WARNING', 20),
  PROFIT_FACTOR_GOOD: getEnvNumber('VITE_METRICS_PROFIT_FACTOR_GOOD', 2),
  PROFIT_FACTOR_WARNING: getEnvNumber('VITE_METRICS_PROFIT_FACTOR_WARNING', 1.5),
} as const;

/**
 * Risk management limits
 */
export const RISK_LIMITS = {
  MAX_POSITION_PERCENT: getEnvNumber('VITE_RISK_MAX_POSITION_PERCENT', 10),
  MAX_LEVERAGE: getEnvNumber('VITE_RISK_MAX_LEVERAGE', 20),
  MAX_DAILY_LOSS_PERCENT: getEnvNumber('VITE_RISK_MAX_DAILY_LOSS_PERCENT', 5),
  MAX_DRAWDOWN_PERCENT: getEnvNumber('VITE_RISK_MAX_DRAWDOWN_PERCENT', 20),
  MIN_RISK_REWARD_RATIO: getEnvNumber('VITE_RISK_MIN_RR_RATIO', 2),
} as const;

/**
 * Notification types
 */
export const NOTIFICATION_TYPES = [
  { value: 'ORDER_FILLED', label: 'Order Filled', enabled: getEnvBoolean('VITE_NOTIF_ORDER_FILLED', true) },
  { value: 'ORDER_CANCELLED', label: 'Order Cancelled', enabled: getEnvBoolean('VITE_NOTIF_ORDER_CANCELLED', true) },
  { value: 'POSITION_OPENED', label: 'Position Opened', enabled: getEnvBoolean('VITE_NOTIF_POSITION_OPENED', true) },
  { value: 'POSITION_CLOSED', label: 'Position Closed', enabled: getEnvBoolean('VITE_NOTIF_POSITION_CLOSED', true) },
  { value: 'STOP_LOSS_HIT', label: 'Stop Loss Hit', enabled: getEnvBoolean('VITE_NOTIF_STOP_LOSS', true) },
  { value: 'TAKE_PROFIT_HIT', label: 'Take Profit Hit', enabled: getEnvBoolean('VITE_NOTIF_TAKE_PROFIT', true) },
  { value: 'SIGNAL_GENERATED', label: 'Signal Generated', enabled: getEnvBoolean('VITE_NOTIF_SIGNAL', false) },
  { value: 'RISK_LIMIT_EXCEEDED', label: 'Risk Limit Exceeded', enabled: getEnvBoolean('VITE_NOTIF_RISK_LIMIT', true) },
] as const;

/**
 * Color palette
 */
export const COLORS = {
  PRIMARY: getEnvVar('VITE_COLOR_PRIMARY', '#3b82f6'),
  SUCCESS: getEnvVar('VITE_COLOR_SUCCESS', '#10b981'),
  WARNING: getEnvVar('VITE_COLOR_WARNING', '#f59e0b'),
  DANGER: getEnvVar('VITE_COLOR_DANGER', '#ef4444'),
  INFO: getEnvVar('VITE_COLOR_INFO', '#3b82f6'),
  BUY: getEnvVar('VITE_COLOR_BUY', '#10b981'),
  SELL: getEnvVar('VITE_COLOR_SELL', '#ef4444'),
} as const;

/**
 * Local storage keys
 */
export const STORAGE_KEYS = {
  THEME: getEnvVar('VITE_STORAGE_KEY_THEME', 'quantum_trader_theme'),
  LAYOUT: getEnvVar('VITE_STORAGE_KEY_LAYOUT', 'quantum_trader_layout'),
  PREFERENCES: getEnvVar('VITE_STORAGE_KEY_PREFERENCES', 'quantum_trader_preferences'),
  CHART_PRESETS: getEnvVar('VITE_STORAGE_KEY_CHART_PRESETS', 'quantum_trader_chart_presets'),
  WATCHLIST: getEnvVar('VITE_STORAGE_KEY_WATCHLIST', 'quantum_trader_watchlist'),
} as const;

/**
 * Feature flags
 */
export const FEATURES = {
  ENABLE_LIVE_TRADING: getEnvBoolean('VITE_FEATURE_LIVE_TRADING', false),
  ENABLE_BACKTESTING: getEnvBoolean('VITE_FEATURE_BACKTESTING', true),
  ENABLE_PAPER_TRADING: getEnvBoolean('VITE_FEATURE_PAPER_TRADING', true),
  ENABLE_SOCIAL_FEATURES: getEnvBoolean('VITE_FEATURE_SOCIAL', false),
  ENABLE_ADVANCED_CHARTS: getEnvBoolean('VITE_FEATURE_ADVANCED_CHARTS', true),
  ENABLE_AI_SIGNALS: getEnvBoolean('VITE_FEATURE_AI_SIGNALS', true),
  ENABLE_NOTIFICATIONS: getEnvBoolean('VITE_FEATURE_NOTIFICATIONS', true),
  ENABLE_ANALYTICS: getEnvBoolean('VITE_FEATURE_ANALYTICS', true),
} as const;

/**
 * Application metadata
 */
export const APP_METADATA = {
  NAME: getEnvVar('VITE_APP_NAME', 'Quantum Trader AI'),
  VERSION: getEnvVar('VITE_APP_VERSION', '1.0.0'),
  ENVIRONMENT: getEnvVar('VITE_ENVIRONMENT', 'development'),
  BUILD_ID: getEnvVar('VITE_BUILD_ID', 'dev'),
  API_VERSION: getEnvVar('VITE_API_VERSION', 'v1'),
} as const;

/**
 * Validation constants
 */
export const VALIDATION = {
  MIN_PASSWORD_LENGTH: getEnvNumber('VITE_MIN_PASSWORD_LENGTH', 8),
  MAX_PASSWORD_LENGTH: getEnvNumber('VITE_MAX_PASSWORD_LENGTH', 128),
  MIN_USERNAME_LENGTH: getEnvNumber('VITE_MIN_USERNAME_LENGTH', 3),
  MAX_USERNAME_LENGTH: getEnvNumber('VITE_MAX_USERNAME_LENGTH', 32),
  EMAIL_REGEX: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
} as const;
