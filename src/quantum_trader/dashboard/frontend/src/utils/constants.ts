/**
 * Application Constants
 *
 * Central location for all application-wide constants, configuration values,
 * and magic numbers used throughout the Quantum Trader dashboard.
 */

import type { TimeFrame, Exchange, StrategyType, TimeframeOption } from '@/types';

// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  WS_URL: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8001',
  TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000', 10),
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
} as const;

// Authentication
export const AUTH_CONFIG = {
  TOKEN_KEY: 'quantum_trader_token',
  REFRESH_TOKEN_KEY: 'quantum_trader_refresh_token',
  USER_KEY: 'quantum_trader_user',
  TOKEN_EXPIRY_BUFFER: 5 * 60 * 1000, // 5 minutes in ms
  SESSION_TIMEOUT: 24 * 60 * 60 * 1000, // 24 hours in ms
  MFA_CODE_LENGTH: 6,
  PASSWORD_MIN_LENGTH: 8,
} as const;

// WebSocket Channels
export const WS_CHANNELS = {
  TRADES: 'trades',
  POSITIONS: 'positions',
  ORDERS: 'orders',
  PORTFOLIO: 'portfolio',
  MARKET_DATA: 'market',
  METRICS: 'metrics',
  ALERTS: 'alerts',
  STRATEGIES: 'strategies',
} as const;

export type WSChannel = typeof WS_CHANNELS[keyof typeof WS_CHANNELS];

// Timeframes
export const TIMEFRAMES: Record<TimeFrame, TimeframeOption> = {
  '1m': { value: '1m', label: '1 Minute', milliseconds: 60 * 1000 },
  '5m': { value: '5m', label: '5 Minutes', milliseconds: 5 * 60 * 1000 },
  '15m': { value: '15m', label: '15 Minutes', milliseconds: 15 * 60 * 1000 },
  '30m': { value: '30m', label: '30 Minutes', milliseconds: 30 * 60 * 1000 },
  '1h': { value: '1h', label: '1 Hour', milliseconds: 60 * 60 * 1000 },
  '4h': { value: '4h', label: '4 Hours', milliseconds: 4 * 60 * 60 * 1000 },
  '1d': { value: '1d', label: '1 Day', milliseconds: 24 * 60 * 60 * 1000 },
  '1w': { value: '1w', label: '1 Week', milliseconds: 7 * 24 * 60 * 60 * 1000 },
  '1M': { value: '1M', label: '1 Month', milliseconds: 30 * 24 * 60 * 60 * 1000 },
} as const;

export const TIMEFRAME_OPTIONS = Object.values(TIMEFRAMES);
export const DEFAULT_TIMEFRAME: TimeFrame = '1h';

// Exchanges
export const EXCHANGES: Record<Exchange, { name: string; logo?: string; color: string }> = {
  binance: { name: 'Binance', color: '#F3BA2F' },
  bybit: { name: 'Bybit', color: '#F7A600' },
  okx: { name: 'OKX', color: '#000000' },
  kucoin: { name: 'KuCoin', color: '#24AE8F' },
  bitget: { name: 'Bitget', color: '#00C087' },
} as const;

export const EXCHANGE_OPTIONS = Object.entries(EXCHANGES).map(([value, { name }]) => ({
  value: value as Exchange,
  label: name,
}));

export const DEFAULT_EXCHANGE: Exchange = 'binance';

// Strategy Types
export const STRATEGY_TYPES: Record<StrategyType, { name: string; description: string; color: string }> = {
  ARBITRAGE: {
    name: 'Arbitrage',
    description: 'Statistical and cross-exchange arbitrage opportunities',
    color: '#3b82f6',
  },
  MOMENTUM: {
    name: 'Momentum',
    description: 'Trend following and momentum-based strategies',
    color: '#10b981',
  },
  MEAN_REVERSION: {
    name: 'Mean Reversion',
    description: 'Statistical mean reversion strategies',
    color: '#8b5cf6',
  },
  MARKET_MAKING: {
    name: 'Market Making',
    description: 'Liquidity provision and spread capture',
    color: '#f59e0b',
  },
  HFT: {
    name: 'High Frequency',
    description: 'High-frequency trading strategies',
    color: '#ef4444',
  },
  ML: {
    name: 'Machine Learning',
    description: 'AI and ML-powered predictive strategies',
    color: '#06b6d4',
  },
  HYBRID: {
    name: 'Hybrid',
    description: 'Combined multi-strategy approaches',
    color: '#ec4899',
  },
  OPTIONS: {
    name: 'Options',
    description: 'Options and derivatives strategies',
    color: '#14b8a6',
  },
  ORDER_FLOW: {
    name: 'Order Flow',
    description: 'Order flow and microstructure analysis',
    color: '#a855f7',
  },
} as const;

export const STRATEGY_TYPE_OPTIONS = Object.entries(STRATEGY_TYPES).map(([value, { name }]) => ({
  value: value as StrategyType,
  label: name,
}));

// Chart Colors
export const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  success: '#10b981',
  danger: '#ef4444',
  warning: '#f59e0b',
  info: '#06b6d4',
  up: '#10b981',
  down: '#ef4444',
  neutral: '#64748b',
  volume: {
    up: 'rgba(16, 185, 129, 0.3)',
    down: 'rgba(239, 68, 68, 0.3)',
  },
} as const;

export const CHART_LINE_COLORS = [
  '#3b82f6',
  '#10b981',
  '#f59e0b',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#14b8a6',
  '#ef4444',
] as const;

// Risk Levels
export const RISK_LEVELS = {
  LOW: { label: 'Low', color: '#10b981', threshold: 0.3 },
  MEDIUM: { label: 'Medium', color: '#f59e0b', threshold: 0.6 },
  HIGH: { label: 'High', color: '#ef4444', threshold: 0.85 },
  CRITICAL: { label: 'Critical', color: '#dc2626', threshold: 1.0 },
} as const;

// Performance Metrics Thresholds
export const METRIC_THRESHOLDS = {
  SHARPE_RATIO: {
    excellent: 2.0,
    good: 1.0,
    acceptable: 0.5,
    poor: 0,
  },
  SORTINO_RATIO: {
    excellent: 3.0,
    good: 1.5,
    acceptable: 0.75,
    poor: 0,
  },
  WIN_RATE: {
    excellent: 0.65,
    good: 0.55,
    acceptable: 0.45,
    poor: 0.35,
  },
  MAX_DRAWDOWN: {
    excellent: 0.05,
    good: 0.1,
    acceptable: 0.15,
    poor: 0.25,
  },
  PROFIT_FACTOR: {
    excellent: 2.0,
    good: 1.5,
    acceptable: 1.2,
    poor: 1.0,
  },
} as const;

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  PAGE_SIZE_OPTIONS: [10, 20, 50, 100, 200],
  MAX_PAGE_SIZE: 1000,
} as const;

// Data Refresh Intervals (in milliseconds)
export const REFRESH_INTERVALS = {
  REALTIME: 1000, // 1 second
  FAST: 5000, // 5 seconds
  NORMAL: 10000, // 10 seconds
  SLOW: 30000, // 30 seconds
  VERY_SLOW: 60000, // 1 minute
  PORTFOLIO: 5000,
  POSITIONS: 5000,
  ORDERS: 2000,
  TRADES: 10000,
  MARKET_DATA: 1000,
  METRICS: 10000,
  STRATEGIES: 15000,
  RISK: 10000,
} as const;

// Number Formatting
export const NUMBER_FORMATS = {
  DECIMAL_PLACES: {
    PRICE: 2,
    PERCENTAGE: 2,
    QUANTITY: 8,
    CURRENCY: 2,
    RATIO: 2,
    LARGE_NUMBER: 0,
  },
  COMPACT_THRESHOLD: 1000000, // 1M
  PERCENTAGE_MULTIPLIER: 100,
} as const;

// Currency Settings
export const CURRENCIES = {
  DEFAULT: 'USD',
  SUPPORTED: ['USD', 'USDT', 'BUSD', 'EUR', 'GBP', 'BTC', 'ETH'],
  SYMBOLS: {
    USD: '$',
    USDT: '₮',
    BUSD: '$',
    EUR: '€',
    GBP: '£',
    BTC: '₿',
    ETH: 'Ξ',
  },
} as const;

// Date/Time Formats
export const DATE_FORMATS = {
  SHORT_DATE: 'MMM dd',
  LONG_DATE: 'MMM dd, yyyy',
  FULL_DATE: 'MMMM dd, yyyy',
  SHORT_TIME: 'HH:mm',
  LONG_TIME: 'HH:mm:ss',
  FULL_DATETIME: 'MMM dd, yyyy HH:mm:ss',
  ISO: "yyyy-MM-dd'T'HH:mm:ss.SSSxxx",
} as const;

// Local Storage Keys
export const STORAGE_KEYS = {
  THEME: 'quantum_trader_theme',
  LANGUAGE: 'quantum_trader_language',
  TIMEZONE: 'quantum_trader_timezone',
  DEFAULT_TIMEFRAME: 'quantum_trader_default_timeframe',
  DEFAULT_EXCHANGE: 'quantum_trader_default_exchange',
  CHART_SETTINGS: 'quantum_trader_chart_settings',
  TABLE_PREFERENCES: 'quantum_trader_table_preferences',
  DASHBOARD_LAYOUT: 'quantum_trader_dashboard_layout',
  SIDEBAR_COLLAPSED: 'quantum_trader_sidebar_collapsed',
  RECENT_SYMBOLS: 'quantum_trader_recent_symbols',
  WATCHLIST: 'quantum_trader_watchlist',
} as const;

// Chart Settings
export const CHART_SETTINGS = {
  DEFAULT_HEIGHT: 400,
  MIN_HEIGHT: 200,
  MAX_HEIGHT: 800,
  DEFAULT_CANDLES: 200,
  MAX_CANDLES: 1000,
  ANIMATION_DURATION: 300,
  DEBOUNCE_DELAY: 300,
} as const;

// Table Settings
export const TABLE_SETTINGS = {
  ROW_HEIGHT: {
    COMPACT: 40,
    NORMAL: 52,
    COMFORTABLE: 64,
  },
  DEFAULT_DENSITY: 'normal' as const,
  STICKY_HEADER: true,
  VIRTUAL_SCROLL_THRESHOLD: 100,
} as const;

// Order Constraints
export const ORDER_CONSTRAINTS = {
  MIN_ORDER_USD: 10,
  MAX_ORDER_USD: 100000,
  MIN_QUANTITY: 0.00000001,
  MAX_SLIPPAGE_PCT: 5,
  DEFAULT_SLIPPAGE_PCT: 0.1,
  ORDER_TIMEOUT_MS: 30000,
} as const;

// Position Constraints
export const POSITION_CONSTRAINTS = {
  MAX_POSITIONS: 20,
  MAX_POSITION_SIZE_PCT: 5,
  MIN_POSITION_USD: 500,
  MAX_LEVERAGE: 2.0,
} as const;

// Risk Limits
export const RISK_LIMITS = {
  MAX_DAILY_LOSS_PCT: 5,
  MAX_INTRADAY_LOSS_PCT: 3,
  MAX_DRAWDOWN_PCT: 15,
  MAX_PORTFOLIO_RISK_PCT: 2,
  MAX_SECTOR_EXPOSURE_PCT: 20,
  MAX_CORRELATION: 0.8,
} as const;

// Notification Settings
export const NOTIFICATION_SETTINGS = {
  MAX_NOTIFICATIONS: 50,
  AUTO_DISMISS_DELAY: 5000,
  SOUND_ENABLED: true,
  TOAST_POSITION: 'top-right' as const,
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  AUTH_FAILED: 'Authentication failed. Please login again.',
  SESSION_EXPIRED: 'Your session has expired. Please login again.',
  PERMISSION_DENIED: 'You do not have permission to perform this action.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  SERVER_ERROR: 'Server error. Please try again later.',
  NOT_FOUND: 'The requested resource was not found.',
  RATE_LIMIT: 'Too many requests. Please try again later.',
  UNKNOWN_ERROR: 'An unexpected error occurred.',
} as const;

// Success Messages
export const SUCCESS_MESSAGES = {
  ORDER_PLACED: 'Order placed successfully',
  ORDER_CANCELLED: 'Order cancelled successfully',
  STRATEGY_STARTED: 'Strategy started successfully',
  STRATEGY_PAUSED: 'Strategy paused successfully',
  SETTINGS_SAVED: 'Settings saved successfully',
  BACKTEST_STARTED: 'Backtest started successfully',
} as const;

// Routes
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  DASHBOARD: '/dashboard',
  PORTFOLIO: '/portfolio',
  TRADING: '/trading',
  STRATEGIES: '/strategies',
  BACKTESTING: '/backtesting',
  RISK: '/risk',
  ANALYTICS: '/analytics',
  SETTINGS: '/settings',
  HELP: '/help',
} as const;

// Feature Flags
export const FEATURE_FLAGS = {
  ENABLE_BACKTESTING: true,
  ENABLE_PAPER_TRADING: true,
  ENABLE_OPTIONS_TRADING: false,
  ENABLE_SOCIAL_TRADING: false,
  ENABLE_ADVANCED_CHARTS: true,
  ENABLE_AI_INSIGHTS: true,
  ENABLE_NOTIFICATIONS: true,
  ENABLE_DARK_MODE: true,
  ENABLE_MULTI_LANGUAGE: false,
} as const;

// Popular Trading Symbols
export const POPULAR_SYMBOLS = [
  'BTC/USDT',
  'ETH/USDT',
  'BNB/USDT',
  'SOL/USDT',
  'XRP/USDT',
  'ADA/USDT',
  'AVAX/USDT',
  'DOT/USDT',
  'MATIC/USDT',
  'LINK/USDT',
] as const;

// Indicator Presets
export const INDICATOR_PRESETS = {
  RSI: {
    period: 14,
    overbought: 70,
    oversold: 30,
  },
  MACD: {
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
  },
  BOLLINGER_BANDS: {
    period: 20,
    stdDev: 2,
  },
  STOCHASTIC: {
    kPeriod: 14,
    dPeriod: 3,
    overbought: 80,
    oversold: 20,
  },
  ATR: {
    period: 14,
  },
  EMA: {
    periods: [9, 21, 50, 200],
  },
  SMA: {
    periods: [20, 50, 100, 200],
  },
} as const;

// Animation Durations
export const ANIMATION = {
  FAST: 150,
  NORMAL: 200,
  SLOW: 300,
  VERY_SLOW: 500,
} as const;

// Z-Index Scale
export const Z_INDEX = {
  DROPDOWN: 1000,
  STICKY: 1020,
  FIXED: 1030,
  MODAL_BACKDROP: 1040,
  MODAL: 1050,
  POPOVER: 1060,
  TOOLTIP: 1070,
} as const;

// Breakpoints (pixels)
export const BREAKPOINTS = {
  XS: 0,
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// Export everything as default for convenient importing
export default {
  API_CONFIG,
  AUTH_CONFIG,
  WS_CHANNELS,
  TIMEFRAMES,
  TIMEFRAME_OPTIONS,
  DEFAULT_TIMEFRAME,
  EXCHANGES,
  EXCHANGE_OPTIONS,
  DEFAULT_EXCHANGE,
  STRATEGY_TYPES,
  STRATEGY_TYPE_OPTIONS,
  CHART_COLORS,
  CHART_LINE_COLORS,
  RISK_LEVELS,
  METRIC_THRESHOLDS,
  PAGINATION,
  REFRESH_INTERVALS,
  NUMBER_FORMATS,
  CURRENCIES,
  DATE_FORMATS,
  STORAGE_KEYS,
  CHART_SETTINGS,
  TABLE_SETTINGS,
  ORDER_CONSTRAINTS,
  POSITION_CONSTRAINTS,
  RISK_LIMITS,
  NOTIFICATION_SETTINGS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  ROUTES,
  FEATURE_FLAGS,
  POPULAR_SYMBOLS,
  INDICATOR_PRESETS,
  ANIMATION,
  Z_INDEX,
  BREAKPOINTS,
} as const;
