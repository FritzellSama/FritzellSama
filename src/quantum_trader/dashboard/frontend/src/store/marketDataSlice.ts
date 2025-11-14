/**
 * Market Data Redux Slice
 *
 * Manages real-time market data state for the Quantum Trader AI dashboard.
 * Handles orderbook, ticker, trades, and OHLCV data from multiple exchanges.
 *
 * @module marketDataSlice
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from './index';

/**
 * Market data interfaces
 */
export interface OrderBookLevel {
  price: string;  // Decimal as string to avoid precision loss
  quantity: string;
  total: string;
}

export interface OrderBook {
  symbol: string;
  exchange: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: string;
  lastUpdateId: number;
}

export interface Ticker {
  symbol: string;
  exchange: string;
  last: string;
  bid: string;
  ask: string;
  volume24h: string;
  high24h: string;
  low24h: string;
  change24h: string;
  changePercent24h: string;
  timestamp: string;
}

export interface Trade {
  id: string;
  symbol: string;
  exchange: string;
  price: string;
  quantity: string;
  side: 'BUY' | 'SELL';
  timestamp: string;
}

export interface OHLCV {
  timestamp: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

export interface MarketDataSubscription {
  symbol: string;
  exchange: string;
  channels: string[];  // e.g., ['orderbook', 'ticker', 'trades']
}

/**
 * Market data state interface
 */
export interface MarketDataState {
  orderbooks: Record<string, OrderBook>;
  tickers: Record<string, Ticker>;
  trades: Record<string, Trade[]>;
  ohlcv: Record<string, OHLCV[]>;
  subscriptions: MarketDataSubscription[];
  loading: Record<string, boolean>;
  errors: Record<string, string | null>;
  connectionStatus: Record<string, 'connected' | 'connecting' | 'disconnected' | 'error'>;
  lastUpdated: Record<string, string>;
}

/**
 * Initial state
 */
const initialState: MarketDataState = {
  orderbooks: {},
  tickers: {},
  trades: {},
  ohlcv: {},
  subscriptions: [],
  loading: {},
  errors: {},
  connectionStatus: {},
  lastUpdated: {},
};

/**
 * Async thunks for API calls
 */

// Subscribe to market data
export const subscribeMarketData = createAsyncThunk(
  'marketData/subscribe',
  async (subscription: MarketDataSubscription, { rejectWithValue }) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
      if (!apiUrl) {
        throw new Error('API_URL not configured');
      }

      const response = await fetch(`${apiUrl}/api/v1/market-data/subscribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(subscription),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to subscribe to market data');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Unsubscribe from market data
export const unsubscribeMarketData = createAsyncThunk(
  'marketData/unsubscribe',
  async (subscription: MarketDataSubscription, { rejectWithValue }) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
      if (!apiUrl) {
        throw new Error('API_URL not configured');
      }

      const response = await fetch(`${apiUrl}/api/v1/market-data/unsubscribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(subscription),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to unsubscribe from market data');
      }

      return subscription;
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Fetch historical OHLCV data
export const fetchOHLCV = createAsyncThunk(
  'marketData/fetchOHLCV',
  async (params: { symbol: string; exchange: string; timeframe: string; limit?: number }, { rejectWithValue }) => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
      if (!apiUrl) {
        throw new Error('API_URL not configured');
      }

      const queryParams = new URLSearchParams({
        symbol: params.symbol,
        exchange: params.exchange,
        timeframe: params.timeframe,
        ...(params.limit && { limit: params.limit.toString() }),
      });

      const response = await fetch(`${apiUrl}/api/v1/market-data/ohlcv?${queryParams}`);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch OHLCV data');
      }

      const data = await response.json();
      return { key: `${params.exchange}:${params.symbol}:${params.timeframe}`, data: data.ohlcv };
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

/**
 * Market data slice
 */
const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    // Update orderbook from WebSocket
    updateOrderBook: (state, action: PayloadAction<OrderBook>) => {
      const key = `${action.payload.exchange}:${action.payload.symbol}`;
      state.orderbooks[key] = action.payload;
      state.lastUpdated[key] = new Date().toISOString();
      state.errors[key] = null;
    },

    // Update ticker from WebSocket
    updateTicker: (state, action: PayloadAction<Ticker>) => {
      const key = `${action.payload.exchange}:${action.payload.symbol}`;
      state.tickers[key] = action.payload;
      state.lastUpdated[key] = new Date().toISOString();
      state.errors[key] = null;
    },

    // Add trade from WebSocket
    addTrade: (state, action: PayloadAction<Trade>) => {
      const key = `${action.payload.exchange}:${action.payload.symbol}`;

      if (!state.trades[key]) {
        state.trades[key] = [];
      }

      // Add new trade to the beginning
      state.trades[key].unshift(action.payload);

      // Keep only the latest N trades (configurable via env var)
      const maxTrades = parseInt(process.env.REACT_APP_MAX_TRADES_HISTORY || '100', 10);
      if (state.trades[key].length > maxTrades) {
        state.trades[key] = state.trades[key].slice(0, maxTrades);
      }

      state.lastUpdated[key] = new Date().toISOString();
    },

    // Update connection status
    updateConnectionStatus: (
      state,
      action: PayloadAction<{ exchange: string; status: 'connected' | 'connecting' | 'disconnected' | 'error' }>
    ) => {
      state.connectionStatus[action.payload.exchange] = action.payload.status;
    },

    // Set error
    setError: (state, action: PayloadAction<{ key: string; error: string }>) => {
      state.errors[action.payload.key] = action.payload.error;
    },

    // Clear error
    clearError: (state, action: PayloadAction<string>) => {
      state.errors[action.payload] = null;
    },

    // Clear all market data
    clearMarketData: (state) => {
      state.orderbooks = {};
      state.tickers = {};
      state.trades = {};
      state.ohlcv = {};
      state.lastUpdated = {};
      state.errors = {};
    },

    // Remove market data for a specific symbol
    removeMarketData: (state, action: PayloadAction<{ exchange: string; symbol: string }>) => {
      const key = `${action.payload.exchange}:${action.payload.symbol}`;
      delete state.orderbooks[key];
      delete state.tickers[key];
      delete state.trades[key];
      delete state.lastUpdated[key];
      delete state.errors[key];
    },
  },
  extraReducers: (builder) => {
    // Subscribe to market data
    builder.addCase(subscribeMarketData.pending, (state, action) => {
      const key = `${action.meta.arg.exchange}:${action.meta.arg.symbol}`;
      state.loading[key] = true;
      state.errors[key] = null;
      state.connectionStatus[action.meta.arg.exchange] = 'connecting';
    });
    builder.addCase(subscribeMarketData.fulfilled, (state, action) => {
      const subscription = action.meta.arg;
      const key = `${subscription.exchange}:${subscription.symbol}`;

      // Add subscription if not already present
      const existingIndex = state.subscriptions.findIndex(
        (sub) => sub.exchange === subscription.exchange && sub.symbol === subscription.symbol
      );

      if (existingIndex === -1) {
        state.subscriptions.push(subscription);
      } else {
        // Merge channels
        state.subscriptions[existingIndex].channels = Array.from(
          new Set([...state.subscriptions[existingIndex].channels, ...subscription.channels])
        );
      }

      state.loading[key] = false;
      state.connectionStatus[subscription.exchange] = 'connected';
    });
    builder.addCase(subscribeMarketData.rejected, (state, action) => {
      const key = `${action.meta.arg.exchange}:${action.meta.arg.symbol}`;
      state.loading[key] = false;
      state.errors[key] = action.payload as string;
      state.connectionStatus[action.meta.arg.exchange] = 'error';
    });

    // Unsubscribe from market data
    builder.addCase(unsubscribeMarketData.fulfilled, (state, action) => {
      const subscription = action.payload;
      state.subscriptions = state.subscriptions.filter(
        (sub) => !(sub.exchange === subscription.exchange && sub.symbol === subscription.symbol)
      );

      // Clean up data
      const key = `${subscription.exchange}:${subscription.symbol}`;
      delete state.orderbooks[key];
      delete state.tickers[key];
      delete state.trades[key];
      delete state.loading[key];
      delete state.errors[key];
      delete state.lastUpdated[key];
    });

    // Fetch OHLCV data
    builder.addCase(fetchOHLCV.pending, (state, action) => {
      const key = `${action.meta.arg.exchange}:${action.meta.arg.symbol}:${action.meta.arg.timeframe}`;
      state.loading[key] = true;
      state.errors[key] = null;
    });
    builder.addCase(fetchOHLCV.fulfilled, (state, action) => {
      const key = action.payload.key;
      state.ohlcv[key] = action.payload.data;
      state.loading[key] = false;
      state.lastUpdated[key] = new Date().toISOString();
    });
    builder.addCase(fetchOHLCV.rejected, (state, action) => {
      const key = `${action.meta.arg.exchange}:${action.meta.arg.symbol}:${action.meta.arg.timeframe}`;
      state.loading[key] = false;
      state.errors[key] = action.payload as string;
    });
  },
});

/**
 * Export actions
 */
export const {
  updateOrderBook,
  updateTicker,
  addTrade,
  updateConnectionStatus,
  setError,
  clearError,
  clearMarketData,
  removeMarketData,
} = marketDataSlice.actions;

/**
 * Selectors
 */
export const selectOrderBook = (state: RootState, exchange: string, symbol: string) =>
  state.marketData.orderbooks[`${exchange}:${symbol}`];

export const selectTicker = (state: RootState, exchange: string, symbol: string) =>
  state.marketData.tickers[`${exchange}:${symbol}`];

export const selectTrades = (state: RootState, exchange: string, symbol: string) =>
  state.marketData.trades[`${exchange}:${symbol}`] || [];

export const selectOHLCV = (state: RootState, exchange: string, symbol: string, timeframe: string) =>
  state.marketData.ohlcv[`${exchange}:${symbol}:${timeframe}`] || [];

export const selectConnectionStatus = (state: RootState, exchange: string) =>
  state.marketData.connectionStatus[exchange] || 'disconnected';

export const selectSubscriptions = (state: RootState) => state.marketData.subscriptions;

export const selectMarketDataError = (state: RootState, key: string) =>
  state.marketData.errors[key];

export const selectIsLoading = (state: RootState, key: string) =>
  state.marketData.loading[key] || false;

/**
 * Export reducer
 */
export default marketDataSlice.reducer;
