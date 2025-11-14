/**
 * Trading Redux Slice
 *
 * Manages trading state including orders, positions, and execution.
 * Handles order placement, modification, cancellation, and position tracking.
 *
 * @module tradingSlice
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from './index';
import type {
  Order,
  OrderCreateRequest,
  Position,
  ExecutionResult,
  PortfolioSummary,
  TradeHistory,
  OrderStatus,
} from '../types/trading.types';

/**
 * Trading state interface
 */
export interface TradingState {
  // Orders
  activeOrders: Record<string, Order>;
  orderHistory: TradeHistory[];
  pendingOrders: string[]; // Order IDs being processed

  // Positions
  positions: Record<string, Position>;
  closedPositions: Position[];

  // Portfolio
  portfolio: PortfolioSummary | null;

  // Execution results
  lastExecution: ExecutionResult | null;

  // UI state
  selectedSymbol: string | null;
  selectedExchange: string | null;
  orderFormVisible: boolean;

  // Loading states
  loading: {
    orders: boolean;
    positions: boolean;
    portfolio: boolean;
    execution: boolean;
  };

  // Errors
  errors: {
    orders: string | null;
    positions: string | null;
    portfolio: string | null;
    execution: string | null;
  };

  // Last updated timestamps
  lastUpdated: {
    orders: string | null;
    positions: string | null;
    portfolio: string | null;
  };
}

/**
 * Initial state
 */
const initialState: TradingState = {
  activeOrders: {},
  orderHistory: [],
  pendingOrders: [],
  positions: {},
  closedPositions: [],
  portfolio: null,
  lastExecution: null,
  selectedSymbol: null,
  selectedExchange: null,
  orderFormVisible: false,
  loading: {
    orders: false,
    positions: false,
    portfolio: false,
    execution: false,
  },
  errors: {
    orders: null,
    positions: null,
    portfolio: null,
    execution: null,
  },
  lastUpdated: {
    orders: null,
    positions: null,
    portfolio: null,
  },
};

/**
 * Helper function to get API URL
 */
const getApiUrl = (): string => {
  const url = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
  if (!url) {
    throw new Error('API_URL not configured');
  }
  return url;
};

/**
 * Helper function to get auth token
 */
const getAuthToken = (): string => {
  const token = localStorage.getItem('auth_token');
  if (!token) {
    throw new Error('Not authenticated');
  }
  return token;
};

/**
 * Async thunks
 */

// Fetch active orders
export const fetchActiveOrders = createAsyncThunk(
  'trading/fetchActiveOrders',
  async (_, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/orders/active`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch orders');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Fetch order history
export const fetchOrderHistory = createAsyncThunk(
  'trading/fetchOrderHistory',
  async (params: { limit?: number; offset?: number }, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const queryParams = new URLSearchParams({
        ...(params.limit && { limit: params.limit.toString() }),
        ...(params.offset && { offset: params.offset.toString() }),
      });

      const response = await fetch(`${apiUrl}/api/v1/trading/orders/history?${queryParams}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch order history');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Create and submit order
export const createOrder = createAsyncThunk(
  'trading/createOrder',
  async (orderRequest: OrderCreateRequest, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/orders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(orderRequest),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create order');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Cancel order
export const cancelOrder = createAsyncThunk(
  'trading/cancelOrder',
  async (orderId: string, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/orders/${orderId}/cancel`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to cancel order');
      }

      return { orderId, result: await response.json() };
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Fetch positions
export const fetchPositions = createAsyncThunk(
  'trading/fetchPositions',
  async (_, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/positions`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch positions');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Close position
export const closePosition = createAsyncThunk(
  'trading/closePosition',
  async (positionId: string, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/positions/${positionId}/close`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to close position');
      }

      return { positionId, result: await response.json() };
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Fetch portfolio summary
export const fetchPortfolio = createAsyncThunk(
  'trading/fetchPortfolio',
  async (_, { rejectWithValue }) => {
    try {
      const apiUrl = getApiUrl();
      const token = getAuthToken();

      const response = await fetch(`${apiUrl}/api/v1/trading/portfolio`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to fetch portfolio');
      }

      return await response.json();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

/**
 * Trading slice
 */
const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    // Update order from WebSocket
    updateOrder: (state, action: PayloadAction<Order>) => {
      const order = action.payload;
      if (order.order_id) {
        state.activeOrders[order.order_id] = order;
      }
    },

    // Update position from WebSocket
    updatePosition: (state, action: PayloadAction<Position>) => {
      const position = action.payload;
      state.positions[position.position_id] = position;
      state.lastUpdated.positions = new Date().toISOString();
    },

    // Remove order (when filled or cancelled)
    removeOrder: (state, action: PayloadAction<string>) => {
      delete state.activeOrders[action.payload];
      state.pendingOrders = state.pendingOrders.filter((id) => id !== action.payload);
    },

    // Remove position (when closed)
    removePosition: (state, action: PayloadAction<string>) => {
      const position = state.positions[action.payload];
      if (position) {
        state.closedPositions.unshift(position);
        delete state.positions[action.payload];
      }
    },

    // Set selected symbol
    setSelectedSymbol: (state, action: PayloadAction<string>) => {
      state.selectedSymbol = action.payload;
    },

    // Set selected exchange
    setSelectedExchange: (state, action: PayloadAction<string>) => {
      state.selectedExchange = action.payload;
    },

    // Toggle order form visibility
    toggleOrderForm: (state) => {
      state.orderFormVisible = !state.orderFormVisible;
    },

    // Set order form visibility
    setOrderFormVisible: (state, action: PayloadAction<boolean>) => {
      state.orderFormVisible = action.payload;
    },

    // Clear execution result
    clearLastExecution: (state) => {
      state.lastExecution = null;
    },

    // Clear errors
    clearErrors: (state) => {
      state.errors = {
        orders: null,
        positions: null,
        portfolio: null,
        execution: null,
      };
    },

    // Clear specific error
    clearError: (state, action: PayloadAction<keyof TradingState['errors']>) => {
      state.errors[action.payload] = null;
    },
  },
  extraReducers: (builder) => {
    // Fetch active orders
    builder.addCase(fetchActiveOrders.pending, (state) => {
      state.loading.orders = true;
      state.errors.orders = null;
    });
    builder.addCase(fetchActiveOrders.fulfilled, (state, action) => {
      state.loading.orders = false;
      state.activeOrders = {};

      // Convert array to record
      if (Array.isArray(action.payload)) {
        action.payload.forEach((order: Order) => {
          if (order.order_id) {
            state.activeOrders[order.order_id] = order;
          }
        });
      }

      state.lastUpdated.orders = new Date().toISOString();
    });
    builder.addCase(fetchActiveOrders.rejected, (state, action) => {
      state.loading.orders = false;
      state.errors.orders = action.payload as string;
    });

    // Fetch order history
    builder.addCase(fetchOrderHistory.pending, (state) => {
      state.loading.orders = true;
      state.errors.orders = null;
    });
    builder.addCase(fetchOrderHistory.fulfilled, (state, action) => {
      state.loading.orders = false;
      state.orderHistory = action.payload;
    });
    builder.addCase(fetchOrderHistory.rejected, (state, action) => {
      state.loading.orders = false;
      state.errors.orders = action.payload as string;
    });

    // Create order
    builder.addCase(createOrder.pending, (state) => {
      state.loading.execution = true;
      state.errors.execution = null;
    });
    builder.addCase(createOrder.fulfilled, (state, action) => {
      state.loading.execution = false;
      state.lastExecution = action.payload;

      // Add to active orders if order ID is present
      if (action.payload.order_id) {
        state.pendingOrders.push(action.payload.order_id);
      }
    });
    builder.addCase(createOrder.rejected, (state, action) => {
      state.loading.execution = false;
      state.errors.execution = action.payload as string;
    });

    // Cancel order
    builder.addCase(cancelOrder.pending, (state, action) => {
      state.pendingOrders.push(action.meta.arg);
    });
    builder.addCase(cancelOrder.fulfilled, (state, action) => {
      const { orderId } = action.payload;
      delete state.activeOrders[orderId];
      state.pendingOrders = state.pendingOrders.filter((id) => id !== orderId);
    });
    builder.addCase(cancelOrder.rejected, (state, action) => {
      const orderId = action.meta.arg;
      state.pendingOrders = state.pendingOrders.filter((id) => id !== orderId);
      state.errors.orders = action.payload as string;
    });

    // Fetch positions
    builder.addCase(fetchPositions.pending, (state) => {
      state.loading.positions = true;
      state.errors.positions = null;
    });
    builder.addCase(fetchPositions.fulfilled, (state, action) => {
      state.loading.positions = false;
      state.positions = {};

      // Convert array to record
      if (Array.isArray(action.payload)) {
        action.payload.forEach((position: Position) => {
          state.positions[position.position_id] = position;
        });
      }

      state.lastUpdated.positions = new Date().toISOString();
    });
    builder.addCase(fetchPositions.rejected, (state, action) => {
      state.loading.positions = false;
      state.errors.positions = action.payload as string;
    });

    // Close position
    builder.addCase(closePosition.fulfilled, (state, action) => {
      const { positionId } = action.payload;
      const position = state.positions[positionId];

      if (position) {
        state.closedPositions.unshift(position);
        delete state.positions[positionId];
      }
    });
    builder.addCase(closePosition.rejected, (state, action) => {
      state.errors.positions = action.payload as string;
    });

    // Fetch portfolio
    builder.addCase(fetchPortfolio.pending, (state) => {
      state.loading.portfolio = true;
      state.errors.portfolio = null;
    });
    builder.addCase(fetchPortfolio.fulfilled, (state, action) => {
      state.loading.portfolio = false;
      state.portfolio = action.payload;
      state.lastUpdated.portfolio = new Date().toISOString();
    });
    builder.addCase(fetchPortfolio.rejected, (state, action) => {
      state.loading.portfolio = false;
      state.errors.portfolio = action.payload as string;
    });
  },
});

/**
 * Export actions
 */
export const {
  updateOrder,
  updatePosition,
  removeOrder,
  removePosition,
  setSelectedSymbol,
  setSelectedExchange,
  toggleOrderForm,
  setOrderFormVisible,
  clearLastExecution,
  clearErrors,
  clearError,
} = tradingSlice.actions;

/**
 * Selectors
 */
export const selectActiveOrders = (state: RootState) => Object.values(state.trading.activeOrders);
export const selectOrderHistory = (state: RootState) => state.trading.orderHistory;
export const selectPositions = (state: RootState) => Object.values(state.trading.positions);
export const selectClosedPositions = (state: RootState) => state.trading.closedPositions;
export const selectPortfolio = (state: RootState) => state.trading.portfolio;
export const selectLastExecution = (state: RootState) => state.trading.lastExecution;
export const selectSelectedSymbol = (state: RootState) => state.trading.selectedSymbol;
export const selectSelectedExchange = (state: RootState) => state.trading.selectedExchange;
export const selectOrderFormVisible = (state: RootState) => state.trading.orderFormVisible;
export const selectTradingLoading = (state: RootState) => state.trading.loading;
export const selectTradingErrors = (state: RootState) => state.trading.errors;
export const selectPendingOrders = (state: RootState) => state.trading.pendingOrders;

// Derived selectors
export const selectOrderById = (state: RootState, orderId: string) =>
  state.trading.activeOrders[orderId];

export const selectPositionById = (state: RootState, positionId: string) =>
  state.trading.positions[positionId];

export const selectOrdersBySymbol = (state: RootState, symbol: string) =>
  Object.values(state.trading.activeOrders).filter((order) => order.symbol === symbol);

export const selectPositionsBySymbol = (state: RootState, symbol: string) =>
  Object.values(state.trading.positions).filter((position) => position.symbol === symbol);

export const selectTotalPnL = (state: RootState) =>
  state.trading.portfolio?.total_pnl || '0';

export const selectPositionsCount = (state: RootState) =>
  Object.keys(state.trading.positions).length;

export const selectActiveOrdersCount = (state: RootState) =>
  Object.keys(state.trading.activeOrders).length;

/**
 * Export reducer
 */
export default tradingSlice.reducer;
