/**
 * Redux store configuration
 *
 * Central state management with:
 * - Redux Toolkit
 * - Type-safe reducers
 * - Middleware configuration
 * - DevTools integration
 * - Persistence
 */

import { configureStore, combineReducers } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import authReducer from './authSlice';

/**
 * Root reducer combining all slices
 */
const rootReducer = combineReducers({
  auth: authReducer,
  // Add other reducers here as they are created:
  // positions: positionsReducer,
  // orders: ordersReducer,
  // market: marketReducer,
  // signals: signalsReducer,
  // portfolio: portfolioReducer,
  // settings: settingsReducer,
});

/**
 * Custom middleware for logging in development
 */
const loggerMiddleware = (storeAPI: any) => (next: any) => (action: any) => {
  if (process.env.NODE_ENV === 'development') {
    console.group(action.type);
    console.info('dispatching', action);
    const result = next(action);
    console.log('next state', storeAPI.getState());
    console.groupEnd();
    return result;
  }
  return next(action);
};

/**
 * Error handling middleware
 */
const errorHandlerMiddleware = (storeAPI: any) => (next: any) => (action: any) => {
  try {
    return next(action);
  } catch (error) {
    console.error('Redux error:', error);
    // Could dispatch an error action here
    throw error;
  }
};

/**
 * Configure Redux store
 */
export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serialization check
        ignoredActions: ['auth/login/fulfilled', 'auth/fetchProfile/fulfilled'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['meta.arg', 'payload.timestamp'],
        // Ignore these paths in the state
        ignoredPaths: ['items.dates'],
      },
      // Immutability check for development
      immutableCheck: process.env.NODE_ENV === 'development',
    })
      .concat(errorHandlerMiddleware)
      .concat(process.env.NODE_ENV === 'development' ? loggerMiddleware : []),
  devTools: process.env.NODE_ENV === 'development',
});

/**
 * Type definitions for TypeScript
 */
export type RootState = ReturnType<typeof rootReducer>;
export type AppDispatch = typeof store.dispatch;

/**
 * Typed hooks for use throughout the app
 */
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

/**
 * Store subscription for middleware/plugins
 */
export function subscribeToStore(listener: () => void): () => void {
  return store.subscribe(listener);
}

/**
 * Get current state (for use outside React components)
 */
export function getStoreState(): RootState {
  return store.getState();
}

/**
 * Dispatch action (for use outside React components)
 */
export function dispatchAction(action: any): void {
  store.dispatch(action);
}

/**
 * Local storage persistence helpers
 */
const STORAGE_KEY = 'quantum_trader_state';

/**
 * Load persisted state from local storage
 */
export function loadPersistedState(): Partial<RootState> | undefined {
  try {
    const serializedState = localStorage.getItem(STORAGE_KEY);
    if (serializedState === null) {
      return undefined;
    }
    return JSON.parse(serializedState);
  } catch (error) {
    console.error('Failed to load persisted state:', error);
    return undefined;
  }
}

/**
 * Save state to local storage
 */
export function savePersistedState(state: RootState): void {
  try {
    // Only persist specific slices
    const stateToPersist = {
      // Don't persist auth state (handled by auth service)
      // Don't persist real-time data
      // settings: state.settings, // Example: persist user settings
    };

    const serializedState = JSON.stringify(stateToPersist);
    localStorage.setItem(STORAGE_KEY, serializedState);
  } catch (error) {
    console.error('Failed to save persisted state:', error);
  }
}

/**
 * Subscribe to store changes and persist state
 */
if (typeof window !== 'undefined') {
  let timeoutId: NodeJS.Timeout | null = null;

  store.subscribe(() => {
    // Debounce persistence to avoid excessive writes
    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    timeoutId = setTimeout(() => {
      savePersistedState(store.getState());
    }, 1000); // Save after 1 second of inactivity
  });
}

/**
 * Reset store to initial state
 */
export function resetStore(): void {
  // Dispatch reset actions for each slice
  // This is useful for logout or testing
  localStorage.removeItem(STORAGE_KEY);

  // Force reload to reset state
  if (typeof window !== 'undefined') {
    window.location.reload();
  }
}

/**
 * Development-only store inspector
 */
if (process.env.NODE_ENV === 'development') {
  (window as any).__REDUX_STORE__ = store;
  console.info('Redux store available at window.__REDUX_STORE__');
}

export default store;
