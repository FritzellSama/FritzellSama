/**
 * Store Index
 *
 * Central export point for all Zustand stores used in the application.
 * This provides a single import location for all state management needs.
 */

export { useAuthStore, selectUser, selectIsAuthenticated, selectIsLoading, selectError, selectIsInitialized } from './authSlice';

export type { default as AuthStore } from './authSlice';
