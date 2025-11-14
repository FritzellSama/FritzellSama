/**
 * Authentication Store Slice (Zustand)
 *
 * Manages authentication state including user data, login status,
 * and authentication actions.
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type { User, LoginCredentials } from '@/types';
import authService from '@/services/auth';

// State Interface
interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  isInitialized: boolean;
}

// Actions Interface
interface AuthActions {
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  updateUser: (updates: Partial<User>) => Promise<void>;
  clearError: () => void;
  setError: (error: string) => void;
  initialize: () => Promise<void>;
}

// Combined Store Type
type AuthStore = AuthState & AuthActions;

// Initial State
const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  isInitialized: false,
};

// Create Auth Store
export const useAuthStore = create<AuthStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        login: async (credentials: LoginCredentials) => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });

          try {
            const user = await authService.login(credentials);

            set((state) => {
              state.user = user;
              state.isAuthenticated = true;
              state.isLoading = false;
              state.error = null;
            });
          } catch (error) {
            const message =
              error instanceof Error ? error.message : 'Login failed';

            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.isLoading = false;
              state.error = message;
            });

            throw error;
          }
        },

        logout: async () => {
          set((state) => {
            state.isLoading = true;
          });

          try {
            await authService.logout();
          } catch (error) {
            console.error('Logout error:', error);
          } finally {
            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.isLoading = false;
              state.error = null;
            });
          }
        },

        refreshUser: async () => {
          if (!authService.isAuthenticated()) {
            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
            });
            return;
          }

          set((state) => {
            state.isLoading = true;
          });

          try {
            const user = await authService.getCurrentUser();

            if (user) {
              set((state) => {
                state.user = user;
                state.isAuthenticated = true;
                state.isLoading = false;
                state.error = null;
              });
            } else {
              set((state) => {
                state.user = null;
                state.isAuthenticated = false;
                state.isLoading = false;
              });
            }
          } catch (error) {
            console.error('Failed to refresh user:', error);

            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.isLoading = false;
              state.error =
                error instanceof Error
                  ? error.message
                  : 'Failed to refresh user';
            });
          }
        },

        updateUser: async (updates: Partial<User>) => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });

          try {
            const updatedUser = await authService.updateUserProfile(updates);

            set((state) => {
              state.user = updatedUser;
              state.isLoading = false;
            });
          } catch (error) {
            const message =
              error instanceof Error
                ? error.message
                : 'Failed to update profile';

            set((state) => {
              state.isLoading = false;
              state.error = message;
            });

            throw error;
          }
        },

        clearError: () => {
          set((state) => {
            state.error = null;
          });
        },

        setError: (error: string) => {
          set((state) => {
            state.error = error;
          });
        },

        initialize: async () => {
          if (get().isInitialized) {
            return;
          }

          set((state) => {
            state.isLoading = true;
          });

          try {
            if (authService.isAuthenticated()) {
              const user = await authService.getCurrentUser();

              if (user) {
                set((state) => {
                  state.user = user;
                  state.isAuthenticated = true;
                  state.isInitialized = true;
                  state.isLoading = false;
                });
                return;
              }
            }

            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.isInitialized = true;
              state.isLoading = false;
            });
          } catch (error) {
            console.error('Failed to initialize auth:', error);

            set((state) => {
              state.user = null;
              state.isAuthenticated = false;
              state.isInitialized = true;
              state.isLoading = false;
            });
          }
        },
      })),
      {
        name: 'auth-storage',
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    {
      name: 'AuthStore',
      enabled: import.meta.env.MODE === 'development',
    }
  )
);

// Selectors
export const selectUser = (state: AuthStore) => state.user;
export const selectIsAuthenticated = (state: AuthStore) => state.isAuthenticated;
export const selectIsLoading = (state: AuthStore) => state.isLoading;
export const selectError = (state: AuthStore) => state.error;
export const selectIsInitialized = (state: AuthStore) => state.isInitialized;

// Export default for convenience
export default useAuthStore;
