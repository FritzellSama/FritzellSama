/**
 * Redux auth slice for state management
 *
 * Manages authentication state with:
 * - Login/logout actions
 * - User profile management
 * - Loading states
 * - Error handling
 * - Type-safe reducers
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { getAuthService, UserProfile, LoginCredentials, AuthenticationError } from '../services/auth';

interface AuthState {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  loginAttempts: number;
  lastLoginAttempt: number | null;
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  loginAttempts: 0,
  lastLoginAttempt: null,
};

/**
 * Async thunk for login
 */
export const loginAsync = createAsyncThunk<
  UserProfile,
  LoginCredentials,
  { rejectValue: string }
>(
  'auth/login',
  async (credentials, { rejectWithValue }) => {
    try {
      const authService = getAuthService();
      const userProfile = await authService.login(credentials);
      return userProfile;
    } catch (error) {
      if (error instanceof AuthenticationError) {
        return rejectWithValue(error.message);
      }
      return rejectWithValue('An unexpected error occurred during login');
    }
  }
);

/**
 * Async thunk for logout
 */
export const logoutAsync = createAsyncThunk<void, void, { rejectValue: string }>(
  'auth/logout',
  async (_, { rejectWithValue }) => {
    try {
      const authService = getAuthService();
      await authService.logout();
    } catch (error) {
      // Even if logout fails, we should clear local state
      console.error('Logout error:', error);
      return rejectWithValue('Logout failed on server, but local session cleared');
    }
  }
);

/**
 * Async thunk for fetching user profile
 */
export const fetchUserProfileAsync = createAsyncThunk<
  UserProfile,
  void,
  { rejectValue: string }
>(
  'auth/fetchProfile',
  async (_, { rejectWithValue }) => {
    try {
      const authService = getAuthService();
      const userProfile = await authService.fetchUserProfile();
      return userProfile;
    } catch (error) {
      if (error instanceof AuthenticationError) {
        return rejectWithValue(error.message);
      }
      return rejectWithValue('Failed to fetch user profile');
    }
  }
);

/**
 * Async thunk for password change
 */
export const changePasswordAsync = createAsyncThunk<
  void,
  { currentPassword: string; newPassword: string },
  { rejectValue: string }
>(
  'auth/changePassword',
  async ({ currentPassword, newPassword }, { rejectWithValue }) => {
    try {
      const authService = getAuthService();
      await authService.changePassword(currentPassword, newPassword);
    } catch (error) {
      if (error instanceof AuthenticationError) {
        return rejectWithValue(error.message);
      }
      return rejectWithValue('Failed to change password');
    }
  }
);

/**
 * Auth slice
 */
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    /**
     * Clear auth error
     */
    clearError: (state) => {
      state.error = null;
    },

    /**
     * Set user (for SSO or pre-authenticated scenarios)
     */
    setUser: (state, action: PayloadAction<UserProfile>) => {
      state.user = action.payload;
      state.isAuthenticated = true;
      state.error = null;
    },

    /**
     * Clear user (for forced logout scenarios)
     */
    clearUser: (state) => {
      state.user = null;
      state.isAuthenticated = false;
    },

    /**
     * Reset login attempts
     */
    resetLoginAttempts: (state) => {
      state.loginAttempts = 0;
      state.lastLoginAttempt = null;
    },
  },
  extraReducers: (builder) => {
    // Login
    builder
      .addCase(loginAsync.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginAsync.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
        state.error = null;
        state.loginAttempts = 0;
        state.lastLoginAttempt = null;
      })
      .addCase(loginAsync.rejected, (state, action) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = action.payload || 'Login failed';
        state.loginAttempts += 1;
        state.lastLoginAttempt = Date.now();
      });

    // Logout
    builder
      .addCase(logoutAsync.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(logoutAsync.fulfilled, (state) => {
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = null;
        state.loginAttempts = 0;
        state.lastLoginAttempt = null;
      })
      .addCase(logoutAsync.rejected, (state, action) => {
        // Even on rejection, clear auth state
        state.isLoading = false;
        state.user = null;
        state.isAuthenticated = false;
        state.error = action.payload || null;
      });

    // Fetch profile
    builder
      .addCase(fetchUserProfileAsync.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchUserProfileAsync.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
        state.isAuthenticated = true;
        state.error = null;
      })
      .addCase(fetchUserProfileAsync.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to fetch profile';
        // Don't clear auth state on profile fetch failure
      });

    // Change password
    builder
      .addCase(changePasswordAsync.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(changePasswordAsync.fulfilled, (state) => {
        state.isLoading = false;
        state.error = null;
      })
      .addCase(changePasswordAsync.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload || 'Failed to change password';
      });
  },
});

// Export actions
export const {
  clearError,
  setUser,
  clearUser,
  resetLoginAttempts,
} = authSlice.actions;

// Selectors
export const selectAuth = (state: { auth: AuthState }) => state.auth;
export const selectUser = (state: { auth: AuthState }) => state.auth.user;
export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated;
export const selectAuthLoading = (state: { auth: AuthState }) => state.auth.isLoading;
export const selectAuthError = (state: { auth: AuthState }) => state.auth.error;
export const selectLoginAttempts = (state: { auth: AuthState }) => state.auth.loginAttempts;

// Check if user has specific permission
export const selectHasPermission = (permission: string) => (state: { auth: AuthState }) => {
  return state.auth.user?.permissions.includes(permission) ?? false;
};

// Check if user has specific role
export const selectHasRole = (role: string) => (state: { auth: AuthState }) => {
  return state.auth.user?.role === role;
};

// Check if rate limited (after 5 failed attempts within 5 minutes)
export const selectIsRateLimited = (state: { auth: AuthState }) => {
  const { loginAttempts, lastLoginAttempt } = state.auth;

  if (loginAttempts < 5) {
    return false;
  }

  if (!lastLoginAttempt) {
    return false;
  }

  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  return lastLoginAttempt > fiveMinutesAgo;
};

export default authSlice.reducer;

// Export types
export type { AuthState };
