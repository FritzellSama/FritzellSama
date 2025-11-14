/**
 * Authentication Hook
 *
 * Custom React hook for managing authentication state and operations.
 * Handles login, logout, token management, and user session.
 *
 * @module useAuth
 */

import { useState, useCallback, useEffect, createContext, useContext, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApi } from './useApi';
import { useNotification } from './useNotification';
import { storage } from '../services/storage';

/**
 * User interface
 */
export interface User {
  id: string;
  username: string;
  email?: string;
  roles?: string[];
  permissions?: string[];
}

/**
 * Login credentials
 */
export interface LoginCredentials {
  username: string;
  password: string;
}

/**
 * Login response
 */
interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user?: User;
}

/**
 * Auth context value
 */
export interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginCredentials) => Promise<boolean>;
  logout: () => Promise<void>;
  refreshAuth: () => Promise<void>;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: string) => boolean;
}

/**
 * Auth context
 */
const AuthContext = createContext<AuthContextValue | null>(null);

/**
 * Storage keys
 */
const STORAGE_KEYS = {
  TOKEN: 'auth_token',
  USER: 'auth_user',
  EXPIRES_AT: 'auth_expires_at',
} as const;

/**
 * Get token expiry time
 */
const getTokenExpiryTime = (expiresIn: number): number => {
  return Date.now() + expiresIn * 1000;
};

/**
 * Check if token is expired
 */
const isTokenExpired = (): boolean => {
  try {
    const expiresAt = localStorage.getItem(STORAGE_KEYS.EXPIRES_AT);
    if (!expiresAt) return true;

    return Date.now() >= parseInt(expiresAt, 10);
  } catch {
    return true;
  }
};

/**
 * Auth Provider Component
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const navigate = useNavigate();
  const { execute } = useApi();
  const { showSuccess, showError } = useNotification();

  /**
   * Initialize auth state from storage
   */
  useEffect(() => {
    const initAuth = async () => {
      try {
        const token = localStorage.getItem(STORAGE_KEYS.TOKEN);

        if (!token || isTokenExpired()) {
          // Clear expired session
          await clearAuth();
          setIsLoading(false);
          return;
        }

        // Load user from storage
        const storedUser = await storage.getItem<User>(STORAGE_KEYS.USER);
        if (storedUser) {
          setUser(storedUser);
        } else {
          // Verify token with backend
          await refreshAuth();
        }
      } catch (err) {
        console.error('Auth initialization failed:', err);
        await clearAuth();
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Setup token refresh timer
   */
  useEffect(() => {
    if (!user) return;

    const expiresAt = localStorage.getItem(STORAGE_KEYS.EXPIRES_AT);
    if (!expiresAt) return;

    const expiresIn = parseInt(expiresAt, 10) - Date.now();
    const refreshThreshold = parseInt(
      process.env.REACT_APP_TOKEN_REFRESH_THRESHOLD || '300000',
      10
    ); // 5 minutes

    // Refresh token before expiry
    if (expiresIn > refreshThreshold) {
      const timerId = setTimeout(() => {
        refreshAuth();
      }, expiresIn - refreshThreshold);

      return () => clearTimeout(timerId);
    }
  }, [user]);

  /**
   * Clear auth state
   */
  const clearAuth = useCallback(async () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEYS.TOKEN);
    localStorage.removeItem(STORAGE_KEYS.EXPIRES_AT);
    await storage.removeItem(STORAGE_KEYS.USER);
  }, []);

  /**
   * Login user
   */
  const login = useCallback(
    async (credentials: LoginCredentials): Promise<boolean> => {
      setIsLoading(true);
      setError(null);

      try {
        const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
        if (!apiUrl) {
          throw new Error('API URL not configured');
        }

        const response = await execute<LoginResponse>('/api/v1/auth/login', {
          method: 'POST',
          body: credentials,
          skipAuth: true,
        });

        if (!response.success || !response.data) {
          throw new Error(response.error || 'Login failed');
        }

        const { access_token, expires_in, user: userData } = response.data;

        // Store token
        localStorage.setItem(STORAGE_KEYS.TOKEN, access_token);
        localStorage.setItem(
          STORAGE_KEYS.EXPIRES_AT,
          getTokenExpiryTime(expires_in).toString()
        );

        // Store user data
        if (userData) {
          await storage.setItem(STORAGE_KEYS.USER, userData);
          setUser(userData);
        } else {
          // Fetch user data
          await refreshAuth();
        }

        showSuccess('Login successful');
        return true;
      } catch (err) {
        const errorMessage = (err as Error).message || 'Login failed';
        setError(errorMessage);
        showError(errorMessage);
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [execute, showSuccess, showError]
  );

  /**
   * Logout user
   */
  const logout = useCallback(async () => {
    setIsLoading(true);

    try {
      // Call logout endpoint
      await execute('/api/v1/auth/logout', {
        method: 'POST',
        skipErrorNotification: true,
      });
    } catch (err) {
      // Ignore logout errors
      console.error('Logout error:', err);
    } finally {
      await clearAuth();
      setIsLoading(false);
      navigate('/login');
      showSuccess('Logged out successfully');
    }
  }, [execute, navigate, clearAuth, showSuccess]);

  /**
   * Refresh authentication
   */
  const refreshAuth = useCallback(async () => {
    try {
      const token = localStorage.getItem(STORAGE_KEYS.TOKEN);
      if (!token || isTokenExpired()) {
        await clearAuth();
        return;
      }

      const response = await execute<{ user: User }>('/api/v1/auth/verify', {
        method: 'GET',
      });

      if (!response.success || !response.data) {
        throw new Error('Auth verification failed');
      }

      const userData = response.data.user;
      await storage.setItem(STORAGE_KEYS.USER, userData);
      setUser(userData);
    } catch (err) {
      console.error('Auth refresh failed:', err);
      await clearAuth();
    }
  }, [execute, clearAuth]);

  /**
   * Check if user has permission
   */
  const hasPermission = useCallback(
    (permission: string): boolean => {
      if (!user || !user.permissions) return false;
      return user.permissions.includes(permission);
    },
    [user]
  );

  /**
   * Check if user has role
   */
  const hasRole = useCallback(
    (role: string): boolean => {
      if (!user || !user.roles) return false;
      return user.roles.includes(role);
    },
    [user]
  );

  const value: AuthContextValue = {
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    login,
    logout,
    refreshAuth,
    hasPermission,
    hasRole,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

/**
 * Use Auth Hook
 *
 * Provides access to authentication state and operations.
 *
 * @example
 * ```typescript
 * const { user, isAuthenticated, login, logout } = useAuth();
 *
 * // Login
 * await login({ username: 'user@example.com', password: 'password' });
 *
 * // Logout
 * await logout();
 *
 * // Check authentication
 * if (isAuthenticated) {
 *   console.log('User:', user);
 * }
 * ```
 */
export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
}

/**
 * Higher-order component for protected routes
 */
export function withAuth<P extends object>(
  Component: React.ComponentType<P>
): React.ComponentType<P> {
  return function AuthenticatedComponent(props: P) {
    const { isAuthenticated, isLoading } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
      if (!isLoading && !isAuthenticated) {
        navigate('/login', { replace: true });
      }
    }, [isAuthenticated, isLoading, navigate]);

    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            <p className="mt-4 text-gray-400">Loading...</p>
          </div>
        </div>
      );
    }

    if (!isAuthenticated) {
      return null;
    }

    return <Component {...props} />;
  };
}

/**
 * Hook for requiring specific permissions
 */
export function useRequirePermission(permission: string): boolean {
  const { hasPermission, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const hasRequiredPermission = isAuthenticated && hasPermission(permission);

  useEffect(() => {
    if (isAuthenticated && !hasRequiredPermission) {
      // Redirect to unauthorized page or dashboard
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, hasRequiredPermission, navigate]);

  return hasRequiredPermission;
}

/**
 * Hook for requiring specific role
 */
export function useRequireRole(role: string): boolean {
  const { hasRole, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const hasRequiredRole = isAuthenticated && hasRole(role);

  useEffect(() => {
    if (isAuthenticated && !hasRequiredRole) {
      // Redirect to unauthorized page or dashboard
      navigate('/dashboard', { replace: true });
    }
  }, [isAuthenticated, hasRequiredRole, navigate]);

  return hasRequiredRole;
}

export default useAuth;
