/**
 * Authentication service for dashboard
 *
 * Production-ready auth service with:
 * - JWT token management
 * - Secure token storage
 * - Automatic token refresh
 * - Request interceptors
 * - Session management
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

interface AuthConfig {
  apiBaseUrl: string;
  tokenStorageKey: string;
  refreshTokenStorageKey: string;
  tokenExpiryKey: string;
  refreshThresholdSeconds: number;
}

interface LoginCredentials {
  username: string;
  password: string;
}

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  expires_in: number;
  token_type: string;
}

interface UserProfile {
  id: string;
  username: string;
  email: string;
  role: string;
  permissions: string[];
}

export class AuthenticationError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'AuthenticationError';
  }
}

class AuthService {
  private config: AuthConfig;
  private apiClient: AxiosInstance;
  private refreshPromise: Promise<string> | null = null;

  constructor(config: AuthConfig) {
    this.config = config;

    // Initialize axios client
    this.apiClient = axios.create({
      baseURL: config.apiBaseUrl,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.apiClient.interceptors.request.use(
      async (config) => {
        const token = await this.getValidToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor to handle 401 errors
    this.apiClient.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest: any = error.config;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newToken = await this.refreshAccessToken();
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.apiClient(originalRequest);
          } catch (refreshError) {
            await this.logout();
            throw new AuthenticationError('Session expired, please login again');
          }
        }

        return Promise.reject(error);
      }
    );
  }

  /**
   * Login with credentials
   */
  async login(credentials: LoginCredentials): Promise<UserProfile> {
    try {
      const response = await this.apiClient.post<TokenResponse>(
        '/auth/login',
        credentials
      );

      const { access_token, refresh_token, expires_in } = response.data;

      // Store tokens securely
      await this.storeTokens(access_token, refresh_token, expires_in);

      // Fetch user profile
      const profile = await this.fetchUserProfile();

      console.info('Login successful', { username: credentials.username });
      return profile;

    } catch (error) {
      console.error('Login failed', { error });

      if (axios.isAxiosError(error)) {
        const statusCode = error.response?.status;
        const details = error.response?.data;

        if (statusCode === 401) {
          throw new AuthenticationError('Invalid credentials', statusCode, details);
        } else if (statusCode === 429) {
          throw new AuthenticationError('Too many login attempts', statusCode, details);
        }
      }

      throw new AuthenticationError('Login failed', undefined, error);
    }
  }

  /**
   * Logout and clear session
   */
  async logout(): Promise<void> {
    try {
      const refreshToken = this.getRefreshToken();

      if (refreshToken) {
        // Invalidate refresh token on server
        await this.apiClient.post('/auth/logout', {
          refresh_token: refreshToken,
        });
      }
    } catch (error) {
      console.error('Logout API call failed', { error });
    } finally {
      // Always clear local storage
      this.clearTokens();
      console.info('Logout successful');
    }
  }

  /**
   * Refresh access token using refresh token
   */
  async refreshAccessToken(): Promise<string> {
    // Prevent multiple simultaneous refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      try {
        const refreshToken = this.getRefreshToken();

        if (!refreshToken) {
          throw new AuthenticationError('No refresh token available');
        }

        const response = await axios.post<TokenResponse>(
          `${this.config.apiBaseUrl}/auth/refresh`,
          { refresh_token: refreshToken }
        );

        const { access_token, refresh_token: new_refresh_token, expires_in } = response.data;

        await this.storeTokens(access_token, new_refresh_token, expires_in);

        console.info('Token refreshed successfully');
        return access_token;

      } catch (error) {
        console.error('Token refresh failed', { error });
        throw new AuthenticationError('Failed to refresh token');
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  /**
   * Get valid access token (refresh if needed)
   */
  async getValidToken(): Promise<string | null> {
    const token = this.getAccessToken();

    if (!token) {
      return null;
    }

    // Check if token needs refresh
    if (this.shouldRefreshToken()) {
      try {
        return await this.refreshAccessToken();
      } catch (error) {
        console.error('Token refresh failed', { error });
        return null;
      }
    }

    return token;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = this.getAccessToken();
    const expiry = this.getTokenExpiry();

    if (!token || !expiry) {
      return false;
    }

    // Check if token is expired
    return Date.now() < expiry;
  }

  /**
   * Fetch current user profile
   */
  async fetchUserProfile(): Promise<UserProfile> {
    try {
      const response = await this.apiClient.get<UserProfile>('/auth/profile');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch user profile', { error });
      throw new AuthenticationError('Failed to fetch user profile');
    }
  }

  /**
   * Change user password
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      await this.apiClient.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });

      console.info('Password changed successfully');
    } catch (error) {
      console.error('Password change failed', { error });

      if (axios.isAxiosError(error) && error.response?.status === 400) {
        throw new AuthenticationError('Invalid current password');
      }

      throw new AuthenticationError('Failed to change password');
    }
  }

  /**
   * Store tokens securely
   */
  private async storeTokens(
    accessToken: string,
    refreshToken: string,
    expiresIn: number
  ): Promise<void> {
    const expiryTime = Date.now() + expiresIn * 1000;

    localStorage.setItem(this.config.tokenStorageKey, accessToken);
    localStorage.setItem(this.config.refreshTokenStorageKey, refreshToken);
    localStorage.setItem(this.config.tokenExpiryKey, expiryTime.toString());
  }

  /**
   * Clear all tokens from storage
   */
  private clearTokens(): void {
    localStorage.removeItem(this.config.tokenStorageKey);
    localStorage.removeItem(this.config.refreshTokenStorageKey);
    localStorage.removeItem(this.config.tokenExpiryKey);
  }

  /**
   * Get access token from storage
   */
  private getAccessToken(): string | null {
    return localStorage.getItem(this.config.tokenStorageKey);
  }

  /**
   * Get refresh token from storage
   */
  private getRefreshToken(): string | null {
    return localStorage.getItem(this.config.refreshTokenStorageKey);
  }

  /**
   * Get token expiry timestamp
   */
  private getTokenExpiry(): number | null {
    const expiry = localStorage.getItem(this.config.tokenExpiryKey);
    return expiry ? parseInt(expiry, 10) : null;
  }

  /**
   * Check if token should be refreshed
   */
  private shouldRefreshToken(): boolean {
    const expiry = this.getTokenExpiry();

    if (!expiry) {
      return false;
    }

    const refreshThresholdMs = this.config.refreshThresholdSeconds * 1000;
    return Date.now() > expiry - refreshThresholdMs;
  }

  /**
   * Get API client instance (for making authenticated requests)
   */
  getApiClient(): AxiosInstance {
    return this.apiClient;
  }
}

// Singleton instance (initialized from config)
let authServiceInstance: AuthService | null = null;

/**
 * Initialize auth service with configuration
 */
export function initializeAuthService(config: AuthConfig): AuthService {
  if (!authServiceInstance) {
    authServiceInstance = new AuthService(config);
  }
  return authServiceInstance;
}

/**
 * Get auth service instance
 */
export function getAuthService(): AuthService {
  if (!authServiceInstance) {
    throw new Error('AuthService not initialized. Call initializeAuthService first.');
  }
  return authServiceInstance;
}

export type { AuthConfig, LoginCredentials, TokenResponse, UserProfile };
