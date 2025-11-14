/**
 * Authentication Service
 *
 * Handles all authentication-related operations including login, logout,
 * token management, and session persistence.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type { User, AuthToken, LoginCredentials } from '@/types';
import { AUTH_CONFIG, API_CONFIG } from '@/utils/constants';

// API Response Types
interface LoginResponse {
  success: boolean;
  data: {
    accessToken: string;
    tokenType: string;
    expiresIn: number;
    refreshToken?: string;
    user: User;
  };
}

interface RefreshTokenResponse {
  success: boolean;
  data: {
    accessToken: string;
    tokenType: string;
    expiresIn: number;
  };
}

interface LogoutResponse {
  success: boolean;
  message: string;
}

interface UserProfileResponse {
  success: boolean;
  data: User;
}

// Token Storage
class TokenStorage {
  setAccessToken(token: string): void {
    localStorage.setItem(AUTH_CONFIG.TOKEN_KEY, token);
  }

  getAccessToken(): string | null {
    return localStorage.getItem(AUTH_CONFIG.TOKEN_KEY);
  }

  setRefreshToken(token: string): void {
    localStorage.setItem(AUTH_CONFIG.REFRESH_TOKEN_KEY, token);
  }

  getRefreshToken(): string | null {
    return localStorage.setItem(AUTH_CONFIG.REFRESH_TOKEN_KEY);
  }

  setUser(user: User): void {
    localStorage.setItem(AUTH_CONFIG.USER_KEY, JSON.stringify(user));
  }

  getUser(): User | null {
    const userStr = localStorage.getItem(AUTH_CONFIG.USER_KEY);
    if (!userStr) return null;

    try {
      return JSON.parse(userStr);
    } catch (error) {
      console.error('Failed to parse user data:', error);
      return null;
    }
  }

  setTokenExpiry(expiresIn: number): void {
    const expiryTime = Date.now() + expiresIn * 1000;
    localStorage.setItem('token_expiry', expiryTime.toString());
  }

  getTokenExpiry(): number | null {
    const expiry = localStorage.getItem('token_expiry');
    return expiry ? parseInt(expiry, 10) : null;
  }

  isTokenExpired(): boolean {
    const expiry = this.getTokenExpiry();
    if (!expiry) return true;

    return Date.now() >= expiry - AUTH_CONFIG.TOKEN_EXPIRY_BUFFER;
  }

  clear(): void {
    localStorage.removeItem(AUTH_CONFIG.TOKEN_KEY);
    localStorage.removeItem(AUTH_CONFIG.REFRESH_TOKEN_KEY);
    localStorage.removeItem(AUTH_CONFIG.USER_KEY);
    localStorage.removeItem('token_expiry');
  }
}

// Authentication Service
class AuthService {
  private api: AxiosInstance;
  private tokenStorage: TokenStorage;
  private refreshPromise: Promise<string> | null = null;

  constructor() {
    this.tokenStorage = new TokenStorage();

    this.api = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.api.interceptors.request.use(
      (config) => {
        const token = this.tokenStorage.getAccessToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    this.api.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as any;

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            const newToken = await this.refreshAccessToken();
            originalRequest.headers.Authorization = `Bearer ${newToken}`;
            return this.api(originalRequest);
          } catch (refreshError) {
            await this.logout();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          }
        }

        return Promise.reject(error);
      }
    );
  }

  async login(credentials: LoginCredentials): Promise<User> {
    try {
      const response = await this.api.post<LoginResponse>(
        '/api/v1/auth/login',
        credentials
      );

      if (!response.data.success) {
        throw new Error('Login failed');
      }

      const { accessToken, tokenType, expiresIn, refreshToken, user } =
        response.data.data;

      this.tokenStorage.setAccessToken(accessToken);
      this.tokenStorage.setTokenExpiry(expiresIn);
      this.tokenStorage.setUser(user);

      if (refreshToken) {
        this.tokenStorage.setRefreshToken(refreshToken);
      }

      return user;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message ||
          'Invalid username or password';
        throw new Error(message);
      }
      throw error;
    }
  }

  async logout(): Promise<void> {
    try {
      await this.api.post<LogoutResponse>('/api/v1/auth/logout');
    } catch (error) {
      console.error('Logout API call failed:', error);
    } finally {
      this.tokenStorage.clear();
      this.refreshPromise = null;
    }
  }

  async refreshAccessToken(): Promise<string> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      try {
        const refreshToken = this.tokenStorage.getRefreshToken();
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }

        const response = await axios.post<RefreshTokenResponse>(
          `${API_CONFIG.BASE_URL}/api/v1/auth/refresh`,
          { refreshToken },
          { timeout: API_CONFIG.TIMEOUT }
        );

        if (!response.data.success) {
          throw new Error('Token refresh failed');
        }

        const { accessToken, expiresIn } = response.data.data;

        this.tokenStorage.setAccessToken(accessToken);
        this.tokenStorage.setTokenExpiry(expiresIn);

        return accessToken;
      } catch (error) {
        this.tokenStorage.clear();
        throw error;
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  async getCurrentUser(): Promise<User | null> {
    const cachedUser = this.tokenStorage.getUser();
    if (cachedUser && !this.tokenStorage.isTokenExpired()) {
      return cachedUser;
    }

    if (!this.isAuthenticated()) {
      return null;
    }

    try {
      const response = await this.api.get<UserProfileResponse>(
        '/api/v1/auth/me'
      );

      if (!response.data.success) {
        throw new Error('Failed to fetch user profile');
      }

      const user = response.data.data;
      this.tokenStorage.setUser(user);

      return user;
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
      this.tokenStorage.clear();
      return null;
    }
  }

  async updateUserProfile(updates: Partial<User>): Promise<User> {
    try {
      const response = await this.api.patch<UserProfileResponse>(
        '/api/v1/auth/me',
        updates
      );

      if (!response.data.success) {
        throw new Error('Failed to update profile');
      }

      const user = response.data.data;
      this.tokenStorage.setUser(user);

      return user;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message || 'Failed to update profile';
        throw new Error(message);
      }
      throw error;
    }
  }

  async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<void> {
    try {
      const response = await this.api.post('/api/v1/auth/change-password', {
        currentPassword,
        newPassword,
      });

      if (!response.data.success) {
        throw new Error('Failed to change password');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message ||
          'Failed to change password';
        throw new Error(message);
      }
      throw error;
    }
  }

  async enable2FA(): Promise<{ qrCode: string; secret: string }> {
    try {
      const response = await this.api.post('/api/v1/auth/2fa/enable');

      if (!response.data.success) {
        throw new Error('Failed to enable 2FA');
      }

      return response.data.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message || 'Failed to enable 2FA';
        throw new Error(message);
      }
      throw error;
    }
  }

  async verify2FA(code: string): Promise<void> {
    try {
      const response = await this.api.post('/api/v1/auth/2fa/verify', {
        code,
      });

      if (!response.data.success) {
        throw new Error('Invalid 2FA code');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message || 'Invalid 2FA code';
        throw new Error(message);
      }
      throw error;
    }
  }

  async disable2FA(code: string): Promise<void> {
    try {
      const response = await this.api.post('/api/v1/auth/2fa/disable', {
        code,
      });

      if (!response.data.success) {
        throw new Error('Failed to disable 2FA');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message || 'Failed to disable 2FA';
        throw new Error(message);
      }
      throw error;
    }
  }

  async requestPasswordReset(email: string): Promise<void> {
    try {
      const response = await axios.post(
        `${API_CONFIG.BASE_URL}/api/v1/auth/password-reset/request`,
        { email },
        { timeout: API_CONFIG.TIMEOUT }
      );

      if (!response.data.success) {
        throw new Error('Failed to request password reset');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message ||
          'Failed to request password reset';
        throw new Error(message);
      }
      throw error;
    }
  }

  async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      const response = await axios.post(
        `${API_CONFIG.BASE_URL}/api/v1/auth/password-reset/confirm`,
        { token, newPassword },
        { timeout: API_CONFIG.TIMEOUT }
      );

      if (!response.data.success) {
        throw new Error('Failed to reset password');
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const message =
          error.response?.data?.error?.message || 'Failed to reset password';
        throw new Error(message);
      }
      throw error;
    }
  }

  isAuthenticated(): boolean {
    const token = this.tokenStorage.getAccessToken();
    return !!token && !this.tokenStorage.isTokenExpired();
  }

  getAccessToken(): string | null {
    return this.tokenStorage.getAccessToken();
  }

  getCachedUser(): User | null {
    return this.tokenStorage.getUser();
  }
}

const authService = new AuthService();

export { authService, TokenStorage };
export default authService;
