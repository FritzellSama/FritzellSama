/**
 * API Hook
 *
 * Custom React hook for making API requests with automatic error handling,
 * retries, loading states, and authentication.
 *
 * @module useApi
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useNotification } from './useNotification';

/**
 * HTTP methods
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/**
 * API request options
 */
export interface ApiRequestOptions {
  method?: HttpMethod;
  body?: any;
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean>;
  skipAuth?: boolean;
  skipErrorNotification?: boolean;
  retries?: number;
  retryDelay?: number;
  timeout?: number;
}

/**
 * API response wrapper
 */
export interface ApiResponse<T = any> {
  data: T | null;
  error: string | null;
  status: number | null;
  success: boolean;
}

/**
 * API hook return type
 */
export interface UseApiReturn<T = any> {
  data: T | null;
  error: string | null;
  loading: boolean;
  status: number | null;
  execute: (endpoint: string, options?: ApiRequestOptions) => Promise<ApiResponse<T>>;
  reset: () => void;
}

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Get API configuration from environment
 */
const getApiConfig = () => {
  return {
    baseUrl: process.env.REACT_APP_API_URL || window.ENV?.API_URL || 'http://localhost:8080',
    timeout: parseInt(process.env.REACT_APP_API_TIMEOUT || '30000', 10),
    retries: parseInt(process.env.REACT_APP_API_RETRY_ATTEMPTS || '3', 10),
    retryDelay: parseInt(process.env.REACT_APP_API_RETRY_DELAY || '1000', 10),
  };
};

/**
 * Get authentication token
 */
const getAuthToken = (): string | null => {
  try {
    return localStorage.getItem('auth_token');
  } catch (error) {
    console.error('Failed to get auth token:', error);
    return null;
  }
};

/**
 * Build query string from params
 */
const buildQueryString = (params: Record<string, string | number | boolean>): string => {
  const searchParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      searchParams.append(key, String(value));
    }
  });

  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
};

/**
 * API Hook
 *
 * Provides a simple interface for making API requests with built-in
 * error handling, loading states, and retry logic.
 *
 * @example
 * ```typescript
 * const { data, loading, error, execute } = useApi<User>();
 *
 * // Make a GET request
 * await execute('/api/users/123');
 *
 * // Make a POST request
 * await execute('/api/users', {
 *   method: 'POST',
 *   body: { name: 'John Doe' },
 * });
 * ```
 */
export function useApi<T = any>(): UseApiReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<number | null>(null);

  const { showError } = useNotification();
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  /**
   * Execute API request
   */
  const execute = useCallback(
    async (endpoint: string, options: ApiRequestOptions = {}): Promise<ApiResponse<T>> => {
      const {
        method = 'GET',
        body,
        headers = {},
        params,
        skipAuth = false,
        skipErrorNotification = false,
        retries: customRetries,
        retryDelay: customRetryDelay,
        timeout: customTimeout,
      } = options;

      const config = getApiConfig();
      const maxRetries = customRetries ?? config.retries;
      const retryDelayMs = customRetryDelay ?? config.retryDelay;
      const timeoutMs = customTimeout ?? config.timeout;

      // Cancel previous request if still pending
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new abort controller
      abortControllerRef.current = new AbortController();

      setLoading(true);
      setError(null);

      let lastError: Error | null = null;
      let attempt = 0;

      while (attempt <= maxRetries) {
        try {
          // Build URL
          const queryString = params ? buildQueryString(params) : '';
          const url = `${config.baseUrl}${endpoint}${queryString}`;

          // Build headers
          const requestHeaders: Record<string, string> = {
            'Content-Type': 'application/json',
            ...headers,
          };

          // Add authentication token
          if (!skipAuth) {
            const token = getAuthToken();
            if (token) {
              requestHeaders['Authorization'] = `Bearer ${token}`;
            }
          }

          // Build request options
          const fetchOptions: RequestInit = {
            method,
            headers: requestHeaders,
            signal: abortControllerRef.current.signal,
          };

          // Add body for non-GET requests
          if (body && method !== 'GET') {
            fetchOptions.body = JSON.stringify(body);
          }

          // Create timeout promise
          const timeoutPromise = new Promise<never>((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), timeoutMs);
          });

          // Make request with timeout
          const response = await Promise.race([
            fetch(url, fetchOptions),
            timeoutPromise,
          ]);

          setStatus(response.status);

          // Handle response
          if (!response.ok) {
            let errorMessage = `Request failed with status ${response.status}`;

            try {
              const errorData = await response.json();
              errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch {
              // If response is not JSON, use status text
              errorMessage = response.statusText || errorMessage;
            }

            throw new Error(errorMessage);
          }

          // Parse response
          let responseData: T;

          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            responseData = await response.json();
          } else {
            responseData = (await response.text()) as any;
          }

          setData(responseData);
          setLoading(false);

          return {
            data: responseData,
            error: null,
            status: response.status,
            success: true,
          };
        } catch (err) {
          lastError = err as Error;

          // Don't retry on abort
          if (lastError.name === 'AbortError') {
            setLoading(false);
            setError('Request cancelled');
            return {
              data: null,
              error: 'Request cancelled',
              status: null,
              success: false,
            };
          }

          // Check if we should retry
          const shouldRetry = attempt < maxRetries;

          if (shouldRetry) {
            attempt++;
            await sleep(retryDelayMs * attempt); // Exponential backoff
            continue;
          }

          // All retries exhausted
          break;
        }
      }

      // Handle final error
      const errorMessage = lastError?.message || 'An unknown error occurred';
      setError(errorMessage);
      setLoading(false);

      // Show error notification unless skipped
      if (!skipErrorNotification) {
        showError(errorMessage);
      }

      return {
        data: null,
        error: errorMessage,
        status: status,
        success: false,
      };
    },
    [showError, status]
  );

  /**
   * Reset state
   */
  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
    setStatus(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  return {
    data,
    error,
    loading,
    status,
    execute,
    reset,
  };
}

/**
 * Convenience hooks for specific HTTP methods
 */

export function useGet<T = any>() {
  const api = useApi<T>();

  const get = useCallback(
    (endpoint: string, options?: Omit<ApiRequestOptions, 'method'>) => {
      return api.execute(endpoint, { ...options, method: 'GET' });
    },
    [api]
  );

  return { ...api, get };
}

export function usePost<T = any>() {
  const api = useApi<T>();

  const post = useCallback(
    (endpoint: string, body?: any, options?: Omit<ApiRequestOptions, 'method' | 'body'>) => {
      return api.execute(endpoint, { ...options, method: 'POST', body });
    },
    [api]
  );

  return { ...api, post };
}

export function usePut<T = any>() {
  const api = useApi<T>();

  const put = useCallback(
    (endpoint: string, body?: any, options?: Omit<ApiRequestOptions, 'method' | 'body'>) => {
      return api.execute(endpoint, { ...options, method: 'PUT', body });
    },
    [api]
  );

  return { ...api, put };
}

export function useDelete<T = any>() {
  const api = useApi<T>();

  const del = useCallback(
    (endpoint: string, options?: Omit<ApiRequestOptions, 'method'>) => {
      return api.execute(endpoint, { ...options, method: 'DELETE' });
    },
    [api]
  );

  return { ...api, delete: del };
}

/**
 * Hook for fetching data on mount
 */
export function useFetch<T = any>(
  endpoint: string,
  options?: ApiRequestOptions
): UseApiReturn<T> {
  const api = useApi<T>();

  useEffect(() => {
    api.execute(endpoint, options);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint]);

  return api;
}

export default useApi;
