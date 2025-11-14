/**
 * API Service
 *
 * Production-ready HTTP client with authentication, retry logic, rate limiting,
 * and request signing for institutional trading platform.
 */

import {
  ApiResponse,
  ApiError,
  RequestConfig,
  RetryConfig,
  RateLimitConfig
} from '../types/api.types';

class ApiService {
  private baseURL: string;
  private authToken: string | null = null;
  private requestQueue: Map<string, Promise<any>> = new Map();
  private rateLimitTokens: number;
  private rateLimitMaxTokens: number;
  private rateLimitRefillRate: number;
  private lastRefillTime: number;
  private retryConfig: RetryConfig;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

    // Rate limiting configuration (token bucket algorithm)
    const rateLimitConfig: RateLimitConfig = {
      maxTokens: parseInt(process.env.REACT_APP_API_RATE_LIMIT_MAX || '100'),
      refillRate: parseInt(process.env.REACT_APP_API_RATE_LIMIT_REFILL || '10'), // tokens per second
      refillInterval: 1000 // milliseconds
    };

    this.rateLimitMaxTokens = rateLimitConfig.maxTokens;
    this.rateLimitTokens = rateLimitConfig.maxTokens;
    this.rateLimitRefillRate = rateLimitConfig.refillRate;
    this.lastRefillTime = Date.now();

    // Retry configuration
    this.retryConfig = {
      maxRetries: parseInt(process.env.REACT_APP_API_MAX_RETRIES || '3'),
      initialDelay: parseInt(process.env.REACT_APP_API_RETRY_DELAY || '1000'),
      maxDelay: parseInt(process.env.REACT_APP_API_RETRY_MAX_DELAY || '10000'),
      backoffMultiplier: parseFloat(process.env.REACT_APP_API_RETRY_BACKOFF || '2')
    };

    // Load auth token from localStorage
    this.authToken = localStorage.getItem('auth_token');

    // Start rate limit token refill
    this.startRateLimitRefill();
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string): void {
    this.authToken = token;
    localStorage.setItem('auth_token', token);
  }

  /**
   * Clear authentication token
   */
  clearAuthToken(): void {
    this.authToken = null;
    localStorage.removeItem('auth_token');
  }

  /**
   * Rate limiting using token bucket algorithm
   */
  private async acquireRateLimitToken(): Promise<void> {
    // Refill tokens based on time elapsed
    const now = Date.now();
    const timeSinceRefill = now - this.lastRefillTime;
    const tokensToAdd = (timeSinceRefill / this.retryConfig.initialDelay) * this.rateLimitRefillRate;

    this.rateLimitTokens = Math.min(
      this.rateLimitMaxTokens,
      this.rateLimitTokens + tokensToAdd
    );
    this.lastRefillTime = now;

    // Wait if no tokens available
    if (this.rateLimitTokens < 1) {
      const waitTime = ((1 - this.rateLimitTokens) / this.rateLimitRefillRate) * 1000;
      await this.sleep(waitTime);
      this.rateLimitTokens = 1;
    }

    this.rateLimitTokens -= 1;
  }

  /**
   * Start periodic rate limit token refill
   */
  private startRateLimitRefill(): void {
    setInterval(() => {
      const tokensToAdd = this.rateLimitRefillRate;
      this.rateLimitTokens = Math.min(
        this.rateLimitMaxTokens,
        this.rateLimitTokens + tokensToAdd
      );
    }, 1000);
  }

  /**
   * Sleep utility for delays
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Calculate exponential backoff delay
   */
  private calculateBackoff(attempt: number): number {
    const delay = this.retryConfig.initialDelay * Math.pow(this.retryConfig.backoffMultiplier, attempt);
    return Math.min(delay, this.retryConfig.maxDelay);
  }

  /**
   * Sign request with HMAC-SHA256 (if API secret is configured)
   */
  private async signRequest(method: string, path: string, body?: any): Promise<string> {
    const apiSecret = process.env.REACT_APP_API_SECRET;
    if (!apiSecret) return '';

    const timestamp = Date.now().toString();
    const message = `${timestamp}${method}${path}${body ? JSON.stringify(body) : ''}`;

    try {
      const encoder = new TextEncoder();
      const key = await crypto.subtle.importKey(
        'raw',
        encoder.encode(apiSecret),
        { name: 'HMAC', hash: 'SHA-256' },
        false,
        ['sign']
      );

      const signature = await crypto.subtle.sign(
        'HMAC',
        key,
        encoder.encode(message)
      );

      const hashArray = Array.from(new Uint8Array(signature));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

      return `${timestamp}.${hashHex}`;
    } catch (error) {
      console.error('Error signing request:', error);
      return '';
    }
  }

  /**
   * Make HTTP request with retry logic, rate limiting, and request signing
   */
  private async request<T>(
    method: string,
    path: string,
    config: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const {
      body,
      headers = {},
      retries = this.retryConfig.maxRetries,
      skipRateLimit = false,
      skipAuth = false,
      timeout = parseInt(process.env.REACT_APP_API_TIMEOUT || '30000')
    } = config;

    // Apply rate limiting
    if (!skipRateLimit) {
      await this.acquireRateLimitToken();
    }

    // Prepare headers
    const requestHeaders: HeadersInit = {
      'Content-Type': 'application/json',
      ...headers
    };

    // Add authentication
    if (!skipAuth && this.authToken) {
      requestHeaders['Authorization'] = `Bearer ${this.authToken}`;
    }

    // Sign request
    const signature = await this.signRequest(method, path, body);
    if (signature) {
      requestHeaders['X-API-Signature'] = signature;
    }

    // Prepare request
    const url = `${this.baseURL}${path}`;
    const requestInit: RequestInit = {
      method,
      headers: requestHeaders,
      ...(body && { body: JSON.stringify(body) })
    };

    let lastError: Error | null = null;

    // Retry loop
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        // Create timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Request timeout')), timeout);
        });

        // Make request with timeout
        const response = await Promise.race([
          fetch(url, requestInit),
          timeoutPromise
        ]) as Response;

        // Handle response
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({
            detail: response.statusText
          }));

          const apiError: ApiError = {
            message: errorData.detail || `HTTP ${response.status}`,
            status: response.status,
            code: errorData.code,
            details: errorData
          };

          // Don't retry client errors (4xx) except 429 (rate limit)
          if (response.status >= 400 && response.status < 500 && response.status !== 429) {
            throw apiError;
          }

          throw apiError;
        }

        // Parse response
        const data = await response.json();

        return {
          data,
          status: response.status,
          headers: Object.fromEntries(response.headers.entries())
        };

      } catch (error: any) {
        lastError = error;

        // Don't retry on client errors
        if (error.status && error.status >= 400 && error.status < 500 && error.status !== 429) {
          throw error;
        }

        // Check if we should retry
        if (attempt < retries) {
          const backoffDelay = this.calculateBackoff(attempt);
          console.warn(`Request failed, retrying in ${backoffDelay}ms (attempt ${attempt + 1}/${retries})...`);
          await this.sleep(backoffDelay);
        }
      }
    }

    // All retries exhausted
    throw lastError || new Error('Request failed after all retries');
  }

  /**
   * GET request
   */
  async get<T>(path: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('GET', path, config);
  }

  /**
   * POST request
   */
  async post<T>(path: string, body?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('POST', path, { ...config, body });
  }

  /**
   * PUT request
   */
  async put<T>(path: string, body?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', path, { ...config, body });
  }

  /**
   * PATCH request
   */
  async patch<T>(path: string, body?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', path, { ...config, body });
  }

  /**
   * DELETE request
   */
  async delete<T>(path: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', path, config);
  }

  /**
   * Request deduplication - prevent duplicate concurrent requests
   */
  async deduplicate<T>(key: string, requestFn: () => Promise<ApiResponse<T>>): Promise<ApiResponse<T>> {
    // Check if request is already in flight
    if (this.requestQueue.has(key)) {
      return this.requestQueue.get(key)!;
    }

    // Execute request
    const promise = requestFn().finally(() => {
      this.requestQueue.delete(key);
    });

    this.requestQueue.set(key, promise);
    return promise;
  }

  /**
   * Batch requests
   */
  async batch<T>(requests: Array<() => Promise<ApiResponse<any>>>): Promise<ApiResponse<T>[]> {
    const results = await Promise.allSettled(requests.map(fn => fn()));

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        throw new Error(`Batch request ${index} failed: ${result.reason}`);
      }
    });
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.get('/health', {
        skipAuth: true,
        skipRateLimit: true,
        retries: 1
      });
      return response.status === 200;
    } catch {
      return false;
    }
  }

  /**
   * Get metrics
   */
  getMetrics() {
    return {
      rateLimitTokens: this.rateLimitTokens,
      rateLimitMaxTokens: this.rateLimitMaxTokens,
      queuedRequests: this.requestQueue.size
    };
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;
