/**
 * Quantum Trader AI Dashboard - Main Entry Point
 *
 * This is the entry point for the React application. It sets up:
 * - React Query for data fetching and caching
 * - React Router for navigation
 * - Zustand store for state management
 * - Global error boundary
 * - Authentication context
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import App from './App';
import './styles/index.css';

// Configure React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Disable automatic refetching on window focus in production
      refetchOnWindowFocus: import.meta.env.MODE === 'development',
      // Retry failed requests up to 3 times
      retry: 3,
      // Exponential backoff for retries
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      // Cache data for 5 minutes by default
      staleTime: 5 * 60 * 1000,
      // Keep unused data in cache for 10 minutes
      cacheTime: 10 * 60 * 1000,
      // Show previous data while refetching
      keepPreviousData: true,
      // Enable suspense mode
      suspense: false,
      // Refetch on mount if data is stale
      refetchOnMount: true,
      // Refetch on reconnect
      refetchOnReconnect: true,
    },
    mutations: {
      // Retry mutations once on failure
      retry: 1,
      // No automatic retry for mutations by default
      retryDelay: 1000,
    },
  },
});

// Global error handler for React Query
queryClient.setDefaultOptions({
  queries: {
    onError: (error: any) => {
      console.error('[React Query Error]', error);
      // Log to error tracking service in production
      if (import.meta.env.PROD) {
        // Sentry.captureException(error);
      }
    },
  },
  mutations: {
    onError: (error: any) => {
      console.error('[React Query Mutation Error]', error);
      if (import.meta.env.PROD) {
        // Sentry.captureException(error);
      }
    },
  },
});

// Error Boundary Component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('[Error Boundary]', error, errorInfo);

    // Log to error tracking service
    if (import.meta.env.PROD) {
      // Sentry.captureException(error, { extra: errorInfo });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-lg p-8 text-center">
            <div className="mb-6">
              <svg
                className="w-16 h-16 mx-auto text-red-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>

            <h1 className="text-2xl font-bold text-slate-100 mb-2">
              Something went wrong
            </h1>

            <p className="text-slate-400 mb-6">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>

            <div className="space-y-3">
              <button
                onClick={() => window.location.reload()}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                Reload Page
              </button>

              <button
                onClick={() => this.setState({ hasError: false, error: null })}
                className="w-full px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors"
              >
                Try Again
              </button>
            </div>

            {import.meta.env.MODE === 'development' && this.state.error && (
              <details className="mt-6 text-left">
                <summary className="cursor-pointer text-sm text-slate-500 hover:text-slate-400">
                  Error Details
                </summary>
                <pre className="mt-2 p-3 bg-slate-950 rounded text-xs text-red-400 overflow-auto max-h-48">
                  {this.state.error.stack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Performance monitoring
const measurePerformance = () => {
  if (typeof window !== 'undefined' && 'performance' in window) {
    window.addEventListener('load', () => {
      const perfData = window.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
      const connectTime = perfData.responseEnd - perfData.requestStart;
      const renderTime = perfData.domComplete - perfData.domLoading;

      console.log('[Performance Metrics]', {
        pageLoadTime: `${pageLoadTime}ms`,
        connectTime: `${connectTime}ms`,
        renderTime: `${renderTime}ms`,
      });

      // Send to analytics in production
      if (import.meta.env.PROD && pageLoadTime > 3000) {
        console.warn('[Performance] Slow page load detected:', pageLoadTime);
      }
    });
  }
};

// Initialize performance monitoring
measurePerformance();

// Hide loading screen after React hydration
const hideLoadingScreen = () => {
  const loadingScreen = document.getElementById('loading-screen');
  if (loadingScreen) {
    setTimeout(() => {
      loadingScreen.classList.add('hidden');
      // Remove from DOM after animation
      setTimeout(() => {
        loadingScreen.remove();
      }, 300);
    }, 500);
  }
};

// Service Worker registration for PWA support
const registerServiceWorker = async () => {
  if ('serviceWorker' in navigator && import.meta.env.PROD) {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js');
      console.log('[Service Worker] Registered successfully:', registration.scope);
    } catch (error) {
      console.error('[Service Worker] Registration failed:', error);
    }
  }
};

// Main render function
const render = () => {
  const rootElement = document.getElementById('root');

  if (!rootElement) {
    throw new Error('Root element not found');
  }

  const root = ReactDOM.createRoot(rootElement);

  root.render(
    <React.StrictMode>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <App />
          </BrowserRouter>

          {/* React Query Devtools - only in development */}
          {import.meta.env.MODE === 'development' && (
            <ReactQueryDevtools
              initialIsOpen={false}
              position="bottom-right"
              toggleButtonProps={{
                style: {
                  marginLeft: '5.5rem',
                  transform: `scale(.7)`,
                  transformOrigin: 'bottom right',
                },
              }}
            />
          )}
        </QueryClientProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );

  // Hide loading screen after render
  hideLoadingScreen();
};

// Enable React DevTools profiling in development
if (import.meta.env.MODE === 'development') {
  // @ts-ignore
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__?.on?.('operations', () => {
    console.log('[React DevTools] Component tree updated');
  });
}

// Hot Module Replacement (HMR) for Vite
if (import.meta.hot) {
  import.meta.hot.accept();
  import.meta.hot.dispose(() => {
    console.log('[HMR] Disposing old module');
  });
}

// Environment validation
const validateEnvironment = () => {
  const requiredEnvVars = [
    'VITE_API_BASE_URL',
    'VITE_WS_BASE_URL',
  ];

  const missingVars = requiredEnvVars.filter(
    (varName) => !import.meta.env[varName]
  );

  if (missingVars.length > 0) {
    console.warn(
      '[Environment] Missing environment variables:',
      missingVars.join(', ')
    );
  }

  // Log environment info in development
  if (import.meta.env.MODE === 'development') {
    console.log('[Environment]', {
      mode: import.meta.env.MODE,
      apiUrl: import.meta.env.VITE_API_BASE_URL,
      wsUrl: import.meta.env.VITE_WS_BASE_URL,
      dev: import.meta.env.DEV,
      prod: import.meta.env.PROD,
    });
  }
};

// Initialize application
const init = async () => {
  try {
    // Validate environment
    validateEnvironment();

    // Register service worker
    await registerServiceWorker();

    // Render application
    render();

    // Log startup message
    if (import.meta.env.MODE === 'development') {
      console.log(
        '%c🚀 Quantum Trader AI Dashboard Started',
        'font-size: 16px; font-weight: bold; color: #3b82f6;'
      );
      console.log(
        '%cRunning in DEVELOPMENT mode',
        'font-size: 12px; color: #f59e0b;'
      );
    }
  } catch (error) {
    console.error('[Initialization Error]', error);

    // Show error to user
    const rootElement = document.getElementById('root');
    if (rootElement) {
      rootElement.innerHTML = `
        <div style="
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #0f172a;
          color: #f8fafc;
          font-family: Inter, sans-serif;
          padding: 1rem;
        ">
          <div style="
            max-width: 500px;
            text-align: center;
            background: #1e293b;
            padding: 2rem;
            border-radius: 0.5rem;
            border: 1px solid #334155;
          ">
            <h1 style="font-size: 1.5rem; margin-bottom: 1rem; color: #ef4444;">
              Initialization Failed
            </h1>
            <p style="color: #cbd5e1; margin-bottom: 1.5rem;">
              ${error instanceof Error ? error.message : 'Unknown error'}
            </p>
            <button
              onclick="window.location.reload()"
              style="
                padding: 0.5rem 1.5rem;
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 0.375rem;
                cursor: pointer;
                font-size: 1rem;
              "
            >
              Reload Page
            </button>
          </div>
        </div>
      `;
    }
  }
};

// Start the application
init();

// Global type declarations for better TypeScript support
declare global {
  interface Window {
    __REACT_DEVTOOLS_GLOBAL_HOOK__?: any;
  }
}

// Export for testing
export { queryClient, ErrorBoundary };
