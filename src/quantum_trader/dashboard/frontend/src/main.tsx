/**
 * Application entry point
 *
 * Initializes:
 * - React application
 * - Redux store
 * - Router
 * - Authentication service
 * - Theme provider
 * - Global error handling
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { store } from './store';
import { initializeAuthService } from './services/auth';
import { API_CONFIG, AUTH_CONFIG, APP_METADATA } from './utils/constants';

// Import global styles
import './styles/index.css';

// Lazy load main App component
const App = React.lazy(() => import('./App'));

/**
 * Initialize application services
 */
function initializeServices(): void {
  try {
    // Initialize authentication service
    initializeAuthService({
      apiBaseUrl: API_CONFIG.BASE_URL,
      tokenStorageKey: AUTH_CONFIG.TOKEN_STORAGE_KEY,
      refreshTokenStorageKey: AUTH_CONFIG.REFRESH_TOKEN_STORAGE_KEY,
      tokenExpiryKey: AUTH_CONFIG.TOKEN_EXPIRY_KEY,
      refreshThresholdSeconds: AUTH_CONFIG.REFRESH_THRESHOLD_SECONDS,
    });

    console.info('Services initialized successfully');
  } catch (error) {
    console.error('Failed to initialize services:', error);
    throw error;
  }
}

/**
 * Setup global error handlers
 */
function setupErrorHandlers(): void {
  // Log unhandled errors
  window.addEventListener('error', (event) => {
    console.error('Unhandled error:', event.error);

    // Send to error tracking service in production
    if (APP_METADATA.ENVIRONMENT === 'production') {
      // TODO: Send to Sentry, LogRocket, etc.
    }
  });

  // Log unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);

    // Send to error tracking service in production
    if (APP_METADATA.ENVIRONMENT === 'production') {
      // TODO: Send to Sentry, LogRocket, etc.
    }
  });

  // React error boundary fallback
  window.addEventListener('react-error', (event: any) => {
    console.error('React error boundary:', event.detail);
  });
}

/**
 * Setup performance monitoring
 */
function setupPerformanceMonitoring(): void {
  if (APP_METADATA.ENVIRONMENT === 'production') {
    // Monitor page load performance
    window.addEventListener('load', () => {
      const perfData = window.performance.timing;
      const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;

      console.info('Page load time:', pageLoadTime, 'ms');

      // Send to analytics service
      // TODO: Send to Google Analytics, etc.
    });

    // Monitor resource timing
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        // Log slow resources
        if (entry.duration > 1000) {
          console.warn('Slow resource:', entry.name, entry.duration, 'ms');
        }
      }
    });

    observer.observe({ entryTypes: ['resource', 'navigation'] });
  }
}

/**
 * Loading fallback component
 */
function LoadingFallback(): JSX.Element {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%)',
      }}
    >
      <div
        style={{
          textAlign: 'center',
          color: '#9ca3af',
        }}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            margin: '0 auto 16px',
            border: '4px solid rgba(59, 130, 246, 0.1)',
            borderTopColor: '#3b82f6',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
          }}
        />
        <p>Loading application...</p>
      </div>
    </div>
  );
}

/**
 * Error fallback component
 */
interface ErrorFallbackProps {
  error: Error;
  resetError: () => void;
}

function ErrorFallback({ error, resetError }: ErrorFallbackProps): JSX.Element {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: '#0f0f0f',
        color: '#ffffff',
      }}
    >
      <div
        style={{
          maxWidth: '500px',
          padding: '32px',
          textAlign: 'center',
        }}
      >
        <div style={{ fontSize: '64px', marginBottom: '24px' }}>⚠️</div>
        <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#ef4444', marginBottom: '16px' }}>
          Application Error
        </h1>
        <p style={{ fontSize: '16px', color: '#9ca3af', marginBottom: '16px', lineHeight: 1.6 }}>
          {error.message || 'An unexpected error occurred'}
        </p>
        {APP_METADATA.ENVIRONMENT === 'development' && (
          <pre
            style={{
              padding: '16px',
              background: '#1a1a1a',
              borderRadius: '8px',
              fontSize: '12px',
              textAlign: 'left',
              overflow: 'auto',
              marginBottom: '24px',
            }}
          >
            {error.stack}
          </pre>
        )}
        <button
          onClick={resetError}
          style={{
            padding: '12px 24px',
            fontSize: '14px',
            fontWeight: 600,
            color: '#ffffff',
            background: '#3b82f6',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
          }}
        >
          Try Again
        </button>
      </div>
    </div>
  );
}

/**
 * Root error boundary
 */
class RootErrorBoundary extends React.Component<
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
    console.error('Error boundary caught error:', error, errorInfo);

    // Send to error tracking
    if (APP_METADATA.ENVIRONMENT === 'production') {
      // TODO: Send to Sentry, etc.
    }

    // Dispatch custom event
    window.dispatchEvent(
      new CustomEvent('react-error', {
        detail: { error, errorInfo },
      })
    );
  }

  resetError = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      return <ErrorFallback error={this.state.error} resetError={this.resetError} />;
    }

    return this.props.children;
  }
}

/**
 * Initialize and render application
 */
function initializeApp(): void {
  try {
    // Setup error handlers
    setupErrorHandlers();

    // Setup performance monitoring
    setupPerformanceMonitoring();

    // Initialize services
    initializeServices();

    // Get root element
    const rootElement = document.getElementById('root');
    if (!rootElement) {
      throw new Error('Root element not found');
    }

    // Create React root
    const root = ReactDOM.createRoot(rootElement);

    // Render application
    root.render(
      <React.StrictMode>
        <RootErrorBoundary>
          <Provider store={store}>
            <BrowserRouter>
              <React.Suspense fallback={<LoadingFallback />}>
                <App />
              </React.Suspense>
            </BrowserRouter>
          </Provider>
        </RootErrorBoundary>
      </React.StrictMode>
    );

    // Hide loading screen after initial render
    setTimeout(() => {
      if (typeof window.__hideLoadingScreen === 'function') {
        window.__hideLoadingScreen();
      }
    }, 500);

    // Log application info
    console.info(
      `%c${APP_METADATA.NAME} v${APP_METADATA.VERSION}`,
      'font-size: 16px; font-weight: bold; color: #3b82f6;'
    );
    console.info(`Environment: ${APP_METADATA.ENVIRONMENT}`);
    console.info(`Build ID: ${APP_METADATA.BUILD_ID}`);
    console.info(`API Version: ${APP_METADATA.API_VERSION}`);

    // Development-only helpers
    if (APP_METADATA.ENVIRONMENT === 'development') {
      console.info('Development mode enabled');
      console.info('Redux store available at window.__REDUX_STORE__');
    }
  } catch (error) {
    console.error('Failed to initialize application:', error);

    // Show error screen
    const errorScreen = document.getElementById('error-screen');
    const errorMessage = document.getElementById('error-message');
    const loadingScreen = document.getElementById('loading-screen');

    if (loadingScreen) {
      loadingScreen.classList.add('hidden');
    }

    if (errorScreen) {
      errorScreen.classList.add('visible');
    }

    if (errorMessage) {
      errorMessage.textContent =
        error instanceof Error ? error.message : 'Failed to initialize application';
    }
  }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

// Hot module replacement (HMR) support
if (import.meta.hot) {
  import.meta.hot.accept();
}

// Export for testing
export { initializeServices, setupErrorHandlers, setupPerformanceMonitoring };
