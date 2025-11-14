/**
 * App Component - Main Application
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Root application component with routing and layout
 */

import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/layout/Header';
import Footer from './components/layout/Footer';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';

/**
 * Error boundary component
 */
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
    console.error('Application Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
          <div className="bg-gray-800 border border-red-500 rounded-lg p-8 max-w-lg">
            <h1 className="text-2xl font-bold text-red-400 mb-4">
              ⚠️ Application Error
            </h1>
            <p className="text-gray-300 mb-4">
              An unexpected error occurred. Please refresh the page or contact support.
            </p>
            {this.state.error && (
              <pre className="bg-gray-900 p-4 rounded text-xs text-gray-400 overflow-auto">
                {this.state.error.toString()}
              </pre>
            )}
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Loading screen component
 */
const LoadingScreen: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="flex flex-col items-center space-y-4">
        <div className="relative">
          <div className="h-16 w-16 rounded-full border-4 border-gray-700"></div>
          <div className="h-16 w-16 rounded-full border-4 border-blue-500 border-t-transparent animate-spin absolute top-0 left-0"></div>
        </div>
        <p className="text-xl font-semibold text-gray-300">
          Loading Quantum Trader AI
        </p>
        <p className="text-sm text-gray-500">Initializing trading systems...</p>
      </div>
    </div>
  );
};

/**
 * Not found page component
 */
const NotFound: React.FC = () => {
  return (
    <div className="flex-grow flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-700">404</h1>
        <p className="text-xl text-gray-400 mt-4">Page Not Found</p>
        <a
          href="/"
          className="mt-6 inline-block px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          Return to Dashboard
        </a>
      </div>
    </div>
  );
};

/**
 * Main application component
 *
 * Features:
 * - React Router for navigation
 * - Error boundary for error handling
 * - Loading states
 * - Consistent layout with Header and Footer
 * - Dark theme optimized for trading
 */
const App: React.FC = () => {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [systemStatus, setSystemStatus] = useState<'online' | 'offline' | 'degraded'>('online');

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Simulate initialization
        await new Promise((resolve) => setTimeout(resolve, 1000));

        // Check system status
        const apiUrl = process.env.REACT_APP_API_URL;
        if (apiUrl) {
          try {
            const response = await fetch(`${apiUrl}/health`, {
              method: 'GET',
              headers: {
                'Content-Type': 'application/json',
              },
            });

            if (response.ok) {
              setSystemStatus('online');
            } else {
              setSystemStatus('degraded');
            }
          } catch (error) {
            console.error('Health check failed:', error);
            setSystemStatus('degraded');
          }
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Initialization error:', error);
        setSystemStatus('offline');
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  // Periodic health check
  useEffect(() => {
    const apiUrl = process.env.REACT_APP_API_URL;
    if (!apiUrl) return;

    const healthCheckInterval = setInterval(async () => {
      try {
        const response = await fetch(`${apiUrl}/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          setSystemStatus('online');
        } else {
          setSystemStatus('degraded');
        }
      } catch (error) {
        console.error('Health check failed:', error);
        setSystemStatus('degraded');
      }
    }, 30000); // Check every 30 seconds

    return () => clearInterval(healthCheckInterval);
  }, []);

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
          {/* Header */}
          <Header systemStatus={systemStatus} />

          {/* Main Content */}
          <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/positions" element={<NotFound />} />
              <Route path="/orders" element={<NotFound />} />
              <Route path="/strategies" element={<NotFound />} />
              <Route path="/settings" element={<NotFound />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </main>

          {/* Footer */}
          <Footer />
        </div>
      </Router>
    </ErrorBoundary>
  );
};

export default App;
