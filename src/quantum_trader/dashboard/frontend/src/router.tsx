/**
 * React Router Configuration
 *
 * Defines routes and navigation structure for the Quantum Trader AI dashboard.
 * Implements lazy loading, protected routes, and error boundaries.
 *
 * @module router
 */

import React, { Suspense, lazy } from 'react';
import { createBrowserRouter, Navigate, Outlet, RouteObject } from 'react-router-dom';

/**
 * Lazy-loaded components for code splitting
 */
const DashboardLayout = lazy(() => import('./layouts/DashboardLayout'));
const LoginPage = lazy(() => import('./pages/LoginPage'));
const OverviewPage = lazy(() => import('./pages/OverviewPage'));
const TradingPage = lazy(() => import('./pages/TradingPage'));
const PortfolioPage = lazy(() => import('./pages/PortfolioPage'));
const StrategiesPage = lazy(() => import('./pages/StrategiesPage'));
const BacktestPage = lazy(() => import('./pages/BacktestPage'));
const RiskManagementPage = lazy(() => import('./pages/RiskManagementPage'));
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'));
const ErrorPage = lazy(() => import('./pages/ErrorPage'));

/**
 * Loading fallback component
 */
const LoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center h-screen bg-gray-900">
    <div className="text-center">
      <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      <p className="mt-4 text-gray-400">Loading...</p>
    </div>
  </div>
);

/**
 * Protected route wrapper
 * Checks authentication status and redirects to login if not authenticated
 */
const ProtectedRoute: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = React.useState<boolean | null>(null);

  React.useEffect(() => {
    const checkAuth = async () => {
      try {
        const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
        if (!apiUrl) {
          throw new Error('API_URL not configured');
        }

        const token = localStorage.getItem('auth_token');
        if (!token) {
          setIsAuthenticated(false);
          return;
        }

        const response = await fetch(`${apiUrl}/api/v1/auth/verify`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        setIsAuthenticated(response.ok);
      } catch (error) {
        console.error('Authentication check failed:', error);
        setIsAuthenticated(false);
      }
    };

    checkAuth();
  }, []);

  if (isAuthenticated === null) {
    return <LoadingFallback />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Outlet />
    </Suspense>
  );
};

/**
 * Public route wrapper
 * Redirects to dashboard if already authenticated
 */
const PublicRoute: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = React.useState<boolean | null>(null);

  React.useEffect(() => {
    const checkAuth = async () => {
      try {
        const apiUrl = process.env.REACT_APP_API_URL || window.ENV?.API_URL;
        if (!apiUrl) {
          setIsAuthenticated(false);
          return;
        }

        const token = localStorage.getItem('auth_token');
        if (!token) {
          setIsAuthenticated(false);
          return;
        }

        const response = await fetch(`${apiUrl}/api/v1/auth/verify`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        setIsAuthenticated(response.ok);
      } catch (error) {
        console.error('Authentication check failed:', error);
        setIsAuthenticated(false);
      }
    };

    checkAuth();
  }, []);

  if (isAuthenticated === null) {
    return <LoadingFallback />;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <Suspense fallback={<LoadingFallback />}>
      <Outlet />
    </Suspense>
  );
};

/**
 * Route definitions
 */
const routes: RouteObject[] = [
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '/login',
    element: <PublicRoute />,
    children: [
      {
        index: true,
        element: <LoginPage />,
      },
    ],
  },
  {
    path: '/dashboard',
    element: <ProtectedRoute />,
    errorElement: (
      <Suspense fallback={<LoadingFallback />}>
        <ErrorPage />
      </Suspense>
    ),
    children: [
      {
        element: <DashboardLayout />,
        children: [
          {
            index: true,
            element: <OverviewPage />,
          },
          {
            path: 'trading',
            element: <TradingPage />,
          },
          {
            path: 'portfolio',
            element: <PortfolioPage />,
          },
          {
            path: 'strategies',
            element: <StrategiesPage />,
          },
          {
            path: 'backtest',
            element: <BacktestPage />,
          },
          {
            path: 'risk',
            element: <RiskManagementPage />,
          },
          {
            path: 'analytics',
            element: <AnalyticsPage />,
          },
          {
            path: 'settings',
            element: <SettingsPage />,
          },
        ],
      },
    ],
  },
  {
    path: '*',
    element: (
      <Suspense fallback={<LoadingFallback />}>
        <NotFoundPage />
      </Suspense>
    ),
  },
];

/**
 * Create and export router
 */
export const router = createBrowserRouter(routes, {
  future: {
    v7_startTransition: true,
    v7_relativeSplatPath: true,
    v7_fetcherPersist: true,
    v7_normalizeFormMethod: true,
    v7_partialHydration: true,
    v7_skipActionErrorRevalidation: true,
  },
});

/**
 * Route path constants for type-safe navigation
 */
export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  DASHBOARD: '/dashboard',
  TRADING: '/dashboard/trading',
  PORTFOLIO: '/dashboard/portfolio',
  STRATEGIES: '/dashboard/strategies',
  BACKTEST: '/dashboard/backtest',
  RISK: '/dashboard/risk',
  ANALYTICS: '/dashboard/analytics',
  SETTINGS: '/dashboard/settings',
} as const;

/**
 * Route metadata for navigation menus
 */
export interface RouteMetadata {
  path: string;
  title: string;
  icon?: string;
  description?: string;
  showInNav?: boolean;
  requiredPermissions?: string[];
}

export const ROUTE_METADATA: Record<string, RouteMetadata> = {
  OVERVIEW: {
    path: ROUTES.DASHBOARD,
    title: 'Overview',
    icon: 'dashboard',
    description: 'System overview and key metrics',
    showInNav: true,
  },
  TRADING: {
    path: ROUTES.TRADING,
    title: 'Trading',
    icon: 'trending_up',
    description: 'Live trading interface',
    showInNav: true,
    requiredPermissions: ['trading.view'],
  },
  PORTFOLIO: {
    path: ROUTES.PORTFOLIO,
    title: 'Portfolio',
    icon: 'account_balance_wallet',
    description: 'Portfolio positions and P&L',
    showInNav: true,
    requiredPermissions: ['portfolio.view'],
  },
  STRATEGIES: {
    path: ROUTES.STRATEGIES,
    title: 'Strategies',
    icon: 'psychology',
    description: 'Trading strategies management',
    showInNav: true,
    requiredPermissions: ['strategies.view'],
  },
  BACKTEST: {
    path: ROUTES.BACKTEST,
    title: 'Backtest',
    icon: 'history',
    description: 'Strategy backtesting',
    showInNav: true,
    requiredPermissions: ['backtest.view'],
  },
  RISK: {
    path: ROUTES.RISK,
    title: 'Risk Management',
    icon: 'security',
    description: 'Risk metrics and controls',
    showInNav: true,
    requiredPermissions: ['risk.view'],
  },
  ANALYTICS: {
    path: ROUTES.ANALYTICS,
    title: 'Analytics',
    icon: 'analytics',
    description: 'Advanced analytics and reporting',
    showInNav: true,
    requiredPermissions: ['analytics.view'],
  },
  SETTINGS: {
    path: ROUTES.SETTINGS,
    title: 'Settings',
    icon: 'settings',
    description: 'System configuration',
    showInNav: true,
    requiredPermissions: ['settings.view'],
  },
};

/**
 * Helper function to get navigation routes
 */
export const getNavigationRoutes = (userPermissions: string[] = []): RouteMetadata[] => {
  return Object.values(ROUTE_METADATA).filter((route) => {
    if (!route.showInNav) return false;

    if (route.requiredPermissions && route.requiredPermissions.length > 0) {
      return route.requiredPermissions.some((permission) => userPermissions.includes(permission));
    }

    return true;
  });
};

export default router;
