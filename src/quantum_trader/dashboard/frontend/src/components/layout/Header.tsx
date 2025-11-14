/**
 * Header Component - Navigation and status bar
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Main navigation header with system status, user info, and controls
 */

import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

export interface HeaderProps {
  /** Application name */
  appName?: string;
  /** User display name */
  userName?: string;
  /** System status */
  systemStatus?: 'online' | 'offline' | 'degraded';
  /** Additional CSS classes */
  className?: string;
}

interface NavItem {
  label: string;
  path: string;
  icon?: string;
}

const navigationItems: NavItem[] = [
  { label: 'Dashboard', path: '/', icon: '📊' },
  { label: 'Analytics', path: '/analytics', icon: '📈' },
  { label: 'Positions', path: '/positions', icon: '💼' },
  { label: 'Orders', path: '/orders', icon: '📋' },
  { label: 'Strategies', path: '/strategies', icon: '🎯' },
  { label: 'Settings', path: '/settings', icon: '⚙️' },
];

/**
 * Main application header with navigation
 *
 * @example
 * ```tsx
 * <Header
 *   appName="Quantum Trader AI"
 *   userName="Admin User"
 *   systemStatus="online"
 * />
 * ```
 */
export const Header: React.FC<HeaderProps> = ({
  appName = process.env.REACT_APP_NAME || 'Quantum Trader AI',
  userName = process.env.REACT_APP_DEFAULT_USER || 'Trader',
  systemStatus = 'online',
  className = '',
}) => {
  const location = useLocation();
  const [currentTime, setCurrentTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString('en-US', {
          hour12: false,
          timeZone: 'UTC',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }) + ' UTC'
      );
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'online':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'offline':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const isActivePath = (path: string): boolean => {
    return location.pathname === path;
  };

  return (
    <header className={`bg-gray-900 border-b border-gray-800 ${className}`}>
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and App Name */}
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-xl font-bold text-white flex items-center">
                <span className="text-2xl mr-2">⚡</span>
                {appName}
              </h1>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-1">
            {navigationItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActivePath(item.path)
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                }`}
              >
                <span className="mr-1">{item.icon}</span>
                {item.label}
              </Link>
            ))}
          </nav>

          {/* Right side - Status and User */}
          <div className="flex items-center space-x-4">
            {/* System Time */}
            <div className="hidden lg:flex items-center text-sm text-gray-400">
              <span className="mr-2">🕒</span>
              {currentTime}
            </div>

            {/* System Status */}
            <div className="flex items-center space-x-2">
              <div className="flex items-center space-x-1">
                <div className={`h-2 w-2 rounded-full ${getStatusColor(systemStatus)} animate-pulse`} />
                <span className="text-xs text-gray-400 capitalize hidden sm:inline">
                  {systemStatus}
                </span>
              </div>
            </div>

            {/* User Menu */}
            <div className="flex items-center space-x-2 border-l border-gray-700 pl-4">
              <div className="flex items-center space-x-2">
                <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <span className="text-white text-sm font-medium">
                    {userName.charAt(0).toUpperCase()}
                  </span>
                </div>
                <span className="text-sm text-gray-300 hidden sm:inline">
                  {userName}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
