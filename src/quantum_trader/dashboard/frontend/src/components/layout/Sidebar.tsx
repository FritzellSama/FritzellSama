/**
 * Sidebar Component
 *
 * Navigation sidebar for the trading dashboard.
 * Production-ready component for institutional trading platform.
 */

import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

interface NavItem {
  id: string;
  label: string;
  icon: string;
  path: string;
  badge?: string;
  children?: NavItem[];
}

interface SidebarProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ collapsed = false, onToggle }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  const navItems: NavItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: '📊',
      path: '/dashboard'
    },
    {
      id: 'trading',
      label: 'Trading',
      icon: '💹',
      path: '/trading'
    },
    {
      id: 'portfolio',
      label: 'Portfolio',
      icon: '💼',
      path: '/portfolio',
      children: [
        {
          id: 'positions',
          label: 'Positions',
          icon: '📍',
          path: '/portfolio/positions'
        },
        {
          id: 'orders',
          label: 'Orders',
          icon: '📝',
          path: '/portfolio/orders'
        },
        {
          id: 'history',
          label: 'History',
          icon: '📜',
          path: '/portfolio/history'
        }
      ]
    },
    {
      id: 'strategies',
      label: 'Strategies',
      icon: '🧠',
      path: '/strategies'
    },
    {
      id: 'analytics',
      label: 'Analytics',
      icon: '📈',
      path: '/analytics',
      children: [
        {
          id: 'performance',
          label: 'Performance',
          icon: '🎯',
          path: '/analytics/performance'
        },
        {
          id: 'backtesting',
          label: 'Backtesting',
          icon: '⏮️',
          path: '/analytics/backtesting'
        },
        {
          id: 'reports',
          label: 'Reports',
          icon: '📄',
          path: '/analytics/reports'
        }
      ]
    },
    {
      id: 'risk',
      label: 'Risk Management',
      icon: '⚠️',
      path: '/risk'
    },
    {
      id: 'market',
      label: 'Market Data',
      icon: '🌐',
      path: '/market',
      children: [
        {
          id: 'watchlist',
          label: 'Watchlist',
          icon: '👁️',
          path: '/market/watchlist'
        },
        {
          id: 'orderbook',
          label: 'Order Book',
          icon: '📖',
          path: '/market/orderbook'
        },
        {
          id: 'heatmap',
          label: 'Heat Map',
          icon: '🔥',
          path: '/market/heatmap'
        }
      ]
    },
    {
      id: 'alerts',
      label: 'Alerts',
      icon: '🔔',
      path: '/alerts',
      badge: '3'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: '⚙️',
      path: '/settings'
    }
  ];

  const toggleExpand = (itemId: string) => {
    setExpandedItems(prev =>
      prev.includes(itemId)
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  };

  const isActive = (path: string): boolean => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  const handleNavigation = (item: NavItem) => {
    if (item.children) {
      toggleExpand(item.id);
    } else {
      navigate(item.path);
    }
  };

  const renderNavItem = (item: NavItem, level: number = 0) => {
    const active = isActive(item.path);
    const expanded = expandedItems.includes(item.id);
    const hasChildren = item.children && item.children.length > 0;

    return (
      <div key={item.id}>
        <button
          onClick={() => handleNavigation(item)}
          className={`w-full flex items-center justify-between px-4 py-3 transition-colors ${
            collapsed ? 'justify-center' : ''
          } ${
            level > 0 ? 'pl-12' : ''
          } ${
            active
              ? 'bg-blue-600 text-white font-semibold'
              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
          }`}
        >
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <span className={`text-xl ${collapsed ? 'text-2xl' : ''}`}>{item.icon}</span>
            {!collapsed && (
              <span className="truncate">{item.label}</span>
            )}
          </div>
          {!collapsed && (
            <div className="flex items-center gap-2">
              {item.badge && (
                <span className="px-2 py-0.5 bg-red-600 text-white text-xs font-bold rounded-full">
                  {item.badge}
                </span>
              )}
              {hasChildren && (
                <span className={`transition-transform ${expanded ? 'rotate-90' : ''}`}>
                  ▶
                </span>
              )}
            </div>
          )}
        </button>

        {hasChildren && expanded && !collapsed && (
          <div className="bg-gray-800/50">
            {item.children!.map(child => renderNavItem(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={`${
        collapsed ? 'w-20' : 'w-64'
      } bg-gray-800 border-r border-gray-700 flex flex-col transition-all duration-300 ease-in-out`}
    >
      {/* Header */}
      <div className={`p-4 border-b border-gray-700 flex items-center ${collapsed ? 'justify-center' : 'justify-between'}`}>
        {!collapsed && (
          <div>
            <h1 className="text-xl font-bold text-white">Quantum Trader</h1>
            <p className="text-xs text-gray-400">AI Trading Platform</p>
          </div>
        )}
        {onToggle && (
          <button
            onClick={onToggle}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <span className="text-lg">{collapsed ? '»' : '«'}</span>
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-800">
        <div className="py-2">
          {navItems.map(item => renderNavItem(item))}
        </div>
      </nav>

      {/* Footer */}
      <div className={`p-4 border-t border-gray-700 ${collapsed ? 'text-center' : ''}`}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold">
            {collapsed ? 'U' : 'QT'}
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <div className="text-white font-semibold truncate">Admin User</div>
              <div className="text-xs text-gray-400 truncate">admin@quantum.ai</div>
            </div>
          )}
        </div>
        {!collapsed && (
          <button
            onClick={() => {
              localStorage.removeItem('auth_token');
              navigate('/login');
            }}
            className="mt-3 w-full px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors text-sm font-semibold"
          >
            Logout
          </button>
        )}
      </div>

      {/* Status Indicator */}
      <div className={`px-4 py-2 bg-gray-900 border-t border-gray-700 ${collapsed ? 'text-center' : ''}`}>
        <div className="flex items-center gap-2">
          <div className="relative">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <div className="absolute top-0 left-0 w-3 h-3 bg-green-500 rounded-full animate-ping opacity-75"></div>
          </div>
          {!collapsed && (
            <div className="text-xs">
              <div className="text-green-400 font-semibold">System Online</div>
              <div className="text-gray-500">All services operational</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
