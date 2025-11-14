import React, { useState, useEffect } from 'react';
import { Link, Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  CpuChipIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon,
  BellIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
}

interface LayoutProps {
  children?: React.ReactNode;
}

interface UserInfo {
  username: string;
  email: string;
  role: string;
}

const navigation: NavigationItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Positions', href: '/positions', icon: ChartBarIcon },
  { name: 'Trading', href: '/trading', icon: CurrencyDollarIcon },
  { name: 'ML Models', href: '/ml-models', icon: CpuChipIcon },
  { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
];

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [notifications, setNotifications] = useState<number>(0);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    fetchUserInfo();
    fetchNotifications();

    const notificationInterval = setInterval(fetchNotifications, 30000);

    return () => clearInterval(notificationInterval);
  }, []);

  const fetchUserInfo = async (): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/user/info`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleLogout();
          return;
        }
        throw new Error(`Failed to fetch user info: ${response.statusText}`);
      }

      const data = await response.json();
      setUserInfo(data);
    } catch (error) {
      console.error('Error fetching user info:', error);
      handleLogout();
    } finally {
      setLoading(false);
    }
  };

  const fetchNotifications = async (): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/notifications/unread`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setNotifications(data.count || 0);
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    }
  };

  const handleLogout = (): void => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('refresh_token');
    navigate('/login');
  };

  const isActive = (path: string): boolean => {
    return location.pathname === path || location.pathname.startsWith(`${path}/`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <aside
        className={`${
          sidebarOpen ? 'w-64' : 'w-20'
        } bg-gray-800 border-r border-gray-700 transition-all duration-300 ease-in-out flex flex-col`}
      >
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-700">
          <div className="flex items-center">
            <CpuChipIcon className="h-8 w-8 text-blue-500" />
            {sidebarOpen && (
              <span className="ml-3 text-xl font-bold text-white">
                Quantum Trader
              </span>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);

            return (
              <Link
                key={item.name}
                to={item.href}
                className={`${
                  active
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:bg-gray-700 hover:text-white'
                } group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors`}
              >
                <Icon
                  className={`${
                    active ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-300'
                  } flex-shrink-0 h-6 w-6`}
                />
                {sidebarOpen && <span className="ml-3">{item.name}</span>}
              </Link>
            );
          })}
        </nav>

        {/* User Section */}
        <div className="border-t border-gray-700 p-4">
          <div className="flex items-center">
            <UserCircleIcon className="h-8 w-8 text-gray-400" />
            {sidebarOpen && userInfo && (
              <div className="ml-3 flex-1">
                <p className="text-sm font-medium text-white truncate">
                  {userInfo.username}
                </p>
                <p className="text-xs text-gray-400 truncate">{userInfo.role}</p>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Header */}
        <header className="bg-gray-800 border-b border-gray-700 h-16 flex items-center justify-between px-6">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-gray-400 hover:text-white focus:outline-none"
          >
            <svg
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>

          <div className="flex items-center space-x-4">
            {/* Notifications */}
            <button className="relative text-gray-400 hover:text-white focus:outline-none">
              <BellIcon className="h-6 w-6" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-500 rounded-full">
                  {notifications > 99 ? '99+' : notifications}
                </span>
              )}
            </button>

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="flex items-center text-gray-400 hover:text-white focus:outline-none"
            >
              <ArrowRightOnRectangleIcon className="h-6 w-6" />
              <span className="ml-2 text-sm">Logout</span>
            </button>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto bg-gray-900 p-6">
          {children || <Outlet />}
        </main>
      </div>
    </div>
  );
};

export default Layout;
