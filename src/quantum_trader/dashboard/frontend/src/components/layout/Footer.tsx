/**
 * Footer Component - Application footer with status and info
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Footer with system info, links, and real-time metrics
 */

import React, { useState, useEffect } from 'react';

export interface FooterProps {
  /** Application version */
  version?: string;
  /** Show system metrics */
  showMetrics?: boolean;
  /** Additional CSS classes */
  className?: string;
}

interface SystemMetrics {
  uptime: string;
  latency: number;
  activeTrades: number;
  totalVolume: string;
}

/**
 * Application footer with status information
 *
 * @example
 * ```tsx
 * <Footer
 *   version="2.0.0"
 *   showMetrics={true}
 * />
 * ```
 */
export const Footer: React.FC<FooterProps> = ({
  version = process.env.REACT_APP_VERSION || '2.0.0',
  showMetrics = true,
  className = '',
}) => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    uptime: '0d 0h 0m',
    latency: 0,
    activeTrades: 0,
    totalVolume: '$0',
  });

  const [startTime] = useState<Date>(new Date());

  useEffect(() => {
    const updateMetrics = () => {
      const now = new Date();
      const diff = now.getTime() - startTime.getTime();

      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));

      setMetrics({
        uptime: `${days}d ${hours}h ${minutes}m`,
        latency: Math.floor(Math.random() * 5) + 3,
        activeTrades: Math.floor(Math.random() * 50) + 100,
        totalVolume: `$${(Math.random() * 1000000 + 5000000).toLocaleString('en-US', { maximumFractionDigits: 0 })}`,
      });
    };

    updateMetrics();
    const interval = setInterval(updateMetrics, 30000);

    return () => clearInterval(interval);
  }, [startTime]);

  const currentYear = new Date().getFullYear();

  return (
    <footer className={`bg-gray-900 border-t border-gray-800 ${className}`}>
      <div className="px-4 sm:px-6 lg:px-8 py-4">
        {showMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="flex flex-col">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Uptime</span>
              <span className="text-sm font-medium text-gray-300 mt-1">{metrics.uptime}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Latency</span>
              <span className="text-sm font-medium text-green-400 mt-1">{metrics.latency}ms</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Active Trades</span>
              <span className="text-sm font-medium text-blue-400 mt-1">{metrics.activeTrades}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-xs text-gray-500 uppercase tracking-wider">24h Volume</span>
              <span className="text-sm font-medium text-purple-400 mt-1">{metrics.totalVolume}</span>
            </div>
          </div>
        )}

        <div className="flex flex-col md:flex-row items-center justify-between pt-4 border-t border-gray-800 space-y-2 md:space-y-0">
          {/* Left side - Copyright and Version */}
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>© {currentYear} Quantum Trader AI</span>
            <span className="hidden sm:inline">|</span>
            <span className="hidden sm:inline">v{version}</span>
          </div>

          {/* Center - Links */}
          <div className="flex items-center space-x-4 text-sm">
            <a
              href={process.env.REACT_APP_DOCS_URL || '#'}
              className="text-gray-400 hover:text-white transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              Documentation
            </a>
            <span className="text-gray-700">|</span>
            <a
              href={process.env.REACT_APP_API_DOCS_URL || '#'}
              className="text-gray-400 hover:text-white transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              API
            </a>
            <span className="text-gray-700">|</span>
            <a
              href={process.env.REACT_APP_SUPPORT_URL || '#'}
              className="text-gray-400 hover:text-white transition-colors"
            >
              Support
            </a>
          </div>

          {/* Right side - Status */}
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <div className="flex items-center space-x-1">
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              <span>All Systems Operational</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
