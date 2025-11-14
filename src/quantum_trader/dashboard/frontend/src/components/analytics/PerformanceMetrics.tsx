import React, { useState, useEffect } from 'react';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ScaleIcon,
  TrophyIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

interface Metrics {
  total_pnl: string;
  total_pnl_percent: string;
  realized_pnl: string;
  unrealized_pnl: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: string;
  average_win: string;
  average_loss: string;
  profit_factor: string;
  sharpe_ratio: string;
  max_drawdown: string;
  max_drawdown_percent: string;
  total_volume: string;
  total_fees: string;
  best_trade: string;
  worst_trade: string;
  average_trade_duration: string;
  roi: string;
}

interface TimeRange {
  label: string;
  value: string;
}

const timeRanges: TimeRange[] = [
  { label: '24H', value: '24h' },
  { label: '7D', value: '7d' },
  { label: '30D', value: '30d' },
  { label: '90D', value: '90d' },
  { label: 'All', value: 'all' },
];

const PerformanceMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [selectedRange, setSelectedRange] = useState<string>('24h');
  const [refreshing, setRefreshing] = useState<boolean>(false);

  useEffect(() => {
    fetchMetrics();
  }, [selectedRange]);

  const fetchMetrics = async (background: boolean = false): Promise<void> => {
    if (!background) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(
        `${apiUrl}/api/v1/analytics/performance?timeframe=${selectedRange}`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch metrics: ${response.statusText}`);
      }

      const data = await response.json();
      setMetrics(data.metrics);
    } catch (err) {
      console.error('Error fetching performance metrics:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load metrics';
      setError(errorMessage);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = (): void => {
    fetchMetrics(true);
  };

  const formatCurrency = (value: string): string => {
    const num = parseFloat(value);
    return num.toLocaleString('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  };

  const formatPercent = (value: string): string => {
    return `${parseFloat(value).toFixed(2)}%`;
  };

  const getColorClass = (value: string): string => {
    const num = parseFloat(value);
    if (num > 0) return 'text-green-400';
    if (num < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500 bg-opacity-10 border border-red-500 rounded-md p-6 text-center">
        <p className="text-red-400">{error}</p>
        <button
          onClick={handleRefresh}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="text-center py-12">
        <ChartBarIcon className="mx-auto h-12 w-12 text-gray-600" />
        <h3 className="mt-2 text-sm font-medium text-gray-400">No data available</h3>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Performance Metrics</h2>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-5 w-5 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Time Range Selector */}
      <div className="flex space-x-2">
        {timeRanges.map((range) => (
          <button
            key={range.value}
            onClick={() => setSelectedRange(range.value)}
            className={`px-4 py-2 rounded-md font-medium transition-colors ${
              selectedRange === range.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
          >
            {range.label}
          </button>
        ))}
      </div>

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total P&L */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <CurrencyDollarIcon className="h-8 w-8 text-blue-500" />
            {parseFloat(metrics.total_pnl) >= 0 ? (
              <ArrowTrendingUpIcon className="h-6 w-6 text-green-400" />
            ) : (
              <ArrowTrendingDownIcon className="h-6 w-6 text-red-400" />
            )}
          </div>
          <p className="text-sm text-gray-400">Total P&L</p>
          <p className={`text-2xl font-bold mt-1 ${getColorClass(metrics.total_pnl)}`}>
            {formatCurrency(metrics.total_pnl)}
          </p>
          <p className={`text-sm mt-1 ${getColorClass(metrics.total_pnl_percent)}`}>
            {formatPercent(metrics.total_pnl_percent)}
          </p>
        </div>

        {/* Win Rate */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <TrophyIcon className="h-8 w-8 text-yellow-500" />
          </div>
          <p className="text-sm text-gray-400">Win Rate</p>
          <p className="text-2xl font-bold text-white mt-1">
            {formatPercent(metrics.win_rate)}
          </p>
          <p className="text-sm text-gray-400 mt-1">
            {metrics.winning_trades}W / {metrics.losing_trades}L
          </p>
        </div>

        {/* Sharpe Ratio */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <ScaleIcon className="h-8 w-8 text-purple-500" />
          </div>
          <p className="text-sm text-gray-400">Sharpe Ratio</p>
          <p className="text-2xl font-bold text-white mt-1">
            {parseFloat(metrics.sharpe_ratio).toFixed(2)}
          </p>
          <p className="text-sm text-gray-400 mt-1">Risk-adjusted return</p>
        </div>

        {/* Max Drawdown */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <ChartBarIcon className="h-8 w-8 text-red-500" />
          </div>
          <p className="text-sm text-gray-400">Max Drawdown</p>
          <p className="text-2xl font-bold text-red-400 mt-1">
            {formatCurrency(metrics.max_drawdown)}
          </p>
          <p className="text-sm text-red-400 mt-1">
            {formatPercent(metrics.max_drawdown_percent)}
          </p>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* P&L Breakdown */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">P&L Breakdown</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Realized P&L:</span>
              <span className={`font-semibold ${getColorClass(metrics.realized_pnl)}`}>
                {formatCurrency(metrics.realized_pnl)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Unrealized P&L:</span>
              <span className={`font-semibold ${getColorClass(metrics.unrealized_pnl)}`}>
                {formatCurrency(metrics.unrealized_pnl)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Volume:</span>
              <span className="font-semibold text-white">
                {formatCurrency(metrics.total_volume)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Fees:</span>
              <span className="font-semibold text-red-400">
                {formatCurrency(metrics.total_fees)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">ROI:</span>
              <span className={`font-semibold ${getColorClass(metrics.roi)}`}>
                {formatPercent(metrics.roi)}
              </span>
            </div>
          </div>
        </div>

        {/* Trade Statistics */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Trade Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Trades:</span>
              <span className="font-semibold text-white">{metrics.total_trades}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Average Win:</span>
              <span className="font-semibold text-green-400">
                {formatCurrency(metrics.average_win)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Average Loss:</span>
              <span className="font-semibold text-red-400">
                {formatCurrency(metrics.average_loss)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Profit Factor:</span>
              <span className="font-semibold text-white">
                {parseFloat(metrics.profit_factor).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Trade Duration:</span>
              <span className="font-semibold text-white">
                {metrics.average_trade_duration}
              </span>
            </div>
          </div>
        </div>

        {/* Best & Worst Trades */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Trade Extremes</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Best Trade:</span>
              <span className="font-semibold text-green-400">
                {formatCurrency(metrics.best_trade)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Worst Trade:</span>
              <span className="font-semibold text-red-400">
                {formatCurrency(metrics.worst_trade)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;
