/**
 * Strategies Page
 *
 * Strategy management and monitoring dashboard.
 * Production-ready component for institutional trading platform.
 */

import React, { useState, useEffect, useCallback } from 'react';

interface StrategyMetrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_pnl: number;
  avg_trade_duration: number;
  daily_pnl: number;
}

interface Strategy {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'paused' | 'stopped';
  description: string;
  timeframe: string;
  symbols: string[];
  capital_allocated: number;
  metrics: StrategyMetrics;
  created_at: string;
  updated_at: string;
}

interface StrategySignal {
  strategy: string;
  symbol: string;
  action: string;
  strength: number;
  confidence: number;
  timestamp: string;
  executed: boolean;
}

const Strategies: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [recentSignals, setRecentSignals] = useState<StrategySignal[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_STRATEGY_REFRESH_INTERVAL || '10000');

  // Fetch strategies
  const fetchStrategies = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategies`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch strategies');
      }

      const data = await response.json();
      setStrategies(data);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching strategies:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  // Fetch recent signals
  const fetchSignals = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategies/signals?limit=20`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch signals');
      }

      const data = await response.json();
      setRecentSignals(data);
    } catch (err) {
      console.error('Error fetching signals:', err);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchStrategies();
    fetchSignals();

    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchStrategies();
        fetchSignals();
      }, REFRESH_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [fetchStrategies, fetchSignals, autoRefresh, REFRESH_INTERVAL]);

  // Control strategy
  const controlStrategy = async (strategyId: string, action: 'start' | 'pause' | 'stop') => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategies/${strategyId}/${action}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error(`Failed to ${action} strategy`);
      }

      await fetchStrategies();
      alert(`Strategy ${action}ed successfully`);
    } catch (err) {
      console.error(`Error ${action}ing strategy:`, err);
      alert(`Failed to ${action} strategy: ${(err as Error).message}`);
    }
  };

  // Delete strategy
  const deleteStrategy = async (strategyId: string) => {
    if (!window.confirm('Are you sure you want to delete this strategy?')) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/strategies/${strategyId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to delete strategy');
      }

      await fetchStrategies();
      setSelectedStrategy(null);
      alert('Strategy deleted successfully');
    } catch (err) {
      console.error('Error deleting strategy:', err);
      alert(`Failed to delete strategy: ${(err as Error).message}`);
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'active': return 'bg-green-600';
      case 'paused': return 'bg-yellow-600';
      case 'stopped': return 'bg-red-600';
      default: return 'bg-gray-600';
    }
  };

  const getStatusTextColor = (status: string): string => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'paused': return 'text-yellow-400';
      case 'stopped': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  if (loading && strategies.length === 0) {
    return (
      <div className="min-h-screen bg-gray-900 p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
          <p className="text-gray-400 mt-4 text-lg">Loading strategies...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Trading Strategies</h1>
            <p className="text-gray-400">Manage and monitor your trading strategies</p>
          </div>
          <div className="flex gap-3">
            <label className="flex items-center gap-2 text-gray-300 bg-gray-800 px-4 py-2 rounded-md">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded bg-gray-700 border-gray-600 text-blue-500"
              />
              <span>Auto Refresh</span>
            </label>
            <button
              onClick={() => {
                fetchStrategies();
                fetchSignals();
              }}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Refresh
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md font-semibold transition-colors"
            >
              + New Strategy
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-300">
            Error: {error.message}
          </div>
        )}
      </div>

      {/* Strategies Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-6">
        {strategies.map((strategy) => (
          <div
            key={strategy.id}
            className={`bg-gray-800/50 border border-gray-700 rounded-lg p-6 transition-all hover:shadow-xl cursor-pointer ${
              selectedStrategy?.id === strategy.id ? 'ring-2 ring-blue-500' : ''
            }`}
            onClick={() => setSelectedStrategy(strategy)}
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-bold text-white mb-1">{strategy.name}</h3>
                <p className="text-sm text-gray-400">{strategy.type}</p>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-bold ${getStatusColor(strategy.status)} text-white uppercase`}>
                {strategy.status}
              </span>
            </div>

            {/* Description */}
            <p className="text-gray-300 text-sm mb-4 line-clamp-2">{strategy.description}</p>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400 text-xs mb-1">Win Rate</div>
                <div className="text-lg font-bold text-green-400">
                  {strategy.metrics.win_rate.toFixed(1)}%
                </div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400 text-xs mb-1">Total P&L</div>
                <div className={`text-lg font-bold ${
                  strategy.metrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  ${strategy.metrics.total_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400 text-xs mb-1">Sharpe</div>
                <div className="text-lg font-bold text-blue-400">
                  {strategy.metrics.sharpe_ratio.toFixed(2)}
                </div>
              </div>
              <div className="bg-gray-700/50 rounded p-2">
                <div className="text-gray-400 text-xs mb-1">Trades</div>
                <div className="text-lg font-bold text-purple-400">
                  {strategy.metrics.total_trades}
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="flex gap-2">
              {strategy.status === 'stopped' && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    controlStrategy(strategy.id, 'start');
                  }}
                  className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
                >
                  Start
                </button>
              )}
              {strategy.status === 'active' && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    controlStrategy(strategy.id, 'pause');
                  }}
                  className="flex-1 px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded transition-colors"
                >
                  Pause
                </button>
              )}
              {strategy.status === 'paused' && (
                <>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      controlStrategy(strategy.id, 'start');
                    }}
                    className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded transition-colors"
                  >
                    Resume
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      controlStrategy(strategy.id, 'stop');
                    }}
                    className="flex-1 px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                  >
                    Stop
                  </button>
                </>
              )}
              {strategy.status !== 'paused' && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteStrategy(strategy.id);
                  }}
                  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm rounded transition-colors"
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Strategy Details */}
      {selectedStrategy && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 mb-6">
          <h2 className="text-2xl font-bold text-white mb-4">Strategy Details: {selectedStrategy.name}</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Info */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-3">Information</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Type:</span>
                  <span className="text-white">{selectedStrategy.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Timeframe:</span>
                  <span className="text-white">{selectedStrategy.timeframe}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Capital Allocated:</span>
                  <span className="text-white">${selectedStrategy.capital_allocated.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Symbols:</span>
                  <span className="text-white">{selectedStrategy.symbols.join(', ')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={`font-semibold ${getStatusTextColor(selectedStrategy.status)}`}>
                    {selectedStrategy.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-3">Performance Metrics</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Trades:</span>
                  <span className="text-white font-mono">{selectedStrategy.metrics.total_trades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Win Rate:</span>
                  <span className="text-green-400 font-mono">{selectedStrategy.metrics.win_rate.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Profit Factor:</span>
                  <span className="text-blue-400 font-mono">{selectedStrategy.metrics.profit_factor.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Sharpe Ratio:</span>
                  <span className="text-purple-400 font-mono">{selectedStrategy.metrics.sharpe_ratio.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Max Drawdown:</span>
                  <span className="text-red-400 font-mono">{selectedStrategy.metrics.max_drawdown.toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Daily P&L:</span>
                  <span className={`font-mono ${
                    selectedStrategy.metrics.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    ${selectedStrategy.metrics.daily_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Signals */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">Recent Signals</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700/50">
              <tr>
                <th className="px-4 py-3 text-left text-gray-300 font-semibold">Time</th>
                <th className="px-4 py-3 text-left text-gray-300 font-semibold">Strategy</th>
                <th className="px-4 py-3 text-left text-gray-300 font-semibold">Symbol</th>
                <th className="px-4 py-3 text-left text-gray-300 font-semibold">Action</th>
                <th className="px-4 py-3 text-right text-gray-300 font-semibold">Strength</th>
                <th className="px-4 py-3 text-right text-gray-300 font-semibold">Confidence</th>
                <th className="px-4 py-3 text-center text-gray-300 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {recentSignals.map((signal, index) => (
                <tr key={index} className="hover:bg-gray-700/30 transition-colors">
                  <td className="px-4 py-3 text-gray-300 text-sm">
                    {new Date(signal.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-4 py-3 text-white">{signal.strategy}</td>
                  <td className="px-4 py-3 text-white font-mono">{signal.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      signal.action === 'BUY' ? 'bg-green-600' :
                      signal.action === 'SELL' ? 'bg-red-600' :
                      'bg-yellow-600'
                    } text-white`}>
                      {signal.action}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-200">
                    {(signal.strength * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-200">
                    {(signal.confidence * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${
                      signal.executed ? 'bg-blue-600' : 'bg-gray-600'
                    } text-white`}>
                      {signal.executed ? 'EXECUTED' : 'PENDING'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Strategies;
