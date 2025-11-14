/**
 * RiskManagement Page
 *
 * Comprehensive risk management dashboard for monitoring and controlling trading risk.
 * Production-ready component for institutional trading platform.
 */

import React, { useState, useEffect, useCallback } from 'react';
import RiskMetrics from '../components/analytics/RiskMetrics';

interface PortfolioRisk {
  total_var: number;
  total_cvar: number;
  max_drawdown: number;
  current_drawdown: number;
  total_exposure: number;
  leverage: number;
  margin_usage: number;
  timestamp: string;
}

interface PositionRisk {
  symbol: string;
  quantity: number;
  market_value: number;
  unrealized_pnl: number;
  var_95: number;
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  strategy: string;
}

interface RiskLimit {
  metric: string;
  current: number;
  limit: number;
  threshold: number;
  status: 'safe' | 'warning' | 'critical';
}

const RiskManagement: React.FC = () => {
  const [portfolioRisk, setPortfolioRisk] = useState<PortfolioRisk | null>(null);
  const [positionRisks, setPositionRisks] = useState<PositionRisk[]>([]);
  const [riskLimits, setRiskLimits] = useState<RiskLimit[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [emergencyMode, setEmergencyMode] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_RISK_REFRESH_INTERVAL || '5000');

  // Fetch risk data
  const fetchRiskData = useCallback(async () => {
    try {
      const [portfolioRes, positionsRes, limitsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/risk/portfolio`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch(`${API_BASE_URL}/api/v1/risk/positions`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch(`${API_BASE_URL}/api/v1/risk/limits`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        })
      ]);

      if (!portfolioRes.ok || !positionsRes.ok || !limitsRes.ok) {
        throw new Error('Failed to fetch risk data');
      }

      const [portfolio, positions, limits] = await Promise.all([
        portfolioRes.json(),
        positionsRes.json(),
        limitsRes.json()
      ]);

      setPortfolioRisk(portfolio);
      setPositionRisks(positions);
      setRiskLimits(limits);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching risk data:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  // Auto-refresh
  useEffect(() => {
    fetchRiskData();

    if (autoRefresh) {
      const interval = setInterval(fetchRiskData, REFRESH_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [fetchRiskData, autoRefresh, REFRESH_INTERVAL]);

  // Emergency stop
  const handleEmergencyStop = async () => {
    if (!window.confirm('Are you sure you want to trigger EMERGENCY STOP? This will halt all trading and close positions.')) {
      return;
    }

    try {
      setEmergencyMode(true);
      const response = await fetch(`${API_BASE_URL}/api/v1/risk/emergency-stop`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to trigger emergency stop');
      }

      alert('Emergency stop activated successfully');
    } catch (err) {
      console.error('Emergency stop failed:', err);
      alert(`Emergency stop failed: ${(err as Error).message}`);
    }
  };

  // Close specific position
  const handleClosePosition = async (symbol: string) => {
    if (!window.confirm(`Close position for ${symbol}?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/positions/${symbol}/close`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to close position');
      }

      await fetchRiskData();
      alert(`Position ${symbol} closed successfully`);
    } catch (err) {
      console.error('Close position failed:', err);
      alert(`Failed to close position: ${(err as Error).message}`);
    }
  };

  const getRiskLimitColor = (status: string): string => {
    switch (status) {
      case 'critical': return 'text-red-400 bg-red-900/20 border-red-700';
      case 'warning': return 'text-yellow-400 bg-yellow-900/20 border-yellow-700';
      default: return 'text-green-400 bg-green-900/20 border-green-700';
    }
  };

  if (loading && !portfolioRisk) {
    return (
      <div className="min-h-screen bg-gray-900 p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
          <p className="text-gray-400 mt-4 text-lg">Loading risk data...</p>
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
            <h1 className="text-3xl font-bold text-white mb-2">Risk Management</h1>
            <p className="text-gray-400">Real-time risk monitoring and controls</p>
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
              onClick={fetchRiskData}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Refresh Now
            </button>
            <button
              onClick={handleEmergencyStop}
              disabled={emergencyMode}
              className={`px-6 py-2 ${
                emergencyMode ? 'bg-gray-700 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700'
              } text-white rounded-md font-bold transition-colors`}
            >
              {emergencyMode ? 'STOPPED' : 'EMERGENCY STOP'}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-300">
            Error: {error.message}
          </div>
        )}
      </div>

      {/* Portfolio Risk Metrics */}
      {portfolioRisk && (
        <div className="mb-6">
          <RiskMetrics
            var95={portfolioRisk.total_var}
            cvar95={portfolioRisk.total_cvar}
            maxDrawdown={portfolioRisk.max_drawdown}
            currentDrawdown={portfolioRisk.current_drawdown}
            exposure={portfolioRisk.total_exposure}
            leverage={portfolioRisk.leverage}
            onRefresh={fetchRiskData}
          />
        </div>
      )}

      {/* Risk Limits */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white mb-4">Risk Limits</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {riskLimits.map((limit, index) => (
            <div
              key={index}
              className={`border rounded-lg p-4 ${getRiskLimitColor(limit.status)}`}
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold">{limit.metric}</h3>
                <span className={`px-2 py-1 rounded text-xs font-bold uppercase ${
                  limit.status === 'critical' ? 'bg-red-700' :
                  limit.status === 'warning' ? 'bg-yellow-700' :
                  'bg-green-700'
                }`}>
                  {limit.status}
                </span>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Current:</span>
                  <span className="font-mono font-bold">{limit.current.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Threshold:</span>
                  <span className="font-mono">{limit.threshold.toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Limit:</span>
                  <span className="font-mono">{limit.limit.toFixed(2)}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                  <div
                    className={`h-2 rounded-full ${
                      limit.status === 'critical' ? 'bg-red-500' :
                      limit.status === 'warning' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${Math.min((limit.current / limit.limit) * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Position Risks */}
      <div>
        <h2 className="text-xl font-bold text-white mb-4">Position Risks</h2>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-gray-300 font-semibold">Symbol</th>
                  <th className="px-4 py-3 text-left text-gray-300 font-semibold">Strategy</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Quantity</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Market Value</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Unrealized P&L</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">VaR 95%</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Delta</th>
                  <th className="px-4 py-3 text-center text-gray-300 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {positionRisks.map((position, index) => (
                  <tr key={index} className="hover:bg-gray-700/30 transition-colors">
                    <td className="px-4 py-3 text-white font-mono">{position.symbol}</td>
                    <td className="px-4 py-3 text-gray-300">{position.strategy}</td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      {position.quantity.toFixed(8)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      ${position.market_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className={`px-4 py-3 text-right font-mono font-bold ${
                      position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.unrealized_pnl >= 0 ? '+' : ''}
                      ${position.unrealized_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-red-400">
                      ${Math.abs(position.var_95).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      {position.delta.toFixed(4)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => handleClosePosition(position.symbol)}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskManagement;
