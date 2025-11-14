/**
 * RiskMetrics Component
 *
 * Displays key risk metrics with visual indicators and alerts.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useMemo } from 'react';

interface RiskMetricsProps {
  var95: number;
  cvar95?: number;
  maxDrawdown: number;
  currentDrawdown: number;
  exposure: number;
  leverage: number;
  sharpeRatio?: number;
  sortinoRatio?: number;
  marginUsage?: number;
  onRefresh?: () => void;
  loading?: boolean;
}

interface MetricConfig {
  label: string;
  value: number;
  format: 'currency' | 'percentage' | 'ratio';
  threshold?: {
    warning: number;
    critical: number;
  };
  inverse?: boolean; // True if lower is better
}

const RiskMetrics: React.FC<RiskMetricsProps> = ({
  var95,
  cvar95,
  maxDrawdown,
  currentDrawdown,
  exposure,
  leverage,
  sharpeRatio,
  sortinoRatio,
  marginUsage,
  onRefresh,
  loading = false
}) => {
  // Load thresholds from environment
  const MAX_LEVERAGE = parseFloat(process.env.REACT_APP_MAX_LEVERAGE || '10');
  const MAX_DRAWDOWN_WARNING = parseFloat(process.env.REACT_APP_MAX_DRAWDOWN_WARNING || '10');
  const MAX_DRAWDOWN_CRITICAL = parseFloat(process.env.REACT_APP_MAX_DRAWDOWN_CRITICAL || '20');
  const MAX_VAR_WARNING = parseFloat(process.env.REACT_APP_MAX_VAR_WARNING || '50000');
  const MAX_VAR_CRITICAL = parseFloat(process.env.REACT_APP_MAX_VAR_CRITICAL || '100000');

  const metrics = useMemo((): MetricConfig[] => {
    const baseMetrics: MetricConfig[] = [
      {
        label: 'Value at Risk (95%)',
        value: var95,
        format: 'currency',
        threshold: {
          warning: MAX_VAR_WARNING,
          critical: MAX_VAR_CRITICAL
        },
        inverse: true
      },
      {
        label: 'Max Drawdown',
        value: maxDrawdown,
        format: 'percentage',
        threshold: {
          warning: MAX_DRAWDOWN_WARNING,
          critical: MAX_DRAWDOWN_CRITICAL
        },
        inverse: true
      },
      {
        label: 'Current Drawdown',
        value: currentDrawdown,
        format: 'percentage',
        threshold: {
          warning: MAX_DRAWDOWN_WARNING * 0.5,
          critical: MAX_DRAWDOWN_WARNING
        },
        inverse: true
      },
      {
        label: 'Total Exposure',
        value: exposure,
        format: 'currency'
      },
      {
        label: 'Leverage',
        value: leverage,
        format: 'ratio',
        threshold: {
          warning: MAX_LEVERAGE * 0.8,
          critical: MAX_LEVERAGE
        },
        inverse: true
      }
    ];

    if (cvar95 !== undefined) {
      baseMetrics.push({
        label: 'CVaR (95%)',
        value: cvar95,
        format: 'currency',
        threshold: {
          warning: MAX_VAR_WARNING * 1.2,
          critical: MAX_VAR_CRITICAL * 1.2
        },
        inverse: true
      });
    }

    if (sharpeRatio !== undefined) {
      baseMetrics.push({
        label: 'Sharpe Ratio',
        value: sharpeRatio,
        format: 'ratio',
        threshold: {
          warning: 1.0,
          critical: 0.5
        }
      });
    }

    if (sortinoRatio !== undefined) {
      baseMetrics.push({
        label: 'Sortino Ratio',
        value: sortinoRatio,
        format: 'ratio',
        threshold: {
          warning: 1.5,
          critical: 0.75
        }
      });
    }

    if (marginUsage !== undefined) {
      baseMetrics.push({
        label: 'Margin Usage',
        value: marginUsage,
        format: 'percentage',
        threshold: {
          warning: 70,
          critical: 85
        },
        inverse: true
      });
    }

    return baseMetrics;
  }, [var95, cvar95, maxDrawdown, currentDrawdown, exposure, leverage, sharpeRatio, sortinoRatio, marginUsage, MAX_LEVERAGE, MAX_DRAWDOWN_WARNING, MAX_DRAWDOWN_CRITICAL, MAX_VAR_WARNING, MAX_VAR_CRITICAL]);

  const formatValue = (value: number, format: 'currency' | 'percentage' | 'ratio'): string => {
    switch (format) {
      case 'currency':
        return `$${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      case 'percentage':
        return `${value.toFixed(2)}%`;
      case 'ratio':
        return value.toFixed(3);
      default:
        return value.toString();
    }
  };

  const getMetricStatus = (metric: MetricConfig): 'safe' | 'warning' | 'critical' => {
    if (!metric.threshold) return 'safe';

    const absValue = Math.abs(metric.value);

    if (metric.inverse) {
      // For inverse metrics (lower is better)
      if (absValue >= metric.threshold.critical) return 'critical';
      if (absValue >= metric.threshold.warning) return 'warning';
      return 'safe';
    } else {
      // For normal metrics (higher is better)
      if (metric.value <= metric.threshold.critical) return 'critical';
      if (metric.value <= metric.threshold.warning) return 'warning';
      return 'safe';
    }
  };

  const getStatusColor = (status: 'safe' | 'warning' | 'critical'): string => {
    switch (status) {
      case 'critical':
        return 'border-red-500 bg-red-900/20';
      case 'warning':
        return 'border-yellow-500 bg-yellow-900/20';
      default:
        return 'border-green-500 bg-green-900/20';
    }
  };

  const getTextColor = (status: 'safe' | 'warning' | 'critical'): string => {
    switch (status) {
      case 'critical':
        return 'text-red-400';
      case 'warning':
        return 'text-yellow-400';
      default:
        return 'text-green-400';
    }
  };

  const getStatusIcon = (status: 'safe' | 'warning' | 'critical'): string => {
    switch (status) {
      case 'critical':
        return '🔴';
      case 'warning':
        return '🟡';
      default:
        return '🟢';
    }
  };

  // Overall risk assessment
  const overallRisk = useMemo(() => {
    const statuses = metrics.map(getMetricStatus);
    if (statuses.includes('critical')) return 'critical';
    if (statuses.includes('warning')) return 'warning';
    return 'safe';
  }, [metrics]);

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Risk Metrics</h2>
          <div className="flex items-center gap-2">
            <span className="text-gray-400">Overall Status:</span>
            <span className={`font-bold ${getTextColor(overallRisk)}`}>
              {getStatusIcon(overallRisk)} {overallRisk.toUpperCase()}
            </span>
          </div>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={loading}
            className={`px-4 py-2 ${
              loading ? 'bg-gray-700 cursor-not-allowed' : 'bg-gray-700 hover:bg-gray-600'
            } text-gray-200 rounded-md transition-colors`}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {metrics.map((metric, index) => {
          const status = getMetricStatus(metric);
          const statusColor = getStatusColor(status);
          const textColor = getTextColor(status);

          return (
            <div
              key={index}
              className={`border-l-4 rounded-lg p-4 bg-gray-700/30 ${statusColor} transition-all hover:shadow-lg`}
            >
              <div className="flex justify-between items-start mb-2">
                <h3 className="text-gray-300 text-sm font-medium">{metric.label}</h3>
                <span className="text-lg">{getStatusIcon(status)}</span>
              </div>

              <div className={`text-3xl font-bold mb-2 ${textColor}`}>
                {formatValue(metric.value, metric.format)}
              </div>

              {metric.threshold && (
                <div className="space-y-1 text-xs text-gray-400">
                  <div className="flex justify-between">
                    <span>Warning:</span>
                    <span className="font-mono">{formatValue(metric.threshold.warning, metric.format)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Critical:</span>
                    <span className="font-mono">{formatValue(metric.threshold.critical, metric.format)}</span>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full bg-gray-600 rounded-full h-2 mt-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        status === 'critical' ? 'bg-red-500' :
                        status === 'warning' ? 'bg-yellow-500' :
                        'bg-green-500'
                      }`}
                      style={{
                        width: `${Math.min(
                          (Math.abs(metric.value) / metric.threshold.critical) * 100,
                          100
                        )}%`
                      }}
                    ></div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Risk Alerts */}
      {overallRisk !== 'safe' && (
        <div className={`mt-6 p-4 rounded-lg border ${
          overallRisk === 'critical'
            ? 'bg-red-900/20 border-red-700'
            : 'bg-yellow-900/20 border-yellow-700'
        }`}>
          <div className="flex items-start gap-3">
            <span className="text-2xl">{overallRisk === 'critical' ? '⚠️' : '⚡'}</span>
            <div>
              <h3 className={`font-bold mb-1 ${
                overallRisk === 'critical' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                {overallRisk === 'critical' ? 'CRITICAL RISK ALERT' : 'Risk Warning'}
              </h3>
              <p className={overallRisk === 'critical' ? 'text-red-300' : 'text-yellow-300'}>
                {overallRisk === 'critical'
                  ? 'One or more risk metrics have exceeded critical thresholds. Immediate action may be required.'
                  : 'Some risk metrics are approaching warning thresholds. Monitor closely.'}
              </p>
              <ul className="mt-2 space-y-1 text-sm">
                {metrics.map((metric, index) => {
                  const status = getMetricStatus(metric);
                  if (status !== 'safe') {
                    return (
                      <li key={index} className="flex items-center gap-2">
                        <span>{getStatusIcon(status)}</span>
                        <span>{metric.label}: {formatValue(metric.value, metric.format)}</span>
                      </li>
                    );
                  }
                  return null;
                })}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RiskMetrics;
