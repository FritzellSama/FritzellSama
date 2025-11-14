/**
 * DrawdownChart Component - Portfolio drawdown visualization
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Displays underwater equity curve showing drawdown periods
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

export interface DrawdownDataPoint {
  /** Timestamp or label */
  time: string;
  /** Drawdown percentage (negative value) */
  drawdown: number;
  /** Portfolio equity value */
  equity?: number;
  /** Peak equity value */
  peak?: number;
}

export interface DrawdownChartProps {
  /** Drawdown data points */
  data: DrawdownDataPoint[];
  /** Chart height in pixels */
  height?: number;
  /** Show grid lines */
  showGrid?: boolean;
  /** Drawdown warning threshold (%) */
  warningThreshold?: number;
  /** Drawdown danger threshold (%) */
  dangerThreshold?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Custom tooltip for drawdown chart
 */
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0].payload;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      <div className="space-y-1 text-xs">
        <div className="flex justify-between space-x-4">
          <span className="text-gray-500">Drawdown:</span>
          <span className="text-red-400 font-mono font-semibold">
            {data.drawdown.toFixed(2)}%
          </span>
        </div>
        {data.equity !== undefined && (
          <div className="flex justify-between space-x-4">
            <span className="text-gray-500">Equity:</span>
            <span className="text-gray-200 font-mono">
              ${data.equity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
        )}
        {data.peak !== undefined && (
          <div className="flex justify-between space-x-4">
            <span className="text-gray-500">Peak:</span>
            <span className="text-blue-400 font-mono">
              ${data.peak.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Drawdown chart visualization
 *
 * @example
 * ```tsx
 * const data = [
 *   { time: '2024-01-01', drawdown: 0, equity: 100000, peak: 100000 },
 *   { time: '2024-01-02', drawdown: -2.5, equity: 97500, peak: 100000 },
 *   { time: '2024-01-03', drawdown: -1.2, equity: 98800, peak: 100000 },
 *   { time: '2024-01-04', drawdown: 0, equity: 101000, peak: 101000 },
 * ];
 *
 * <DrawdownChart
 *   data={data}
 *   height={300}
 *   warningThreshold={-10}
 *   dangerThreshold={-20}
 * />
 * ```
 */
export const DrawdownChart: React.FC<DrawdownChartProps> = ({
  data,
  height = 300,
  showGrid = true,
  warningThreshold = -10,
  dangerThreshold = -20,
  className = '',
}) => {
  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;

    const maxDrawdown = Math.min(...data.map((d) => d.drawdown));
    const currentDrawdown = data[data.length - 1].drawdown;

    // Calculate drawdown duration
    let currentDuration = 0;
    let maxDuration = 0;
    let tempDuration = 0;

    for (let i = data.length - 1; i >= 0; i--) {
      if (data[i].drawdown < 0) {
        tempDuration++;
        if (i === data.length - 1) {
          currentDuration = tempDuration;
        }
      } else {
        maxDuration = Math.max(maxDuration, tempDuration);
        tempDuration = 0;
      }
    }
    maxDuration = Math.max(maxDuration, tempDuration);

    // Count periods in warning/danger zones
    const warningPeriods = data.filter(
      (d) => d.drawdown <= warningThreshold && d.drawdown > dangerThreshold
    ).length;
    const dangerPeriods = data.filter((d) => d.drawdown <= dangerThreshold).length;

    return {
      maxDrawdown,
      currentDrawdown,
      currentDuration,
      maxDuration,
      warningPeriods,
      dangerPeriods,
    };
  }, [data, warningThreshold, dangerThreshold]);

  if (!data || data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800/50 rounded ${className}`}
        style={{ height }}
      >
        <p className="text-gray-500">No drawdown data available</p>
      </div>
    );
  }

  return (
    <div className={className}>
      {/* Stats Header */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Current Drawdown</span>
            <span
              className={`text-lg font-semibold ${
                stats.currentDrawdown === 0
                  ? 'text-green-400'
                  : stats.currentDrawdown > warningThreshold
                  ? 'text-yellow-400'
                  : stats.currentDrawdown > dangerThreshold
                  ? 'text-orange-400'
                  : 'text-red-400'
              }`}
            >
              {stats.currentDrawdown.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Max Drawdown</span>
            <span className="text-lg font-semibold text-red-400">
              {stats.maxDrawdown.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Current Duration</span>
            <span className="text-lg font-semibold text-gray-300">
              {stats.currentDuration} periods
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Max Duration</span>
            <span className="text-lg font-semibold text-gray-300">
              {stats.maxDuration} periods
            </span>
          </div>
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={data}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
          )}
          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            tick={{ fill: '#9ca3af' }}
          />
          <YAxis
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            tick={{ fill: '#9ca3af' }}
            tickFormatter={(value) => `${value.toFixed(0)}%`}
            domain={['auto', 0]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }}
            iconType="circle"
          />

          {/* Zero line */}
          <ReferenceLine
            y={0}
            stroke="#6b7280"
            strokeWidth={2}
            label={{ value: 'Peak', fill: '#9ca3af', fontSize: 12 }}
          />

          {/* Warning threshold */}
          {warningThreshold && (
            <ReferenceLine
              y={warningThreshold}
              stroke="#f59e0b"
              strokeDasharray="5 5"
              label={{ value: 'Warning', fill: '#f59e0b', fontSize: 11 }}
            />
          )}

          {/* Danger threshold */}
          {dangerThreshold && (
            <ReferenceLine
              y={dangerThreshold}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{ value: 'Danger', fill: '#ef4444', fontSize: 11 }}
            />
          )}

          {/* Drawdown area */}
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#colorDrawdown)"
            name="Drawdown %"
          />

          {/* Gradient definitions */}
          <defs>
            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
            </linearGradient>
          </defs>
        </AreaChart>
      </ResponsiveContainer>

      {/* Risk indicators */}
      {stats && (stats.warningPeriods > 0 || stats.dangerPeriods > 0) && (
        <div className="mt-4 flex items-center justify-center space-x-6 text-sm">
          {stats.warningPeriods > 0 && (
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-yellow-400" />
              <span className="text-gray-400">
                Warning: {stats.warningPeriods} periods
              </span>
            </div>
          )}
          {stats.dangerPeriods > 0 && (
            <div className="flex items-center space-x-2">
              <div className="h-3 w-3 rounded-full bg-red-400" />
              <span className="text-gray-400">
                Danger: {stats.dangerPeriods} periods
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DrawdownChart;
