/**
 * EquityChart Component - Portfolio equity curve
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Real-time portfolio equity tracking with profit zones
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

export interface EquityDataPoint {
  /** Timestamp or label */
  time: string;
  /** Portfolio equity value */
  equity: number;
  /** Benchmark equity (optional) */
  benchmark?: number;
  /** Daily PnL (optional) */
  pnl?: number;
}

export interface EquityChartProps {
  /** Equity data points */
  data: EquityDataPoint[];
  /** Chart height in pixels */
  height?: number;
  /** Show benchmark comparison */
  showBenchmark?: boolean;
  /** Show grid lines */
  showGrid?: boolean;
  /** Chart type */
  chartType?: 'area' | 'line';
  /** Initial equity value for reference */
  initialEquity?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Custom tooltip for equity chart
 */
const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  const equityData = payload.find((p: any) => p.dataKey === 'equity');
  const benchmarkData = payload.find((p: any) => p.dataKey === 'benchmark');
  const pnlData = payload.find((p: any) => p.dataKey === 'pnl');

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-sm text-gray-400 mb-2">{label}</p>
      <div className="space-y-1 text-xs">
        {equityData && (
          <div className="flex justify-between space-x-4">
            <span className="text-gray-500">Equity:</span>
            <span className="text-blue-400 font-mono font-semibold">
              ${equityData.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
        )}
        {benchmarkData && (
          <div className="flex justify-between space-x-4">
            <span className="text-gray-500">Benchmark:</span>
            <span className="text-purple-400 font-mono">
              ${benchmarkData.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
        )}
        {pnlData !== undefined && (
          <div className="flex justify-between space-x-4 pt-1 border-t border-gray-700">
            <span className="text-gray-500">P&L:</span>
            <span className={`font-mono ${pnlData.value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {pnlData.value >= 0 ? '+' : ''}${pnlData.value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Portfolio equity curve chart
 *
 * @example
 * ```tsx
 * const data = [
 *   { time: '2024-01-01', equity: 100000, benchmark: 100000, pnl: 0 },
 *   { time: '2024-01-02', equity: 102500, benchmark: 100800, pnl: 2500 },
 *   { time: '2024-01-03', equity: 105200, benchmark: 101200, pnl: 2700 },
 * ];
 *
 * <EquityChart
 *   data={data}
 *   height={400}
 *   showBenchmark={true}
 *   initialEquity={100000}
 * />
 * ```
 */
export const EquityChart: React.FC<EquityChartProps> = ({
  data,
  height = 400,
  showBenchmark = false,
  showGrid = true,
  chartType = 'area',
  initialEquity,
  className = '',
}) => {
  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;

    const currentEquity = data[data.length - 1].equity;
    const startEquity = initialEquity || data[0].equity;
    const totalReturn = currentEquity - startEquity;
    const returnPercent = ((currentEquity - startEquity) / startEquity) * 100;

    const maxEquity = Math.max(...data.map((d) => d.equity));
    const maxDrawdown = data.reduce((max, point) => {
      const dd = ((point.equity - maxEquity) / maxEquity) * 100;
      return dd < max ? dd : max;
    }, 0);

    return {
      currentEquity,
      totalReturn,
      returnPercent,
      maxDrawdown,
    };
  }, [data, initialEquity]);

  if (!data || data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800/50 rounded ${className}`}
        style={{ height }}
      >
        <p className="text-gray-500">No equity data available</p>
      </div>
    );
  }

  const ChartComponent = chartType === 'area' ? AreaChart : LineChart;

  return (
    <div className={className}>
      {/* Stats Header */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Current Equity</span>
            <span className="text-lg font-semibold text-blue-400">
              ${stats.currentEquity.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Total Return</span>
            <span className={`text-lg font-semibold ${stats.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats.totalReturn >= 0 ? '+' : ''}${stats.totalReturn.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Return %</span>
            <span className={`text-lg font-semibold ${stats.returnPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats.returnPercent >= 0 ? '+' : ''}{stats.returnPercent.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Max Drawdown</span>
            <span className="text-lg font-semibold text-red-400">
              {stats.maxDrawdown.toFixed(2)}%
            </span>
          </div>
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent
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
            tickFormatter={(value) => `$${(value / 1000).toFixed(0)}K`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }}
            iconType="circle"
          />

          {/* Initial equity reference line */}
          {initialEquity && (
            <ReferenceLine
              y={initialEquity}
              stroke="#6b7280"
              strokeDasharray="3 3"
              label={{ value: 'Initial', fill: '#9ca3af', fontSize: 12 }}
            />
          )}

          {/* Equity line/area */}
          {chartType === 'area' ? (
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#colorEquity)"
              name="Equity"
            />
          ) : (
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              name="Equity"
            />
          )}

          {/* Benchmark comparison */}
          {showBenchmark && (
            chartType === 'area' ? (
              <Area
                type="monotone"
                dataKey="benchmark"
                stroke="#a855f7"
                strokeWidth={1.5}
                fill="url(#colorBenchmark)"
                name="Benchmark"
              />
            ) : (
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="#a855f7"
                strokeWidth={1.5}
                strokeDasharray="5 5"
                dot={false}
                name="Benchmark"
              />
            )
          )}

          {/* Gradient definitions */}
          <defs>
            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorBenchmark" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
            </linearGradient>
          </defs>
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
};

export default EquityChart;
