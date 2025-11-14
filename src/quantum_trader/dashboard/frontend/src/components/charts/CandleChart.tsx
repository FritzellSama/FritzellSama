/**
 * CandleChart Component - Professional candlestick chart
 *
 * Quantum Trader AI - Institutional Trading Platform
 * High-performance OHLC candlestick chart with volume overlay
 */

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts';

export interface CandleData {
  /** Timestamp or label */
  time: string;
  /** Open price */
  open: number;
  /** High price */
  high: number;
  /** Low price */
  low: number;
  /** Close price */
  close: number;
  /** Trading volume */
  volume?: number;
}

export interface CandleChartProps {
  /** Candlestick data array */
  data: CandleData[];
  /** Chart height in pixels */
  height?: number;
  /** Show volume bars */
  showVolume?: boolean;
  /** Show grid lines */
  showGrid?: boolean;
  /** Up candle color */
  upColor?: string;
  /** Down candle color */
  downColor?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Custom candlestick shape component
 */
const CandleStick: React.FC<any> = (props) => {
  const { x, y, width, height, payload, upColor, downColor } = props;

  if (!payload) return null;

  const { open, close, high, low } = payload;
  const isUp = close >= open;
  const color = isUp ? upColor : downColor;

  const wickX = x + width / 2;
  const bodyTop = Math.min(open, close);
  const bodyBottom = Math.max(open, close);
  const bodyHeight = Math.abs(close - open);

  // Calculate positions (inverted Y axis)
  const chartHeight = 300;
  const maxPrice = Math.max(high, open, close, low);
  const minPrice = Math.min(high, open, close, low);
  const range = maxPrice - minPrice || 1;

  const getY = (price: number) => {
    return chartHeight - ((price - minPrice) / range) * chartHeight;
  };

  return (
    <g>
      {/* Wick */}
      <line
        x1={wickX}
        y1={getY(high)}
        x2={wickX}
        y2={getY(low)}
        stroke={color}
        strokeWidth={1}
      />
      {/* Body */}
      <rect
        x={x}
        y={getY(bodyBottom)}
        width={width}
        height={Math.max(bodyHeight * (chartHeight / range), 1)}
        fill={color}
        stroke={color}
        strokeWidth={1}
      />
    </g>
  );
};

/**
 * Custom tooltip for candlestick chart
 */
const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0].payload;
  const isUp = data.close >= data.open;

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-sm text-gray-400 mb-2">{data.time}</p>
      <div className="space-y-1 text-xs">
        <div className="flex justify-between space-x-4">
          <span className="text-gray-500">Open:</span>
          <span className="text-gray-200 font-mono">${data.open.toFixed(2)}</span>
        </div>
        <div className="flex justify-between space-x-4">
          <span className="text-gray-500">High:</span>
          <span className="text-green-400 font-mono">${data.high.toFixed(2)}</span>
        </div>
        <div className="flex justify-between space-x-4">
          <span className="text-gray-500">Low:</span>
          <span className="text-red-400 font-mono">${data.low.toFixed(2)}</span>
        </div>
        <div className="flex justify-between space-x-4">
          <span className="text-gray-500">Close:</span>
          <span className={`font-mono ${isUp ? 'text-green-400' : 'text-red-400'}`}>
            ${data.close.toFixed(2)}
          </span>
        </div>
        {data.volume !== undefined && (
          <div className="flex justify-between space-x-4 pt-1 border-t border-gray-700">
            <span className="text-gray-500">Volume:</span>
            <span className="text-gray-200 font-mono">
              {data.volume.toLocaleString()}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Candlestick chart with optional volume overlay
 *
 * @example
 * ```tsx
 * const data = [
 *   { time: '09:00', open: 45000, high: 45500, low: 44800, close: 45200, volume: 1200 },
 *   { time: '10:00', open: 45200, high: 45800, low: 45100, close: 45600, volume: 1500 },
 * ];
 *
 * <CandleChart
 *   data={data}
 *   height={400}
 *   showVolume={true}
 * />
 * ```
 */
export const CandleChart: React.FC<CandleChartProps> = ({
  data,
  height = 400,
  showVolume = true,
  showGrid = true,
  upColor = '#22c55e',
  downColor = '#ef4444',
  className = '',
}) => {
  // Transform data for chart rendering
  const chartData = useMemo(() => {
    return data.map((item) => ({
      ...item,
      candleColor: item.close >= item.open ? upColor : downColor,
    }));
  }, [data, upColor, downColor]);

  if (!data || data.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-800/50 rounded ${className}`}
        style={{ height }}
      >
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={chartData}
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
            yAxisId="price"
            domain={['auto', 'auto']}
            stroke="#9ca3af"
            style={{ fontSize: '12px' }}
            tick={{ fill: '#9ca3af' }}
            tickFormatter={(value) => `$${value.toFixed(0)}`}
          />
          {showVolume && (
            <YAxis
              yAxisId="volume"
              orientation="right"
              stroke="#9ca3af"
              style={{ fontSize: '12px' }}
              tick={{ fill: '#9ca3af' }}
              tickFormatter={(value) => `${(value / 1000).toFixed(0)}K`}
            />
          )}
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: '#9ca3af' }}
            iconType="circle"
          />

          {/* Volume bars */}
          {showVolume && (
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="#4b5563"
              opacity={0.3}
              name="Volume"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.candleColor} opacity={0.2} />
              ))}
            </Bar>
          )}

          {/* Candlestick representation using bars */}
          <Bar
            yAxisId="price"
            dataKey="high"
            fill="transparent"
            name="Price"
          >
            {chartData.map((entry, index) => (
              <Cell key={`candle-${index}`} fill={entry.candleColor} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CandleChart;
