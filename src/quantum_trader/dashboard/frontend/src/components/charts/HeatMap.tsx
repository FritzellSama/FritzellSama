/**
 * HeatMap Component - Performance heatmap visualization
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Calendar-style heatmap for returns, PnL, or other metrics
 */

import React, { useMemo } from 'react';

export interface HeatMapCell {
  /** Row label (e.g., strategy name, symbol) */
  row: string;
  /** Column label (e.g., date, timeframe) */
  column: string;
  /** Numeric value for coloring */
  value: number;
  /** Optional display label */
  label?: string;
}

export interface HeatMapProps {
  /** Heatmap data cells */
  data: HeatMapCell[];
  /** Chart title */
  title?: string;
  /** Minimum value for color scale */
  minValue?: number;
  /** Maximum value for color scale */
  maxValue?: number;
  /** Show cell values */
  showValues?: boolean;
  /** Cell size in pixels */
  cellSize?: number;
  /** Color scheme */
  colorScheme?: 'default' | 'performance' | 'risk';
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get color based on value and color scheme
 */
const getColor = (
  value: number,
  min: number,
  max: number,
  scheme: 'default' | 'performance' | 'risk'
): string => {
  const normalized = (value - min) / (max - min || 1);

  if (scheme === 'performance') {
    // Red (negative) to Green (positive)
    if (normalized < 0.5) {
      const intensity = Math.floor((0.5 - normalized) * 2 * 255);
      return `rgb(${Math.min(239, 200 + intensity)}, ${Math.max(68, 100 - intensity)}, 68)`;
    } else {
      const intensity = Math.floor((normalized - 0.5) * 2 * 255);
      return `rgb(${Math.max(34, 100 - intensity)}, ${Math.min(197, 150 + intensity)}, ${Math.max(94, 100 - intensity)})`;
    }
  } else if (scheme === 'risk') {
    // Green (low) to Red (high)
    const intensity = Math.floor(normalized * 255);
    return `rgb(${Math.min(239, 100 + intensity)}, ${Math.max(68, 200 - intensity)}, 68)`;
  } else {
    // Blue gradient
    const intensity = Math.floor(normalized * 200);
    return `rgb(${Math.max(30, 100 - intensity)}, ${Math.max(82, 150 - intensity)}, ${Math.min(246, 150 + intensity)})`;
  }
};

/**
 * Format value for display
 */
const formatValue = (value: number): string => {
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toFixed(1)}K`;
  } else if (Math.abs(value) >= 1) {
    return value.toFixed(1);
  } else {
    return value.toFixed(2);
  }
};

/**
 * Heatmap visualization component
 *
 * @example
 * ```tsx
 * const data = [
 *   { row: 'Strategy A', column: 'Mon', value: 1250.50 },
 *   { row: 'Strategy A', column: 'Tue', value: -340.20 },
 *   { row: 'Strategy B', column: 'Mon', value: 890.75 },
 *   { row: 'Strategy B', column: 'Tue', value: 1120.30 },
 * ];
 *
 * <HeatMap
 *   data={data}
 *   title="Strategy Performance"
 *   colorScheme="performance"
 *   showValues={true}
 * />
 * ```
 */
export const HeatMap: React.FC<HeatMapProps> = ({
  data,
  title,
  minValue,
  maxValue,
  showValues = true,
  cellSize = 60,
  colorScheme = 'default',
  className = '',
}) => {
  // Process data into grid structure
  const { rows, columns, grid, min, max } = useMemo(() => {
    if (!data || data.length === 0) {
      return { rows: [], columns: [], grid: new Map(), min: 0, max: 0 };
    }

    const rowSet = new Set<string>();
    const columnSet = new Set<string>();
    const gridMap = new Map<string, HeatMapCell>();

    let dataMin = minValue ?? Infinity;
    let dataMax = maxValue ?? -Infinity;

    data.forEach((cell) => {
      rowSet.add(cell.row);
      columnSet.add(cell.column);
      gridMap.set(`${cell.row}:${cell.column}`, cell);

      if (minValue === undefined) {
        dataMin = Math.min(dataMin, cell.value);
      }
      if (maxValue === undefined) {
        dataMax = Math.max(dataMax, cell.value);
      }
    });

    return {
      rows: Array.from(rowSet),
      columns: Array.from(columnSet),
      grid: gridMap,
      min: minValue ?? dataMin,
      max: maxValue ?? dataMax,
    };
  }, [data, minValue, maxValue]);

  if (!data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center bg-gray-800/50 rounded p-8 ${className}`}>
        <p className="text-gray-500">No heatmap data available</p>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800/50 rounded-lg p-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-100 mb-4">{title}</h3>
      )}

      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header row */}
          <div className="flex">
            <div
              className="flex-shrink-0 flex items-center justify-end pr-2"
              style={{ width: `${cellSize * 2}px` }}
            />
            {columns.map((col) => (
              <div
                key={col}
                className="flex-shrink-0 flex items-center justify-center text-xs text-gray-400 font-medium"
                style={{ width: `${cellSize}px` }}
              >
                {col}
              </div>
            ))}
          </div>

          {/* Data rows */}
          {rows.map((row) => (
            <div key={row} className="flex mt-1">
              {/* Row label */}
              <div
                className="flex-shrink-0 flex items-center justify-end pr-2 text-xs text-gray-400 font-medium truncate"
                style={{ width: `${cellSize * 2}px` }}
                title={row}
              >
                {row}
              </div>

              {/* Cells */}
              {columns.map((col) => {
                const cell = grid.get(`${row}:${col}`);
                const value = cell?.value ?? 0;
                const bgColor = cell ? getColor(value, min, max, colorScheme) : '#1f2937';

                return (
                  <div
                    key={`${row}:${col}`}
                    className="flex-shrink-0 mx-0.5 rounded flex items-center justify-center text-xs font-medium transition-all hover:scale-110 hover:z-10 cursor-pointer"
                    style={{
                      width: `${cellSize - 4}px`,
                      height: `${cellSize - 4}px`,
                      backgroundColor: bgColor,
                    }}
                    title={cell ? `${row} - ${col}: ${value.toFixed(2)}` : 'No data'}
                  >
                    {showValues && cell && (
                      <span className="text-white text-shadow">
                        {cell.label || formatValue(value)}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center justify-center space-x-4">
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">Min:</span>
          <span className="text-sm font-mono text-gray-300">{formatValue(min)}</span>
        </div>
        <div className="flex items-center space-x-1">
          {[0, 0.25, 0.5, 0.75, 1].map((normalized) => {
            const value = min + (max - min) * normalized;
            return (
              <div
                key={normalized}
                className="w-8 h-4 rounded"
                style={{
                  backgroundColor: getColor(value, min, max, colorScheme),
                }}
                title={formatValue(value)}
              />
            );
          })}
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-500">Max:</span>
          <span className="text-sm font-mono text-gray-300">{formatValue(max)}</span>
        </div>
      </div>
    </div>
  );
};

export default HeatMap;
