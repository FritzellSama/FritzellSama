/**
 * CorrelationMatrix Component - Asset correlation visualization
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Displays correlation matrix between trading pairs or strategies
 */

import React, { useMemo } from 'react';

export interface CorrelationData {
  /** Asset/strategy names in order */
  labels: string[];
  /** Correlation matrix (2D array, values -1 to 1) */
  matrix: number[][];
}

export interface CorrelationMatrixProps {
  /** Correlation data */
  data: CorrelationData;
  /** Chart title */
  title?: string;
  /** Cell size in pixels */
  cellSize?: number;
  /** Show correlation values */
  showValues?: boolean;
  /** Highlight threshold (absolute value) */
  highlightThreshold?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get color based on correlation value (-1 to 1)
 */
const getCorrelationColor = (value: number): string => {
  // Normalize from [-1, 1] to [0, 1]
  const normalized = (value + 1) / 2;

  if (value > 0) {
    // Positive correlation: white to green
    const intensity = Math.floor(normalized * 255);
    return `rgb(${Math.max(34, 150 - intensity)}, ${Math.min(197, 100 + intensity * 0.8)}, ${Math.max(94, 150 - intensity)})`;
  } else if (value < 0) {
    // Negative correlation: white to red
    const intensity = Math.floor((1 - normalized) * 255);
    return `rgb(${Math.min(239, 100 + intensity)}, ${Math.max(68, 150 - intensity)}, ${Math.max(68, 150 - intensity)})`;
  } else {
    // Zero correlation: white/gray
    return '#6b7280';
  }
};

/**
 * Format correlation value for display
 */
const formatCorrelation = (value: number): string => {
  return value.toFixed(2);
};

/**
 * Get text color based on background brightness
 */
const getTextColor = (correlation: number): string => {
  const absCorr = Math.abs(correlation);
  return absCorr > 0.5 ? '#ffffff' : '#000000';
};

/**
 * Correlation matrix visualization
 *
 * @example
 * ```tsx
 * const data = {
 *   labels: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
 *   matrix: [
 *     [1.00, 0.85, 0.72],
 *     [0.85, 1.00, 0.68],
 *     [0.72, 0.68, 1.00],
 *   ],
 * };
 *
 * <CorrelationMatrix
 *   data={data}
 *   title="Asset Correlation"
 *   showValues={true}
 *   highlightThreshold={0.7}
 * />
 * ```
 */
export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  data,
  title,
  cellSize = 70,
  showValues = true,
  highlightThreshold = 0.7,
  className = '',
}) => {
  // Validate data structure
  const isValid = useMemo(() => {
    if (!data || !data.labels || !data.matrix) return false;
    if (data.labels.length === 0) return false;
    if (data.matrix.length !== data.labels.length) return false;
    return data.matrix.every((row) => row.length === data.labels.length);
  }, [data]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (!isValid) return null;

    let sum = 0;
    let count = 0;
    let maxCorr = -Infinity;
    let minCorr = Infinity;
    let maxPair = ['', ''];
    let minPair = ['', ''];

    data.matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        if (i !== j) {
          // Exclude diagonal (self-correlation)
          sum += value;
          count++;

          if (value > maxCorr) {
            maxCorr = value;
            maxPair = [data.labels[i], data.labels[j]];
          }
          if (value < minCorr) {
            minCorr = value;
            minPair = [data.labels[i], data.labels[j]];
          }
        }
      });
    });

    return {
      average: sum / count,
      max: maxCorr,
      min: minCorr,
      maxPair,
      minPair,
    };
  }, [data, isValid]);

  if (!isValid) {
    return (
      <div className={`flex items-center justify-center bg-gray-800/50 rounded p-8 ${className}`}>
        <p className="text-gray-500">Invalid or missing correlation data</p>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800/50 rounded-lg p-4 ${className}`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-100 mb-4">{title}</h3>
      )}

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Avg Correlation</span>
            <span className="text-base font-semibold text-gray-300">
              {formatCorrelation(stats.average)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Strongest</span>
            <span className="text-base font-semibold text-green-400">
              {formatCorrelation(stats.max)}
            </span>
            <span className="text-xs text-gray-500 truncate" title={`${stats.maxPair[0]} - ${stats.maxPair[1]}`}>
              {stats.maxPair[0]} - {stats.maxPair[1]}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500">Weakest</span>
            <span className="text-base font-semibold text-red-400">
              {formatCorrelation(stats.min)}
            </span>
            <span className="text-xs text-gray-500 truncate" title={`${stats.minPair[0]} - ${stats.minPair[1]}`}>
              {stats.minPair[0]} - {stats.minPair[1]}
            </span>
          </div>
        </div>
      )}

      {/* Matrix */}
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          {/* Header row */}
          <div className="flex">
            <div
              className="flex-shrink-0"
              style={{ width: `${cellSize}px` }}
            />
            {data.labels.map((label) => (
              <div
                key={`header-${label}`}
                className="flex-shrink-0 flex items-center justify-center text-xs text-gray-400 font-medium transform -rotate-45 origin-bottom-left"
                style={{ width: `${cellSize}px`, height: `${cellSize}px` }}
              >
                <span className="truncate" title={label}>
                  {label}
                </span>
              </div>
            ))}
          </div>

          {/* Data rows */}
          {data.matrix.map((row, rowIndex) => (
            <div key={`row-${rowIndex}`} className="flex">
              {/* Row label */}
              <div
                className="flex-shrink-0 flex items-center justify-end pr-2 text-xs text-gray-400 font-medium truncate"
                style={{ width: `${cellSize}px` }}
                title={data.labels[rowIndex]}
              >
                {data.labels[rowIndex]}
              </div>

              {/* Cells */}
              {row.map((value, colIndex) => {
                const bgColor = getCorrelationColor(value);
                const textColor = getTextColor(value);
                const isHighlighted = Math.abs(value) >= highlightThreshold && rowIndex !== colIndex;
                const isDiagonal = rowIndex === colIndex;

                return (
                  <div
                    key={`cell-${rowIndex}-${colIndex}`}
                    className={`flex-shrink-0 m-0.5 rounded flex items-center justify-center text-xs font-medium transition-all hover:scale-110 hover:z-10 cursor-pointer ${
                      isHighlighted ? 'ring-2 ring-yellow-400 ring-offset-1 ring-offset-gray-800' : ''
                    } ${isDiagonal ? 'opacity-50' : ''}`}
                    style={{
                      width: `${cellSize - 4}px`,
                      height: `${cellSize - 4}px`,
                      backgroundColor: bgColor,
                      color: textColor,
                    }}
                    title={`${data.labels[rowIndex]} vs ${data.labels[colIndex]}: ${formatCorrelation(value)}`}
                  >
                    {showValues && (
                      <span className="font-mono font-semibold">
                        {formatCorrelation(value)}
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
          <span className="text-xs text-gray-500">-1.00</span>
          <div className="flex items-center space-x-1">
            {[-1, -0.5, 0, 0.5, 1].map((value) => (
              <div
                key={value}
                className="w-8 h-4 rounded"
                style={{ backgroundColor: getCorrelationColor(value) }}
                title={formatCorrelation(value)}
              />
            ))}
          </div>
          <span className="text-xs text-gray-500">+1.00</span>
        </div>
        {highlightThreshold > 0 && (
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <div className="w-3 h-3 border-2 border-yellow-400 rounded" />
            <span>|r| ≥ {formatCorrelation(highlightThreshold)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default CorrelationMatrix;
