/**
 * ReturnDistribution Component
 *
 * Displays distribution of trading returns with histogram and statistical metrics.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useEffect, useState, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface ReturnData {
  timestamp: string;
  return_pct: number;
  strategy: string;
}

interface DistributionStats {
  mean: number;
  median: number;
  std_dev: number;
  skewness: number;
  kurtosis: number;
  min: number;
  max: number;
  var_95: number;
  var_99: number;
}

interface ReturnDistributionProps {
  data: ReturnData[];
  bins?: number;
  timeframe?: string;
  loading?: boolean;
  error?: Error | null;
  onRefresh?: () => void;
}

const ReturnDistribution: React.FC<ReturnDistributionProps> = ({
  data,
  bins = parseInt(process.env.REACT_APP_CHART_DEFAULT_BINS || '50'),
  timeframe = '1D',
  loading = false,
  error = null,
  onRefresh
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string>('ALL');
  const [showNormal, setShowNormal] = useState(true);

  // Calculate distribution statistics
  const stats = useMemo((): DistributionStats | null => {
    if (!data || data.length === 0) return null;

    const filteredData = selectedStrategy === 'ALL'
      ? data
      : data.filter(d => d.strategy === selectedStrategy);

    const returns = filteredData.map(d => d.return_pct).sort((a, b) => a - b);
    const n = returns.length;

    if (n === 0) return null;

    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const median = n % 2 === 0
      ? (returns[n / 2 - 1] + returns[n / 2]) / 2
      : returns[Math.floor(n / 2)];

    const variance = returns.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / n;
    const std_dev = Math.sqrt(variance);

    const skewness = returns.reduce((acc, val) => acc + Math.pow((val - mean) / std_dev, 3), 0) / n;
    const kurtosis = returns.reduce((acc, val) => acc + Math.pow((val - mean) / std_dev, 4), 0) / n - 3;

    const var_95_idx = Math.floor(n * 0.05);
    const var_99_idx = Math.floor(n * 0.01);

    return {
      mean,
      median,
      std_dev,
      skewness,
      kurtosis,
      min: returns[0],
      max: returns[n - 1],
      var_95: returns[var_95_idx],
      var_99: returns[var_99_idx]
    };
  }, [data, selectedStrategy]);

  // Calculate histogram data
  const histogramData = useMemo(() => {
    if (!data || data.length === 0 || !stats) {
      return { labels: [], datasets: [] };
    }

    const filteredData = selectedStrategy === 'ALL'
      ? data
      : data.filter(d => d.strategy === selectedStrategy);

    const returns = filteredData.map(d => d.return_pct);
    const min = stats.min;
    const max = stats.max;
    const binWidth = (max - min) / bins;

    const histogram = new Array(bins).fill(0);
    const binLabels = new Array(bins).fill(0).map((_, i) => {
      const start = min + i * binWidth;
      const end = start + binWidth;
      return `${start.toFixed(2)}%`;
    });

    returns.forEach(ret => {
      const binIndex = Math.min(Math.floor((ret - min) / binWidth), bins - 1);
      histogram[binIndex]++;
    });

    const datasets = [
      {
        label: 'Return Frequency',
        data: histogram,
        backgroundColor: histogram.map((_, i) => {
          const binCenter = min + (i + 0.5) * binWidth;
          return binCenter < 0
            ? `rgba(239, 68, 68, ${0.5 + Math.abs(binCenter) / Math.abs(min) * 0.5})`
            : `rgba(34, 197, 94, ${0.5 + binCenter / max * 0.5})`;
        }),
        borderColor: histogram.map((_, i) => {
          const binCenter = min + (i + 0.5) * binWidth;
          return binCenter < 0 ? 'rgba(239, 68, 68, 1)' : 'rgba(34, 197, 94, 1)';
        }),
        borderWidth: 1
      }
    ];

    // Add normal distribution overlay if enabled
    if (showNormal && stats) {
      const normalDist = binLabels.map((_, i) => {
        const x = min + (i + 0.5) * binWidth;
        const exponent = -Math.pow(x - stats.mean, 2) / (2 * Math.pow(stats.std_dev, 2));
        const y = (1 / (stats.std_dev * Math.sqrt(2 * Math.PI))) * Math.exp(exponent);
        return y * returns.length * binWidth;
      });

      datasets.push({
        label: 'Normal Distribution',
        data: normalDist,
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2
      } as any);
    }

    return { labels: binLabels, datasets };
  }, [data, bins, selectedStrategy, stats, showNormal]);

  const chartOptions: ChartOptions<'bar'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: process.env.REACT_APP_CHART_TEXT_COLOR || '#E5E7EB',
          font: {
            family: process.env.REACT_APP_CHART_FONT_FAMILY || 'Inter, sans-serif',
            size: 12
          }
        }
      },
      title: {
        display: true,
        text: `Return Distribution - ${timeframe}`,
        color: process.env.REACT_APP_CHART_TEXT_COLOR || '#E5E7EB',
        font: {
          family: process.env.REACT_APP_CHART_FONT_FAMILY || 'Inter, sans-serif',
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: process.env.REACT_APP_TOOLTIP_BG || 'rgba(17, 24, 39, 0.95)',
        titleColor: '#F9FAFB',
        bodyColor: '#E5E7EB',
        borderColor: 'rgba(75, 85, 99, 0.5)',
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y || 0;
            return `${label}: ${value.toFixed(0)} trades`;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)'
        },
        ticks: {
          color: '#9CA3AF',
          maxRotation: 45,
          minRotation: 45,
          font: {
            size: 10
          }
        }
      },
      y: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)'
        },
        ticks: {
          color: '#9CA3AF',
          font: {
            size: 11
          }
        }
      }
    }
  }), [timeframe]);

  // Get unique strategies
  const strategies = useMemo(() => {
    if (!data) return [];
    const uniqueStrategies = [...new Set(data.map(d => d.strategy))];
    return ['ALL', ...uniqueStrategies];
  }, [data]);

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
        <h3 className="text-red-400 font-semibold mb-2">Error Loading Distribution</h3>
        <p className="text-red-300 text-sm mb-4">{error.message}</p>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded-md transition-colors"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center">
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
        <p className="text-gray-400 mt-4">Loading distribution data...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
      {/* Controls */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex gap-4 items-center">
          <select
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            className="bg-gray-700 border border-gray-600 text-gray-200 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {strategies.map(strategy => (
              <option key={strategy} value={strategy}>{strategy}</option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={showNormal}
              onChange={(e) => setShowNormal(e.target.checked)}
              className="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
            />
            <span>Show Normal Distribution</span>
          </label>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md transition-colors"
          >
            Refresh
          </button>
        )}
      </div>

      {/* Chart */}
      <div className="h-96 mb-6">
        <Bar data={histogramData} options={chartOptions} />
      </div>

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Mean</div>
            <div className={`text-2xl font-bold ${stats.mean >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats.mean.toFixed(3)}%
            </div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Std Dev</div>
            <div className="text-2xl font-bold text-blue-400">{stats.std_dev.toFixed(3)}%</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Skewness</div>
            <div className="text-2xl font-bold text-purple-400">{stats.skewness.toFixed(3)}</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Kurtosis</div>
            <div className="text-2xl font-bold text-orange-400">{stats.kurtosis.toFixed(3)}</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">VaR 95%</div>
            <div className="text-2xl font-bold text-red-400">{stats.var_95.toFixed(3)}%</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">VaR 99%</div>
            <div className="text-2xl font-bold text-red-500">{stats.var_99.toFixed(3)}%</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Min</div>
            <div className="text-2xl font-bold text-red-400">{stats.min.toFixed(3)}%</div>
          </div>
          <div className="bg-gray-700/50 rounded-lg p-4">
            <div className="text-gray-400 text-sm mb-1">Max</div>
            <div className="text-2xl font-bold text-green-400">{stats.max.toFixed(3)}%</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReturnDistribution;
