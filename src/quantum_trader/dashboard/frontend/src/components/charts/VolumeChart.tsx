/**
 * VolumeChart Component
 *
 * Displays trading volume over time with customizable timeframes.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
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
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

interface VolumeData {
  timestamp: string;
  volume: number;
  buy_volume: number;
  sell_volume: number;
  price: number;
}

interface VolumeChartProps {
  symbol: string;
  timeframe?: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  limit?: number;
  height?: number;
  showPriceOverlay?: boolean;
  autoRefresh?: boolean;
}

const VolumeChart: React.FC<VolumeChartProps> = ({
  symbol,
  timeframe = '1h',
  limit = parseInt(process.env.REACT_APP_VOLUME_CHART_LIMIT || '24'),
  height = parseInt(process.env.REACT_APP_CHART_HEIGHT || '300'),
  showPriceOverlay = true,
  autoRefresh = true
}) => {
  const [data, setData] = useState<VolumeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_VOLUME_CHART_REFRESH || '60000');

  // Fetch volume data
  const fetchVolumeData = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        timeframe,
        limit: limit.toString()
      });

      const response = await fetch(`${API_BASE_URL}/api/v1/market/${symbol}/volume?${params}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch volume data');
      }

      const volumeData = await response.json();
      setData(volumeData);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching volume data:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL, symbol, timeframe, limit]);

  useEffect(() => {
    fetchVolumeData();

    if (autoRefresh) {
      const interval = setInterval(fetchVolumeData, REFRESH_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [fetchVolumeData, autoRefresh, REFRESH_INTERVAL]);

  // Prepare chart data
  const chartData = useMemo(() => {
    if (!data || data.length === 0) {
      return { labels: [], datasets: [] };
    }

    const labels = data.map(d => {
      const date = new Date(d.timestamp);
      switch (timeframe) {
        case '1m':
        case '5m':
        case '15m':
          return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        case '1h':
        case '4h':
          return date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        case '1d':
          return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        default:
          return date.toLocaleString();
      }
    });

    const datasets = [
      {
        type: 'bar' as const,
        label: 'Buy Volume',
        data: data.map(d => d.buy_volume),
        backgroundColor: 'rgba(34, 197, 94, 0.7)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 1,
        yAxisID: 'y'
      },
      {
        type: 'bar' as const,
        label: 'Sell Volume',
        data: data.map(d => d.sell_volume),
        backgroundColor: 'rgba(239, 68, 68, 0.7)',
        borderColor: 'rgba(239, 68, 68, 1)',
        borderWidth: 1,
        yAxisID: 'y'
      }
    ];

    // Add price overlay if enabled
    if (showPriceOverlay) {
      datasets.push({
        type: 'line' as const,
        label: 'Price',
        data: data.map(d => d.price),
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
        yAxisID: 'y1',
        pointRadius: 0,
        pointHoverRadius: 4
      } as any);
    }

    return { labels, datasets };
  }, [data, timeframe, showPriceOverlay]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;

    const totalVolume = data.reduce((sum, d) => sum + d.volume, 0);
    const totalBuyVolume = data.reduce((sum, d) => sum + d.buy_volume, 0);
    const totalSellVolume = data.reduce((sum, d) => sum + d.sell_volume, 0);
    const avgVolume = totalVolume / data.length;
    const maxVolume = Math.max(...data.map(d => d.volume));
    const buyPressure = (totalBuyVolume / totalVolume) * 100;

    return {
      totalVolume,
      totalBuyVolume,
      totalSellVolume,
      avgVolume,
      maxVolume,
      buyPressure
    };
  }, [data]);

  const chartOptions: ChartOptions<'bar'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: process.env.REACT_APP_CHART_TEXT_COLOR || '#E5E7EB',
          font: {
            family: process.env.REACT_APP_CHART_FONT_FAMILY || 'Inter, sans-serif',
            size: 12
          },
          usePointStyle: true,
          padding: 15
        }
      },
      title: {
        display: true,
        text: `${symbol} Volume - ${timeframe}`,
        color: process.env.REACT_APP_CHART_TEXT_COLOR || '#E5E7EB',
        font: {
          family: process.env.REACT_APP_CHART_FONT_FAMILY || 'Inter, sans-serif',
          size: 16,
          weight: 'bold'
        },
        padding: 20
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

            if (label === 'Price') {
              return `${label}: $${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            } else {
              return `${label}: ${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            }
          }
        }
      }
    },
    scales: {
      x: {
        stacked: true,
        grid: {
          color: 'rgba(75, 85, 99, 0.2)',
          display: false
        },
        ticks: {
          color: '#9CA3AF',
          maxRotation: 45,
          minRotation: 0,
          font: {
            size: 10
          }
        }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        stacked: true,
        grid: {
          color: 'rgba(75, 85, 99, 0.2)'
        },
        ticks: {
          color: '#9CA3AF',
          font: {
            size: 11
          },
          callback: function(value: any) {
            if (value >= 1000000) {
              return (value / 1000000).toFixed(1) + 'M';
            } else if (value >= 1000) {
              return (value / 1000).toFixed(1) + 'K';
            }
            return value;
          }
        }
      },
      ...(showPriceOverlay && {
        y1: {
          type: 'linear' as const,
          display: true,
          position: 'right' as const,
          grid: {
            drawOnChartArea: false
          },
          ticks: {
            color: '#60A5FA',
            font: {
              size: 11
            },
            callback: function(value: any) {
              return '$' + value.toLocaleString();
            }
          }
        }
      })
    }
  }), [symbol, timeframe, showPriceOverlay]);

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
        <h3 className="text-red-400 font-semibold mb-2">Error Loading Volume Data</h3>
        <p className="text-red-300 text-sm">{error.message}</p>
        <button
          onClick={fetchVolumeData}
          className="mt-4 px-4 py-2 bg-red-700 hover:bg-red-600 text-white rounded-md transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (loading && data.length === 0) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-12 text-center" style={{ height }}>
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
        <p className="text-gray-400 mt-4">Loading volume data...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-4">
          <div className="bg-gray-700/50 rounded p-3">
            <div className="text-gray-400 text-xs mb-1">Total Volume</div>
            <div className="text-lg font-bold text-white">
              {stats.totalVolume >= 1000000
                ? `${(stats.totalVolume / 1000000).toFixed(2)}M`
                : `${(stats.totalVolume / 1000).toFixed(2)}K`}
            </div>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <div className="text-gray-400 text-xs mb-1">Avg Volume</div>
            <div className="text-lg font-bold text-blue-400">
              {stats.avgVolume >= 1000000
                ? `${(stats.avgVolume / 1000000).toFixed(2)}M`
                : `${(stats.avgVolume / 1000).toFixed(2)}K`}
            </div>
          </div>
          <div className="bg-gray-700/50 rounded p-3">
            <div className="text-gray-400 text-xs mb-1">Buy Pressure</div>
            <div className={`text-lg font-bold ${
              stats.buyPressure > 50 ? 'text-green-400' : 'text-red-400'
            }`}>
              {stats.buyPressure.toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div style={{ height }}>
        <Bar data={chartData} options={chartOptions} />
      </div>

      {/* Buy/Sell Ratio Indicator */}
      {stats && (
        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Sell Pressure</span>
            <span>Buy Pressure</span>
          </div>
          <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden flex">
            <div
              className="bg-red-500 transition-all"
              style={{ width: `${100 - stats.buyPressure}%` }}
            ></div>
            <div
              className="bg-green-500 transition-all"
              style={{ width: `${stats.buyPressure}%` }}
            ></div>
          </div>
          <div className="flex justify-between text-xs mt-1">
            <span className="text-red-400 font-mono">
              {(100 - stats.buyPressure).toFixed(1)}%
            </span>
            <span className="text-green-400 font-mono">
              {stats.buyPressure.toFixed(1)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default VolumeChart;
