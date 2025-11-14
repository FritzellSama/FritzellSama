import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface PnLDataPoint {
  timestamp: string;
  pnl: string;
  cumulative_pnl: string;
  balance: string;
}

interface PnLChartProps {
  timeframe?: string;
  height?: number;
}

const PnLChart: React.FC<PnLChartProps> = ({
  timeframe = '24h',
  height = 400,
}) => {
  const [data, setData] = useState<PnLDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState<string>(timeframe);
  const chartRef = useRef<ChartJS<'line'>>(null);

  const timeframes = [
    { label: '1H', value: '1h' },
    { label: '24H', value: '24h' },
    { label: '7D', value: '7d' },
    { label: '30D', value: '30d' },
    { label: '90D', value: '90d' },
  ];

  useEffect(() => {
    fetchPnLData();

    const interval = setInterval(fetchPnLData, 30000);

    return () => clearInterval(interval);
  }, [selectedTimeframe]);

  const fetchPnLData = async (background: boolean = false): Promise<void> => {
    if (!background) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(
        `${apiUrl}/api/v1/analytics/pnl?timeframe=${selectedTimeframe}`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch P&L data: ${response.statusText}`);
      }

      const responseData = await response.json();
      setData(responseData.data || []);
    } catch (err) {
      console.error('Error fetching P&L data:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load P&L data';
      setError(errorMessage);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = (): void => {
    fetchPnLData(true);
  };

  const formatTimestamp = (timestamp: string): string => {
    try {
      const date = new Date(timestamp);

      if (selectedTimeframe === '1h') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else if (selectedTimeframe === '24h') {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      } else {
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
      }
    } catch {
      return timestamp;
    }
  };

  const chartData = {
    labels: data.map((point) => formatTimestamp(point.timestamp)),
    datasets: [
      {
        label: 'Cumulative P&L',
        data: data.map((point) => parseFloat(point.cumulative_pnl)),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
      {
        label: 'Period P&L',
        data: data.map((point) => parseFloat(point.pnl)),
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        pointHoverRadius: 6,
        borderWidth: 2,
      },
    ],
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'rgb(156, 163, 175)',
          font: {
            size: 12,
          },
        },
      },
      title: {
        display: false,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(31, 41, 55, 0.95)',
        titleColor: 'rgb(255, 255, 255)',
        bodyColor: 'rgb(209, 213, 219)',
        borderColor: 'rgb(75, 85, 99)',
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label: function (context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += '$' + context.parsed.y.toFixed(2);
            }
            return label;
          },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          maxRotation: 45,
          minRotation: 0,
        },
      },
      y: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)',
        },
        ticks: {
          color: 'rgb(156, 163, 175)',
          callback: function (value) {
            return '$' + (value as number).toFixed(0);
          },
        },
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false,
    },
  };

  const getCurrentPnL = (): string => {
    if (data.length === 0) return '$0.00';
    const latest = data[data.length - 1];
    const pnl = parseFloat(latest.cumulative_pnl);
    return pnl >= 0 ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`;
  };

  const getPnLChange = (): string => {
    if (data.length < 2) return '0.00%';
    const first = parseFloat(data[0].balance);
    const last = parseFloat(data[data.length - 1].balance);
    const change = ((last - first) / first) * 100;
    return change >= 0 ? `+${change.toFixed(2)}%` : `${change.toFixed(2)}%`;
  };

  const getPnLColor = (): string => {
    if (data.length === 0) return 'text-gray-400';
    const latest = parseFloat(data[data.length - 1].cumulative_pnl);
    return latest >= 0 ? 'text-green-400' : 'text-red-400';
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center" style={{ height }}>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="text-center" style={{ height }}>
          <p className="text-red-400 mb-4">{error}</p>
          <button
            onClick={handleRefresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-xl font-semibold text-white">Profit & Loss</h3>
            <div className="flex items-center mt-2">
              <span className={`text-2xl font-bold ${getPnLColor()}`}>
                {getCurrentPnL()}
              </span>
              <span className={`ml-3 text-sm ${getPnLColor()}`}>
                {getPnLChange()}
              </span>
            </div>
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="p-2 bg-gray-700 text-gray-400 hover:bg-gray-600 rounded-md transition-colors disabled:opacity-50"
          >
            <ArrowPathIcon className={`h-5 w-5 ${refreshing ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Timeframe Selector */}
        <div className="flex space-x-2">
          {timeframes.map((tf) => (
            <button
              key={tf.value}
              onClick={() => setSelectedTimeframe(tf.value)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                selectedTimeframe === tf.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
            >
              {tf.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="p-4">
        {data.length === 0 ? (
          <div className="flex items-center justify-center" style={{ height }}>
            <p className="text-gray-400">No P&L data available</p>
          </div>
        ) : (
          <div style={{ height }}>
            <Line ref={chartRef} data={chartData} options={options} />
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-gray-700 p-4">
        <div className="flex justify-between items-center text-xs text-gray-400">
          <div>
            Last updated:{' '}
            {data.length > 0
              ? new Date(data[data.length - 1].timestamp).toLocaleString()
              : 'N/A'}
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
            Live
          </div>
        </div>
      </div>
    </div>
  );
};

export default PnLChart;
