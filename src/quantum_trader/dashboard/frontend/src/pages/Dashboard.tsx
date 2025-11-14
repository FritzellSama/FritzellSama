/**
 * Dashboard Page - Main trading dashboard
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Real-time overview of portfolio, positions, and market data
 */

import React, { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import EquityChart from '../components/charts/EquityChart';
import CandleChart from '../components/charts/CandleChart';
import HeatMap from '../components/charts/HeatMap';

interface PortfolioStats {
  totalEquity: number;
  dailyPnl: number;
  dailyPnlPercent: number;
  totalPositions: number;
  activeOrders: number;
  winRate: number;
}

interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

/**
 * Dashboard page component
 *
 * Features:
 * - Real-time portfolio statistics
 * - Equity curve chart
 * - Active positions table
 * - Market overview with candlestick charts
 * - Performance heatmap
 */
const Dashboard: React.FC = () => {
  const [portfolioStats, setPortfolioStats] = useState<PortfolioStats>({
    totalEquity: 1000000,
    dailyPnl: 15250.75,
    dailyPnlPercent: 1.55,
    totalPositions: 12,
    activeOrders: 8,
    winRate: 68.5,
  });

  const [positions, setPositions] = useState<Position[]>([
    {
      symbol: 'BTC/USDT',
      quantity: 2.5,
      entryPrice: 42000,
      currentPrice: 43500,
      pnl: 3750,
      pnlPercent: 3.57,
    },
    {
      symbol: 'ETH/USDT',
      quantity: 15,
      entryPrice: 2200,
      currentPrice: 2280,
      pnl: 1200,
      pnlPercent: 3.64,
    },
    {
      symbol: 'SOL/USDT',
      quantity: 100,
      entryPrice: 95,
      currentPrice: 92,
      pnl: -300,
      pnlPercent: -3.16,
    },
  ]);

  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      const apiUrl = process.env.REACT_APP_API_URL;
      if (!apiUrl) return;

      try {
        setIsLoading(true);

        // Fetch portfolio stats
        const statsResponse = await fetch(`${apiUrl}/portfolio/stats`);
        if (statsResponse.ok) {
          const stats = await statsResponse.json();
          setPortfolioStats(stats);
        }

        // Fetch positions
        const positionsResponse = await fetch(`${apiUrl}/positions`);
        if (positionsResponse.ok) {
          const positionsData = await positionsResponse.json();
          setPositions(positionsData);
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setIsLoading(false);
      }
    };

    fetchData();

    // Update data every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  // Generate sample equity data
  const equityData = Array.from({ length: 30 }, (_, i) => {
    const baseEquity = 950000;
    const trend = i * 2000;
    const variance = Math.random() * 10000 - 5000;
    const equity = baseEquity + trend + variance;

    return {
      time: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      equity: equity,
      pnl: equity - baseEquity,
    };
  });

  // Generate sample candle data
  const candleData = Array.from({ length: 20 }, (_, i) => {
    const basePrice = 43000;
    const open = basePrice + Math.random() * 1000 - 500;
    const close = open + Math.random() * 800 - 400;
    const high = Math.max(open, close) + Math.random() * 300;
    const low = Math.min(open, close) - Math.random() * 300;

    return {
      time: `${9 + Math.floor(i / 2)}:${i % 2 === 0 ? '00' : '30'}`,
      open,
      high,
      low,
      close,
      volume: Math.floor(Math.random() * 2000) + 1000,
    };
  });

  // Generate sample heatmap data
  const heatmapData = [
    { row: 'Momentum', column: 'Mon', value: 1250 },
    { row: 'Momentum', column: 'Tue', value: -340 },
    { row: 'Momentum', column: 'Wed', value: 890 },
    { row: 'Momentum', column: 'Thu', value: 1120 },
    { row: 'Momentum', column: 'Fri', value: 2340 },
    { row: 'Mean Rev', column: 'Mon', value: -450 },
    { row: 'Mean Rev', column: 'Tue', value: 1230 },
    { row: 'Mean Rev', column: 'Wed', value: -120 },
    { row: 'Mean Rev', column: 'Thu', value: 780 },
    { row: 'Mean Rev', column: 'Fri', value: 1560 },
    { row: 'Arbitrage', column: 'Mon', value: 340 },
    { row: 'Arbitrage', column: 'Tue', value: 520 },
    { row: 'Arbitrage', column: 'Wed', value: 380 },
    { row: 'Arbitrage', column: 'Thu', value: 410 },
    { row: 'Arbitrage', column: 'Fri', value: 490 },
  ];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Real-time portfolio overview and market insights
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="ghost" size="sm" onClick={() => window.location.reload()}>
            🔄 Refresh
          </Button>
          <Button variant="primary" size="sm">
            ⚙️ Settings
          </Button>
        </div>
      </div>

      {/* Portfolio Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Total Equity
            </span>
            <span className="text-2xl font-bold text-blue-400 mt-2">
              ${portfolioStats.totalEquity.toLocaleString('en-US')}
            </span>
          </div>
        </Card>

        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Daily P&L
            </span>
            <span
              className={`text-2xl font-bold mt-2 ${
                portfolioStats.dailyPnl >= 0 ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {portfolioStats.dailyPnl >= 0 ? '+' : ''}$
              {portfolioStats.dailyPnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </span>
            <span
              className={`text-sm mt-1 ${
                portfolioStats.dailyPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {portfolioStats.dailyPnlPercent >= 0 ? '+' : ''}
              {portfolioStats.dailyPnlPercent.toFixed(2)}%
            </span>
          </div>
        </Card>

        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Positions
            </span>
            <span className="text-2xl font-bold text-purple-400 mt-2">
              {portfolioStats.totalPositions}
            </span>
          </div>
        </Card>

        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Active Orders
            </span>
            <span className="text-2xl font-bold text-yellow-400 mt-2">
              {portfolioStats.activeOrders}
            </span>
          </div>
        </Card>

        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Win Rate
            </span>
            <span className="text-2xl font-bold text-green-400 mt-2">
              {portfolioStats.winRate.toFixed(1)}%
            </span>
          </div>
        </Card>

        <Card padding="md" hoverable>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Status
            </span>
            <div className="flex items-center mt-2">
              <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse mr-2" />
              <span className="text-lg font-semibold text-green-400">Active</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Equity Chart */}
        <Card title="Portfolio Equity" subtitle="30-day performance">
          <EquityChart data={equityData} height={300} initialEquity={950000} />
        </Card>

        {/* Price Chart */}
        <Card title="BTC/USDT" subtitle="Live market data">
          <CandleChart data={candleData} height={300} showVolume={true} />
        </Card>
      </div>

      {/* Positions Table */}
      <Card
        title="Active Positions"
        subtitle={`${positions.length} open positions`}
        headerAction={
          <Button variant="ghost" size="sm">
            View All
          </Button>
        }
      >
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-700">
            <thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Quantity
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Entry Price
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Current Price
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">
                  P&L %
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {positions.map((position) => (
                <tr key={position.symbol} className="hover:bg-gray-700/50 transition-colors">
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className="text-sm font-medium text-gray-200">
                      {position.symbol}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span className="text-sm text-gray-300">{position.quantity}</span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span className="text-sm font-mono text-gray-300">
                      ${position.entryPrice.toLocaleString('en-US')}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span className="text-sm font-mono text-gray-300">
                      ${position.currentPrice.toLocaleString('en-US')}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span
                      className={`text-sm font-mono font-semibold ${
                        position.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {position.pnl >= 0 ? '+' : ''}$
                      {position.pnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span
                      className={`text-sm font-semibold ${
                        position.pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {position.pnlPercent >= 0 ? '+' : ''}
                      {position.pnlPercent.toFixed(2)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Strategy Performance Heatmap */}
      <Card title="Strategy Performance" subtitle="Daily P&L by strategy">
        <HeatMap
          data={heatmapData}
          colorScheme="performance"
          showValues={true}
          cellSize={70}
        />
      </Card>
    </div>
  );
};

export default Dashboard;
