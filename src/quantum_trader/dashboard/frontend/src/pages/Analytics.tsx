/**
 * Analytics Page - Advanced performance analytics
 *
 * Quantum Trader AI - Institutional Trading Platform
 * In-depth analysis with correlation matrix, drawdown, and risk metrics
 */

import React, { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import EquityChart from '../components/charts/EquityChart';
import CorrelationMatrix from '../components/analytics/CorrelationMatrix';
import DrawdownChart from '../components/analytics/DrawdownChart';
import HeatMap from '../components/charts/HeatMap';

interface RiskMetrics {
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  volatility: number;
  var95: number;
  cvar95: number;
}

interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
}

/**
 * Analytics page component
 *
 * Features:
 * - Advanced performance metrics
 * - Risk analysis and VaR
 * - Correlation matrix between assets
 * - Drawdown analysis
 * - Strategy performance heatmaps
 */
const Analytics: React.FC = () => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    sharpeRatio: 2.45,
    sortinoRatio: 3.12,
    maxDrawdown: -8.5,
    volatility: 12.3,
    var95: -2.8,
    cvar95: -4.2,
  });

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics>({
    totalReturn: 45.8,
    annualizedReturn: 38.2,
    winRate: 68.5,
    profitFactor: 2.34,
    avgWin: 1250.5,
    avgLoss: -534.2,
  });

  const [selectedTimeframe, setSelectedTimeframe] = useState<string>('30D');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Fetch analytics data
  useEffect(() => {
    const fetchData = async () => {
      const apiUrl = process.env.REACT_APP_API_URL;
      if (!apiUrl) return;

      try {
        setIsLoading(true);

        const metricsResponse = await fetch(`${apiUrl}/analytics/metrics?timeframe=${selectedTimeframe}`);
        if (metricsResponse.ok) {
          const metrics = await metricsResponse.json();
          setRiskMetrics(metrics.risk);
          setPerformanceMetrics(metrics.performance);
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching analytics data:', error);
        setIsLoading(false);
      }
    };

    fetchData();
  }, [selectedTimeframe]);

  // Generate sample equity data with benchmark
  const equityData = Array.from({ length: 90 }, (_, i) => {
    const baseEquity = 950000;
    const trend = i * 1200;
    const variance = Math.random() * 8000 - 4000;
    const equity = baseEquity + trend + variance;
    const benchmark = baseEquity + i * 800;

    return {
      time: new Date(Date.now() - (89 - i) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      equity: equity,
      benchmark: benchmark,
      pnl: equity - baseEquity,
    };
  });

  // Generate sample drawdown data
  const drawdownData = equityData.map((point, i) => {
    const maxEquity = Math.max(...equityData.slice(0, i + 1).map((p) => p.equity));
    const drawdown = ((point.equity - maxEquity) / maxEquity) * 100;

    return {
      time: point.time,
      drawdown: drawdown,
      equity: point.equity,
      peak: maxEquity,
    };
  });

  // Generate correlation matrix data
  const correlationData = {
    labels: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT'],
    matrix: [
      [1.0, 0.85, 0.72, 0.68, 0.61],
      [0.85, 1.0, 0.78, 0.71, 0.66],
      [0.72, 0.78, 1.0, 0.64, 0.58],
      [0.68, 0.71, 0.64, 1.0, 0.69],
      [0.61, 0.66, 0.58, 0.69, 1.0],
    ],
  };

  // Generate strategy performance heatmap
  const strategyHeatmapData = [
    { row: 'Momentum', column: 'Week 1', value: 3250 },
    { row: 'Momentum', column: 'Week 2', value: -1240 },
    { row: 'Momentum', column: 'Week 3', value: 4890 },
    { row: 'Momentum', column: 'Week 4', value: 2120 },
    { row: 'Mean Rev', column: 'Week 1', value: 1450 },
    { row: 'Mean Rev', column: 'Week 2', value: 2230 },
    { row: 'Mean Rev', column: 'Week 3', value: -820 },
    { row: 'Mean Rev', column: 'Week 4', value: 1780 },
    { row: 'Arbitrage', column: 'Week 1', value: 540 },
    { row: 'Arbitrage', column: 'Week 2', value: 720 },
    { row: 'Arbitrage', column: 'Week 3', value: 680 },
    { row: 'Arbitrage', column: 'Week 4', value: 810 },
    { row: 'ML Model', column: 'Week 1', value: 2340 },
    { row: 'ML Model', column: 'Week 2', value: 3120 },
    { row: 'ML Model', column: 'Week 3', value: 1890 },
    { row: 'ML Model', column: 'Week 4', value: 2560 },
  ];

  const timeframes = ['7D', '30D', '90D', '1Y', 'ALL'];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Analytics</h1>
          <p className="text-gray-400 mt-1">
            Advanced performance analysis and risk metrics
          </p>
        </div>
        <div className="flex space-x-2">
          {timeframes.map((tf) => (
            <Button
              key={tf}
              variant={selectedTimeframe === tf ? 'primary' : 'ghost'}
              size="sm"
              onClick={() => setSelectedTimeframe(tf)}
            >
              {tf}
            </Button>
          ))}
        </div>
      </div>

      {/* Performance Metrics */}
      <Card title="Performance Metrics" subtitle="Key performance indicators">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Total Return
            </span>
            <span className="text-2xl font-bold text-green-400 mt-2">
              +{performanceMetrics.totalReturn.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Annualized Return
            </span>
            <span className="text-2xl font-bold text-green-400 mt-2">
              +{performanceMetrics.annualizedReturn.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Win Rate
            </span>
            <span className="text-2xl font-bold text-blue-400 mt-2">
              {performanceMetrics.winRate.toFixed(1)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Profit Factor
            </span>
            <span className="text-2xl font-bold text-purple-400 mt-2">
              {performanceMetrics.profitFactor.toFixed(2)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Avg Win
            </span>
            <span className="text-2xl font-bold text-green-400 mt-2">
              ${performanceMetrics.avgWin.toFixed(0)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Avg Loss
            </span>
            <span className="text-2xl font-bold text-red-400 mt-2">
              ${performanceMetrics.avgLoss.toFixed(0)}
            </span>
          </div>
        </div>
      </Card>

      {/* Risk Metrics */}
      <Card title="Risk Metrics" subtitle="Risk-adjusted performance">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Sharpe Ratio
            </span>
            <span className="text-2xl font-bold text-blue-400 mt-2">
              {riskMetrics.sharpeRatio.toFixed(2)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Sortino Ratio
            </span>
            <span className="text-2xl font-bold text-blue-400 mt-2">
              {riskMetrics.sortinoRatio.toFixed(2)}
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Max Drawdown
            </span>
            <span className="text-2xl font-bold text-red-400 mt-2">
              {riskMetrics.maxDrawdown.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              Volatility
            </span>
            <span className="text-2xl font-bold text-yellow-400 mt-2">
              {riskMetrics.volatility.toFixed(1)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              VaR (95%)
            </span>
            <span className="text-2xl font-bold text-orange-400 mt-2">
              {riskMetrics.var95.toFixed(2)}%
            </span>
          </div>
          <div className="flex flex-col">
            <span className="text-xs text-gray-500 uppercase tracking-wider">
              CVaR (95%)
            </span>
            <span className="text-2xl font-bold text-red-400 mt-2">
              {riskMetrics.cvar95.toFixed(2)}%
            </span>
          </div>
        </div>
      </Card>

      {/* Equity Curve with Benchmark */}
      <Card title="Portfolio Equity vs Benchmark" subtitle="90-day comparison">
        <EquityChart
          data={equityData}
          height={400}
          showBenchmark={true}
          chartType="area"
          initialEquity={950000}
        />
      </Card>

      {/* Drawdown Analysis */}
      <Card title="Drawdown Analysis" subtitle="Underwater equity curve">
        <DrawdownChart
          data={drawdownData}
          height={350}
          warningThreshold={-10}
          dangerThreshold={-20}
        />
      </Card>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Correlation Matrix */}
        <Card title="Asset Correlation" subtitle="Pearson correlation coefficients">
          <CorrelationMatrix
            data={correlationData}
            showValues={true}
            highlightThreshold={0.7}
            cellSize={65}
          />
        </Card>

        {/* Strategy Performance Heatmap */}
        <Card title="Strategy Performance" subtitle="Weekly P&L breakdown">
          <HeatMap
            data={strategyHeatmapData}
            colorScheme="performance"
            showValues={true}
            cellSize={80}
          />
        </Card>
      </div>

      {/* Additional Insights */}
      <Card title="Performance Insights" subtitle="AI-generated analysis">
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              <div className="h-2 w-2 rounded-full bg-green-400" />
            </div>
            <div>
              <p className="text-sm text-gray-300">
                <span className="font-semibold text-green-400">Strong Performance:</span>{' '}
                Portfolio outperforming benchmark by{' '}
                <span className="font-mono">
                  {((equityData[equityData.length - 1].equity - equityData[equityData.length - 1].benchmark) /
                    equityData[equityData.length - 1].benchmark *
                    100
                  ).toFixed(2)}%
                </span>{' '}
                over the selected period.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              <div className="h-2 w-2 rounded-full bg-blue-400" />
            </div>
            <div>
              <p className="text-sm text-gray-300">
                <span className="font-semibold text-blue-400">Risk-Adjusted Returns:</span>{' '}
                Sharpe ratio of {riskMetrics.sharpeRatio.toFixed(2)} indicates excellent
                risk-adjusted performance.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              <div className="h-2 w-2 rounded-full bg-yellow-400" />
            </div>
            <div>
              <p className="text-sm text-gray-300">
                <span className="font-semibold text-yellow-400">Correlation Analysis:</span>{' '}
                High correlation detected between BTC/USDT and ETH/USDT (0.85). Consider
                diversification strategies.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 mt-1">
              <div className="h-2 w-2 rounded-full bg-purple-400" />
            </div>
            <div>
              <p className="text-sm text-gray-300">
                <span className="font-semibold text-purple-400">Strategy Recommendation:</span>{' '}
                ML Model and Momentum strategies showing consistent positive returns. Consider
                increasing allocation.
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Analytics;
