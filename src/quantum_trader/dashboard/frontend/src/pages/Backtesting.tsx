/**
 * Backtesting Page
 *
 * Comprehensive backtesting interface for testing trading strategies
 * against historical data with detailed performance analytics.
 */

import React, { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type {
  BacktestConfig,
  BacktestResult,
  StrategyType,
  Exchange,
  TimeFrame,
} from '@/types';
import {
  formatCurrency,
  formatPercentage,
  formatNumber,
  formatDate,
  formatRatio,
} from '@/utils/formatters';
import {
  STRATEGY_TYPES,
  EXCHANGES,
  TIMEFRAMES,
  POPULAR_SYMBOLS,
  CHART_COLORS,
} from '@/utils/constants';

interface BacktestFormData {
  strategyType: StrategyType;
  symbols: string[];
  exchanges: Exchange[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  commission: number;
  slippage: number;
  parameters: Record<string, any>;
}

const Backtesting: React.FC = () => {
  const queryClient = useQueryClient();

  const [selectedResult, setSelectedResult] = useState<BacktestResult | null>(
    null
  );
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [formData, setFormData] = useState<BacktestFormData>({
    strategyType: 'MOMENTUM',
    symbols: ['BTC/USDT'],
    exchanges: ['binance'],
    startDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000)
      .toISOString()
      .split('T')[0],
    endDate: new Date().toISOString().split('T')[0],
    initialCapital: 100000,
    commission: 0.001,
    slippage: 0.001,
    parameters: {},
  });

  const { data: backtestList, isLoading: isLoadingList } = useQuery({
    queryKey: ['backtests'],
    queryFn: async () => {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/v1/backtests`,
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('quantum_trader_token')}`,
          },
        }
      );
      if (!response.ok) throw new Error('Failed to fetch backtests');
      const data = await response.json();
      return data.backtests as BacktestResult[];
    },
    refetchInterval: 10000,
  });

  const runBacktestMutation = useMutation({
    mutationFn: async (config: BacktestConfig) => {
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/api/v1/backtests`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${localStorage.getItem('quantum_trader_token')}`,
          },
          body: JSON.stringify(config),
        }
      );
      if (!response.ok) throw new Error('Failed to start backtest');
      const data = await response.json();
      return data.backtest_id;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['backtests'] });
      setIsFormOpen(false);
    },
  });

  const handleRunBacktest = () => {
    const config: BacktestConfig = {
      strategyId: crypto.randomUUID(),
      strategyType: formData.strategyType,
      symbols: formData.symbols,
      exchanges: formData.exchanges,
      startDate: formData.startDate,
      endDate: formData.endDate,
      initialCapital: formData.initialCapital,
      commission: formData.commission,
      slippage: formData.slippage,
      parameters: formData.parameters,
    };

    runBacktestMutation.mutate(config);
  };

  const equityCurveData = useMemo(() => {
    if (!selectedResult?.equityCurve) return [];
    return selectedResult.equityCurve.map((point) => ({
      date: new Date(point.timestamp).getTime(),
      equity: point.equity,
      drawdown: point.drawdownPct * 100,
      benchmark: point.benchmark || selectedResult.initialCapital,
    }));
  }, [selectedResult]);

  const monthlyReturnsData = useMemo(() => {
    if (!selectedResult?.metrics) return [];

    const monthlyReturns = new Map<string, number>();

    selectedResult.metrics.forEach((metric) => {
      const date = new Date(metric.timestamp);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;

      if (!monthlyReturns.has(monthKey)) {
        monthlyReturns.set(monthKey, 0);
      }

      monthlyReturns.set(
        monthKey,
        monthlyReturns.get(monthKey)! + metric.dailyReturnPct
      );
    });

    return Array.from(monthlyReturns.entries()).map(([month, return_]) => ({
      month,
      return: return_ * 100,
    }));
  }, [selectedResult]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Strategy Backtesting</h1>
          <p className="text-slate-400">
            Test trading strategies against historical data and analyze
            performance metrics
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-2">
            {selectedResult ? (
              <div className="space-y-6">
                <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-xl font-semibold mb-2">
                        {selectedResult.id}
                      </h2>
                      <p className="text-slate-400 text-sm">
                        {formatDate(selectedResult.startDate)} -{' '}
                        {formatDate(selectedResult.endDate)}
                      </p>
                    </div>
                    <button
                      onClick={() => setSelectedResult(null)}
                      className="text-slate-400 hover:text-slate-200"
                    >
                      <svg
                        className="w-6 h-6"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <MetricCard
                      label="Total Return"
                      value={formatPercentage(
                        selectedResult.totalReturnPct / 100
                      )}
                      valueColor={
                        selectedResult.totalReturnPct > 0
                          ? 'text-green-500'
                          : 'text-red-500'
                      }
                    />
                    <MetricCard
                      label="Sharpe Ratio"
                      value={formatRatio(selectedResult.sharpeRatio)}
                      valueColor={
                        selectedResult.sharpeRatio > 1
                          ? 'text-green-500'
                          : 'text-yellow-500'
                      }
                    />
                    <MetricCard
                      label="Max Drawdown"
                      value={formatPercentage(
                        selectedResult.maxDrawdownPct / 100
                      )}
                      valueColor="text-red-500"
                    />
                    <MetricCard
                      label="Win Rate"
                      value={formatPercentage(selectedResult.winRatePct / 100)}
                      valueColor={
                        selectedResult.winRatePct > 50
                          ? 'text-green-500'
                          : 'text-yellow-500'
                      }
                    />
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-4">
                      Equity Curve
                    </h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={equityCurveData}>
                        <defs>
                          <linearGradient
                            id="equityGradient"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="5%"
                              stopColor={CHART_COLORS.primary}
                              stopOpacity={0.3}
                            />
                            <stop
                              offset="95%"
                              stopColor={CHART_COLORS.primary}
                              stopOpacity={0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#334155"
                        />
                        <XAxis
                          dataKey="date"
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          tickFormatter={(value) =>
                            formatDate(new Date(value), 'MMM dd')
                          }
                          stroke="#94a3b8"
                        />
                        <YAxis
                          tickFormatter={(value) =>
                            formatCurrency(value, 'USD', 0, false)
                          }
                          stroke="#94a3b8"
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '0.5rem',
                          }}
                          labelFormatter={(value) =>
                            formatDate(new Date(value))
                          }
                          formatter={(value: any) =>
                            formatCurrency(value, 'USD', 2)
                          }
                        />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="equity"
                          stroke={CHART_COLORS.primary}
                          fill="url(#equityGradient)"
                          name="Portfolio Value"
                        />
                        <Area
                          type="monotone"
                          dataKey="benchmark"
                          stroke={CHART_COLORS.neutral}
                          fill="none"
                          strokeDasharray="5 5"
                          name="Benchmark"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mb-6">
                    <h3 className="text-lg font-semibold mb-4">
                      Drawdown Analysis
                    </h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={equityCurveData}>
                        <defs>
                          <linearGradient
                            id="drawdownGradient"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="5%"
                              stopColor={CHART_COLORS.danger}
                              stopOpacity={0.3}
                            />
                            <stop
                              offset="95%"
                              stopColor={CHART_COLORS.danger}
                              stopOpacity={0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#334155"
                        />
                        <XAxis
                          dataKey="date"
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          tickFormatter={(value) =>
                            formatDate(new Date(value), 'MMM dd')
                          }
                          stroke="#94a3b8"
                        />
                        <YAxis
                          tickFormatter={(value) => `${value}%`}
                          stroke="#94a3b8"
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: '#1e293b',
                            border: '1px solid #334155',
                            borderRadius: '0.5rem',
                          }}
                          labelFormatter={(value) =>
                            formatDate(new Date(value))
                          }
                          formatter={(value: any) => `${value.toFixed(2)}%`}
                        />
                        <Area
                          type="monotone"
                          dataKey="drawdown"
                          stroke={CHART_COLORS.danger}
                          fill="url(#drawdownGradient)"
                          name="Drawdown"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {monthlyReturnsData.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold mb-4">
                        Monthly Returns
                      </h3>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={monthlyReturnsData}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke="#334155"
                          />
                          <XAxis dataKey="month" stroke="#94a3b8" />
                          <YAxis
                            tickFormatter={(value) => `${value}%`}
                            stroke="#94a3b8"
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#1e293b',
                              border: '1px solid #334155',
                              borderRadius: '0.5rem',
                            }}
                            formatter={(value: any) => `${value.toFixed(2)}%`}
                          />
                          <Bar
                            dataKey="return"
                            fill={CHART_COLORS.primary}
                            radius={[4, 4, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
                  <h3 className="text-lg font-semibold mb-4">
                    Performance Metrics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <MetricRow
                      label="Initial Capital"
                      value={formatCurrency(selectedResult.initialCapital)}
                    />
                    <MetricRow
                      label="Final Capital"
                      value={formatCurrency(selectedResult.finalCapital)}
                    />
                    <MetricRow
                      label="Total Trades"
                      value={formatNumber(selectedResult.numTrades, 0)}
                    />
                    <MetricRow
                      label="Avg Win"
                      value={formatCurrency(selectedResult.avgWin)}
                    />
                    <MetricRow
                      label="Avg Loss"
                      value={formatCurrency(selectedResult.avgLoss)}
                    />
                    <MetricRow
                      label="Profit Factor"
                      value={formatRatio(selectedResult.profitFactor)}
                    />
                    <MetricRow
                      label="Sortino Ratio"
                      value={formatRatio(selectedResult.sortinoRatio)}
                    />
                    <MetricRow
                      label="Annualized Return"
                      value={formatPercentage(
                        selectedResult.annualizedReturnPct / 100
                      )}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-12 text-center">
                <svg
                  className="w-16 h-16 mx-auto mb-4 text-slate-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <h3 className="text-xl font-semibold mb-2">
                  No Backtest Selected
                </h3>
                <p className="text-slate-400 mb-6">
                  Select a backtest from the list or create a new one to view
                  results
                </p>
                <button
                  onClick={() => setIsFormOpen(true)}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                >
                  Run New Backtest
                </button>
              </div>
            )}
          </div>

          <div className="space-y-6">
            <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Backtest History</h3>
                <button
                  onClick={() => setIsFormOpen(true)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
                >
                  New Backtest
                </button>
              </div>

              {isLoadingList ? (
                <div className="text-center py-8 text-slate-400">
                  Loading...
                </div>
              ) : backtestList && backtestList.length > 0 ? (
                <div className="space-y-2">
                  {backtestList.map((result) => (
                    <button
                      key={result.id}
                      onClick={() => setSelectedResult(result)}
                      className={`w-full text-left p-4 rounded-lg border transition-colors ${
                        selectedResult?.id === result.id
                          ? 'bg-blue-500/10 border-blue-500'
                          : 'bg-slate-800 border-slate-700 hover:bg-slate-800/80'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-medium">
                          {result.id.slice(0, 8)}
                        </span>
                        <span
                          className={`text-sm ${
                            result.totalReturnPct > 0
                              ? 'text-green-500'
                              : 'text-red-500'
                          }`}
                        >
                          {formatPercentage(result.totalReturnPct / 100)}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400">
                        {formatDate(result.completedAt || result.endDate)}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-400">
                  No backtests yet
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {isFormOpen && (
        <BacktestFormModal
          formData={formData}
          setFormData={setFormData}
          onSubmit={handleRunBacktest}
          onClose={() => setIsFormOpen(false)}
          isSubmitting={runBacktestMutation.isPending}
        />
      )}
    </div>
  );
};

const MetricCard: React.FC<{
  label: string;
  value: string;
  valueColor?: string;
}> = ({ label, value, valueColor = 'text-slate-100' }) => (
  <div>
    <div className="text-sm text-slate-400 mb-1">{label}</div>
    <div className={`text-2xl font-bold ${valueColor}`}>{value}</div>
  </div>
);

const MetricRow: React.FC<{ label: string; value: string }> = ({
  label,
  value,
}) => (
  <div className="flex justify-between items-center py-2 border-b border-slate-800 last:border-0">
    <span className="text-slate-400">{label}</span>
    <span className="font-semibold">{value}</span>
  </div>
);

const BacktestFormModal: React.FC<{
  formData: BacktestFormData;
  setFormData: React.Dispatch<React.SetStateAction<BacktestFormData>>;
  onSubmit: () => void;
  onClose: () => void;
  isSubmitting: boolean;
}> = ({ formData, setFormData, onSubmit, onClose, isSubmitting }) => (
  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
    <div className="bg-slate-900 rounded-lg border border-slate-800 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold">New Backtest</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Start Date
              </label>
              <input
                type="date"
                value={formData.startDate}
                onChange={(e) =>
                  setFormData({ ...formData, startDate: e.target.value })
                }
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                End Date
              </label>
              <input
                type="date"
                value={formData.endDate}
                onChange={(e) =>
                  setFormData({ ...formData, endDate: e.target.value })
                }
                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Initial Capital
            </label>
            <input
              type="number"
              value={formData.initialCapital}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  initialCapital: parseFloat(e.target.value),
                })
              }
              className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg"
            />
          </div>

          <div className="flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-6 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={onSubmit}
              disabled={isSubmitting}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
            >
              {isSubmitting ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
);

export default Backtesting;
