/**
 * TradeHistory Component
 *
 * Displays historical trade data with filtering and export capabilities.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Table, { Column } from '../common/Table';

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  total: number;
  fee: number;
  pnl?: number;
  strategy: string;
  exchange: string;
  order_type: string;
  status: 'FILLED' | 'PARTIAL' | 'CANCELLED' | 'REJECTED';
}

interface TradeHistoryProps {
  limit?: number;
  autoRefresh?: boolean;
  onTradeClick?: (trade: Trade) => void;
}

const TradeHistory: React.FC<TradeHistoryProps> = ({
  limit = parseInt(process.env.REACT_APP_TRADE_HISTORY_LIMIT || '100'),
  autoRefresh = true,
  onTradeClick
}) => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [dateFilter, setDateFilter] = useState<'all' | 'today' | 'week' | 'month'>('all');
  const [sideFilter, setSideFilter] = useState<'all' | 'BUY' | 'SELL'>('all');
  const [symbolFilter, setSymbolFilter] = useState<string>('');
  const [strategyFilter, setStrategyFilter] = useState<string>('all');

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_TRADE_HISTORY_REFRESH || '30000');

  // Fetch trades
  const fetchTrades = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        limit: limit.toString()
      });

      const response = await fetch(`${API_BASE_URL}/api/v1/trades?${params}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch trade history');
      }

      const data = await response.json();
      setTrades(data);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching trades:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL, limit]);

  useEffect(() => {
    fetchTrades();

    if (autoRefresh) {
      const interval = setInterval(fetchTrades, REFRESH_INTERVAL);
      return () => clearInterval(interval);
    }
  }, [fetchTrades, autoRefresh, REFRESH_INTERVAL]);

  // Filter trades
  const filteredTrades = useMemo(() => {
    let filtered = trades;

    // Date filter
    if (dateFilter !== 'all') {
      const now = new Date();
      const cutoff = new Date();

      switch (dateFilter) {
        case 'today':
          cutoff.setHours(0, 0, 0, 0);
          break;
        case 'week':
          cutoff.setDate(now.getDate() - 7);
          break;
        case 'month':
          cutoff.setMonth(now.getMonth() - 1);
          break;
      }

      filtered = filtered.filter(trade => new Date(trade.timestamp) >= cutoff);
    }

    // Side filter
    if (sideFilter !== 'all') {
      filtered = filtered.filter(trade => trade.side === sideFilter);
    }

    // Symbol filter
    if (symbolFilter) {
      filtered = filtered.filter(trade =>
        trade.symbol.toLowerCase().includes(symbolFilter.toLowerCase())
      );
    }

    // Strategy filter
    if (strategyFilter !== 'all') {
      filtered = filtered.filter(trade => trade.strategy === strategyFilter);
    }

    return filtered;
  }, [trades, dateFilter, sideFilter, symbolFilter, strategyFilter]);

  // Get unique strategies
  const strategies = useMemo(() => {
    const unique = new Set(trades.map(t => t.strategy));
    return ['all', ...Array.from(unique)];
  }, [trades]);

  // Calculate statistics
  const stats = useMemo(() => {
    const totalTrades = filteredTrades.length;
    const buyTrades = filteredTrades.filter(t => t.side === 'BUY').length;
    const sellTrades = filteredTrades.filter(t => t.side === 'SELL').length;
    const totalVolume = filteredTrades.reduce((sum, t) => sum + t.total, 0);
    const totalFees = filteredTrades.reduce((sum, t) => sum + t.fee, 0);
    const totalPnL = filteredTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const winningTrades = filteredTrades.filter(t => (t.pnl || 0) > 0).length;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    return {
      totalTrades,
      buyTrades,
      sellTrades,
      totalVolume,
      totalFees,
      totalPnL,
      winRate
    };
  }, [filteredTrades]);

  // Export to CSV
  const exportToCSV = useCallback(() => {
    const headers = ['Timestamp', 'Symbol', 'Side', 'Quantity', 'Price', 'Total', 'Fee', 'P&L', 'Strategy', 'Exchange', 'Status'];
    const rows = filteredTrades.map(trade => [
      trade.timestamp,
      trade.symbol,
      trade.side,
      trade.quantity,
      trade.price,
      trade.total,
      trade.fee,
      trade.pnl || 0,
      trade.strategy,
      trade.exchange,
      trade.status
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `trade_history_${new Date().toISOString()}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [filteredTrades]);

  // Table columns
  const columns: Column<Trade>[] = [
    {
      key: 'timestamp',
      label: 'Time',
      sortable: true,
      render: (trade) => (
        <span className="text-gray-200 text-sm">
          {new Date(trade.timestamp).toLocaleString()}
        </span>
      )
    },
    {
      key: 'symbol',
      label: 'Symbol',
      sortable: true,
      render: (trade) => (
        <span className="text-white font-mono font-semibold">{trade.symbol}</span>
      )
    },
    {
      key: 'side',
      label: 'Side',
      sortable: true,
      align: 'center',
      render: (trade) => (
        <span className={`px-2 py-1 rounded text-xs font-bold ${
          trade.side === 'BUY' ? 'bg-green-600' : 'bg-red-600'
        } text-white`}>
          {trade.side}
        </span>
      )
    },
    {
      key: 'quantity',
      label: 'Quantity',
      sortable: true,
      align: 'right',
      render: (trade) => (
        <span className="text-gray-200 font-mono">{trade.quantity.toFixed(8)}</span>
      )
    },
    {
      key: 'price',
      label: 'Price',
      sortable: true,
      align: 'right',
      render: (trade) => (
        <span className="text-gray-200 font-mono">
          ${trade.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      )
    },
    {
      key: 'total',
      label: 'Total',
      sortable: true,
      align: 'right',
      render: (trade) => (
        <span className="text-white font-mono font-semibold">
          ${trade.total.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
      )
    },
    {
      key: 'fee',
      label: 'Fee',
      sortable: true,
      align: 'right',
      render: (trade) => (
        <span className="text-gray-400 font-mono text-sm">
          ${trade.fee.toFixed(2)}
        </span>
      )
    },
    {
      key: 'pnl',
      label: 'P&L',
      sortable: true,
      align: 'right',
      render: (trade) => {
        if (!trade.pnl) return <span className="text-gray-500">-</span>;
        return (
          <span className={`font-mono font-bold ${
            trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {trade.pnl >= 0 ? '+' : ''}
            ${trade.pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        );
      }
    },
    {
      key: 'strategy',
      label: 'Strategy',
      sortable: true,
      render: (trade) => (
        <span className="text-gray-300">{trade.strategy}</span>
      )
    },
    {
      key: 'exchange',
      label: 'Exchange',
      sortable: true,
      render: (trade) => (
        <span className="text-gray-300">{trade.exchange}</span>
      )
    },
    {
      key: 'status',
      label: 'Status',
      sortable: true,
      align: 'center',
      render: (trade) => (
        <span className={`px-2 py-1 rounded text-xs font-bold ${
          trade.status === 'FILLED' ? 'bg-blue-600' :
          trade.status === 'PARTIAL' ? 'bg-yellow-600' :
          trade.status === 'CANCELLED' ? 'bg-gray-600' :
          'bg-red-600'
        } text-white`}>
          {trade.status}
        </span>
      )
    }
  ];

  return (
    <div className="space-y-4">
      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Total Trades</div>
          <div className="text-2xl font-bold text-white">{stats.totalTrades}</div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Buy Trades</div>
          <div className="text-2xl font-bold text-green-400">{stats.buyTrades}</div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Sell Trades</div>
          <div className="text-2xl font-bold text-red-400">{stats.sellTrades}</div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Total Volume</div>
          <div className="text-2xl font-bold text-blue-400">
            ${(stats.totalVolume / 1000).toFixed(1)}K
          </div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Total Fees</div>
          <div className="text-2xl font-bold text-orange-400">
            ${stats.totalFees.toFixed(2)}
          </div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Total P&L</div>
          <div className={`text-2xl font-bold ${
            stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {stats.totalPnL >= 0 ? '+' : ''}${stats.totalPnL.toFixed(2)}
          </div>
        </div>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
          <div className="text-gray-400 text-xs mb-1">Win Rate</div>
          <div className="text-2xl font-bold text-purple-400">
            {stats.winRate.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="block text-gray-400 text-sm mb-2">Date Range</label>
            <select
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value as any)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Time</option>
              <option value="today">Today</option>
              <option value="week">Last 7 Days</option>
              <option value="month">Last 30 Days</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Side</label>
            <select
              value={sideFilter}
              onChange={(e) => setSideFilter(e.target.value as any)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Sides</option>
              <option value="BUY">Buy Only</option>
              <option value="SELL">Sell Only</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Symbol</label>
            <input
              type="text"
              placeholder="Filter symbol..."
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Strategy</label>
            <select
              value={strategyFilter}
              onChange={(e) => setStrategyFilter(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {strategies.map(strategy => (
                <option key={strategy} value={strategy}>
                  {strategy === 'all' ? 'All Strategies' : strategy}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end gap-2">
            <button
              onClick={fetchTrades}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
            >
              Refresh
            </button>
            <button
              onClick={exportToCSV}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors"
            >
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Trade Table */}
      <Table
        data={filteredTrades}
        columns={columns}
        keyExtractor={(trade) => trade.id}
        loading={loading}
        error={error}
        emptyMessage="No trades found"
        onRowClick={onTradeClick}
        sortable={true}
        filterable={false}
        paginated={true}
        pageSize={20}
      />
    </div>
  );
};

export default TradeHistory;
