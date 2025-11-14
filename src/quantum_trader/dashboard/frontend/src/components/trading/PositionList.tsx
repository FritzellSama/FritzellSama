import React, { useState, useEffect } from 'react';
import {
  XMarkIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
} from '@heroicons/react/24/outline';
import { ConfirmModal } from '../common/Modal';

interface Position {
  position_id: string;
  symbol: string;
  quantity: string;
  entry_price: string;
  current_price: string;
  pnl: string;
  pnl_percent: string;
  exchange: string;
  strategy: string;
  opened_at: string;
  side: 'LONG' | 'SHORT';
  leverage: string;
  liquidation_price: string;
}

interface PositionListProps {
  symbol?: string;
  refreshInterval?: number;
  onPositionClosed?: (positionId: string) => void;
}

const PositionList: React.FC<PositionListProps> = ({
  symbol,
  refreshInterval = 3000,
  onPositionClosed,
}) => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterExchange, setFilterExchange] = useState<string>('ALL');
  const [filterStrategy, setFilterStrategy] = useState<string>('ALL');
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [closeModalOpen, setCloseModalOpen] = useState<boolean>(false);
  const [selectedPositionId, setSelectedPositionId] = useState<string>('');
  const [closing, setClosing] = useState<boolean>(false);

  useEffect(() => {
    fetchPositions();

    const interval = setInterval(fetchPositions, refreshInterval);

    return () => clearInterval(interval);
  }, [symbol, refreshInterval]);

  const fetchPositions = async (showLoader: boolean = false): Promise<void> => {
    if (showLoader) {
      setLoading(true);
    }
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      let url = `${apiUrl}/api/v1/positions`;

      if (symbol) {
        url += `?symbol=${encodeURIComponent(symbol)}`;
      }

      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch positions: ${response.statusText}`);
      }

      const data = await response.json();
      setPositions(data.positions || []);
    } catch (err) {
      console.error('Error fetching positions:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load positions';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleClosePosition = async (positionId: string): Promise<void> => {
    setClosing(true);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const maxRetries = parseInt(process.env.REACT_APP_MAX_ORDER_RETRIES || '3', 10);

      let lastError: Error | null = null;

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const response = await fetch(`${apiUrl}/api/v1/positions/${positionId}/close`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
              'Content-Type': 'application/json',
            },
          });

          if (!response.ok) {
            const errorData = await response.json();

            if (response.status >= 500) {
              throw new Error(errorData.detail || 'Server error. Retrying...');
            } else {
              setError(errorData.detail || 'Failed to close position');
              return;
            }
          }

          await fetchPositions();
          setCloseModalOpen(false);

          if (onPositionClosed) {
            onPositionClosed(positionId);
          }

          return;
        } catch (err) {
          lastError = err as Error;

          if (attempt < maxRetries - 1) {
            const delay = Math.pow(2, attempt) * 1000;
            await new Promise((resolve) => setTimeout(resolve, delay));
          }
        }
      }

      throw lastError;
    } catch (err) {
      console.error('Error closing position:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to close position';
      setError(errorMessage);
    } finally {
      setClosing(false);
    }
  };

  const openCloseModal = (positionId: string): void => {
    setSelectedPositionId(positionId);
    setCloseModalOpen(true);
  };

  const getPnLColor = (pnl: string): string => {
    const value = parseFloat(pnl);
    if (value > 0) return 'text-green-400';
    if (value < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getSideColor = (side: string): string => {
    return side === 'LONG' ? 'text-green-400 bg-green-400/10 border-green-400/20' : 'text-red-400 bg-red-400/10 border-red-400/20';
  };

  const formatCurrency = (value: string): string => {
    return `$${parseFloat(value).toFixed(2)}`;
  };

  const formatPercent = (value: string): string => {
    const num = parseFloat(value);
    return num >= 0 ? `+${num.toFixed(2)}%` : `${num.toFixed(2)}%`;
  };

  const formatDate = (dateString: string): string => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  const getUniqueExchanges = (): string[] => {
    return Array.from(new Set(positions.map((p) => p.exchange)));
  };

  const getUniqueStrategies = (): string[] => {
    return Array.from(new Set(positions.map((p) => p.strategy)));
  };

  const filteredPositions = positions.filter((position) => {
    const matchesSearch =
      searchTerm === '' ||
      position.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      position.position_id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesExchange =
      filterExchange === 'ALL' || position.exchange === filterExchange;

    const matchesStrategy =
      filterStrategy === 'ALL' || position.strategy === filterStrategy;

    return matchesSearch && matchesExchange && matchesStrategy;
  });

  const getTotalPnL = (): string => {
    const total = filteredPositions.reduce((sum, pos) => sum + parseFloat(pos.pnl), 0);
    return formatCurrency(total.toString());
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h3 className="text-xl font-semibold text-white">Open Positions</h3>
            <p className="text-sm text-gray-400 mt-1">
              Total P&L: <span className={getPnLColor(getTotalPnL())}>{getTotalPnL()}</span>
            </p>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`p-2 rounded-md transition-colors ${
                showFilters
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
            >
              <FunnelIcon className="h-5 w-5" />
            </button>
            <button
              onClick={() => fetchPositions(true)}
              disabled={loading}
              className="p-2 bg-gray-700 text-gray-400 hover:bg-gray-600 rounded-md transition-colors"
            >
              <ArrowPathIcon className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search by symbol or position ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Filters */}
        {showFilters && (
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Exchange
              </label>
              <select
                value={filterExchange}
                onChange={(e) => setFilterExchange(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ALL">All</option>
                {getUniqueExchanges().map((exchange) => (
                  <option key={exchange} value={exchange}>
                    {exchange}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Strategy
              </label>
              <select
                value={filterStrategy}
                onChange={(e) => setFilterStrategy(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ALL">All</option>
                {getUniqueStrategies().map((strategy) => (
                  <option key={strategy} value={strategy}>
                    {strategy}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500 bg-opacity-10 border-b border-red-500 p-4">
          <span className="text-sm text-red-400">{error}</span>
        </div>
      )}

      {/* Position Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-700 text-gray-300 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">Symbol</th>
              <th className="px-4 py-3 text-left">Side</th>
              <th className="px-4 py-3 text-right">Quantity</th>
              <th className="px-4 py-3 text-right">Entry Price</th>
              <th className="px-4 py-3 text-right">Current Price</th>
              <th className="px-4 py-3 text-right">P&L</th>
              <th className="px-4 py-3 text-right">P&L %</th>
              <th className="px-4 py-3 text-center">Leverage</th>
              <th className="px-4 py-3 text-left">Exchange</th>
              <th className="px-4 py-3 text-left">Strategy</th>
              <th className="px-4 py-3 text-left">Opened</th>
              <th className="px-4 py-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {loading && positions.length === 0 ? (
              <tr>
                <td colSpan={12} className="px-4 py-8 text-center text-gray-400">
                  <div className="flex justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  </div>
                </td>
              </tr>
            ) : filteredPositions.length === 0 ? (
              <tr>
                <td colSpan={12} className="px-4 py-8 text-center text-gray-400">
                  No open positions
                </td>
              </tr>
            ) : (
              filteredPositions.map((position) => (
                <tr
                  key={position.position_id}
                  className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors"
                >
                  <td className="px-4 py-3 text-white font-medium">
                    {position.symbol}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 text-xs font-semibold rounded border ${getSideColor(position.side)}`}>
                      {position.side}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-white font-mono">
                    {parseFloat(position.quantity).toFixed(4)}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">
                    {formatCurrency(position.entry_price)}
                  </td>
                  <td className="px-4 py-3 text-right text-white font-mono">
                    {formatCurrency(position.current_price)}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono font-semibold ${getPnLColor(position.pnl)}`}>
                    {formatCurrency(position.pnl)}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono ${getPnLColor(position.pnl_percent)}`}>
                    {formatPercent(position.pnl_percent)}
                  </td>
                  <td className="px-4 py-3 text-center text-white">
                    {position.leverage}x
                  </td>
                  <td className="px-4 py-3 text-gray-300">{position.exchange}</td>
                  <td className="px-4 py-3 text-gray-300">{position.strategy}</td>
                  <td className="px-4 py-3 text-gray-300">
                    {formatDate(position.opened_at)}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <button
                      onClick={() => openCloseModal(position.position_id)}
                      className="p-1 text-red-400 hover:text-red-300 transition-colors"
                      title="Close position"
                    >
                      <XMarkIcon className="h-5 w-5" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Close Confirmation Modal */}
      <ConfirmModal
        isOpen={closeModalOpen}
        onClose={() => setCloseModalOpen(false)}
        onConfirm={() => handleClosePosition(selectedPositionId)}
        title="Close Position"
        message="Are you sure you want to close this position? This action cannot be undone."
        confirmText="Close Position"
        confirmButtonClass="bg-red-600 hover:bg-red-700"
        loading={closing}
      />
    </div>
  );
};

export default PositionList;
