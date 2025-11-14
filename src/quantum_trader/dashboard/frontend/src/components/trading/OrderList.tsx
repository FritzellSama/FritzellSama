import React, { useState, useEffect } from 'react';
import {
  XMarkIcon,
  ArrowPathIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline';
import { ConfirmModal } from '../common/Modal';

interface Order {
  order_id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  order_type: 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'TAKE_PROFIT' | 'STOP_LIMIT';
  quantity: string;
  filled_quantity: string;
  price: string;
  average_price: string;
  status: 'PENDING' | 'OPEN' | 'PARTIAL' | 'FILLED' | 'CANCELLED' | 'REJECTED' | 'EXPIRED' | 'FAILED';
  exchange: string;
  strategy: string;
  timestamp: string;
  updated_at: string;
}

interface OrderListProps {
  symbol?: string;
  refreshInterval?: number;
}

const OrderList: React.FC<OrderListProps> = ({
  symbol,
  refreshInterval = 5000,
}) => {
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [filterStatus, setFilterStatus] = useState<string>('ALL');
  const [filterSide, setFilterSide] = useState<string>('ALL');
  const [showFilters, setShowFilters] = useState<boolean>(false);
  const [cancelModalOpen, setCancelModalOpen] = useState<boolean>(false);
  const [selectedOrderId, setSelectedOrderId] = useState<string>('');
  const [cancelling, setCancelling] = useState<boolean>(false);

  useEffect(() => {
    fetchOrders();

    const interval = setInterval(fetchOrders, refreshInterval);

    return () => clearInterval(interval);
  }, [symbol, refreshInterval]);

  const fetchOrders = async (showLoader: boolean = false): Promise<void> => {
    if (showLoader) {
      setLoading(true);
    }
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      let url = `${apiUrl}/api/v1/orders`;

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
        throw new Error(`Failed to fetch orders: ${response.statusText}`);
      }

      const data = await response.json();
      setOrders(data.orders || []);
    } catch (err) {
      console.error('Error fetching orders:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load orders';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCancelOrder = async (orderId: string): Promise<void> => {
    setCancelling(true);

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const maxRetries = parseInt(process.env.REACT_APP_MAX_ORDER_RETRIES || '3', 10);

      let lastError: Error | null = null;

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const response = await fetch(`${apiUrl}/api/v1/orders/${orderId}`, {
            method: 'DELETE',
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
              setError(errorData.detail || 'Failed to cancel order');
              return;
            }
          }

          await fetchOrders();
          setCancelModalOpen(false);
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
      console.error('Error cancelling order:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to cancel order';
      setError(errorMessage);
    } finally {
      setCancelling(false);
    }
  };

  const openCancelModal = (orderId: string): void => {
    setSelectedOrderId(orderId);
    setCancelModalOpen(true);
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'FILLED':
        return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'OPEN':
      case 'PARTIAL':
        return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
      case 'CANCELLED':
      case 'EXPIRED':
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
      case 'REJECTED':
      case 'FAILED':
        return 'text-red-400 bg-red-400/10 border-red-400/20';
      case 'PENDING':
        return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      default:
        return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
    }
  };

  const getSideColor = (side: string): string => {
    return side === 'BUY' ? 'text-green-400' : 'text-red-400';
  };

  const formatDate = (dateString: string): string => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch {
      return dateString;
    }
  };

  const canCancelOrder = (status: string): boolean => {
    return status === 'OPEN' || status === 'PARTIAL' || status === 'PENDING';
  };

  const filteredOrders = orders.filter((order) => {
    const matchesSearch =
      searchTerm === '' ||
      order.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      order.order_id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus =
      filterStatus === 'ALL' || order.status === filterStatus;

    const matchesSide =
      filterSide === 'ALL' || order.side === filterSide;

    return matchesSearch && matchesStatus && matchesSide;
  });

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-xl font-semibold text-white">Orders</h3>
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
              onClick={() => fetchOrders(true)}
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
            placeholder="Search by symbol or order ID..."
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
                Status
              </label>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ALL">All</option>
                <option value="PENDING">Pending</option>
                <option value="OPEN">Open</option>
                <option value="PARTIAL">Partial</option>
                <option value="FILLED">Filled</option>
                <option value="CANCELLED">Cancelled</option>
                <option value="REJECTED">Rejected</option>
                <option value="EXPIRED">Expired</option>
                <option value="FAILED">Failed</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Side
              </label>
              <select
                value={filterSide}
                onChange={(e) => setFilterSide(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ALL">All</option>
                <option value="BUY">Buy</option>
                <option value="SELL">Sell</option>
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

      {/* Order Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-700 text-gray-300 text-xs uppercase">
            <tr>
              <th className="px-4 py-3 text-left">Time</th>
              <th className="px-4 py-3 text-left">Symbol</th>
              <th className="px-4 py-3 text-left">Side</th>
              <th className="px-4 py-3 text-left">Type</th>
              <th className="px-4 py-3 text-right">Quantity</th>
              <th className="px-4 py-3 text-right">Filled</th>
              <th className="px-4 py-3 text-right">Price</th>
              <th className="px-4 py-3 text-left">Status</th>
              <th className="px-4 py-3 text-left">Exchange</th>
              <th className="px-4 py-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {loading && orders.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-4 py-8 text-center text-gray-400">
                  <div className="flex justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  </div>
                </td>
              </tr>
            ) : filteredOrders.length === 0 ? (
              <tr>
                <td colSpan={10} className="px-4 py-8 text-center text-gray-400">
                  No orders found
                </td>
              </tr>
            ) : (
              filteredOrders.map((order) => (
                <tr
                  key={order.order_id}
                  className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors"
                >
                  <td className="px-4 py-3 text-gray-300">
                    {formatDate(order.timestamp)}
                  </td>
                  <td className="px-4 py-3 text-white font-medium">
                    {order.symbol}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`font-semibold ${getSideColor(order.side)}`}>
                      {order.side}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-300">{order.order_type}</td>
                  <td className="px-4 py-3 text-right text-white font-mono">
                    {parseFloat(order.quantity).toFixed(8)}
                  </td>
                  <td className="px-4 py-3 text-right text-gray-300 font-mono">
                    {parseFloat(order.filled_quantity).toFixed(8)}
                  </td>
                  <td className="px-4 py-3 text-right text-white font-mono">
                    {order.price ? `$${parseFloat(order.price).toFixed(2)}` : 'Market'}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded border ${getStatusColor(
                        order.status
                      )}`}
                    >
                      {order.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-300">{order.exchange}</td>
                  <td className="px-4 py-3 text-center">
                    {canCancelOrder(order.status) && (
                      <button
                        onClick={() => openCancelModal(order.order_id)}
                        className="p-1 text-red-400 hover:text-red-300 transition-colors"
                        title="Cancel order"
                      >
                        <XMarkIcon className="h-5 w-5" />
                      </button>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Cancel Confirmation Modal */}
      <ConfirmModal
        isOpen={cancelModalOpen}
        onClose={() => setCancelModalOpen(false)}
        onConfirm={() => handleCancelOrder(selectedOrderId)}
        title="Cancel Order"
        message="Are you sure you want to cancel this order?"
        confirmText="Cancel Order"
        confirmButtonClass="bg-red-600 hover:bg-red-700"
        loading={cancelling}
      />
    </div>
  );
};

export default OrderList;
