/**
 * Trading Page
 *
 * Main trading interface with order placement, positions, and market data.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useState, useEffect, useCallback } from 'react';
import TradeHistory from '../components/trading/TradeHistory';
import VolumeChart from '../components/charts/VolumeChart';

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  pnl_percent: number;
  strategy: string;
  exchange: string;
}

interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

interface OrderBook {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  timestamp: string;
}

interface Ticker {
  symbol: string;
  last_price: number;
  bid: number;
  ask: number;
  volume_24h: number;
  change_24h: number;
  high_24h: number;
  low_24h: number;
}

const Trading: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>(
    process.env.REACT_APP_DEFAULT_SYMBOL || 'BTC/USDT'
  );
  const [positions, setPositions] = useState<Position[]>([]);
  const [orderBook, setOrderBook] = useState<OrderBook | null>(null);
  const [ticker, setTicker] = useState<Ticker | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Order form state
  const [orderSide, setOrderSide] = useState<'BUY' | 'SELL'>('BUY');
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('LIMIT');
  const [orderQuantity, setOrderQuantity] = useState<string>('');
  const [orderPrice, setOrderPrice] = useState<string>('');
  const [submitting, setSubmitting] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const REFRESH_INTERVAL = parseInt(process.env.REACT_APP_TRADING_REFRESH_INTERVAL || '1000');

  // Available symbols
  const symbols = (process.env.REACT_APP_TRADING_SYMBOLS || 'BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT').split(',');

  // Fetch positions
  const fetchPositions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/positions`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch positions');
      }

      const data = await response.json();
      setPositions(data);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching positions:', err);
    }
  }, [API_BASE_URL]);

  // Fetch order book
  const fetchOrderBook = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/market/${selectedSymbol}/orderbook`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch order book');
      }

      const data = await response.json();
      setOrderBook(data);
    } catch (err) {
      console.error('Error fetching order book:', err);
    }
  }, [API_BASE_URL, selectedSymbol]);

  // Fetch ticker
  const fetchTicker = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/market/${selectedSymbol}/ticker`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch ticker');
      }

      const data = await response.json();
      setTicker(data);

      // Auto-fill price for limit orders
      if (orderType === 'LIMIT' && !orderPrice) {
        setOrderPrice(data.last_price.toFixed(2));
      }
    } catch (err) {
      console.error('Error fetching ticker:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL, selectedSymbol, orderType, orderPrice]);

  useEffect(() => {
    fetchPositions();
    fetchOrderBook();
    fetchTicker();

    const interval = setInterval(() => {
      fetchOrderBook();
      fetchTicker();
    }, REFRESH_INTERVAL);

    return () => clearInterval(interval);
  }, [fetchPositions, fetchOrderBook, fetchTicker, REFRESH_INTERVAL]);

  // Submit order
  const submitOrder = async () => {
    if (!orderQuantity || (orderType === 'LIMIT' && !orderPrice)) {
      alert('Please fill in all required fields');
      return;
    }

    if (!window.confirm(`Place ${orderSide} order for ${orderQuantity} ${selectedSymbol}?`)) {
      return;
    }

    try {
      setSubmitting(true);

      const orderData = {
        symbol: selectedSymbol,
        side: orderSide,
        quantity: parseFloat(orderQuantity),
        order_type: orderType,
        ...(orderType === 'LIMIT' && { price: parseFloat(orderPrice) })
      };

      const response = await fetch(`${API_BASE_URL}/api/v1/orders`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(orderData)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to place order');
      }

      const result = await response.json();
      alert(`Order placed successfully! Order ID: ${result.order_id}`);

      // Reset form
      setOrderQuantity('');
      setOrderPrice('');

      // Refresh positions
      await fetchPositions();
    } catch (err) {
      console.error('Error placing order:', err);
      alert(`Failed to place order: ${(err as Error).message}`);
    } finally {
      setSubmitting(false);
    }
  };

  // Close position
  const closePosition = async (symbol: string) => {
    if (!window.confirm(`Close position for ${symbol}?`)) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/positions/${symbol}/close`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Failed to close position');
      }

      alert(`Position ${symbol} closed successfully`);
      await fetchPositions();
    } catch (err) {
      console.error('Error closing position:', err);
      alert(`Failed to close position: ${(err as Error).message}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Trading</h1>
        <p className="text-gray-400">Execute trades and manage positions</p>
      </div>

      {error && (
        <div className="mb-4 bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-300">
          Error: {error.message}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Order Entry */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Place Order</h2>

            {/* Symbol Selection */}
            <div className="mb-4">
              <label className="block text-gray-300 mb-2">Symbol</label>
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>

            {/* Current Price */}
            {ticker && (
              <div className="mb-4 p-4 bg-gray-700/50 rounded-lg">
                <div className="text-gray-400 text-sm mb-1">Current Price</div>
                <div className="text-3xl font-bold text-white mb-2">
                  ${ticker.last_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className={`text-sm font-semibold ${
                  ticker.change_24h >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {ticker.change_24h >= 0 ? '+' : ''}{ticker.change_24h.toFixed(2)}% (24h)
                </div>
              </div>
            )}

            {/* Side Selection */}
            <div className="mb-4">
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setOrderSide('BUY')}
                  className={`px-4 py-3 rounded-md font-semibold transition-colors ${
                    orderSide === 'BUY'
                      ? 'bg-green-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  BUY
                </button>
                <button
                  onClick={() => setOrderSide('SELL')}
                  className={`px-4 py-3 rounded-md font-semibold transition-colors ${
                    orderSide === 'SELL'
                      ? 'bg-red-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  SELL
                </button>
              </div>
            </div>

            {/* Order Type */}
            <div className="mb-4">
              <label className="block text-gray-300 mb-2">Order Type</label>
              <select
                value={orderType}
                onChange={(e) => setOrderType(e.target.value as 'MARKET' | 'LIMIT')}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="LIMIT">Limit</option>
                <option value="MARKET">Market</option>
              </select>
            </div>

            {/* Price (for limit orders) */}
            {orderType === 'LIMIT' && (
              <div className="mb-4">
                <label className="block text-gray-300 mb-2">Price (USDT)</label>
                <input
                  type="number"
                  step="0.01"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(e.target.value)}
                  placeholder="0.00"
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}

            {/* Quantity */}
            <div className="mb-4">
              <label className="block text-gray-300 mb-2">Quantity</label>
              <input
                type="number"
                step="0.00000001"
                value={orderQuantity}
                onChange={(e) => setOrderQuantity(e.target.value)}
                placeholder="0.00"
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Total */}
            {orderQuantity && (orderType === 'MARKET' || orderPrice) && (
              <div className="mb-4 p-3 bg-gray-700/50 rounded-lg">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Total (est.)</span>
                  <span className="text-white font-mono">
                    ${(parseFloat(orderQuantity) * (orderType === 'MARKET' ? (ticker?.last_price || 0) : parseFloat(orderPrice))).toFixed(2)}
                  </span>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={submitOrder}
              disabled={submitting || loading}
              className={`w-full px-6 py-3 rounded-md font-bold text-white transition-colors ${
                orderSide === 'BUY'
                  ? 'bg-green-600 hover:bg-green-700 disabled:bg-gray-700'
                  : 'bg-red-600 hover:bg-red-700 disabled:bg-gray-700'
              } disabled:cursor-not-allowed`}
            >
              {submitting ? 'Placing Order...' : `${orderSide} ${selectedSymbol}`}
            </button>
          </div>
        </div>

        {/* Order Book */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
            <h2 className="text-xl font-bold text-white mb-4">Order Book</h2>
            {orderBook && (
              <div className="space-y-4">
                {/* Asks (Sell Orders) */}
                <div>
                  <div className="text-sm text-gray-400 mb-2">Asks</div>
                  <div className="space-y-1">
                    {orderBook.asks.slice(0, 10).reverse().map((ask, index) => (
                      <div
                        key={index}
                        className="flex justify-between text-sm bg-red-900/10 hover:bg-red-900/20 px-2 py-1 rounded"
                      >
                        <span className="text-red-400 font-mono">{ask.price.toFixed(2)}</span>
                        <span className="text-gray-300 font-mono">{ask.quantity.toFixed(8)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Spread */}
                {ticker && (
                  <div className="py-3 bg-gray-700/50 rounded-lg text-center">
                    <div className="text-gray-400 text-xs mb-1">Spread</div>
                    <div className="text-white font-mono font-semibold">
                      ${(ticker.ask - ticker.bid).toFixed(2)}
                    </div>
                  </div>
                )}

                {/* Bids (Buy Orders) */}
                <div>
                  <div className="text-sm text-gray-400 mb-2">Bids</div>
                  <div className="space-y-1">
                    {orderBook.bids.slice(0, 10).map((bid, index) => (
                      <div
                        key={index}
                        className="flex justify-between text-sm bg-green-900/10 hover:bg-green-900/20 px-2 py-1 rounded"
                      >
                        <span className="text-green-400 font-mono">{bid.price.toFixed(2)}</span>
                        <span className="text-gray-300 font-mono">{bid.quantity.toFixed(8)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Market Info */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-white mb-4">Market Info</h2>
            {ticker && (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">24h High</span>
                  <span className="text-white font-mono">${ticker.high_24h.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">24h Low</span>
                  <span className="text-white font-mono">${ticker.low_24h.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">24h Volume</span>
                  <span className="text-white font-mono">${(ticker.volume_24h / 1000000).toFixed(2)}M</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Bid</span>
                  <span className="text-green-400 font-mono">${ticker.bid.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Ask</span>
                  <span className="text-red-400 font-mono">${ticker.ask.toFixed(2)}</span>
                </div>
              </div>
            )}
          </div>

          {/* Volume Chart */}
          {ticker && (
            <VolumeChart
              symbol={selectedSymbol}
              timeframe="1h"
            />
          )}
        </div>
      </div>

      {/* Open Positions */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-4">Open Positions</h2>
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-gray-300 font-semibold">Symbol</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Quantity</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Entry Price</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Current Price</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">Unrealized P&L</th>
                  <th className="px-4 py-3 text-right text-gray-300 font-semibold">P&L %</th>
                  <th className="px-4 py-3 text-left text-gray-300 font-semibold">Strategy</th>
                  <th className="px-4 py-3 text-center text-gray-300 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {positions.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-4 py-12 text-center text-gray-400">
                      No open positions
                    </td>
                  </tr>
                )}
                {positions.map((position, index) => (
                  <tr key={index} className="hover:bg-gray-700/30 transition-colors">
                    <td className="px-4 py-3 text-white font-mono font-semibold">{position.symbol}</td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      {position.quantity.toFixed(8)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      ${position.entry_price.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-gray-200">
                      ${position.current_price.toFixed(2)}
                    </td>
                    <td className={`px-4 py-3 text-right font-mono font-bold ${
                      position.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.unrealized_pnl >= 0 ? '+' : ''}
                      ${position.unrealized_pnl.toFixed(2)}
                    </td>
                    <td className={`px-4 py-3 text-right font-mono font-bold ${
                      position.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.pnl_percent >= 0 ? '+' : ''}{position.pnl_percent.toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 text-gray-300">{position.strategy}</td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => closePosition(position.symbol)}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                      >
                        Close
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Trade History */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-4">Trade History</h2>
        <TradeHistory limit={50} />
      </div>
    </div>
  );
};

export default Trading;
