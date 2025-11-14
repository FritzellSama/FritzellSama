import React, { useState, useEffect, useRef } from 'react';
import { ExclamationCircleIcon } from '@heroicons/react/24/outline';

interface OrderBookLevel {
  price: string;
  quantity: string;
  total: string;
}

interface OrderBookData {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: string;
}

interface OrderBookProps {
  symbol: string;
  depth?: number;
  updateInterval?: number;
}

const OrderBook: React.FC<OrderBookProps> = ({
  symbol,
  depth = 20,
  updateInterval = 1000,
}) => {
  const [orderBook, setOrderBook] = useState<OrderBookData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');
  const [spread, setSpread] = useState<string>('0.00');
  const [spreadPercent, setSpreadPercent] = useState<string>('0.00');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);

  useEffect(() => {
    connectWebSocket();

    return () => {
      disconnectWebSocket();
    };
  }, [symbol]);

  useEffect(() => {
    if (orderBook && orderBook.asks.length > 0 && orderBook.bids.length > 0) {
      calculateSpread();
    }
  }, [orderBook]);

  const connectWebSocket = (): void => {
    try {
      const wsUrl = process.env.REACT_APP_WS_URL || window.REACT_APP_WS_URL;
      const token = localStorage.getItem('auth_token');

      const ws = new WebSocket(`${wsUrl}/ws/orderbook?symbol=${symbol}&token=${token}`);

      ws.onopen = () => {
        console.log('OrderBook WebSocket connected');
        setError('');
        reconnectAttemptsRef.current = 0;
        setLoading(false);

        ws.send(
          JSON.stringify({
            action: 'subscribe',
            symbol: symbol,
            depth: depth,
          })
        );
      };

      ws.onmessage = (event) => {
        try {
          const data: OrderBookData = JSON.parse(event.data);
          setOrderBook(data);
        } catch (err) {
          console.error('Error parsing orderbook data:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('OrderBook WebSocket error:', event);
        setError('WebSocket connection error');
      };

      ws.onclose = () => {
        console.log('OrderBook WebSocket closed');
        wsRef.current = null;

        const maxRetries = parseInt(process.env.REACT_APP_WS_MAX_RETRIES || '5', 10);
        if (reconnectAttemptsRef.current < maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          reconnectAttemptsRef.current++;

          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxRetries})...`);
            connectWebSocket();
          }, delay);
        } else {
          setError('Failed to connect to orderbook feed');
        }
      };

      wsRef.current = ws;
    } catch (err) {
      console.error('Error connecting to WebSocket:', err);
      setError('Failed to establish WebSocket connection');
      setLoading(false);
    }
  };

  const disconnectWebSocket = (): void => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const calculateSpread = (): void => {
    if (!orderBook || orderBook.asks.length === 0 || orderBook.bids.length === 0) {
      return;
    }

    const bestAsk = parseFloat(orderBook.asks[0].price);
    const bestBid = parseFloat(orderBook.bids[0].price);
    const spreadValue = bestAsk - bestBid;
    const spreadPercentValue = (spreadValue / bestBid) * 100;

    setSpread(spreadValue.toFixed(2));
    setSpreadPercent(spreadPercentValue.toFixed(4));
  };

  const getMaxTotal = (): number => {
    if (!orderBook) return 0;

    const maxBidTotal = Math.max(...orderBook.bids.map((b) => parseFloat(b.total)));
    const maxAskTotal = Math.max(...orderBook.asks.map((a) => parseFloat(a.total)));

    return Math.max(maxBidTotal, maxAskTotal);
  };

  const getDepthBarWidth = (total: string, maxTotal: number): number => {
    if (maxTotal === 0) return 0;
    return (parseFloat(total) / maxTotal) * 100;
  };

  if (loading) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <ExclamationCircleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-400">{error}</p>
            <button
              onClick={connectWebSocket}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  const maxTotal = getMaxTotal();

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="border-b border-gray-700 p-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold text-white">Order Book</h3>
          <div className="text-sm">
            <span className="text-gray-400">Spread: </span>
            <span className="text-white font-mono">{spread}</span>
            <span className="text-gray-400 ml-2">({spreadPercent}%)</span>
          </div>
        </div>
        <p className="text-sm text-gray-400 mt-1">{symbol}</p>
      </div>

      {/* Order Book Table */}
      <div className="grid grid-cols-2 divide-x divide-gray-700">
        {/* Bids (Buy Orders) */}
        <div className="p-4">
          <div className="text-xs text-gray-400 grid grid-cols-3 gap-2 mb-2 font-semibold">
            <div className="text-left">Price</div>
            <div className="text-right">Amount</div>
            <div className="text-right">Total</div>
          </div>
          <div className="space-y-1">
            {orderBook?.bids.slice(0, depth).map((bid, index) => (
              <div key={index} className="relative">
                {/* Depth bar */}
                <div
                  className="absolute inset-0 bg-green-500 bg-opacity-10"
                  style={{
                    width: `${getDepthBarWidth(bid.total, maxTotal)}%`,
                  }}
                ></div>
                {/* Order data */}
                <div className="relative grid grid-cols-3 gap-2 text-sm font-mono py-1">
                  <div className="text-green-400 text-left">{parseFloat(bid.price).toFixed(2)}</div>
                  <div className="text-white text-right">{parseFloat(bid.quantity).toFixed(4)}</div>
                  <div className="text-gray-400 text-right">{parseFloat(bid.total).toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Asks (Sell Orders) */}
        <div className="p-4">
          <div className="text-xs text-gray-400 grid grid-cols-3 gap-2 mb-2 font-semibold">
            <div className="text-left">Price</div>
            <div className="text-right">Amount</div>
            <div className="text-right">Total</div>
          </div>
          <div className="space-y-1">
            {orderBook?.asks.slice(0, depth).map((ask, index) => (
              <div key={index} className="relative">
                {/* Depth bar */}
                <div
                  className="absolute inset-0 bg-red-500 bg-opacity-10"
                  style={{
                    width: `${getDepthBarWidth(ask.total, maxTotal)}%`,
                  }}
                ></div>
                {/* Order data */}
                <div className="relative grid grid-cols-3 gap-2 text-sm font-mono py-1">
                  <div className="text-red-400 text-left">{parseFloat(ask.price).toFixed(2)}</div>
                  <div className="text-white text-right">{parseFloat(ask.quantity).toFixed(4)}</div>
                  <div className="text-gray-400 text-right">{parseFloat(ask.total).toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-700 p-4">
        <div className="flex justify-between items-center text-xs text-gray-400">
          <div>
            Last updated:{' '}
            {orderBook?.timestamp
              ? new Date(orderBook.timestamp).toLocaleTimeString()
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

export default OrderBook;
