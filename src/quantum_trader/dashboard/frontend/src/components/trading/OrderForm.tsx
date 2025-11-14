import React, { useState, useEffect } from 'react';
import { ExclamationCircleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';

interface OrderFormData {
  symbol: string;
  side: 'BUY' | 'SELL';
  orderType: 'MARKET' | 'LIMIT' | 'STOP_LOSS' | 'TAKE_PROFIT';
  quantity: string;
  price: string;
  stopPrice: string;
}

interface Balance {
  currency: string;
  available: string;
  locked: string;
}

interface OrderFormProps {
  symbol?: string;
  onOrderPlaced?: (orderId: string) => void;
}

const OrderForm: React.FC<OrderFormProps> = ({ symbol: initialSymbol = 'BTC/USDT', onOrderPlaced }) => {
  const [formData, setFormData] = useState<OrderFormData>({
    symbol: initialSymbol,
    side: 'BUY',
    orderType: 'LIMIT',
    quantity: '',
    price: '',
    stopPrice: '',
  });
  const [balance, setBalance] = useState<Balance[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [currentPrice, setCurrentPrice] = useState<string>('0.00');
  const [estimatedTotal, setEstimatedTotal] = useState<string>('0.00');

  useEffect(() => {
    fetchBalance();
    fetchCurrentPrice();

    const priceInterval = setInterval(fetchCurrentPrice, 5000);

    return () => clearInterval(priceInterval);
  }, [formData.symbol]);

  useEffect(() => {
    calculateEstimatedTotal();
  }, [formData.quantity, formData.price, formData.orderType]);

  const fetchBalance = async (): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/account/balance`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch balance');
      }

      const data = await response.json();
      setBalance(data.balances || []);
    } catch (err) {
      console.error('Error fetching balance:', err);
    }
  };

  const fetchCurrentPrice = async (): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(
        `${apiUrl}/api/v1/market/ticker?symbol=${encodeURIComponent(formData.symbol)}`,
        {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (!response.ok) {
        throw new Error('Failed to fetch current price');
      }

      const data = await response.json();
      setCurrentPrice(data.last_price || '0.00');
    } catch (err) {
      console.error('Error fetching current price:', err);
    }
  };

  const calculateEstimatedTotal = (): void => {
    const quantity = parseFloat(formData.quantity) || 0;
    const price =
      formData.orderType === 'MARKET'
        ? parseFloat(currentPrice)
        : parseFloat(formData.price) || 0;

    const total = quantity * price;
    setEstimatedTotal(total.toFixed(2));
  };

  const getAvailableBalance = (currency: string): string => {
    const bal = balance.find((b) => b.currency === currency);
    return bal ? parseFloat(bal.available).toFixed(8) : '0.00000000';
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ): void => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    setError('');
    setSuccess('');
  };

  const validateForm = (): boolean => {
    if (!formData.symbol.trim()) {
      setError('Symbol is required');
      return false;
    }

    const quantity = parseFloat(formData.quantity);
    if (isNaN(quantity) || quantity <= 0) {
      setError('Quantity must be greater than 0');
      return false;
    }

    if (formData.orderType === 'LIMIT' || formData.orderType === 'STOP_LOSS' || formData.orderType === 'TAKE_PROFIT') {
      const price = parseFloat(formData.price);
      if (isNaN(price) || price <= 0) {
        setError('Price must be greater than 0');
        return false;
      }
    }

    if (formData.orderType === 'STOP_LOSS' || formData.orderType === 'TAKE_PROFIT') {
      const stopPrice = parseFloat(formData.stopPrice);
      if (isNaN(stopPrice) || stopPrice <= 0) {
        setError('Stop price must be greater than 0');
        return false;
      }
    }

    const [base, quote] = formData.symbol.split('/');
    const requiredCurrency = formData.side === 'BUY' ? quote : base;
    const availableBal = parseFloat(getAvailableBalance(requiredCurrency));

    const requiredAmount =
      formData.side === 'BUY' ? parseFloat(estimatedTotal) : quantity;

    if (requiredAmount > availableBal) {
      setError(`Insufficient ${requiredCurrency} balance`);
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const maxRetries = parseInt(process.env.REACT_APP_MAX_ORDER_RETRIES || '3', 10);

      let lastError: Error | null = null;

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const payload: any = {
            symbol: formData.symbol,
            side: formData.side,
            order_type: formData.orderType,
            quantity: formData.quantity,
          };

          if (formData.orderType !== 'MARKET') {
            payload.price = formData.price;
          }

          if (formData.orderType === 'STOP_LOSS' || formData.orderType === 'TAKE_PROFIT') {
            payload.stop_price = formData.stopPrice;
          }

          const response = await fetch(`${apiUrl}/api/v1/orders`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
          });

          if (!response.ok) {
            const errorData = await response.json();

            if (response.status >= 500) {
              throw new Error(errorData.detail || 'Server error. Retrying...');
            } else {
              setError(errorData.detail || 'Failed to place order');
              setLoading(false);
              return;
            }
          }

          const data = await response.json();
          setSuccess(`Order placed successfully! Order ID: ${data.order_id}`);

          setFormData((prev) => ({
            ...prev,
            quantity: '',
            price: '',
            stopPrice: '',
          }));

          await fetchBalance();

          if (onOrderPlaced) {
            onOrderPlaced(data.order_id);
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
      console.error('Order submission error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to place order';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleUseMaxBalance = (): void => {
    const [base, quote] = formData.symbol.split('/');
    const currency = formData.side === 'BUY' ? quote : base;
    const availableBal = parseFloat(getAvailableBalance(currency));

    if (formData.side === 'BUY') {
      const price =
        formData.orderType === 'MARKET'
          ? parseFloat(currentPrice)
          : parseFloat(formData.price) || parseFloat(currentPrice);

      if (price > 0) {
        const maxQuantity = availableBal / price;
        setFormData((prev) => ({
          ...prev,
          quantity: maxQuantity.toFixed(8),
        }));
      }
    } else {
      setFormData((prev) => ({
        ...prev,
        quantity: availableBal.toFixed(8),
      }));
    }
  };

  const [base, quote] = formData.symbol.split('/');

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <h3 className="text-xl font-semibold text-white mb-6">Place Order</h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Error/Success Messages */}
        {error && (
          <div className="bg-red-500 bg-opacity-10 border border-red-500 rounded-md p-3 flex items-start">
            <ExclamationCircleIcon className="h-5 w-5 text-red-500 mt-0.5 mr-2 flex-shrink-0" />
            <span className="text-sm text-red-400">{error}</span>
          </div>
        )}

        {success && (
          <div className="bg-green-500 bg-opacity-10 border border-green-500 rounded-md p-3 flex items-start">
            <CheckCircleIcon className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
            <span className="text-sm text-green-400">{success}</span>
          </div>
        )}

        {/* Symbol */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Symbol
          </label>
          <input
            type="text"
            name="symbol"
            value={formData.symbol}
            onChange={handleChange}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="BTC/USDT"
            disabled={loading}
          />
          <p className="mt-1 text-xs text-gray-400">
            Current Price: <span className="font-mono">${currentPrice}</span>
          </p>
        </div>

        {/* Order Side */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => setFormData((prev) => ({ ...prev, side: 'BUY' }))}
            className={`py-3 rounded-md font-semibold transition-colors ${
              formData.side === 'BUY'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            disabled={loading}
          >
            BUY
          </button>
          <button
            type="button"
            onClick={() => setFormData((prev) => ({ ...prev, side: 'SELL' }))}
            className={`py-3 rounded-md font-semibold transition-colors ${
              formData.side === 'SELL'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
            }`}
            disabled={loading}
          >
            SELL
          </button>
        </div>

        {/* Order Type */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Order Type
          </label>
          <select
            name="orderType"
            value={formData.orderType}
            onChange={handleChange}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          >
            <option value="MARKET">Market</option>
            <option value="LIMIT">Limit</option>
            <option value="STOP_LOSS">Stop Loss</option>
            <option value="TAKE_PROFIT">Take Profit</option>
          </select>
        </div>

        {/* Quantity */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Quantity ({base})
          </label>
          <div className="relative">
            <input
              type="number"
              name="quantity"
              value={formData.quantity}
              onChange={handleChange}
              step="0.00000001"
              min="0"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="0.00000000"
              disabled={loading}
            />
            <button
              type="button"
              onClick={handleUseMaxBalance}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 text-xs text-blue-400 hover:text-blue-300"
              disabled={loading}
            >
              MAX
            </button>
          </div>
          <p className="mt-1 text-xs text-gray-400">
            Available: <span className="font-mono">{getAvailableBalance(formData.side === 'BUY' ? quote : base)}</span> {formData.side === 'BUY' ? quote : base}
          </p>
        </div>

        {/* Price (for LIMIT orders) */}
        {(formData.orderType === 'LIMIT' ||
          formData.orderType === 'STOP_LOSS' ||
          formData.orderType === 'TAKE_PROFIT') && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Price ({quote})
            </label>
            <input
              type="number"
              name="price"
              value={formData.price}
              onChange={handleChange}
              step="0.01"
              min="0"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="0.00"
              disabled={loading}
            />
          </div>
        )}

        {/* Stop Price (for STOP orders) */}
        {(formData.orderType === 'STOP_LOSS' || formData.orderType === 'TAKE_PROFIT') && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Stop Price ({quote})
            </label>
            <input
              type="number"
              name="stopPrice"
              value={formData.stopPrice}
              onChange={handleChange}
              step="0.01"
              min="0"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="0.00"
              disabled={loading}
            />
          </div>
        )}

        {/* Estimated Total */}
        <div className="bg-gray-700 rounded-md p-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Estimated Total:</span>
            <span className="text-lg font-semibold text-white font-mono">
              {estimatedTotal} {quote}
            </span>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-3 rounded-md font-semibold transition-colors ${
            formData.side === 'BUY'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-red-600 hover:bg-red-700'
          } text-white ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Placing Order...
            </span>
          ) : (
            `${formData.side} ${formData.symbol}`
          )}
        </button>
      </form>
    </div>
  );
};

export default OrderForm;
