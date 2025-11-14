import React, { useState, useEffect } from 'react';
import {
  BoltIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';

interface QuickTradeFormData {
  symbol: string;
  side: 'BUY' | 'SELL';
  amount: string;
  useMarketPrice: boolean;
}

interface MarketPrice {
  symbol: string;
  bid: string;
  ask: string;
  last: string;
}

interface Balance {
  currency: string;
  available: string;
}

interface QuickTradeProps {
  defaultSymbol?: string;
  onTradeExecuted?: (orderId: string) => void;
}

const QuickTrade: React.FC<QuickTradeProps> = ({
  defaultSymbol = 'BTC/USDT',
  onTradeExecuted,
}) => {
  const [formData, setFormData] = useState<QuickTradeFormData>({
    symbol: defaultSymbol,
    side: 'BUY',
    amount: '',
    useMarketPrice: true,
  });
  const [marketPrice, setMarketPrice] = useState<MarketPrice | null>(null);
  const [balance, setBalance] = useState<Balance[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [estimatedCost, setEstimatedCost] = useState<string>('0.00');

  useEffect(() => {
    fetchMarketPrice();
    fetchBalance();

    const priceInterval = setInterval(fetchMarketPrice, 2000);

    return () => clearInterval(priceInterval);
  }, [formData.symbol]);

  useEffect(() => {
    calculateEstimatedCost();
  }, [formData.amount, formData.side, marketPrice]);

  const fetchMarketPrice = async (): Promise<void> => {
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

      if (response.ok) {
        const data = await response.json();
        setMarketPrice({
          symbol: formData.symbol,
          bid: data.bid || '0',
          ask: data.ask || '0',
          last: data.last || '0',
        });
      }
    } catch (err) {
      console.error('Error fetching market price:', err);
    }
  };

  const fetchBalance = async (): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/account/balance`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        setBalance(data.balances || []);
      }
    } catch (err) {
      console.error('Error fetching balance:', err);
    }
  };

  const calculateEstimatedCost = (): void => {
    if (!marketPrice || !formData.amount) {
      setEstimatedCost('0.00');
      return;
    }

    const amount = parseFloat(formData.amount) || 0;
    const price =
      formData.side === 'BUY'
        ? parseFloat(marketPrice.ask)
        : parseFloat(marketPrice.bid);

    const cost = amount * price;
    setEstimatedCost(cost.toFixed(2));
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

  const handleSideToggle = (side: 'BUY' | 'SELL'): void => {
    setFormData((prev) => ({
      ...prev,
      side,
    }));
    setError('');
    setSuccess('');
  };

  const validateForm = (): boolean => {
    if (!formData.symbol.trim()) {
      setError('Symbol is required');
      return false;
    }

    const amount = parseFloat(formData.amount);
    if (isNaN(amount) || amount <= 0) {
      setError('Amount must be greater than 0');
      return false;
    }

    const [base, quote] = formData.symbol.split('/');
    const requiredCurrency = formData.side === 'BUY' ? quote : base;
    const availableBal = parseFloat(getAvailableBalance(requiredCurrency));

    const requiredAmount =
      formData.side === 'BUY' ? parseFloat(estimatedCost) : amount;

    if (requiredAmount > availableBal) {
      setError(`Insufficient ${requiredCurrency} balance`);
      return false;
    }

    return true;
  };

  const handleQuickTrade = async (): Promise<void> => {
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
          const payload = {
            symbol: formData.symbol,
            side: formData.side,
            order_type: 'MARKET',
            quantity: formData.amount,
          };

          const response = await fetch(`${apiUrl}/api/v1/orders/quick`, {
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
              setError(errorData.detail || 'Failed to execute quick trade');
              setLoading(false);
              return;
            }
          }

          const data = await response.json();
          setSuccess(
            `${formData.side} order executed! ${formData.amount} ${formData.symbol.split('/')[0]} @ ${marketPrice?.last || 'Market'}`
          );

          setFormData((prev) => ({
            ...prev,
            amount: '',
          }));

          await fetchBalance();

          if (onTradeExecuted) {
            onTradeExecuted(data.order_id);
          }

          setTimeout(() => {
            setSuccess('');
          }, 5000);

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
      console.error('Quick trade error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to execute quick trade';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleUseMaxBalance = (): void => {
    const [base, quote] = formData.symbol.split('/');

    if (formData.side === 'BUY') {
      const availableQuote = parseFloat(getAvailableBalance(quote));
      const price = parseFloat(marketPrice?.ask || '0');

      if (price > 0) {
        const maxAmount = availableQuote / price;
        setFormData((prev) => ({
          ...prev,
          amount: maxAmount.toFixed(8),
        }));
      }
    } else {
      const availableBase = parseFloat(getAvailableBalance(base));
      setFormData((prev) => ({
        ...prev,
        amount: availableBase.toFixed(8),
      }));
    }
  };

  const [base, quote] = formData.symbol.split('/');

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center mb-6">
        <BoltIcon className="h-6 w-6 text-yellow-500 mr-2" />
        <h3 className="text-xl font-semibold text-white">Quick Trade</h3>
      </div>

      <div className="space-y-4">
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
          {marketPrice && (
            <div className="mt-2 text-xs text-gray-400 flex justify-between">
              <span>
                Bid: <span className="text-red-400 font-mono">${parseFloat(marketPrice.bid).toFixed(2)}</span>
              </span>
              <span>
                Ask: <span className="text-green-400 font-mono">${parseFloat(marketPrice.ask).toFixed(2)}</span>
              </span>
              <span>
                Last: <span className="text-white font-mono">${parseFloat(marketPrice.last).toFixed(2)}</span>
              </span>
            </div>
          )}
        </div>

        {/* Buy/Sell Toggle */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => handleSideToggle('BUY')}
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
            onClick={() => handleSideToggle('SELL')}
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

        {/* Amount */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Amount ({base})
          </label>
          <div className="relative">
            <input
              type="number"
              name="amount"
              value={formData.amount}
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
            Available:{' '}
            <span className="font-mono">
              {getAvailableBalance(formData.side === 'BUY' ? quote : base)}
            </span>{' '}
            {formData.side === 'BUY' ? quote : base}
          </p>
        </div>

        {/* Estimated Cost */}
        <div className="bg-gray-700 rounded-md p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-400">Order Type:</span>
            <span className="text-sm font-semibold text-white">MARKET</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">
              Estimated {formData.side === 'BUY' ? 'Cost' : 'Receive'}:
            </span>
            <span className="text-lg font-semibold text-white font-mono">
              {estimatedCost} {quote}
            </span>
          </div>
        </div>

        {/* Execute Button */}
        <button
          type="button"
          onClick={handleQuickTrade}
          disabled={loading}
          className={`w-full py-3 rounded-md font-semibold transition-colors flex items-center justify-center ${
            formData.side === 'BUY'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-red-600 hover:bg-red-700'
          } text-white ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {loading ? (
            <span className="flex items-center">
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
              Executing...
            </span>
          ) : (
            <>
              <BoltIcon className="h-5 w-5 mr-2" />
              Quick {formData.side} {formData.symbol}
            </>
          )}
        </button>

        {/* Warning */}
        <div className="bg-yellow-500 bg-opacity-10 border border-yellow-500 rounded-md p-3">
          <p className="text-xs text-yellow-400">
            <strong>Warning:</strong> Quick trades execute at market price. Price may vary from estimate.
          </p>
        </div>
      </div>
    </div>
  );
};

export default QuickTrade;
