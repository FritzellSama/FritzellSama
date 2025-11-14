/**
 * Formatting utilities for trading dashboard
 *
 * Production-ready formatters with:
 * - Currency formatting
 * - Date/time formatting
 * - Number formatting
 * - Percentage formatting
 * - Precision handling
 */

import Big from 'big.js';
import { NUMBER_FORMATS, DATE_FORMATS } from './constants';

/**
 * Format currency value
 */
export function formatCurrency(
  value: number | string,
  currency: string = 'USD',
  decimals?: number
): string {
  try {
    const num = new Big(value);
    const decimalPlaces = decimals !== undefined ? decimals : NUMBER_FORMATS.CURRENCY_DECIMALS;

    const formatted = num.toFixed(decimalPlaces);
    const parts = formatted.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');

    const withDecimals = parts.join('.');

    // Currency symbol mapping
    const symbols: Record<string, string> = {
      USD: '$',
      EUR: '€',
      GBP: '£',
      JPY: '¥',
      USDT: '$',
      USDC: '$',
      BTC: '₿',
      ETH: 'Ξ',
    };

    const symbol = symbols[currency] || currency;

    return `${symbol}${withDecimals}`;
  } catch (error) {
    console.error('Currency formatting error:', error);
    return '$0.00';
  }
}

/**
 * Format price value
 */
export function formatPrice(
  value: number | string,
  decimals?: number
): string {
  try {
    const num = new Big(value);
    const decimalPlaces = decimals !== undefined ? decimals : NUMBER_FORMATS.PRICE_DECIMALS;

    const formatted = num.toFixed(decimalPlaces);
    const parts = formatted.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');

    return parts.join('.');
  } catch (error) {
    console.error('Price formatting error:', error);
    return '0.00';
  }
}

/**
 * Format quantity value
 */
export function formatQuantity(
  value: number | string,
  decimals?: number
): string {
  try {
    const num = new Big(value);
    const decimalPlaces = decimals !== undefined ? decimals : NUMBER_FORMATS.QUANTITY_DECIMALS;

    // Remove trailing zeros
    let formatted = num.toFixed(decimalPlaces);
    formatted = formatted.replace(/\.?0+$/, '');

    const parts = formatted.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');

    return parts.join('.');
  } catch (error) {
    console.error('Quantity formatting error:', error);
    return '0';
  }
}

/**
 * Format percentage value
 */
export function formatPercentage(
  value: number | string,
  decimals?: number,
  includeSign: boolean = false
): string {
  try {
    const num = new Big(value);
    const decimalPlaces = decimals !== undefined ? decimals : NUMBER_FORMATS.PERCENTAGE_DECIMALS;

    const formatted = num.toFixed(decimalPlaces);
    const sign = includeSign && num.gt(0) ? '+' : '';

    return `${sign}${formatted}%`;
  } catch (error) {
    console.error('Percentage formatting error:', error);
    return '0.00%';
  }
}

/**
 * Format large numbers with abbreviations (K, M, B)
 */
export function formatCompactNumber(
  value: number | string,
  decimals: number = 2
): string {
  try {
    const num = new Big(value);
    const absNum = num.abs();

    const sign = num.lt(0) ? '-' : '';

    if (absNum.gte(1e9)) {
      return `${sign}${absNum.div(1e9).toFixed(decimals)}B`;
    } else if (absNum.gte(1e6)) {
      return `${sign}${absNum.div(1e6).toFixed(decimals)}M`;
    } else if (absNum.gte(1e3)) {
      return `${sign}${absNum.div(1e3).toFixed(decimals)}K`;
    } else {
      return `${sign}${absNum.toFixed(decimals)}`;
    }
  } catch (error) {
    console.error('Compact number formatting error:', error);
    return '0';
  }
}

/**
 * Format date/time
 */
export function formatDateTime(
  timestamp: number | string | Date,
  format: 'full' | 'date' | 'time' | 'short' = 'full'
): string {
  try {
    const date = typeof timestamp === 'number'
      ? new Date(timestamp)
      : typeof timestamp === 'string'
      ? new Date(parseInt(timestamp))
      : timestamp;

    if (isNaN(date.getTime())) {
      throw new Error('Invalid date');
    }

    const pad = (n: number) => n.toString().padStart(2, '0');

    const year = date.getFullYear();
    const month = pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hours = pad(date.getHours());
    const minutes = pad(date.getMinutes());
    const seconds = pad(date.getSeconds());

    switch (format) {
      case 'full':
        return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
      case 'date':
        return `${year}-${month}-${day}`;
      case 'time':
        return `${hours}:${minutes}:${seconds}`;
      case 'short':
        return `${month}/${day} ${hours}:${minutes}`;
      default:
        return date.toISOString();
    }
  } catch (error) {
    console.error('DateTime formatting error:', error);
    return 'Invalid Date';
  }
}

/**
 * Format relative time (e.g., "5 minutes ago")
 */
export function formatRelativeTime(timestamp: number | Date): string {
  try {
    const date = typeof timestamp === 'number' ? new Date(timestamp) : timestamp;
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 7) {
      return formatDateTime(date, 'date');
    } else if (days > 0) {
      return `${days} day${days > 1 ? 's' : ''} ago`;
    } else if (hours > 0) {
      return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    } else if (minutes > 0) {
      return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    } else if (seconds > 5) {
      return `${seconds} seconds ago`;
    } else {
      return 'just now';
    }
  } catch (error) {
    console.error('Relative time formatting error:', error);
    return 'Unknown';
  }
}

/**
 * Format duration (milliseconds to human readable)
 */
export function formatDuration(milliseconds: number): string {
  try {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h`;
    } else if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  } catch (error) {
    console.error('Duration formatting error:', error);
    return '0s';
  }
}

/**
 * Format PnL with color indication
 */
export function formatPnL(
  value: number | string,
  includeSign: boolean = true
): { text: string; color: string } {
  try {
    const num = new Big(value);
    const formatted = formatCurrency(value, 'USD', 2);

    let text = formatted;
    if (includeSign && num.gt(0)) {
      text = `+${formatted}`;
    }

    const color = num.gt(0) ? '#10b981' : num.lt(0) ? '#ef4444' : '#6b7280';

    return { text, color };
  } catch (error) {
    console.error('PnL formatting error:', error);
    return { text: '$0.00', color: '#6b7280' };
  }
}

/**
 * Format order side with color
 */
export function formatOrderSide(side: string): { text: string; color: string } {
  const sideUpper = side.toUpperCase();

  switch (sideUpper) {
    case 'BUY':
      return { text: 'Buy', color: '#10b981' };
    case 'SELL':
      return { text: 'Sell', color: '#ef4444' };
    default:
      return { text: side, color: '#6b7280' };
  }
}

/**
 * Format order status with color
 */
export function formatOrderStatus(status: string): { text: string; color: string } {
  const statusUpper = status.toUpperCase();

  const statusMap: Record<string, { text: string; color: string }> = {
    PENDING: { text: 'Pending', color: '#6b7280' },
    OPEN: { text: 'Open', color: '#3b82f6' },
    PARTIAL: { text: 'Partially Filled', color: '#f59e0b' },
    FILLED: { text: 'Filled', color: '#10b981' },
    CANCELLED: { text: 'Cancelled', color: '#6b7280' },
    REJECTED: { text: 'Rejected', color: '#ef4444' },
    EXPIRED: { text: 'Expired', color: '#6b7280' },
    FAILED: { text: 'Failed', color: '#ef4444' },
  };

  return statusMap[statusUpper] || { text: status, color: '#6b7280' };
}

/**
 * Format symbol (e.g., "BTCUSDT" -> "BTC/USDT")
 */
export function formatSymbol(symbol: string): string {
  try {
    // Already formatted
    if (symbol.includes('/')) {
      return symbol;
    }

    // Common quote currencies
    const quoteCurrencies = ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'EUR', 'GBP'];

    for (const quote of quoteCurrencies) {
      if (symbol.endsWith(quote)) {
        const base = symbol.slice(0, -quote.length);
        return `${base}/${quote}`;
      }
    }

    return symbol;
  } catch (error) {
    console.error('Symbol formatting error:', error);
    return symbol;
  }
}

/**
 * Format address (truncate middle)
 */
export function formatAddress(
  address: string,
  startChars: number = 6,
  endChars: number = 4
): string {
  try {
    if (address.length <= startChars + endChars) {
      return address;
    }

    const start = address.slice(0, startChars);
    const end = address.slice(-endChars);

    return `${start}...${end}`;
  } catch (error) {
    console.error('Address formatting error:', error);
    return address;
  }
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number): string {
  try {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  } catch (error) {
    console.error('File size formatting error:', error);
    return '0 B';
  }
}

/**
 * Format leverage (e.g., "5x", "10x")
 */
export function formatLeverage(value: number | string): string {
  try {
    const num = new Big(value);
    return `${num.toFixed(0)}x`;
  } catch (error) {
    console.error('Leverage formatting error:', error);
    return '1x';
  }
}

/**
 * Format risk/reward ratio
 */
export function formatRiskReward(value: number | string): string {
  try {
    const num = new Big(value);
    return `1:${num.toFixed(2)}`;
  } catch (error) {
    console.error('Risk/reward formatting error:', error);
    return '1:0.00';
  }
}

/**
 * Parse formatted number back to string
 */
export function parseFormattedNumber(formatted: string): string {
  try {
    // Remove currency symbols, commas, and percentage signs
    const cleaned = formatted
      .replace(/[$€£¥₿Ξ,]/g, '')
      .replace(/%$/, '')
      .trim();

    // Handle abbreviations
    if (cleaned.endsWith('B')) {
      return new Big(cleaned.slice(0, -1)).times(1e9).toString();
    } else if (cleaned.endsWith('M')) {
      return new Big(cleaned.slice(0, -1)).times(1e6).toString();
    } else if (cleaned.endsWith('K')) {
      return new Big(cleaned.slice(0, -1)).times(1e3).toString();
    }

    return new Big(cleaned).toString();
  } catch (error) {
    console.error('Parse formatted number error:', error);
    return '0';
  }
}

/**
 * Clamp number to range
 */
export function clampNumber(
  value: number | string,
  min: number | string,
  max: number | string
): string {
  try {
    const num = new Big(value);
    const minVal = new Big(min);
    const maxVal = new Big(max);

    if (num.lt(minVal)) {
      return minVal.toString();
    } else if (num.gt(maxVal)) {
      return maxVal.toString();
    }

    return num.toString();
  } catch (error) {
    console.error('Clamp number error:', error);
    return '0';
  }
}

/**
 * Format volume with appropriate unit
 */
export function formatVolume(value: number | string): string {
  try {
    const num = new Big(value);

    if (num.gte(1e9)) {
      return `${num.div(1e9).toFixed(2)}B`;
    } else if (num.gte(1e6)) {
      return `${num.div(1e6).toFixed(2)}M`;
    } else if (num.gte(1e3)) {
      return `${num.div(1e3).toFixed(2)}K`;
    } else {
      return num.toFixed(2);
    }
  } catch (error) {
    console.error('Volume formatting error:', error);
    return '0';
  }
}

/**
 * Validate and sanitize numeric input
 */
export function sanitizeNumericInput(
  input: string,
  maxDecimals?: number
): string {
  try {
    // Remove non-numeric characters except decimal point and minus
    let cleaned = input.replace(/[^0-9.-]/g, '');

    // Ensure only one decimal point
    const parts = cleaned.split('.');
    if (parts.length > 2) {
      cleaned = parts[0] + '.' + parts.slice(1).join('');
    }

    // Ensure only one minus sign at the beginning
    if (cleaned.indexOf('-') !== cleaned.lastIndexOf('-')) {
      cleaned = '-' + cleaned.replace(/-/g, '');
    } else if (cleaned.indexOf('-') > 0) {
      cleaned = cleaned.replace(/-/g, '');
    }

    // Limit decimal places
    if (maxDecimals !== undefined && parts.length === 2) {
      const decimals = parts[1].slice(0, maxDecimals);
      cleaned = parts[0] + '.' + decimals;
    }

    return cleaned;
  } catch (error) {
    console.error('Sanitize numeric input error:', error);
    return '';
  }
}
