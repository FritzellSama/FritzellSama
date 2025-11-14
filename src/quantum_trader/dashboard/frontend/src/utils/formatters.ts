/**
 * Formatting Utilities
 *
 * Comprehensive formatting functions for numbers, currencies, dates,
 * percentages, and other data types used in the dashboard.
 */

import { format, formatDistance, formatRelative, parseISO } from 'date-fns';
import type { TimeFrame } from '@/types';
import { CURRENCIES, NUMBER_FORMATS } from './constants';

// Number Formatting

export function formatNumber(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.DECIMAL,
  options?: Intl.NumberFormatOptions
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    ...options,
  }).format(value);
}

export function formatCompactNumber(
  value: number | null | undefined,
  decimals: number = 1
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const absValue = Math.abs(value);

  if (absValue >= 1e12) {
    return `${formatNumber(value / 1e12, decimals)}T`;
  } else if (absValue >= 1e9) {
    return `${formatNumber(value / 1e9, decimals)}B`;
  } else if (absValue >= 1e6) {
    return `${formatNumber(value / 1e6, decimals)}M`;
  } else if (absValue >= 1e3) {
    return `${formatNumber(value / 1e3, decimals)}K`;
  }

  return formatNumber(value, decimals);
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return '-';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatDecimal(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.DECIMAL
): string {
  return formatNumber(value, decimals);
}

// Currency Formatting

export function formatCurrency(
  value: number | null | undefined,
  currency: string = CURRENCIES.DEFAULT,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.CURRENCY,
  showSymbol: boolean = true
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const formatted = formatNumber(value, decimals);

  if (!showSymbol) return formatted;

  const symbol = CURRENCIES.SYMBOLS[currency as keyof typeof CURRENCIES.SYMBOLS] || '$';
  const isNegative = value < 0;
  const absoluteFormatted = formatNumber(Math.abs(value), decimals);

  return isNegative ? `-${symbol}${absoluteFormatted}` : `${symbol}${formatted}`;
}

export function formatCompactCurrency(
  value: number | null | undefined,
  currency: string = CURRENCIES.DEFAULT,
  decimals: number = 1
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const symbol = CURRENCIES.SYMBOLS[currency as keyof typeof CURRENCIES.SYMBOLS] || '$';
  const compactNumber = formatCompactNumber(value, decimals);

  return value < 0
    ? `-${symbol}${compactNumber.replace('-', '')}`
    : `${symbol}${compactNumber}`;
}

export function formatPrice(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.PRICE
): string {
  return formatCurrency(value, CURRENCIES.DEFAULT, decimals);
}

export function formatPnL(
  value: number | null | undefined,
  currency: string = CURRENCIES.DEFAULT,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.CURRENCY,
  showSign: boolean = true
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const formatted = formatCurrency(Math.abs(value), currency, decimals);
  const symbol = CURRENCIES.SYMBOLS[currency as keyof typeof CURRENCIES.SYMBOLS] || '$';

  if (!showSign) return formatted;

  if (value > 0) {
    return `+${formatted}`;
  } else if (value < 0) {
    return `-${symbol}${formatNumber(Math.abs(value), decimals)}`;
  }

  return formatted;
}

// Percentage Formatting

export function formatPercentage(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.PERCENTAGE,
  showSign: boolean = false,
  showSymbol: boolean = true
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const percentage = value * NUMBER_FORMATS.PERCENTAGE_MULTIPLIER;
  const formatted = formatNumber(Math.abs(percentage), decimals);

  let result = '';

  if (showSign || value < 0) {
    result += value >= 0 ? '+' : '-';
  }

  result += formatted;

  if (showSymbol) {
    result += '%';
  }

  return result;
}

export function formatPercentageChange(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.PERCENTAGE
): string {
  return formatPercentage(value, decimals, true, true);
}

export function formatBasisPoints(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return '-';
  const bps = value * 10000;
  return `${formatNumber(bps, 0)} bps`;
}

// Quantity Formatting

export function formatQuantity(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.QUANTITY
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  const absValue = Math.abs(value);

  if (absValue >= 1) {
    decimals = 2;
  } else if (absValue >= 0.01) {
    decimals = 4;
  } else if (absValue >= 0.0001) {
    decimals = 6;
  } else {
    decimals = 8;
  }

  return formatNumber(value, decimals);
}

export function formatVolume(
  value: number | null | undefined,
  compact: boolean = true
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  if (compact && value >= NUMBER_FORMATS.COMPACT_THRESHOLD) {
    return formatCompactNumber(value, 2);
  }

  return formatInteger(value);
}

// Date and Time Formatting

export function formatDate(
  date: string | Date | null | undefined,
  formatStr: string = 'MMM dd, yyyy'
): string {
  if (!date) return '-';

  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return format(dateObj, formatStr);
  } catch (error) {
    console.error('Date formatting error:', error);
    return '-';
  }
}

export function formatTime(
  date: string | Date | null | undefined,
  formatStr: string = 'HH:mm:ss'
): string {
  if (!date) return '-';

  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return format(dateObj, formatStr);
  } catch (error) {
    console.error('Time formatting error:', error);
    return '-';
  }
}

export function formatDateTime(
  date: string | Date | null | undefined,
  formatStr: string = 'MMM dd, yyyy HH:mm:ss'
): string {
  if (!date) return '-';

  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return format(dateObj, formatStr);
  } catch (error) {
    console.error('DateTime formatting error:', error);
    return '-';
  }
}

export function formatRelativeTime(
  date: string | Date | null | undefined
): string {
  if (!date) return '-';

  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return formatDistance(dateObj, new Date(), { addSuffix: true });
  } catch (error) {
    console.error('Relative time formatting error:', error);
    return '-';
  }
}

export function formatRelativeDate(
  date: string | Date | null | undefined
): string {
  if (!date) return '-';

  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    return formatRelative(dateObj, new Date());
  } catch (error) {
    console.error('Relative date formatting error:', error);
    return '-';
  }
}

export function formatDuration(milliseconds: number): string {
  if (isNaN(milliseconds) || milliseconds < 0) return '-';

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
}

export function formatHoldingPeriod(hours: number | null | undefined): string {
  if (hours === null || hours === undefined || isNaN(hours)) return '-';

  if (hours < 1) {
    return `${Math.round(hours * 60)}m`;
  } else if (hours < 24) {
    return `${formatNumber(hours, 1)}h`;
  } else {
    const days = hours / 24;
    return `${formatNumber(days, 1)}d`;
  }
}

// Risk Level Formatting

export function formatRiskLevel(
  score: number,
  thresholds: { low: number; medium: number; high: number } = {
    low: 0.3,
    medium: 0.6,
    high: 0.85,
  }
): string {
  if (score <= thresholds.low) return 'Low';
  if (score <= thresholds.medium) return 'Medium';
  if (score <= thresholds.high) return 'High';
  return 'Critical';
}

export function formatRiskScore(
  score: number | null | undefined,
  max: number = 100
): string {
  if (score === null || score === undefined || isNaN(score)) return '-';
  return `${formatNumber(score, 0)}/${max}`;
}

// Ratio Formatting

export function formatRatio(
  value: number | null | undefined,
  decimals: number = NUMBER_FORMATS.DECIMAL_PLACES.RATIO
): string {
  if (value === null || value === undefined || isNaN(value)) return '-';

  if (!isFinite(value)) {
    return value > 0 ? '∞' : '-∞';
  }

  return formatNumber(value, decimals);
}

export function formatSharpeRatio(value: number | null | undefined): string {
  return formatRatio(value, 2);
}

export function formatSortinoRatio(value: number | null | undefined): string {
  return formatRatio(value, 2);
}

// Symbol Formatting

export function formatSymbol(symbol: string): string {
  if (!symbol) return '-';
  return symbol.replace('/', ' / ').toUpperCase();
}

export function formatExchange(exchange: string): string {
  if (!exchange) return '-';
  return exchange.charAt(0).toUpperCase() + exchange.slice(1).toLowerCase();
}

export function formatOrderSide(side: string): string {
  if (!side) return '-';
  return side.charAt(0).toUpperCase() + side.slice(1).toLowerCase();
}

export function formatOrderType(type: string): string {
  if (!type) return '-';
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

export function formatOrderStatus(status: string): string {
  if (!status) return '-';
  return status
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

export function formatStrategyType(type: string): string {
  if (!type) return '-';
  return type
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

export function formatTimeframe(timeframe: TimeFrame): string {
  const labels: Record<TimeFrame, string> = {
    '1m': '1 Minute',
    '5m': '5 Minutes',
    '15m': '15 Minutes',
    '30m': '30 Minutes',
    '1h': '1 Hour',
    '4h': '4 Hours',
    '1d': '1 Day',
    '1w': '1 Week',
    '1M': '1 Month',
  };

  return labels[timeframe] || timeframe;
}

// Address Formatting

export function formatAddress(
  address: string,
  startChars: number = 6,
  endChars: number = 4
): string {
  if (!address || address.length <= startChars + endChars) return address;
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
}

export function formatTransactionHash(hash: string): string {
  return formatAddress(hash, 8, 6);
}

// File Size Formatting

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${formatNumber(bytes / Math.pow(k, i), 2)} ${sizes[i]}`;
}

// Color Formatting for Values

export function getColorForValue(
  value: number,
  thresholds?: { positive: number; negative: number }
): string {
  if (value > (thresholds?.positive || 0)) {
    return 'text-green-500';
  } else if (value < (thresholds?.negative || 0)) {
    return 'text-red-500';
  }
  return 'text-slate-400';
}

export function getColorForPercentage(value: number): string {
  return getColorForValue(value);
}

export function getColorForPnL(value: number): string {
  return getColorForValue(value);
}

export function getBackgroundColorForValue(
  value: number,
  thresholds?: { positive: number; negative: number }
): string {
  if (value > (thresholds?.positive || 0)) {
    return 'bg-green-500/10';
  } else if (value < (thresholds?.negative || 0)) {
    return 'bg-red-500/10';
  }
  return 'bg-slate-500/10';
}

// Chart Label Formatting

export function formatChartValue(
  value: number,
  type: 'currency' | 'percentage' | 'number' | 'compact' = 'number',
  decimals?: number
): string {
  switch (type) {
    case 'currency':
      return formatCurrency(value, CURRENCIES.DEFAULT, decimals);
    case 'percentage':
      return formatPercentage(value, decimals);
    case 'compact':
      return formatCompactNumber(value, decimals);
    case 'number':
    default:
      return formatNumber(value, decimals);
  }
}

export function formatAxisLabel(value: number | string): string {
  if (typeof value === 'string') {
    if (value.length > 10) {
      return formatDate(value, 'MMM dd');
    }
    return value;
  }

  if (Math.abs(value) >= 1e6) {
    return formatCompactNumber(value, 1);
  }

  return formatNumber(value, 2);
}

export function formatTooltipValue(
  value: number,
  name: string,
  decimals: number = 2
): string {
  const lowerName = name.toLowerCase();

  if (lowerName.includes('price') || lowerName.includes('value')) {
    return formatCurrency(value, CURRENCIES.DEFAULT, decimals);
  } else if (
    lowerName.includes('percent') ||
    lowerName.includes('rate') ||
    lowerName.includes('ratio')
  ) {
    return formatPercentage(value / 100, decimals);
  } else if (lowerName.includes('volume')) {
    return formatVolume(value, true);
  }

  return formatNumber(value, decimals);
}

// Validation Helpers

export function isValidNumber(value: any): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

export function isValidDate(value: any): value is Date {
  return value instanceof Date && !isNaN(value.getTime());
}

export function parseNumberInput(input: string): number | null {
  const cleaned = input.replace(/[^0-9.-]/g, '');
  const parsed = parseFloat(cleaned);
  return isValidNumber(parsed) ? parsed : null;
}

// Clipboard Formatting

export function formatForClipboard(
  value: number | string | null | undefined,
  type: 'number' | 'currency' | 'percentage' = 'number'
): string {
  if (value === null || value === undefined) return '';

  if (typeof value === 'string') return value;

  switch (type) {
    case 'currency':
      return formatNumber(value, NUMBER_FORMATS.DECIMAL_PLACES.CURRENCY);
    case 'percentage':
      return formatNumber(
        value * NUMBER_FORMATS.PERCENTAGE_MULTIPLIER,
        NUMBER_FORMATS.DECIMAL_PLACES.PERCENTAGE
      );
    case 'number':
    default:
      return value.toString();
  }
}

// Export all formatters
export default {
  formatNumber,
  formatCompactNumber,
  formatInteger,
  formatDecimal,
  formatCurrency,
  formatCompactCurrency,
  formatPrice,
  formatPnL,
  formatPercentage,
  formatPercentageChange,
  formatBasisPoints,
  formatQuantity,
  formatVolume,
  formatDate,
  formatTime,
  formatDateTime,
  formatRelativeTime,
  formatRelativeDate,
  formatDuration,
  formatHoldingPeriod,
  formatRiskLevel,
  formatRiskScore,
  formatRatio,
  formatSharpeRatio,
  formatSortinoRatio,
  formatSymbol,
  formatExchange,
  formatOrderSide,
  formatOrderType,
  formatOrderStatus,
  formatStrategyType,
  formatTimeframe,
  formatAddress,
  formatTransactionHash,
  formatFileSize,
  getColorForValue,
  getColorForPercentage,
  getColorForPnL,
  getBackgroundColorForValue,
  formatChartValue,
  formatAxisLabel,
  formatTooltipValue,
  isValidNumber,
  isValidDate,
  parseNumberInput,
  formatForClipboard,
};
