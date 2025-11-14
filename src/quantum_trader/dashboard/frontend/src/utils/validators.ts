/**
 * Input validation and sanitization utilities
 * Production-grade validation for trading platform inputs
 */

import Decimal from 'decimal.js';

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Symbol validation options
 */
interface SymbolValidationOptions {
  minLength?: number;
  maxLength?: number;
  allowedPattern?: RegExp;
}

/**
 * Price validation options
 */
interface PriceValidationOptions {
  min?: string | Decimal;
  max?: string | Decimal;
  maxDecimals?: number;
}

/**
 * Quantity validation options
 */
interface QuantityValidationOptions {
  min?: string | Decimal;
  max?: string | Decimal;
  maxDecimals?: number;
}

// Load validation config from environment
const SYMBOL_MIN_LENGTH = parseInt(process.env.VITE_SYMBOL_MIN_LENGTH || '3', 10);
const SYMBOL_MAX_LENGTH = parseInt(process.env.VITE_SYMBOL_MAX_LENGTH || '20', 10);
const SYMBOL_PATTERN = new RegExp(process.env.VITE_SYMBOL_PATTERN || '^[A-Z0-9/]+$');

const PRICE_MIN = new Decimal(process.env.VITE_PRICE_MIN || '0.00000001');
const PRICE_MAX = new Decimal(process.env.VITE_PRICE_MAX || '1000000000');
const PRICE_MAX_DECIMALS = parseInt(process.env.VITE_PRICE_MAX_DECIMALS || '8', 10);

const QUANTITY_MIN = new Decimal(process.env.VITE_QUANTITY_MIN || '0.00000001');
const QUANTITY_MAX = new Decimal(process.env.VITE_QUANTITY_MAX || '1000000000');
const QUANTITY_MAX_DECIMALS = parseInt(process.env.VITE_QUANTITY_MAX_DECIMALS || '8', 10);

const PERCENTAGE_MIN = new Decimal(process.env.VITE_PERCENTAGE_MIN || '0');
const PERCENTAGE_MAX = new Decimal(process.env.VITE_PERCENTAGE_MAX || '100');

/**
 * Validate trading symbol
 *
 * @param symbol - Trading pair symbol (e.g., 'BTC/USDT')
 * @param options - Validation options
 * @returns Validation result
 *
 * @example
 * ```typescript
 * const result = validateSymbol('BTC/USDT');
 * if (!result.valid) {
 *   console.error(result.error);
 * }
 * ```
 */
export const validateSymbol = (
  symbol: string,
  options: SymbolValidationOptions = {}
): ValidationResult => {
  try {
    const {
      minLength = SYMBOL_MIN_LENGTH,
      maxLength = SYMBOL_MAX_LENGTH,
      allowedPattern = SYMBOL_PATTERN,
    } = options;

    if (!symbol || typeof symbol !== 'string') {
      return { valid: false, error: 'Symbol must be a non-empty string' };
    }

    const trimmed = symbol.trim().toUpperCase();

    if (trimmed.length < minLength) {
      return { valid: false, error: `Symbol must be at least ${minLength} characters` };
    }

    if (trimmed.length > maxLength) {
      return { valid: false, error: `Symbol must be at most ${maxLength} characters` };
    }

    if (!allowedPattern.test(trimmed)) {
      return { valid: false, error: 'Symbol contains invalid characters' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Symbol validation error: ${error}` };
  }
};

/**
 * Validate price value
 *
 * @param price - Price value (string or Decimal)
 * @param options - Validation options
 * @returns Validation result
 *
 * @example
 * ```typescript
 * const result = validatePrice('50000.50');
 * if (!result.valid) {
 *   console.error(result.error);
 * }
 * ```
 */
export const validatePrice = (
  price: string | number | Decimal,
  options: PriceValidationOptions = {}
): ValidationResult => {
  try {
    const {
      min = PRICE_MIN,
      max = PRICE_MAX,
      maxDecimals = PRICE_MAX_DECIMALS,
    } = options;

    if (price === null || price === undefined || price === '') {
      return { valid: false, error: 'Price is required' };
    }

    let priceDecimal: Decimal;
    try {
      priceDecimal = new Decimal(price);
    } catch {
      return { valid: false, error: 'Price must be a valid number' };
    }

    if (priceDecimal.isNaN() || !priceDecimal.isFinite()) {
      return { valid: false, error: 'Price must be a valid number' };
    }

    if (priceDecimal.isNegative() || priceDecimal.isZero()) {
      return { valid: false, error: 'Price must be greater than zero' };
    }

    const minDecimal = new Decimal(min);
    if (priceDecimal.lessThan(minDecimal)) {
      return { valid: false, error: `Price must be at least ${minDecimal.toString()}` };
    }

    const maxDecimal = new Decimal(max);
    if (priceDecimal.greaterThan(maxDecimal)) {
      return { valid: false, error: `Price must be at most ${maxDecimal.toString()}` };
    }

    const decimalPlaces = priceDecimal.decimalPlaces();
    if (decimalPlaces > maxDecimals) {
      return { valid: false, error: `Price can have at most ${maxDecimals} decimal places` };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Price validation error: ${error}` };
  }
};

/**
 * Validate quantity value
 *
 * @param quantity - Quantity value (string or Decimal)
 * @param options - Validation options
 * @returns Validation result
 *
 * @example
 * ```typescript
 * const result = validateQuantity('1.5');
 * if (!result.valid) {
 *   console.error(result.error);
 * }
 * ```
 */
export const validateQuantity = (
  quantity: string | number | Decimal,
  options: QuantityValidationOptions = {}
): ValidationResult => {
  try {
    const {
      min = QUANTITY_MIN,
      max = QUANTITY_MAX,
      maxDecimals = QUANTITY_MAX_DECIMALS,
    } = options;

    if (quantity === null || quantity === undefined || quantity === '') {
      return { valid: false, error: 'Quantity is required' };
    }

    let quantityDecimal: Decimal;
    try {
      quantityDecimal = new Decimal(quantity);
    } catch {
      return { valid: false, error: 'Quantity must be a valid number' };
    }

    if (quantityDecimal.isNaN() || !quantityDecimal.isFinite()) {
      return { valid: false, error: 'Quantity must be a valid number' };
    }

    if (quantityDecimal.isNegative() || quantityDecimal.isZero()) {
      return { valid: false, error: 'Quantity must be greater than zero' };
    }

    const minDecimal = new Decimal(min);
    if (quantityDecimal.lessThan(minDecimal)) {
      return { valid: false, error: `Quantity must be at least ${minDecimal.toString()}` };
    }

    const maxDecimal = new Decimal(max);
    if (quantityDecimal.greaterThan(maxDecimal)) {
      return { valid: false, error: `Quantity must be at most ${maxDecimal.toString()}` };
    }

    const decimalPlaces = quantityDecimal.decimalPlaces();
    if (decimalPlaces > maxDecimals) {
      return { valid: false, error: `Quantity can have at most ${maxDecimals} decimal places` };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Quantity validation error: ${error}` };
  }
};

/**
 * Validate percentage value
 *
 * @param percentage - Percentage value (0-100)
 * @param options - Min/max overrides
 * @returns Validation result
 *
 * @example
 * ```typescript
 * const result = validatePercentage('25.5');
 * if (!result.valid) {
 *   console.error(result.error);
 * }
 * ```
 */
export const validatePercentage = (
  percentage: string | number | Decimal,
  options: { min?: Decimal; max?: Decimal } = {}
): ValidationResult => {
  try {
    const { min = PERCENTAGE_MIN, max = PERCENTAGE_MAX } = options;

    if (percentage === null || percentage === undefined || percentage === '') {
      return { valid: false, error: 'Percentage is required' };
    }

    let percentageDecimal: Decimal;
    try {
      percentageDecimal = new Decimal(percentage);
    } catch {
      return { valid: false, error: 'Percentage must be a valid number' };
    }

    if (percentageDecimal.isNaN() || !percentageDecimal.isFinite()) {
      return { valid: false, error: 'Percentage must be a valid number' };
    }

    if (percentageDecimal.lessThan(min)) {
      return { valid: false, error: `Percentage must be at least ${min.toString()}` };
    }

    if (percentageDecimal.greaterThan(max)) {
      return { valid: false, error: `Percentage must be at most ${max.toString()}` };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Percentage validation error: ${error}` };
  }
};

/**
 * Validate exchange name
 *
 * @param exchange - Exchange name
 * @returns Validation result
 */
export const validateExchange = (exchange: string): ValidationResult => {
  try {
    const validExchanges = (process.env.VITE_VALID_EXCHANGES || 'BINANCE,BYBIT,OKX,KUCOIN,BITGET').split(',');

    if (!exchange || typeof exchange !== 'string') {
      return { valid: false, error: 'Exchange must be a non-empty string' };
    }

    const upperExchange = exchange.trim().toUpperCase();

    if (!validExchanges.includes(upperExchange)) {
      return { valid: false, error: `Invalid exchange. Must be one of: ${validExchanges.join(', ')}` };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Exchange validation error: ${error}` };
  }
};

/**
 * Validate timeframe
 *
 * @param timeframe - Timeframe string (e.g., '1m', '5m', '1h')
 * @returns Validation result
 */
export const validateTimeframe = (timeframe: string): ValidationResult => {
  try {
    const validTimeframes = (process.env.VITE_VALID_TIMEFRAMES || '1m,5m,15m,30m,1h,4h,1d,1w,1M').split(',');

    if (!timeframe || typeof timeframe !== 'string') {
      return { valid: false, error: 'Timeframe must be a non-empty string' };
    }

    const trimmed = timeframe.trim();

    if (!validTimeframes.includes(trimmed)) {
      return { valid: false, error: `Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}` };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: `Timeframe validation error: ${error}` };
  }
};

/**
 * Sanitize string input
 *
 * @param input - String to sanitize
 * @param maxLength - Maximum allowed length
 * @returns Sanitized string
 */
export const sanitizeString = (input: string, maxLength: number = 1000): string => {
  if (!input || typeof input !== 'string') {
    return '';
  }

  let sanitized = input.trim();

  // Remove control characters
  sanitized = sanitized.replace(/[\x00-\x1F\x7F]/g, '');

  // Limit length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength);
  }

  return sanitized;
};

/**
 * Validate and sanitize order input
 *
 * @param order - Order input data
 * @returns Validation result with sanitized data
 */
export interface OrderInput {
  symbol: string;
  side: string;
  quantity: string | Decimal;
  price?: string | Decimal;
  orderType: string;
  exchange: string;
}

export const validateOrder = (order: Partial<OrderInput>): ValidationResult & { data?: OrderInput } => {
  try {
    if (!order || typeof order !== 'object') {
      return { valid: false, error: 'Order must be an object' };
    }

    const symbolResult = validateSymbol(order.symbol || '');
    if (!symbolResult.valid) {
      return symbolResult;
    }

    const validSides = (process.env.VITE_VALID_ORDER_SIDES || 'BUY,SELL').split(',');
    const side = (order.side || '').trim().toUpperCase();
    if (!validSides.includes(side)) {
      return { valid: false, error: `Invalid order side. Must be one of: ${validSides.join(', ')}` };
    }

    const quantityResult = validateQuantity(order.quantity || '');
    if (!quantityResult.valid) {
      return quantityResult;
    }

    if (order.price) {
      const priceResult = validatePrice(order.price);
      if (!priceResult.valid) {
        return priceResult;
      }
    }

    const validOrderTypes = (process.env.VITE_VALID_ORDER_TYPES || 'MARKET,LIMIT').split(',');
    const orderType = (order.orderType || '').trim().toUpperCase();
    if (!validOrderTypes.includes(orderType)) {
      return { valid: false, error: `Invalid order type. Must be one of: ${validOrderTypes.join(', ')}` };
    }

    const exchangeResult = validateExchange(order.exchange || '');
    if (!exchangeResult.valid) {
      return exchangeResult;
    }

    return {
      valid: true,
      data: {
        symbol: (order.symbol || '').trim().toUpperCase(),
        side,
        quantity: order.quantity!,
        price: order.price,
        orderType,
        exchange: (order.exchange || '').trim().toUpperCase(),
      },
    };
  } catch (error) {
    return { valid: false, error: `Order validation error: ${error}` };
  }
};

export default {
  validateSymbol,
  validatePrice,
  validateQuantity,
  validatePercentage,
  validateExchange,
  validateTimeframe,
  validateOrder,
  sanitizeString,
};
