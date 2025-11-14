/**
 * Financial calculation utilities for trading dashboard
 *
 * Production-ready calculations with:
 * - Precise decimal arithmetic
 * - Type-safe operations
 * - Error handling
 * - Performance optimization
 */

import Big from 'big.js';

// Configure Big.js for financial precision
Big.DP = 8; // 8 decimal places
Big.RM = Big.roundDown; // Round down for conservative calculations

/**
 * Calculate profit/loss (P&L)
 */
export function calculatePnL(
  entryPrice: number | string,
  currentPrice: number | string,
  quantity: number | string,
  side: 'long' | 'short'
): string {
  try {
    const entry = new Big(entryPrice);
    const current = new Big(currentPrice);
    const qty = new Big(quantity);

    let pnl: Big;
    if (side === 'long') {
      pnl = current.minus(entry).times(qty);
    } else {
      pnl = entry.minus(current).times(qty);
    }

    return pnl.toFixed(2);
  } catch (error) {
    console.error('PnL calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate P&L percentage
 */
export function calculatePnLPercent(
  entryPrice: number | string,
  currentPrice: number | string,
  side: 'long' | 'short'
): string {
  try {
    const entry = new Big(entryPrice);
    const current = new Big(currentPrice);

    if (entry.eq(0)) {
      return '0.00';
    }

    let pnlPercent: Big;
    if (side === 'long') {
      pnlPercent = current.minus(entry).div(entry).times(100);
    } else {
      pnlPercent = entry.minus(current).div(entry).times(100);
    }

    return pnlPercent.toFixed(2);
  } catch (error) {
    console.error('PnL percent calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate position value
 */
export function calculatePositionValue(
  price: number | string,
  quantity: number | string
): string {
  try {
    const p = new Big(price);
    const q = new Big(quantity);
    return p.times(q).toFixed(2);
  } catch (error) {
    console.error('Position value calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate total portfolio value
 */
export function calculatePortfolioValue(
  positions: Array<{
    currentPrice: number | string;
    quantity: number | string;
  }>,
  cashBalance: number | string
): string {
  try {
    let total = new Big(cashBalance);

    for (const position of positions) {
      const posValue = new Big(position.currentPrice).times(new Big(position.quantity));
      total = total.plus(posValue.abs());
    }

    return total.toFixed(2);
  } catch (error) {
    console.error('Portfolio value calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate weighted average entry price
 */
export function calculateAverageEntryPrice(
  trades: Array<{
    price: number | string;
    quantity: number | string;
  }>
): string {
  try {
    if (trades.length === 0) {
      return '0.00';
    }

    let totalCost = new Big(0);
    let totalQuantity = new Big(0);

    for (const trade of trades) {
      const price = new Big(trade.price);
      const quantity = new Big(trade.quantity);
      totalCost = totalCost.plus(price.times(quantity));
      totalQuantity = totalQuantity.plus(quantity);
    }

    if (totalQuantity.eq(0)) {
      return '0.00';
    }

    return totalCost.div(totalQuantity).toFixed(2);
  } catch (error) {
    console.error('Average entry price calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate position size based on risk
 */
export function calculatePositionSize(
  accountBalance: number | string,
  riskPercent: number | string,
  entryPrice: number | string,
  stopLossPrice: number | string
): string {
  try {
    const balance = new Big(accountBalance);
    const risk = new Big(riskPercent).div(100);
    const entry = new Big(entryPrice);
    const stopLoss = new Big(stopLossPrice);

    const riskAmount = balance.times(risk);
    const priceRisk = entry.minus(stopLoss).abs();

    if (priceRisk.eq(0)) {
      return '0.00';
    }

    const positionSize = riskAmount.div(priceRisk);
    return positionSize.toFixed(8);
  } catch (error) {
    console.error('Position size calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate leverage
 */
export function calculateLeverage(
  positionValue: number | string,
  accountBalance: number | string
): string {
  try {
    const position = new Big(positionValue);
    const balance = new Big(accountBalance);

    if (balance.eq(0)) {
      return '0.00';
    }

    return position.div(balance).toFixed(2);
  } catch (error) {
    console.error('Leverage calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate Sharpe ratio
 */
export function calculateSharpeRatio(
  returns: (number | string)[],
  riskFreeRate: number | string = '0.02'
): string {
  try {
    if (returns.length === 0) {
      return '0.00';
    }

    // Convert to Big
    const bigReturns = returns.map(r => new Big(r));
    const rfRate = new Big(riskFreeRate);

    // Calculate mean return
    let sum = new Big(0);
    for (const ret of bigReturns) {
      sum = sum.plus(ret);
    }
    const meanReturn = sum.div(returns.length);

    // Calculate standard deviation
    let variance = new Big(0);
    for (const ret of bigReturns) {
      const diff = ret.minus(meanReturn);
      variance = variance.plus(diff.times(diff));
    }
    variance = variance.div(returns.length);

    const stdDev = new Big(Math.sqrt(parseFloat(variance.toString())));

    if (stdDev.eq(0)) {
      return '0.00';
    }

    // Sharpe ratio = (mean return - risk-free rate) / std dev
    const sharpe = meanReturn.minus(rfRate).div(stdDev);
    return sharpe.toFixed(2);
  } catch (error) {
    console.error('Sharpe ratio calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate maximum drawdown
 */
export function calculateMaxDrawdown(
  equityCurve: (number | string)[]
): string {
  try {
    if (equityCurve.length === 0) {
      return '0.00';
    }

    const values = equityCurve.map(v => new Big(v));
    let maxDrawdown = new Big(0);
    let peak = values[0];

    for (const value of values) {
      if (value.gt(peak)) {
        peak = value;
      }

      const drawdown = peak.minus(value).div(peak).times(100);
      if (drawdown.gt(maxDrawdown)) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown.toFixed(2);
  } catch (error) {
    console.error('Max drawdown calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate win rate
 */
export function calculateWinRate(
  trades: Array<{ pnl: number | string }>
): string {
  try {
    if (trades.length === 0) {
      return '0.00';
    }

    let wins = 0;
    for (const trade of trades) {
      const pnl = new Big(trade.pnl);
      if (pnl.gt(0)) {
        wins++;
      }
    }

    const winRate = new Big(wins).div(trades.length).times(100);
    return winRate.toFixed(2);
  } catch (error) {
    console.error('Win rate calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate profit factor
 */
export function calculateProfitFactor(
  trades: Array<{ pnl: number | string }>
): string {
  try {
    if (trades.length === 0) {
      return '0.00';
    }

    let grossProfit = new Big(0);
    let grossLoss = new Big(0);

    for (const trade of trades) {
      const pnl = new Big(trade.pnl);
      if (pnl.gt(0)) {
        grossProfit = grossProfit.plus(pnl);
      } else {
        grossLoss = grossLoss.plus(pnl.abs());
      }
    }

    if (grossLoss.eq(0)) {
      return grossProfit.gt(0) ? '999.99' : '0.00';
    }

    return grossProfit.div(grossLoss).toFixed(2);
  } catch (error) {
    console.error('Profit factor calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate fees (maker/taker)
 */
export function calculateFees(
  orderValue: number | string,
  feeRate: number | string
): string {
  try {
    const value = new Big(orderValue);
    const rate = new Big(feeRate);
    return value.times(rate).toFixed(8);
  } catch (error) {
    console.error('Fee calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate return on investment (ROI)
 */
export function calculateROI(
  initialValue: number | string,
  currentValue: number | string
): string {
  try {
    const initial = new Big(initialValue);
    const current = new Big(currentValue);

    if (initial.eq(0)) {
      return '0.00';
    }

    const roi = current.minus(initial).div(initial).times(100);
    return roi.toFixed(2);
  } catch (error) {
    console.error('ROI calculation error:', error);
    return '0.00';
  }
}

/**
 * Calculate compound annual growth rate (CAGR)
 */
export function calculateCAGR(
  initialValue: number | string,
  finalValue: number | string,
  years: number | string
): string {
  try {
    const initial = new Big(initialValue);
    const final = new Big(finalValue);
    const y = new Big(years);

    if (initial.eq(0) || y.eq(0)) {
      return '0.00';
    }

    // CAGR = (final/initial)^(1/years) - 1
    const ratio = parseFloat(final.div(initial).toString());
    const exponent = parseFloat(new Big(1).div(y).toString());
    const cagr = (Math.pow(ratio, exponent) - 1) * 100;

    return cagr.toFixed(2);
  } catch (error) {
    console.error('CAGR calculation error:', error);
    return '0.00';
  }
}

/**
 * Format number with commas and decimals
 */
export function formatNumber(
  value: number | string,
  decimals: number = 2
): string {
  try {
    const num = new Big(value);
    const formatted = num.toFixed(decimals);

    // Add thousand separators
    const parts = formatted.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');

    return parts.join('.');
  } catch (error) {
    console.error('Number formatting error:', error);
    return '0.00';
  }
}

/**
 * Safe division (returns 0 if divisor is 0)
 */
export function safeDivide(
  numerator: number | string,
  denominator: number | string,
  decimals: number = 2
): string {
  try {
    const num = new Big(numerator);
    const den = new Big(denominator);

    if (den.eq(0)) {
      return '0.00';
    }

    return num.div(den).toFixed(decimals);
  } catch (error) {
    console.error('Division error:', error);
    return '0.00';
  }
}
