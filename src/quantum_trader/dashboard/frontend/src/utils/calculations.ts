/**
 * Financial Calculations and Metrics
 *
 * Comprehensive suite of financial calculations for trading analytics,
 * risk metrics, performance analysis, and statistical computations.
 */

import type { PerformanceMetrics, Position, Trade } from '@/types';

// Statistical Calculations

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function standardDeviation(values: number[]): number {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const squareDiffs = values.map((value) => Math.pow(value - avg, 2));
  return Math.sqrt(mean(squareDiffs));
}

export function variance(values: number[]): number {
  if (values.length === 0) return 0;
  const avg = mean(values);
  const squareDiffs = values.map((value) => Math.pow(value - avg, 2));
  return mean(squareDiffs);
}

export function covariance(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
  const xMean = mean(x);
  const yMean = mean(y);
  return mean(x.map((xi, i) => (xi - xMean) * (y[i] - yMean)));
}

export function correlation(x: number[], y: number[]): number {
  const cov = covariance(x, y);
  const xStd = standardDeviation(x);
  const yStd = standardDeviation(y);
  if (xStd === 0 || yStd === 0) return 0;
  return cov / (xStd * yStd);
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index % 1;
  if (lower === upper) return sorted[lower];
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

// Return Calculations

export function simpleReturn(startValue: number, endValue: number): number {
  if (startValue === 0) return 0;
  return (endValue - startValue) / startValue;
}

export function logReturn(startValue: number, endValue: number): number {
  if (startValue === 0 || endValue === 0) return 0;
  return Math.log(endValue / startValue);
}

export function cumulativeReturn(returns: number[]): number {
  return returns.reduce((cum, ret) => (1 + cum) * (1 + ret) - 1, 0);
}

export function annualizedReturn(totalReturn: number, days: number): number {
  const years = days / 365;
  if (years === 0) return 0;
  return Math.pow(1 + totalReturn, 1 / years) - 1;
}

export function compoundAnnualGrowthRate(
  startValue: number,
  endValue: number,
  years: number
): number {
  if (startValue === 0 || years === 0) return 0;
  return Math.pow(endValue / startValue, 1 / years) - 1;
}

// Risk Metrics

export function sharpeRatio(
  returns: number[],
  riskFreeRate: number = 0
): number {
  if (returns.length === 0) return 0;
  const excessReturns = returns.map((r) => r - riskFreeRate);
  const avgExcessReturn = mean(excessReturns);
  const stdDev = standardDeviation(returns);
  if (stdDev === 0) return 0;
  return avgExcessReturn / stdDev;
}

export function annualizedSharpeRatio(
  returns: number[],
  riskFreeRate: number = 0,
  periodsPerYear: number = 252
): number {
  const sr = sharpeRatio(returns, riskFreeRate);
  return sr * Math.sqrt(periodsPerYear);
}

export function sortinoRatio(
  returns: number[],
  targetReturn: number = 0,
  periodsPerYear: number = 252
): number {
  if (returns.length === 0) return 0;
  const excessReturns = returns.map((r) => r - targetReturn);
  const avgExcessReturn = mean(excessReturns);
  const downside = returns.filter((r) => r < targetReturn);
  if (downside.length === 0) return Infinity;
  const downsideDeviation = standardDeviation(downside);
  if (downsideDeviation === 0) return 0;
  return (avgExcessReturn / downsideDeviation) * Math.sqrt(periodsPerYear);
}

export function maxDrawdown(equityCurve: number[]): {
  maxDrawdown: number;
  maxDrawdownPct: number;
  peak: number;
  trough: number;
  peakIndex: number;
  troughIndex: number;
} {
  let peak = equityCurve[0];
  let peakIndex = 0;
  let maxDD = 0;
  let maxDDPct = 0;
  let trough = equityCurve[0];
  let troughIndex = 0;

  for (let i = 0; i < equityCurve.length; i++) {
    if (equityCurve[i] > peak) {
      peak = equityCurve[i];
      peakIndex = i;
    }
    const dd = peak - equityCurve[i];
    const ddPct = peak > 0 ? dd / peak : 0;
    if (dd > maxDD) {
      maxDD = dd;
      maxDDPct = ddPct;
      trough = equityCurve[i];
      troughIndex = i;
    }
  }

  return {
    maxDrawdown: maxDD,
    maxDrawdownPct: maxDDPct,
    peak,
    trough,
    peakIndex,
    troughIndex,
  };
}

export function calmarRatio(
  annualizedReturn: number,
  maxDrawdownPct: number
): number {
  if (maxDrawdownPct === 0) return 0;
  return annualizedReturn / maxDrawdownPct;
}

export function valueAtRisk(
  returns: number[],
  confidenceLevel: number = 0.95
): number {
  if (returns.length === 0) return 0;
  return -percentile(returns, (1 - confidenceLevel) * 100);
}

export function conditionalValueAtRisk(
  returns: number[],
  confidenceLevel: number = 0.95
): number {
  const var95 = valueAtRisk(returns, confidenceLevel);
  const tailLosses = returns.filter((r) => r <= -var95);
  return tailLosses.length > 0 ? -mean(tailLosses) : 0;
}

export function beta(assetReturns: number[], marketReturns: number[]): number {
  const cov = covariance(assetReturns, marketReturns);
  const marketVar = variance(marketReturns);
  if (marketVar === 0) return 0;
  return cov / marketVar;
}

export function alpha(
  assetReturn: number,
  riskFreeRate: number,
  beta: number,
  marketReturn: number
): number {
  return assetReturn - (riskFreeRate + beta * (marketReturn - riskFreeRate));
}

export function informationRatio(
  portfolioReturns: number[],
  benchmarkReturns: number[]
): number {
  if (portfolioReturns.length !== benchmarkReturns.length) return 0;
  const excessReturns = portfolioReturns.map(
    (pr, i) => pr - benchmarkReturns[i]
  );
  const avgExcessReturn = mean(excessReturns);
  const trackingError = standardDeviation(excessReturns);
  if (trackingError === 0) return 0;
  return avgExcessReturn / trackingError;
}

export function trackingError(
  portfolioReturns: number[],
  benchmarkReturns: number[]
): number {
  if (portfolioReturns.length !== benchmarkReturns.length) return 0;
  const excessReturns = portfolioReturns.map(
    (pr, i) => pr - benchmarkReturns[i]
  );
  return standardDeviation(excessReturns);
}

// Trading Performance Metrics

export function profitFactor(wins: number[], losses: number[]): number {
  const totalWins = wins.reduce((sum, win) => sum + Math.abs(win), 0);
  const totalLosses = losses.reduce((sum, loss) => sum + Math.abs(loss), 0);
  if (totalLosses === 0) return totalWins > 0 ? Infinity : 0;
  return totalWins / totalLosses;
}

export function winRate(numWins: number, numTrades: number): number {
  if (numTrades === 0) return 0;
  return numWins / numTrades;
}

export function expectancy(
  avgWin: number,
  avgLoss: number,
  winRate: number
): number {
  return winRate * avgWin - (1 - winRate) * Math.abs(avgLoss);
}

export function payoffRatio(avgWin: number, avgLoss: number): number {
  if (avgLoss === 0) return avgWin > 0 ? Infinity : 0;
  return avgWin / Math.abs(avgLoss);
}

export function kellyPercentage(
  winRate: number,
  payoffRatio: number
): number {
  if (payoffRatio === 0) return 0;
  return winRate - (1 - winRate) / payoffRatio;
}

export function calculateTradeMetrics(trades: Trade[]): {
  numTrades: number;
  numWins: number;
  numLosses: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  expectancy: number;
  payoffRatio: number;
} {
  const wins = trades
    .filter((t) => t.realizedPnL && t.realizedPnL > 0)
    .map((t) => t.realizedPnL!);
  const losses = trades
    .filter((t) => t.realizedPnL && t.realizedPnL < 0)
    .map((t) => t.realizedPnL!);

  const numTrades = trades.length;
  const numWins = wins.length;
  const numLosses = losses.length;
  const wr = winRate(numWins, numTrades);
  const avgWin = wins.length > 0 ? mean(wins) : 0;
  const avgLoss = losses.length > 0 ? mean(losses) : 0;
  const largestWin = wins.length > 0 ? Math.max(...wins) : 0;
  const largestLoss = losses.length > 0 ? Math.min(...losses) : 0;
  const pf = profitFactor(wins, losses);
  const exp = expectancy(avgWin, avgLoss, wr);
  const pr = payoffRatio(avgWin, avgLoss);

  return {
    numTrades,
    numWins,
    numLosses,
    winRate: wr,
    avgWin,
    avgLoss,
    largestWin,
    largestLoss,
    profitFactor: pf,
    expectancy: exp,
    payoffRatio: pr,
  };
}

// Position Calculations

export function positionValue(quantity: number, currentPrice: number): number {
  return quantity * currentPrice;
}

export function unrealizedPnL(
  quantity: number,
  entryPrice: number,
  currentPrice: number,
  side: 'BUY' | 'SELL'
): number {
  const diff = currentPrice - entryPrice;
  return side === 'BUY' ? quantity * diff : quantity * -diff;
}

export function unrealizedPnLPercentage(
  entryPrice: number,
  currentPrice: number,
  side: 'BUY' | 'SELL'
): number {
  if (entryPrice === 0) return 0;
  const diff = currentPrice - entryPrice;
  const pct = diff / entryPrice;
  return side === 'BUY' ? pct : -pct;
}

export function calculatePositionMetrics(position: Position): {
  marketValue: number;
  costBasis: number;
  unrealizedPnL: number;
  unrealizedPnLPct: number;
  returnOnInvestment: number;
} {
  const marketValue = positionValue(position.quantity, position.currentPrice);
  const costBasis = positionValue(position.quantity, position.entryPrice);
  const pnl = unrealizedPnL(
    position.quantity,
    position.entryPrice,
    position.currentPrice,
    position.side
  );
  const pnlPct = unrealizedPnLPercentage(
    position.entryPrice,
    position.currentPrice,
    position.side
  );
  const roi = costBasis > 0 ? pnl / costBasis : 0;

  return {
    marketValue,
    costBasis,
    unrealizedPnL: pnl,
    unrealizedPnLPct: pnlPct,
    returnOnInvestment: roi,
  };
}

// Portfolio Calculations

export function portfolioTotalValue(
  cashBalance: number,
  positions: Position[]
): number {
  const positionsValue = positions.reduce(
    (sum, pos) => sum + positionValue(pos.quantity, pos.currentPrice),
    0
  );
  return cashBalance + positionsValue;
}

export function portfolioWeights(positions: Position[]): Map<string, number> {
  const totalValue = positions.reduce(
    (sum, pos) => sum + positionValue(pos.quantity, pos.currentPrice),
    0
  );

  const weights = new Map<string, number>();
  positions.forEach((pos) => {
    const value = positionValue(pos.quantity, pos.currentPrice);
    weights.set(pos.symbol, totalValue > 0 ? value / totalValue : 0);
  });

  return weights;
}

export function concentrationRisk(positions: Position[]): number {
  const weights = Array.from(portfolioWeights(positions).values());
  weights.sort((a, b) => b - a);
  const top5 = weights.slice(0, 5);
  return top5.reduce((sum, w) => sum + w, 0);
}

export function portfolioBeta(
  positions: Position[],
  betas: Map<string, number>
): number {
  const weights = portfolioWeights(positions);
  let portfolioBeta = 0;
  weights.forEach((weight, symbol) => {
    const beta = betas.get(symbol) || 1;
    portfolioBeta += weight * beta;
  });
  return portfolioBeta;
}

// Risk Calculations

export function positionSizeKelly(
  winRate: number,
  payoffRatio: number,
  accountBalance: number,
  kellyFraction: number = 0.25
): number {
  const kelly = kellyPercentage(winRate, payoffRatio);
  const safeKelly = Math.max(0, Math.min(kelly, 0.25)) * kellyFraction;
  return accountBalance * safeKelly;
}

export function positionSizeFixedRisk(
  accountBalance: number,
  riskPercentage: number,
  entryPrice: number,
  stopLossPrice: number
): number {
  const riskAmount = accountBalance * riskPercentage;
  const riskPerShare = Math.abs(entryPrice - stopLossPrice);
  if (riskPerShare === 0) return 0;
  return riskAmount / riskPerShare;
}

export function positionSizeVolatilityBased(
  accountBalance: number,
  targetVolatility: number,
  assetVolatility: number,
  currentPrice: number
): number {
  if (assetVolatility === 0) return 0;
  const targetDollarVolatility = accountBalance * targetVolatility;
  return targetDollarVolatility / (currentPrice * assetVolatility);
}

export function leverageRatio(
  totalPositionValue: number,
  equity: number
): number {
  if (equity === 0) return 0;
  return totalPositionValue / equity;
}

export function marginRequired(
  positionValue: number,
  leverage: number
): number {
  if (leverage === 0) return positionValue;
  return positionValue / leverage;
}

// Technical Indicators (Simple Implementations)

export function simpleMovingAverage(
  values: number[],
  period: number
): number[] {
  const sma: number[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      sma.push(NaN);
    } else {
      const slice = values.slice(i - period + 1, i + 1);
      sma.push(mean(slice));
    }
  }
  return sma;
}

export function exponentialMovingAverage(
  values: number[],
  period: number
): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  for (let i = 0; i < values.length; i++) {
    if (i === 0) {
      ema.push(values[0]);
    } else if (i < period) {
      ema.push(mean(values.slice(0, i + 1)));
    } else {
      ema.push((values[i] - ema[i - 1]) * multiplier + ema[i - 1]);
    }
  }
  return ema;
}

export function relativeStrengthIndex(
  prices: number[],
  period: number = 14
): number[] {
  const rsi: number[] = [];
  const changes: number[] = [];

  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  for (let i = 0; i < prices.length; i++) {
    if (i < period) {
      rsi.push(NaN);
    } else {
      const slice = changes.slice(i - period, i);
      const gains = slice.filter((c) => c > 0);
      const losses = slice.filter((c) => c < 0).map((c) => Math.abs(c));
      const avgGain = gains.length > 0 ? mean(gains) : 0;
      const avgLoss = losses.length > 0 ? mean(losses) : 0;
      if (avgLoss === 0) {
        rsi.push(100);
      } else {
        const rs = avgGain / avgLoss;
        rsi.push(100 - 100 / (1 + rs));
      }
    }
  }

  return rsi;
}

// Utility Functions

export function roundToDecimalPlaces(value: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function normalize(value: number, min: number, max: number): number {
  if (max === min) return 0;
  return (value - min) / (max - min);
}

export function lerp(start: number, end: number, t: number): number {
  return start + (end - start) * t;
}

export function movingSum(values: number[], period: number): number[] {
  const result: number[] = [];
  let sum = 0;

  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) {
      sum -= values[i - period];
    }
    result.push(i < period - 1 ? NaN : sum);
  }

  return result;
}

export function rateOfChange(values: number[], period: number = 1): number[] {
  const roc: number[] = [];

  for (let i = 0; i < values.length; i++) {
    if (i < period) {
      roc.push(NaN);
    } else {
      const change = values[i] - values[i - period];
      roc.push(values[i - period] !== 0 ? change / values[i - period] : 0);
    }
  }

  return roc;
}

// Export all functions as a namespace for convenience
export default {
  mean,
  median,
  standardDeviation,
  variance,
  covariance,
  correlation,
  percentile,
  simpleReturn,
  logReturn,
  cumulativeReturn,
  annualizedReturn,
  compoundAnnualGrowthRate,
  sharpeRatio,
  annualizedSharpeRatio,
  sortinoRatio,
  maxDrawdown,
  calmarRatio,
  valueAtRisk,
  conditionalValueAtRisk,
  beta,
  alpha,
  informationRatio,
  trackingError,
  profitFactor,
  winRate,
  expectancy,
  payoffRatio,
  kellyPercentage,
  calculateTradeMetrics,
  positionValue,
  unrealizedPnL,
  unrealizedPnLPercentage,
  calculatePositionMetrics,
  portfolioTotalValue,
  portfolioWeights,
  concentrationRisk,
  portfolioBeta,
  positionSizeKelly,
  positionSizeFixedRisk,
  positionSizeVolatilityBased,
  leverageRatio,
  marginRequired,
  simpleMovingAverage,
  exponentialMovingAverage,
  relativeStrengthIndex,
  roundToDecimalPlaces,
  clamp,
  normalize,
  lerp,
  movingSum,
  rateOfChange,
};
