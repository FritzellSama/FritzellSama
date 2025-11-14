/**
 * Chart Type Definitions
 *
 * Type definitions for charting components using Recharts and Lightweight Charts
 */

import { CandlestickData, LineData, HistogramData, Time } from 'lightweight-charts';

// Time-based data types
export type TimeStamp = string | number | Date;
export type ChartTime = Time;

// OHLCV Data Types
export interface OHLCVData {
  timestamp: TimeStamp;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface CandlestickChartData extends OHLCVData {
  time: ChartTime;
}

// Line Chart Data
export interface LineChartData {
  timestamp: TimeStamp;
  value: number;
  label?: string;
}

export interface MultiLineChartData {
  timestamp: TimeStamp;
  [key: string]: number | TimeStamp;
}

// Bar/Histogram Chart Data
export interface BarChartData {
  timestamp: TimeStamp;
  value: number;
  color?: string;
  label?: string;
}

export interface HistogramChartData extends HistogramData {
  time: ChartTime;
}

// Area Chart Data
export interface AreaChartData {
  timestamp: TimeStamp;
  value: number;
  area?: number;
  label?: string;
}

// Scatter Plot Data
export interface ScatterData {
  x: number;
  y: number;
  z?: number;
  label?: string;
  color?: string;
}

// Heatmap Data
export interface HeatmapCell {
  x: number;
  y: number;
  value: number;
  label?: string;
}

export interface HeatmapData {
  xLabels: string[];
  yLabels: string[];
  data: number[][];
}

// Portfolio Performance Data
export interface PortfolioPerformanceData {
  timestamp: TimeStamp;
  totalValue: number;
  cashBalance: number;
  investedValue: number;
  unrealizedPnL: number;
  realizedPnL: number;
  dailyReturn?: number;
  cumulativeReturn?: number;
}

// Equity Curve Data
export interface EquityCurveData {
  timestamp: TimeStamp;
  equity: number;
  drawdown: number;
  drawdownPercent: number;
  benchmark?: number;
}

// PnL Data
export interface PnLData {
  timestamp: TimeStamp;
  realizedPnL: number;
  unrealizedPnL: number;
  totalPnL: number;
  cumulativePnL: number;
}

// Risk Metrics Chart Data
export interface RiskMetricsData {
  timestamp: TimeStamp;
  var95: number;
  cvar95: number;
  sharpeRatio?: number;
  sortinoRatio?: number;
  maxDrawdown?: number;
}

// Strategy Performance Data
export interface StrategyPerformanceData {
  strategyName: string;
  returns: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  trades: number;
  pnl: number;
}

// Asset Allocation Data
export interface AllocationData {
  name: string;
  value: number;
  percentage: number;
  color?: string;
  category?: string;
}

// Trade Distribution Data
export interface TradeDistributionData {
  range: string;
  count: number;
  profit: number;
}

// Volume Profile Data
export interface VolumeProfileData {
  price: number;
  volume: number;
  buyVolume?: number;
  sellVolume?: number;
}

// Order Book Data
export interface OrderBookLevel {
  price: number;
  quantity: number;
  total: number;
}

export interface OrderBookData {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: TimeStamp;
}

// Chart Configuration
export interface ChartConfig {
  width?: number;
  height?: number;
  responsive?: boolean;
  margin?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  xAxis?: AxisConfig;
  yAxis?: AxisConfig;
  grid?: GridConfig;
  tooltip?: TooltipConfig;
  legend?: LegendConfig;
  theme?: 'light' | 'dark';
}

export interface AxisConfig {
  show?: boolean;
  label?: string;
  tickCount?: number;
  tickFormat?: (value: any) => string;
  domain?: [number | 'auto', number | 'auto'];
  scale?: 'linear' | 'log' | 'time';
}

export interface GridConfig {
  show?: boolean;
  strokeDasharray?: string;
  stroke?: string;
  opacity?: number;
}

export interface TooltipConfig {
  show?: boolean;
  formatter?: (value: any, name: string, props: any) => any;
  labelFormatter?: (label: any) => string;
  cursor?: boolean | object;
}

export interface LegendConfig {
  show?: boolean;
  position?: 'top' | 'bottom' | 'left' | 'right';
  align?: 'left' | 'center' | 'right';
  verticalAlign?: 'top' | 'middle' | 'bottom';
}

// Lightweight Charts Configuration
export interface LightweightChartConfig {
  width?: number;
  height?: number;
  layout?: {
    background?: { color: string };
    textColor?: string;
  };
  grid?: {
    vertLines?: { color: string; visible?: boolean };
    horzLines?: { color: string; visible?: boolean };
  };
  crosshair?: {
    mode?: number;
    vertLine?: { color?: string; width?: number; style?: number; visible?: boolean };
    horzLine?: { color?: string; width?: number; style?: number; visible?: boolean };
  };
  timeScale?: {
    timeVisible?: boolean;
    secondsVisible?: boolean;
    borderColor?: string;
    rightOffset?: number;
    barSpacing?: number;
    minBarSpacing?: number;
  };
  rightPriceScale?: {
    borderColor?: string;
    scaleMargins?: { top: number; bottom: number };
    mode?: number;
  };
  leftPriceScale?: {
    visible?: boolean;
    borderColor?: string;
  };
  handleScroll?: boolean | { mouseWheel?: boolean; pressedMouseMove?: boolean; horzTouchDrag?: boolean; vertTouchDrag?: boolean };
  handleScale?: boolean | { axisPressedMouseMove?: boolean; mouseWheel?: boolean; pinch?: boolean };
}

// Indicator Data Types
export interface RSIData {
  timestamp: TimeStamp;
  rsi: number;
  overbought?: number;
  oversold?: number;
}

export interface MACDData {
  timestamp: TimeStamp;
  macd: number;
  signal: number;
  histogram: number;
}

export interface BollingerBandsData {
  timestamp: TimeStamp;
  upper: number;
  middle: number;
  lower: number;
  price: number;
}

export interface MovingAverageData {
  timestamp: TimeStamp;
  value: number;
  period: number;
  type: 'SMA' | 'EMA' | 'WMA';
}

export interface StochasticData {
  timestamp: TimeStamp;
  k: number;
  d: number;
  overbought?: number;
  oversold?: number;
}

export interface ATRData {
  timestamp: TimeStamp;
  atr: number;
}

export interface VWAPData {
  timestamp: TimeStamp;
  vwap: number;
  price: number;
}

// Chart Series Types
export type SeriesType =
  | 'line'
  | 'area'
  | 'bar'
  | 'candlestick'
  | 'histogram'
  | 'scatter'
  | 'heatmap';

export interface ChartSeries<T = any> {
  id: string;
  name: string;
  type: SeriesType;
  data: T[];
  color?: string;
  visible?: boolean;
  yAxisId?: string;
  strokeWidth?: number;
  fillOpacity?: number;
  options?: any;
}

// Chart Reference Type
export interface ChartHandle {
  updateData: (data: any) => void;
  resize: (width: number, height: number) => void;
  fitContent: () => void;
  scrollToPosition: (position: number) => void;
  takeScreenshot: () => string;
}

// Timeframe Types
export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

export interface TimeframeOption {
  value: Timeframe;
  label: string;
  milliseconds: number;
}

// Chart Annotations
export interface ChartAnnotation {
  id: string;
  type: 'line' | 'rect' | 'circle' | 'text' | 'arrow';
  timestamp?: TimeStamp;
  price?: number;
  startTimestamp?: TimeStamp;
  endTimestamp?: TimeStamp;
  startPrice?: number;
  endPrice?: number;
  text?: string;
  color?: string;
  strokeWidth?: number;
  fillOpacity?: number;
}

// Chart Drawing Tool
export type DrawingTool =
  | 'none'
  | 'trendline'
  | 'horizontal'
  | 'vertical'
  | 'rectangle'
  | 'fibonacci'
  | 'text';

export interface DrawingState {
  tool: DrawingTool;
  isDrawing: boolean;
  annotations: ChartAnnotation[];
}

// Price Alert
export interface PriceAlert {
  id: string;
  symbol: string;
  price: number;
  condition: 'above' | 'below' | 'crosses';
  enabled: boolean;
  triggered: boolean;
  createdAt: Date;
}

// Chart Color Schemes
export interface ColorScheme {
  background: string;
  text: string;
  grid: string;
  upColor: string;
  downColor: string;
  volumeUpColor: string;
  volumeDownColor: string;
  borderColor: string;
  crosshairColor: string;
  axisColor: string;
}

export const darkColorScheme: ColorScheme = {
  background: '#0f172a',
  text: '#f8fafc',
  grid: '#334155',
  upColor: '#10b981',
  downColor: '#ef4444',
  volumeUpColor: 'rgba(16, 185, 129, 0.3)',
  volumeDownColor: 'rgba(239, 68, 68, 0.3)',
  borderColor: '#334155',
  crosshairColor: '#64748b',
  axisColor: '#94a3b8',
};

export const lightColorScheme: ColorScheme = {
  background: '#ffffff',
  text: '#1e293b',
  grid: '#e2e8f0',
  upColor: '#10b981',
  downColor: '#ef4444',
  volumeUpColor: 'rgba(16, 185, 129, 0.3)',
  volumeDownColor: 'rgba(239, 68, 68, 0.3)',
  borderColor: '#cbd5e1',
  crosshairColor: '#94a3b8',
  axisColor: '#64748b',
};

// Export aggregated types
export type ChartData =
  | OHLCVData
  | LineChartData
  | BarChartData
  | AreaChartData
  | ScatterData
  | PortfolioPerformanceData
  | EquityCurveData
  | PnLData
  | RiskMetricsData;
