/**
 * Chart type definitions for trading dashboard
 *
 * Type-safe chart data structures for:
 * - OHLCV candlestick charts
 * - Performance charts
 * - Indicator overlays
 * - Real-time updates
 */

/**
 * OHLCV candlestick data point
 */
export interface CandlestickData {
  timestamp: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

/**
 * Time series data point
 */
export interface TimeSeriesPoint {
  timestamp: number;
  value: string;
  label?: string;
}

/**
 * Line chart data
 */
export interface LineChartData {
  id: string;
  name: string;
  data: TimeSeriesPoint[];
  color?: string;
  strokeWidth?: number;
  dashed?: boolean;
}

/**
 * Area chart data
 */
export interface AreaChartData extends LineChartData {
  fillOpacity?: number;
  gradient?: boolean;
}

/**
 * Bar chart data point
 */
export interface BarDataPoint {
  label: string;
  value: string;
  color?: string;
}

/**
 * Bar chart data
 */
export interface BarChartData {
  id: string;
  name: string;
  data: BarDataPoint[];
  stacked?: boolean;
}

/**
 * Technical indicator data
 */
export interface IndicatorData {
  id: string;
  name: string;
  type: 'overlay' | 'separate';
  data: TimeSeriesPoint[] | MultiLineData;
  config?: IndicatorConfig;
}

/**
 * Multi-line indicator data (e.g., Bollinger Bands)
 */
export interface MultiLineData {
  [key: string]: TimeSeriesPoint[];
}

/**
 * Indicator configuration
 */
export interface IndicatorConfig {
  color?: string;
  strokeWidth?: number;
  opacity?: number;
  visible?: boolean;
  parameters?: Record<string, any>;
}

/**
 * Chart annotation (support/resistance, etc.)
 */
export interface ChartAnnotation {
  id: string;
  type: 'horizontal' | 'vertical' | 'trend' | 'text';
  value: string | number;
  label?: string;
  color?: string;
  style?: 'solid' | 'dashed' | 'dotted';
  timestamp?: number;
}

/**
 * Trade marker on chart
 */
export interface TradeMarker {
  id: string;
  timestamp: number;
  price: string;
  side: 'buy' | 'sell';
  quantity: string;
  label?: string;
  color?: string;
}

/**
 * Chart configuration
 */
export interface ChartConfig {
  type: 'candlestick' | 'line' | 'area' | 'bar';
  timeframe: string;
  height?: number;
  showVolume?: boolean;
  showGrid?: boolean;
  showCrosshair?: boolean;
  showLegend?: boolean;
  theme?: 'light' | 'dark';
  dateFormat?: string;
  priceDecimals?: number;
  volumeDecimals?: number;
}

/**
 * Chart data container
 */
export interface ChartData {
  symbol: string;
  timeframe: string;
  candlesticks?: CandlestickData[];
  indicators?: IndicatorData[];
  annotations?: ChartAnnotation[];
  tradeMarkers?: TradeMarker[];
  lastUpdate: number;
}

/**
 * Performance chart data
 */
export interface PerformanceChartData {
  equity: TimeSeriesPoint[];
  drawdown: TimeSeriesPoint[];
  returns: TimeSeriesPoint[];
  benchmark?: TimeSeriesPoint[];
}

/**
 * Portfolio allocation chart data
 */
export interface AllocationData {
  symbol: string;
  value: string;
  percentage: string;
  color?: string;
}

/**
 * Heatmap data point
 */
export interface HeatmapPoint {
  x: number | string;
  y: number | string;
  value: string;
  color?: string;
}

/**
 * Heatmap data
 */
export interface HeatmapData {
  id: string;
  name: string;
  data: HeatmapPoint[];
  colorScale?: string[];
}

/**
 * Chart tooltip data
 */
export interface TooltipData {
  timestamp: number;
  values: Record<string, string>;
  formattedDate?: string;
}

/**
 * Chart axis configuration
 */
export interface AxisConfig {
  type: 'linear' | 'log' | 'time';
  min?: number;
  max?: number;
  tickFormat?: string;
  gridLines?: boolean;
  label?: string;
}

/**
 * Chart viewport (zoom/pan state)
 */
export interface ChartViewport {
  startTime: number;
  endTime: number;
  minPrice?: string;
  maxPrice?: string;
  scale?: number;
}

/**
 * Chart update event
 */
export interface ChartUpdateEvent {
  symbol: string;
  timeframe: string;
  type: 'candle' | 'indicator' | 'trade' | 'annotation';
  data: any;
  timestamp: number;
}

/**
 * Chart error
 */
export interface ChartError {
  code: string;
  message: string;
  timestamp: number;
}

/**
 * Supported timeframes
 */
export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w' | '1M';

/**
 * Supported chart themes
 */
export type ChartTheme = 'light' | 'dark' | 'custom';

/**
 * Chart loading state
 */
export interface ChartLoadingState {
  isLoading: boolean;
  progress?: number;
  message?: string;
}

/**
 * Real-time chart subscription
 */
export interface ChartSubscription {
  symbol: string;
  timeframe: Timeframe;
  indicators?: string[];
  callback: (event: ChartUpdateEvent) => void;
  errorCallback?: (error: ChartError) => void;
}

/**
 * Chart export options
 */
export interface ChartExportOptions {
  format: 'png' | 'svg' | 'pdf' | 'csv';
  width?: number;
  height?: number;
  filename?: string;
  includeIndicators?: boolean;
  includeAnnotations?: boolean;
}

/**
 * Indicator parameter definition
 */
export interface IndicatorParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'select';
  default: any;
  min?: number;
  max?: number;
  options?: string[];
  description?: string;
}

/**
 * Available indicators
 */
export interface IndicatorDefinition {
  id: string;
  name: string;
  category: 'trend' | 'momentum' | 'volatility' | 'volume';
  description: string;
  parameters: IndicatorParameter[];
  overlay: boolean;
}

/**
 * Chart preset (saved configuration)
 */
export interface ChartPreset {
  id: string;
  name: string;
  description?: string;
  config: ChartConfig;
  indicators: Array<{
    id: string;
    parameters: Record<string, any>;
  }>;
  annotations?: ChartAnnotation[];
  createdAt: number;
  updatedAt: number;
}

/**
 * Chart comparison data (multiple symbols)
 */
export interface ComparisonChartData {
  symbols: string[];
  normalized: boolean;
  baseValue: string;
  data: Record<string, TimeSeriesPoint[]>;
}

/**
 * Order book depth data
 */
export interface OrderBookLevel {
  price: string;
  quantity: string;
  total: string;
}

/**
 * Order book chart data
 */
export interface OrderBookData {
  symbol: string;
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  midPrice: string;
  spread: string;
  timestamp: number;
}

/**
 * Volume profile data
 */
export interface VolumeProfileLevel {
  price: string;
  volume: string;
  percentage: string;
}

/**
 * Volume profile chart data
 */
export interface VolumeProfileData {
  symbol: string;
  timeframe: string;
  levels: VolumeProfileLevel[];
  pocPrice: string; // Point of Control
  valueAreaHigh: string;
  valueAreaLow: string;
}
