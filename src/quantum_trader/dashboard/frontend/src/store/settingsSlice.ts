/**
 * Settings Redux Slice
 *
 * Manages user preferences and system settings for the dashboard.
 * Handles theme, display preferences, notifications, and API configurations.
 *
 * @module settingsSlice
 */

import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from './index';
import { storage } from '../services/storage';

/**
 * Theme options
 */
export type Theme = 'light' | 'dark' | 'auto';

/**
 * Display settings
 */
export interface DisplaySettings {
  theme: Theme;
  compactMode: boolean;
  showAnimations: boolean;
  fontSize: 'small' | 'medium' | 'large';
  colorScheme: 'default' | 'colorblind' | 'high-contrast';
  chartType: 'candlestick' | 'line' | 'area';
  showGrid: boolean;
  showVolume: boolean;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  enabled: boolean;
  sound: boolean;
  desktop: boolean;
  email: boolean;
  tradingAlerts: boolean;
  priceAlerts: boolean;
  riskAlerts: boolean;
  systemAlerts: boolean;
  volume: number; // 0-100
}

/**
 * Trading settings
 */
export interface TradingSettings {
  defaultExchange: string;
  defaultSymbol: string;
  defaultOrderType: 'MARKET' | 'LIMIT';
  confirmOrders: boolean;
  autoRefreshInterval: number; // seconds
  maxSlippagePercent: string; // Decimal as string
  defaultLeverage: number;
  enableHotkeys: boolean;
}

/**
 * API settings
 */
export interface APISettings {
  baseUrl: string;
  wsUrl: string;
  timeout: number; // milliseconds
  retryAttempts: number;
  retryDelay: number; // milliseconds
}

/**
 * Risk settings
 */
export interface RiskSettings {
  maxPositionSize: string; // Decimal as string
  maxDailyLoss: string; // Decimal as string
  maxDrawdown: string; // Decimal as string
  enableRiskChecks: boolean;
  enableEmergencyStop: boolean;
  emergencyStopThreshold: string; // Decimal as string
}

/**
 * Advanced settings
 */
export interface AdvancedSettings {
  debugMode: boolean;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  enableAnalytics: boolean;
  cacheEnabled: boolean;
  cacheDuration: number; // minutes
  maxHistoryDays: number;
}

/**
 * Complete settings state
 */
export interface SettingsState {
  display: DisplaySettings;
  notifications: NotificationSettings;
  trading: TradingSettings;
  api: APISettings;
  risk: RiskSettings;
  advanced: AdvancedSettings;
  loading: boolean;
  error: string | null;
  lastSaved: string | null;
}

/**
 * Default settings
 */
const defaultSettings: SettingsState = {
  display: {
    theme: 'dark',
    compactMode: false,
    showAnimations: true,
    fontSize: 'medium',
    colorScheme: 'default',
    chartType: 'candlestick',
    showGrid: true,
    showVolume: true,
  },
  notifications: {
    enabled: true,
    sound: true,
    desktop: true,
    email: false,
    tradingAlerts: true,
    priceAlerts: true,
    riskAlerts: true,
    systemAlerts: true,
    volume: 50,
  },
  trading: {
    defaultExchange: process.env.REACT_APP_DEFAULT_EXCHANGE || 'BINANCE',
    defaultSymbol: process.env.REACT_APP_DEFAULT_SYMBOL || 'BTC/USDT',
    defaultOrderType: 'LIMIT',
    confirmOrders: true,
    autoRefreshInterval: parseInt(process.env.REACT_APP_AUTO_REFRESH_INTERVAL || '5', 10),
    maxSlippagePercent: process.env.REACT_APP_MAX_SLIPPAGE || '0.5',
    defaultLeverage: 1,
    enableHotkeys: true,
  },
  api: {
    baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8080',
    wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws',
    timeout: parseInt(process.env.REACT_APP_API_TIMEOUT || '30000', 10),
    retryAttempts: parseInt(process.env.REACT_APP_API_RETRY_ATTEMPTS || '3', 10),
    retryDelay: parseInt(process.env.REACT_APP_API_RETRY_DELAY || '1000', 10),
  },
  risk: {
    maxPositionSize: process.env.REACT_APP_MAX_POSITION_SIZE || '10000',
    maxDailyLoss: process.env.REACT_APP_MAX_DAILY_LOSS || '1000',
    maxDrawdown: process.env.REACT_APP_MAX_DRAWDOWN || '5000',
    enableRiskChecks: true,
    enableEmergencyStop: true,
    emergencyStopThreshold: process.env.REACT_APP_EMERGENCY_STOP_THRESHOLD || '10000',
  },
  advanced: {
    debugMode: process.env.NODE_ENV === 'development',
    logLevel: (process.env.REACT_APP_LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error') || 'info',
    enableAnalytics: process.env.REACT_APP_ENABLE_ANALYTICS === 'true',
    cacheEnabled: true,
    cacheDuration: parseInt(process.env.REACT_APP_CACHE_DURATION || '30', 10),
    maxHistoryDays: parseInt(process.env.REACT_APP_MAX_HISTORY_DAYS || '90', 10),
  },
  loading: false,
  error: null,
  lastSaved: null,
};

/**
 * Async thunks
 */

// Load settings from storage
export const loadSettings = createAsyncThunk(
  'settings/load',
  async (_, { rejectWithValue }) => {
    try {
      const stored = await storage.getItem<SettingsState>('settings');
      return stored || defaultSettings;
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Save settings to storage
export const saveSettings = createAsyncThunk(
  'settings/save',
  async (settings: SettingsState, { rejectWithValue }) => {
    try {
      await storage.setItem('settings', settings);
      return new Date().toISOString();
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Reset settings to defaults
export const resetSettings = createAsyncThunk(
  'settings/reset',
  async (_, { rejectWithValue }) => {
    try {
      await storage.removeItem('settings');
      return defaultSettings;
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Export settings to file
export const exportSettings = createAsyncThunk(
  'settings/export',
  async (settings: SettingsState, { rejectWithValue }) => {
    try {
      const dataStr = JSON.stringify(settings, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);

      const link = document.createElement('a');
      link.href = url;
      link.download = `quantum-trader-settings-${new Date().toISOString()}.json`;
      link.click();

      URL.revokeObjectURL(url);

      return true;
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

// Import settings from file
export const importSettings = createAsyncThunk(
  'settings/import',
  async (file: File, { rejectWithValue }) => {
    try {
      const text = await file.text();
      const settings = JSON.parse(text) as SettingsState;

      // Validate settings structure
      if (!settings.display || !settings.notifications || !settings.trading) {
        throw new Error('Invalid settings file format');
      }

      await storage.setItem('settings', settings);

      return settings;
    } catch (error) {
      return rejectWithValue((error as Error).message);
    }
  }
);

/**
 * Settings slice
 */
const settingsSlice = createSlice({
  name: 'settings',
  initialState: defaultSettings,
  reducers: {
    // Update display settings
    updateDisplaySettings: (state, action: PayloadAction<Partial<DisplaySettings>>) => {
      state.display = { ...state.display, ...action.payload };
    },

    // Update notification settings
    updateNotificationSettings: (state, action: PayloadAction<Partial<NotificationSettings>>) => {
      state.notifications = { ...state.notifications, ...action.payload };
    },

    // Update trading settings
    updateTradingSettings: (state, action: PayloadAction<Partial<TradingSettings>>) => {
      state.trading = { ...state.trading, ...action.payload };
    },

    // Update API settings
    updateAPISettings: (state, action: PayloadAction<Partial<APISettings>>) => {
      state.api = { ...state.api, ...action.payload };
    },

    // Update risk settings
    updateRiskSettings: (state, action: PayloadAction<Partial<RiskSettings>>) => {
      state.risk = { ...state.risk, ...action.payload };
    },

    // Update advanced settings
    updateAdvancedSettings: (state, action: PayloadAction<Partial<AdvancedSettings>>) => {
      state.advanced = { ...state.advanced, ...action.payload };
    },

    // Set theme
    setTheme: (state, action: PayloadAction<Theme>) => {
      state.display.theme = action.payload;
    },

    // Toggle compact mode
    toggleCompactMode: (state) => {
      state.display.compactMode = !state.display.compactMode;
    },

    // Set notification volume
    setNotificationVolume: (state, action: PayloadAction<number>) => {
      state.notifications.volume = Math.max(0, Math.min(100, action.payload));
    },

    // Clear error
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    // Load settings
    builder.addCase(loadSettings.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(loadSettings.fulfilled, (state, action) => {
      Object.assign(state, action.payload);
      state.loading = false;
    });
    builder.addCase(loadSettings.rejected, (state, action) => {
      state.loading = false;
      state.error = action.payload as string;
    });

    // Save settings
    builder.addCase(saveSettings.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(saveSettings.fulfilled, (state, action) => {
      state.loading = false;
      state.lastSaved = action.payload;
    });
    builder.addCase(saveSettings.rejected, (state, action) => {
      state.loading = false;
      state.error = action.payload as string;
    });

    // Reset settings
    builder.addCase(resetSettings.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(resetSettings.fulfilled, (state, action) => {
      Object.assign(state, action.payload);
      state.loading = false;
      state.lastSaved = new Date().toISOString();
    });
    builder.addCase(resetSettings.rejected, (state, action) => {
      state.loading = false;
      state.error = action.payload as string;
    });

    // Export settings
    builder.addCase(exportSettings.rejected, (state, action) => {
      state.error = action.payload as string;
    });

    // Import settings
    builder.addCase(importSettings.pending, (state) => {
      state.loading = true;
      state.error = null;
    });
    builder.addCase(importSettings.fulfilled, (state, action) => {
      Object.assign(state, action.payload);
      state.loading = false;
      state.lastSaved = new Date().toISOString();
    });
    builder.addCase(importSettings.rejected, (state, action) => {
      state.loading = false;
      state.error = action.payload as string;
    });
  },
});

/**
 * Export actions
 */
export const {
  updateDisplaySettings,
  updateNotificationSettings,
  updateTradingSettings,
  updateAPISettings,
  updateRiskSettings,
  updateAdvancedSettings,
  setTheme,
  toggleCompactMode,
  setNotificationVolume,
  clearError,
} = settingsSlice.actions;

/**
 * Selectors
 */
export const selectDisplaySettings = (state: RootState) => state.settings.display;
export const selectNotificationSettings = (state: RootState) => state.settings.notifications;
export const selectTradingSettings = (state: RootState) => state.settings.trading;
export const selectAPISettings = (state: RootState) => state.settings.api;
export const selectRiskSettings = (state: RootState) => state.settings.risk;
export const selectAdvancedSettings = (state: RootState) => state.settings.advanced;
export const selectTheme = (state: RootState) => state.settings.display.theme;
export const selectIsCompactMode = (state: RootState) => state.settings.display.compactMode;
export const selectSettingsLoading = (state: RootState) => state.settings.loading;
export const selectSettingsError = (state: RootState) => state.settings.error;
export const selectLastSaved = (state: RootState) => state.settings.lastSaved;

/**
 * Export reducer
 */
export default settingsSlice.reducer;
