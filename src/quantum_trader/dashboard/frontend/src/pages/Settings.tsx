/**
 * Settings Page
 *
 * Configuration and settings management for the trading platform.
 * Production-ready component for institutional trading dashboard.
 */

import React, { useState, useEffect, useCallback } from 'react';

interface RiskSettings {
  max_position_size: number;
  max_leverage: number;
  max_drawdown: number;
  max_daily_loss: number;
  stop_loss_enabled: boolean;
  stop_loss_percentage: number;
  take_profit_enabled: boolean;
  take_profit_percentage: number;
}

interface TradingSettings {
  auto_trading_enabled: boolean;
  max_open_positions: number;
  default_order_type: string;
  slippage_tolerance: number;
  min_order_size: number;
  max_order_size: number;
}

interface NotificationSettings {
  email_enabled: boolean;
  email_address: string;
  telegram_enabled: boolean;
  telegram_chat_id: string;
  webhook_enabled: boolean;
  webhook_url: string;
  alert_on_trade: boolean;
  alert_on_error: boolean;
  alert_on_risk_breach: boolean;
}

interface ExchangeConfig {
  exchange: string;
  enabled: boolean;
  api_key: string;
  api_secret: string;
  testnet: boolean;
}

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'risk' | 'trading' | 'notifications' | 'exchanges'>('risk');
  const [riskSettings, setRiskSettings] = useState<RiskSettings | null>(null);
  const [tradingSettings, setTradingSettings] = useState<TradingSettings | null>(null);
  const [notificationSettings, setNotificationSettings] = useState<NotificationSettings | null>(null);
  const [exchangeConfigs, setExchangeConfigs] = useState<ExchangeConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Fetch settings
  const fetchSettings = useCallback(async () => {
    try {
      setLoading(true);
      const [riskRes, tradingRes, notifRes, exchangesRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/settings/risk`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch(`${API_BASE_URL}/api/v1/settings/trading`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch(`${API_BASE_URL}/api/v1/settings/notifications`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        }),
        fetch(`${API_BASE_URL}/api/v1/settings/exchanges`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json'
          }
        })
      ]);

      if (!riskRes.ok || !tradingRes.ok || !notifRes.ok || !exchangesRes.ok) {
        throw new Error('Failed to fetch settings');
      }

      const [risk, trading, notif, exchanges] = await Promise.all([
        riskRes.json(),
        tradingRes.json(),
        notifRes.json(),
        exchangesRes.json()
      ]);

      setRiskSettings(risk);
      setTradingSettings(trading);
      setNotificationSettings(notif);
      setExchangeConfigs(exchanges);
      setError(null);
    } catch (err) {
      setError(err as Error);
      console.error('Error fetching settings:', err);
    } finally {
      setLoading(false);
    }
  }, [API_BASE_URL]);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // Save settings
  const saveSettings = async (settingsType: string, data: any) => {
    try {
      setSaving(true);
      setSuccessMessage(null);

      const response = await fetch(`${API_BASE_URL}/api/v1/settings/${settingsType}`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error('Failed to save settings');
      }

      setSuccessMessage('Settings saved successfully');
      setTimeout(() => setSuccessMessage(null), 3000);
      await fetchSettings();
    } catch (err) {
      setError(err as Error);
      console.error('Error saving settings:', err);
    } finally {
      setSaving(false);
    }
  };

  // Test connection
  const testExchangeConnection = async (exchange: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/exchanges/${exchange}/test`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        throw new Error('Connection test failed');
      }

      alert(`${exchange} connection successful`);
    } catch (err) {
      alert(`${exchange} connection failed: ${(err as Error).message}`);
    }
  };

  const tabs = [
    { id: 'risk', label: 'Risk Management', icon: '⚠️' },
    { id: 'trading', label: 'Trading', icon: '📈' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'exchanges', label: 'Exchanges', icon: '🔗' }
  ] as const;

  if (loading && !riskSettings) {
    return (
      <div className="min-h-screen bg-gray-900 p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
          <p className="text-gray-400 mt-4 text-lg">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
        <p className="text-gray-400">Configure platform settings and preferences</p>
      </div>

      {/* Success/Error Messages */}
      {successMessage && (
        <div className="mb-4 bg-green-900/20 border border-green-700 rounded-lg p-4 text-green-300">
          {successMessage}
        </div>
      )}
      {error && (
        <div className="mb-4 bg-red-900/20 border border-red-700 rounded-lg p-4 text-red-300">
          Error: {error.message}
        </div>
      )}

      {/* Tabs */}
      <div className="mb-6">
        <div className="flex gap-2 border-b border-gray-700">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 font-semibold transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Risk Settings Tab */}
      {activeTab === 'risk' && riskSettings && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-6">Risk Management Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-gray-300 mb-2">Max Position Size ($)</label>
              <input
                type="number"
                value={riskSettings.max_position_size}
                onChange={(e) => setRiskSettings({ ...riskSettings, max_position_size: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Leverage</label>
              <input
                type="number"
                value={riskSettings.max_leverage}
                onChange={(e) => setRiskSettings({ ...riskSettings, max_leverage: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Drawdown (%)</label>
              <input
                type="number"
                value={riskSettings.max_drawdown}
                onChange={(e) => setRiskSettings({ ...riskSettings, max_drawdown: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Daily Loss ($)</label>
              <input
                type="number"
                value={riskSettings.max_daily_loss}
                onChange={(e) => setRiskSettings({ ...riskSettings, max_daily_loss: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div className="md:col-span-2">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={riskSettings.stop_loss_enabled}
                  onChange={(e) => setRiskSettings({ ...riskSettings, stop_loss_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span>Enable Stop Loss</span>
              </label>
              {riskSettings.stop_loss_enabled && (
                <input
                  type="number"
                  value={riskSettings.stop_loss_percentage}
                  onChange={(e) => setRiskSettings({ ...riskSettings, stop_loss_percentage: parseFloat(e.target.value) })}
                  placeholder="Stop Loss %"
                  className="mt-2 w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>
            <div className="md:col-span-2">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={riskSettings.take_profit_enabled}
                  onChange={(e) => setRiskSettings({ ...riskSettings, take_profit_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span>Enable Take Profit</span>
              </label>
              {riskSettings.take_profit_enabled && (
                <input
                  type="number"
                  value={riskSettings.take_profit_percentage}
                  onChange={(e) => setRiskSettings({ ...riskSettings, take_profit_percentage: parseFloat(e.target.value) })}
                  placeholder="Take Profit %"
                  className="mt-2 w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>
          </div>
          <button
            onClick={() => saveSettings('risk', riskSettings)}
            disabled={saving}
            className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-md font-semibold transition-colors"
          >
            {saving ? 'Saving...' : 'Save Risk Settings'}
          </button>
        </div>
      )}

      {/* Trading Settings Tab */}
      {activeTab === 'trading' && tradingSettings && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-6">Trading Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="md:col-span-2">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={tradingSettings.auto_trading_enabled}
                  onChange={(e) => setTradingSettings({ ...tradingSettings, auto_trading_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span className="font-semibold">Enable Auto Trading</span>
              </label>
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Open Positions</label>
              <input
                type="number"
                value={tradingSettings.max_open_positions}
                onChange={(e) => setTradingSettings({ ...tradingSettings, max_open_positions: parseInt(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Default Order Type</label>
              <select
                value={tradingSettings.default_order_type}
                onChange={(e) => setTradingSettings({ ...tradingSettings, default_order_type: e.target.value })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="MARKET">Market</option>
                <option value="LIMIT">Limit</option>
                <option value="STOP_LOSS">Stop Loss</option>
                <option value="TAKE_PROFIT">Take Profit</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Slippage Tolerance (%)</label>
              <input
                type="number"
                step="0.1"
                value={tradingSettings.slippage_tolerance}
                onChange={(e) => setTradingSettings({ ...tradingSettings, slippage_tolerance: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Min Order Size ($)</label>
              <input
                type="number"
                value={tradingSettings.min_order_size}
                onChange={(e) => setTradingSettings({ ...tradingSettings, min_order_size: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-gray-300 mb-2">Max Order Size ($)</label>
              <input
                type="number"
                value={tradingSettings.max_order_size}
                onChange={(e) => setTradingSettings({ ...tradingSettings, max_order_size: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          <button
            onClick={() => saveSettings('trading', tradingSettings)}
            disabled={saving}
            className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-md font-semibold transition-colors"
          >
            {saving ? 'Saving...' : 'Save Trading Settings'}
          </button>
        </div>
      )}

      {/* Notification Settings Tab */}
      {activeTab === 'notifications' && notificationSettings && (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-bold text-white mb-6">Notification Settings</h2>
          <div className="space-y-6">
            {/* Email */}
            <div className="border-b border-gray-700 pb-6">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer mb-4">
                <input
                  type="checkbox"
                  checked={notificationSettings.email_enabled}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, email_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span className="font-semibold">Enable Email Notifications</span>
              </label>
              {notificationSettings.email_enabled && (
                <input
                  type="email"
                  value={notificationSettings.email_address}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, email_address: e.target.value })}
                  placeholder="email@example.com"
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>

            {/* Telegram */}
            <div className="border-b border-gray-700 pb-6">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer mb-4">
                <input
                  type="checkbox"
                  checked={notificationSettings.telegram_enabled}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, telegram_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span className="font-semibold">Enable Telegram Notifications</span>
              </label>
              {notificationSettings.telegram_enabled && (
                <input
                  type="text"
                  value={notificationSettings.telegram_chat_id}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, telegram_chat_id: e.target.value })}
                  placeholder="Telegram Chat ID"
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>

            {/* Webhook */}
            <div className="border-b border-gray-700 pb-6">
              <label className="flex items-center gap-2 text-gray-300 cursor-pointer mb-4">
                <input
                  type="checkbox"
                  checked={notificationSettings.webhook_enabled}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, webhook_enabled: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600 text-blue-500"
                />
                <span className="font-semibold">Enable Webhook Notifications</span>
              </label>
              {notificationSettings.webhook_enabled && (
                <input
                  type="url"
                  value={notificationSettings.webhook_url}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, webhook_url: e.target.value })}
                  placeholder="https://example.com/webhook"
                  className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
            </div>

            {/* Alert Triggers */}
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Alert Triggers</h3>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notificationSettings.alert_on_trade}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, alert_on_trade: e.target.checked })}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                  <span>Alert on Trade Execution</span>
                </label>
                <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notificationSettings.alert_on_error}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, alert_on_error: e.target.checked })}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                  <span>Alert on Errors</span>
                </label>
                <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notificationSettings.alert_on_risk_breach}
                    onChange={(e) => setNotificationSettings({ ...notificationSettings, alert_on_risk_breach: e.target.checked })}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                  <span>Alert on Risk Limit Breach</span>
                </label>
              </div>
            </div>
          </div>
          <button
            onClick={() => saveSettings('notifications', notificationSettings)}
            disabled={saving}
            className="mt-6 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-md font-semibold transition-colors"
          >
            {saving ? 'Saving...' : 'Save Notification Settings'}
          </button>
        </div>
      )}

      {/* Exchange Settings Tab */}
      {activeTab === 'exchanges' && (
        <div className="space-y-4">
          {exchangeConfigs.map((config, index) => (
            <div key={index} className="bg-gray-800/50 border border-gray-700 rounded-lg p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-bold text-white">{config.exchange}</h3>
                <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.enabled}
                    onChange={(e) => {
                      const newConfigs = [...exchangeConfigs];
                      newConfigs[index].enabled = e.target.checked;
                      setExchangeConfigs(newConfigs);
                    }}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                  <span>Enabled</span>
                </label>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-gray-300 mb-2">API Key</label>
                  <input
                    type="password"
                    value={config.api_key}
                    onChange={(e) => {
                      const newConfigs = [...exchangeConfigs];
                      newConfigs[index].api_key = e.target.value;
                      setExchangeConfigs(newConfigs);
                    }}
                    placeholder="••••••••"
                    className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 mb-2">API Secret</label>
                  <input
                    type="password"
                    value={config.api_secret}
                    onChange={(e) => {
                      const newConfigs = [...exchangeConfigs];
                      newConfigs[index].api_secret = e.target.value;
                      setExchangeConfigs(newConfigs);
                    }}
                    placeholder="••••••••"
                    className="w-full bg-gray-700 border border-gray-600 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.testnet}
                    onChange={(e) => {
                      const newConfigs = [...exchangeConfigs];
                      newConfigs[index].testnet = e.target.checked;
                      setExchangeConfigs(newConfigs);
                    }}
                    className="rounded bg-gray-700 border-gray-600 text-blue-500"
                  />
                  <span>Use Testnet</span>
                </label>
                <button
                  onClick={() => testExchangeConnection(config.exchange)}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md transition-colors"
                >
                  Test Connection
                </button>
              </div>
            </div>
          ))}
          <button
            onClick={() => saveSettings('exchanges', exchangeConfigs)}
            disabled={saving}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-md font-semibold transition-colors"
          >
            {saving ? 'Saving...' : 'Save Exchange Settings'}
          </button>
        </div>
      )}
    </div>
  );
};

export default Settings;
