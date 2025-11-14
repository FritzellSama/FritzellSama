/**
 * Notification Hook
 *
 * Custom React hook for managing notifications and alerts.
 * Supports toast notifications, desktop notifications, and sound alerts.
 *
 * @module useNotification
 */

import { useState, useCallback, createContext, useContext, ReactNode, useEffect } from 'react';
import { useSelector } from 'react-redux';
import type { RootState } from '../store';

/**
 * Notification type
 */
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

/**
 * Notification interface
 */
export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  title?: string;
  duration?: number;
  timestamp: number;
  dismissible?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

/**
 * Notification context value
 */
export interface NotificationContextValue {
  notifications: Notification[];
  showNotification: (
    type: NotificationType,
    message: string,
    options?: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'timestamp'>>
  ) => string;
  showSuccess: (message: string, title?: string) => string;
  showError: (message: string, title?: string) => string;
  showWarning: (message: string, title?: string) => string;
  showInfo: (message: string, title?: string) => string;
  dismiss: (id: string) => void;
  dismissAll: () => void;
  playSound: (type: NotificationType) => void;
  requestDesktopPermission: () => Promise<boolean>;
}

/**
 * Notification context
 */
const NotificationContext = createContext<NotificationContextValue | null>(null);

/**
 * Generate unique notification ID
 */
const generateId = (): string => {
  return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Get default duration based on type
 */
const getDefaultDuration = (type: NotificationType): number => {
  const durations = {
    success: parseInt(process.env.REACT_APP_NOTIFICATION_SUCCESS_DURATION || '3000', 10),
    error: parseInt(process.env.REACT_APP_NOTIFICATION_ERROR_DURATION || '5000', 10),
    warning: parseInt(process.env.REACT_APP_NOTIFICATION_WARNING_DURATION || '4000', 10),
    info: parseInt(process.env.REACT_APP_NOTIFICATION_INFO_DURATION || '3000', 10),
  };

  return durations[type];
};

/**
 * Notification Provider Component
 */
export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<Notification[]>();

  // Get notification settings from Redux store
  const notificationSettings = useSelector(
    (state: RootState) => state.settings?.notifications
  );

  /**
   * Show notification
   */
  const showNotification = useCallback(
    (
      type: NotificationType,
      message: string,
      options: Partial<Omit<Notification, 'id' | 'type' | 'message' | 'timestamp'>> = {}
    ): string => {
      const id = generateId();
      const duration = options.duration ?? getDefaultDuration(type);

      const notification: Notification = {
        id,
        type,
        message,
        title: options.title,
        duration,
        timestamp: Date.now(),
        dismissible: options.dismissible ?? true,
        action: options.action,
      };

      setNotifications((prev) => [...prev, notification]);

      // Auto-dismiss if duration is set
      if (duration > 0) {
        setTimeout(() => {
          dismiss(id);
        }, duration);
      }

      // Play sound if enabled
      if (notificationSettings?.sound) {
        playSound(type);
      }

      // Show desktop notification if enabled and permitted
      if (notificationSettings?.desktop && Notification.permission === 'granted') {
        showDesktopNotification(type, message, options.title);
      }

      return id;
    },
    [notificationSettings]
  );

  /**
   * Show success notification
   */
  const showSuccess = useCallback(
    (message: string, title?: string): string => {
      return showNotification('success', message, { title });
    },
    [showNotification]
  );

  /**
   * Show error notification
   */
  const showError = useCallback(
    (message: string, title?: string): string => {
      return showNotification('error', message, { title });
    },
    [showNotification]
  );

  /**
   * Show warning notification
   */
  const showWarning = useCallback(
    (message: string, title?: string): string => {
      return showNotification('warning', message, { title });
    },
    [showNotification]
  );

  /**
   * Show info notification
   */
  const showInfo = useCallback(
    (message: string, title?: string): string => {
      return showNotification('info', message, { title });
    },
    [showNotification]
  );

  /**
   * Dismiss notification
   */
  const dismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  /**
   * Dismiss all notifications
   */
  const dismissAll = useCallback(() => {
    setNotifications([]);
  }, []);

  /**
   * Play notification sound
   */
  const playSound = useCallback(
    (type: NotificationType) => {
      if (!notificationSettings?.sound) return;

      try {
        const volume = (notificationSettings.volume || 50) / 100;

        // Sound file paths from environment or defaults
        const soundPaths = {
          success: process.env.REACT_APP_SOUND_SUCCESS || '/sounds/success.mp3',
          error: process.env.REACT_APP_SOUND_ERROR || '/sounds/error.mp3',
          warning: process.env.REACT_APP_SOUND_WARNING || '/sounds/warning.mp3',
          info: process.env.REACT_APP_SOUND_INFO || '/sounds/info.mp3',
        };

        const audio = new Audio(soundPaths[type]);
        audio.volume = volume;
        audio.play().catch((err) => {
          console.error('Failed to play notification sound:', err);
        });
      } catch (err) {
        console.error('Error playing notification sound:', err);
      }
    },
    [notificationSettings]
  );

  /**
   * Show desktop notification
   */
  const showDesktopNotification = useCallback(
    (type: NotificationType, message: string, title?: string) => {
      if (!('Notification' in window)) {
        console.warn('Desktop notifications not supported');
        return;
      }

      if (Notification.permission !== 'granted') {
        return;
      }

      try {
        const icons = {
          success: '/icons/success.png',
          error: '/icons/error.png',
          warning: '/icons/warning.png',
          info: '/icons/info.png',
        };

        const notificationTitle = title || 'Quantum Trader AI';
        const options: NotificationOptions = {
          body: message,
          icon: icons[type],
          badge: '/icons/badge.png',
          tag: type,
          requireInteraction: type === 'error',
        };

        const notification = new Notification(notificationTitle, options);

        // Auto-close after duration
        setTimeout(() => {
          notification.close();
        }, getDefaultDuration(type));
      } catch (err) {
        console.error('Failed to show desktop notification:', err);
      }
    },
    []
  );

  /**
   * Request desktop notification permission
   */
  const requestDesktopPermission = useCallback(async (): Promise<boolean> => {
    if (!('Notification' in window)) {
      console.warn('Desktop notifications not supported');
      return false;
    }

    if (Notification.permission === 'granted') {
      return true;
    }

    if (Notification.permission === 'denied') {
      return false;
    }

    try {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    } catch (err) {
      console.error('Failed to request notification permission:', err);
      return false;
    }
  }, []);

  const value: NotificationContextValue = {
    notifications,
    showNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    dismiss,
    dismissAll,
    playSound,
    requestDesktopPermission,
  };

  return <NotificationContext.Provider value={value}>{children}</NotificationContext.Provider>;
}

/**
 * Use Notification Hook
 *
 * Provides access to notification system.
 *
 * @example
 * ```typescript
 * const { showSuccess, showError } = useNotification();
 *
 * // Show success notification
 * showSuccess('Order executed successfully');
 *
 * // Show error notification
 * showError('Failed to place order', 'Order Error');
 *
 * // Show custom notification
 * showNotification('warning', 'High volatility detected', {
 *   title: 'Market Alert',
 *   duration: 10000,
 *   action: {
 *     label: 'View',
 *     onClick: () => navigate('/risk'),
 *   },
 * });
 * ```
 */
export function useNotification(): NotificationContextValue {
  const context = useContext(NotificationContext);

  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }

  return context;
}

/**
 * Hook for trading-specific notifications
 */
export function useTradingNotifications() {
  const { showNotification, showSuccess, showError, showWarning } = useNotification();

  const notifyOrderPlaced = useCallback(
    (symbol: string, side: string, quantity: string) => {
      showSuccess(`${side} order placed: ${quantity} ${symbol}`, 'Order Placed');
    },
    [showSuccess]
  );

  const notifyOrderFilled = useCallback(
    (symbol: string, side: string, quantity: string, price: string) => {
      showSuccess(
        `${side} order filled: ${quantity} ${symbol} @ ${price}`,
        'Order Filled'
      );
    },
    [showSuccess]
  );

  const notifyOrderCancelled = useCallback(
    (symbol: string) => {
      showWarning(`Order cancelled: ${symbol}`, 'Order Cancelled');
    },
    [showWarning]
  );

  const notifyOrderFailed = useCallback(
    (symbol: string, error: string) => {
      showError(`Failed to place order for ${symbol}: ${error}`, 'Order Failed');
    },
    [showError]
  );

  const notifyPositionOpened = useCallback(
    (symbol: string, side: string, quantity: string) => {
      showSuccess(`Position opened: ${side} ${quantity} ${symbol}`, 'Position Opened');
    },
    [showSuccess]
  );

  const notifyPositionClosed = useCallback(
    (symbol: string, pnl: string) => {
      const isProfit = parseFloat(pnl) >= 0;
      showNotification(
        isProfit ? 'success' : 'warning',
        `Position closed: ${symbol} | P&L: ${pnl}`,
        { title: 'Position Closed' }
      );
    },
    [showNotification]
  );

  const notifyRiskAlert = useCallback(
    (message: string) => {
      showNotification('error', message, {
        title: 'Risk Alert',
        duration: 0, // Don't auto-dismiss
        dismissible: true,
      });
    },
    [showNotification]
  );

  const notifyPriceAlert = useCallback(
    (symbol: string, price: string, condition: string) => {
      showNotification('info', `${symbol} ${condition} ${price}`, {
        title: 'Price Alert',
      });
    },
    [showNotification]
  );

  return {
    notifyOrderPlaced,
    notifyOrderFilled,
    notifyOrderCancelled,
    notifyOrderFailed,
    notifyPositionOpened,
    notifyPositionClosed,
    notifyRiskAlert,
    notifyPriceAlert,
  };
}

/**
 * Hook for system notifications
 */
export function useSystemNotifications() {
  const { showInfo, showWarning, showError } = useNotification();

  const notifyConnectionStatus = useCallback(
    (exchange: string, connected: boolean) => {
      if (connected) {
        showInfo(`Connected to ${exchange}`, 'Exchange Connected');
      } else {
        showWarning(`Disconnected from ${exchange}`, 'Exchange Disconnected');
      }
    },
    [showInfo, showWarning]
  );

  const notifySystemError = useCallback(
    (message: string) => {
      showError(message, 'System Error');
    },
    [showError]
  );

  const notifyMaintenance = useCallback(
    (message: string) => {
      showWarning(message, 'Maintenance Notice');
    },
    [showWarning]
  );

  return {
    notifyConnectionStatus,
    notifySystemError,
    notifyMaintenance,
  };
}

export default useNotification;
