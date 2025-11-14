/**
 * Alert Component - Reusable alert/notification component for the dashboard.
 *
 * Displays alerts, warnings, errors, and success messages with
 * configurable severity levels and auto-dismiss functionality.
 */

import React, { useEffect, useState } from 'react';

/**
 * Alert severity levels matching system AlertSeverity enum
 */
export type AlertSeverity = 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

/**
 * Alert component props
 */
export interface AlertProps {
  /** Alert message content */
  message: string;
  /** Alert severity level */
  severity: AlertSeverity;
  /** Optional title */
  title?: string;
  /** Whether alert is dismissible */
  dismissible?: boolean;
  /** Auto-dismiss after milliseconds (0 = no auto-dismiss) */
  autoDismiss?: number;
  /** Callback when alert is dismissed */
  onDismiss?: () => void;
  /** Additional CSS classes */
  className?: string;
  /** Optional icon override */
  icon?: React.ReactNode;
}

/**
 * Get alert styling based on severity
 */
const getAlertStyles = (severity: AlertSeverity): { bg: string; border: string; text: string; icon: string } => {
  const styles = {
    INFO: {
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      border: 'border-blue-200 dark:border-blue-800',
      text: 'text-blue-800 dark:text-blue-200',
      icon: 'text-blue-600 dark:text-blue-400'
    },
    WARNING: {
      bg: 'bg-yellow-50 dark:bg-yellow-900/20',
      border: 'border-yellow-200 dark:border-yellow-800',
      text: 'text-yellow-800 dark:text-yellow-200',
      icon: 'text-yellow-600 dark:text-yellow-400'
    },
    ERROR: {
      bg: 'bg-red-50 dark:bg-red-900/20',
      border: 'border-red-200 dark:border-red-800',
      text: 'text-red-800 dark:text-red-200',
      icon: 'text-red-600 dark:text-red-400'
    },
    CRITICAL: {
      bg: 'bg-red-100 dark:bg-red-900/40',
      border: 'border-red-400 dark:border-red-700',
      text: 'text-red-900 dark:text-red-100',
      icon: 'text-red-700 dark:text-red-300'
    }
  };

  return styles[severity];
};

/**
 * Get default icon for severity level
 */
const getDefaultIcon = (severity: AlertSeverity): string => {
  const icons = {
    INFO: '&#9432;',  // ⓘ
    WARNING: '&#9888;',  // ⚠
    ERROR: '&#10006;',  // ✖
    CRITICAL: '&#128680;'  // 🚨
  };

  return icons[severity];
};

/**
 * Alert Component
 *
 * @example
 * ```tsx
 * <Alert
 *   message="Order executed successfully"
 *   severity="INFO"
 *   dismissible={true}
 *   autoDismiss={5000}
 * />
 * ```
 */
export const Alert: React.FC<AlertProps> = ({
  message,
  severity,
  title,
  dismissible = true,
  autoDismiss = 0,
  onDismiss,
  className = '',
  icon
}) => {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (autoDismiss > 0) {
      const timer = setTimeout(() => {
        handleDismiss();
      }, autoDismiss);

      return () => clearTimeout(timer);
    }
  }, [autoDismiss]);

  const handleDismiss = () => {
    setVisible(false);
    if (onDismiss) {
      onDismiss();
    }
  };

  if (!visible) {
    return null;
  }

  const styles = getAlertStyles(severity);
  const defaultIcon = getDefaultIcon(severity);

  return (
    <div
      className={`
        flex items-start p-4 mb-4 rounded-lg border
        ${styles.bg} ${styles.border} ${styles.text}
        ${className}
      `}
      role="alert"
      aria-live="polite"
    >
      {/* Icon */}
      <div className={`flex-shrink-0 ${styles.icon}`}>
        {icon || (
          <span
            className="text-xl"
            dangerouslySetInnerHTML={{ __html: defaultIcon }}
          />
        )}
      </div>

      {/* Content */}
      <div className="ml-3 flex-1">
        {title && (
          <h3 className="text-sm font-medium mb-1">
            {title}
          </h3>
        )}
        <div className="text-sm">
          {message}
        </div>
      </div>

      {/* Dismiss button */}
      {dismissible && (
        <button
          type="button"
          className={`
            ml-auto -mx-1.5 -my-1.5 rounded-lg p-1.5
            inline-flex h-8 w-8 items-center justify-center
            hover:bg-black/10 dark:hover:bg-white/10
            focus:outline-none focus:ring-2 focus:ring-offset-2
            ${styles.text}
          `}
          onClick={handleDismiss}
          aria-label="Dismiss"
        >
          <span className="sr-only">Dismiss</span>
          <svg
            className="w-5 h-5"
            fill="currentColor"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      )}
    </div>
  );
};

/**
 * Alert container for managing multiple alerts
 */
export interface AlertContainerProps {
  /** Array of alerts to display */
  alerts: Array<AlertProps & { id: string }>;
  /** Position of alert container */
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left' | 'top-center';
  /** Maximum number of alerts to show */
  maxAlerts?: number;
}

/**
 * Alert Container Component
 *
 * Manages multiple alerts with positioning
 */
export const AlertContainer: React.FC<AlertContainerProps> = ({
  alerts,
  position = 'top-right',
  maxAlerts = 5
}) => {
  const positionClasses = {
    'top-right': 'top-4 right-4',
    'top-left': 'top-4 left-4',
    'bottom-right': 'bottom-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'top-center': 'top-4 left-1/2 -translate-x-1/2'
  };

  // Limit number of displayed alerts
  const displayedAlerts = alerts.slice(0, maxAlerts);

  return (
    <div
      className={`
        fixed z-50 w-full max-w-md space-y-2
        ${positionClasses[position]}
      `}
    >
      {displayedAlerts.map((alert) => (
        <Alert key={alert.id} {...alert} />
      ))}
    </div>
  );
};

/**
 * Hook for managing alerts
 */
export const useAlerts = () => {
  const [alerts, setAlerts] = useState<Array<AlertProps & { id: string }>>([]);

  const addAlert = (alert: Omit<AlertProps, 'onDismiss'>) => {
    const id = `alert-${Date.now()}-${Math.random()}`;
    const newAlert = {
      ...alert,
      id,
      onDismiss: () => removeAlert(id)
    };

    setAlerts((prev) => [...prev, newAlert]);

    return id;
  };

  const removeAlert = (id: string) => {
    setAlerts((prev) => prev.filter((alert) => alert.id !== id));
  };

  const clearAlerts = () => {
    setAlerts([]);
  };

  return {
    alerts,
    addAlert,
    removeAlert,
    clearAlerts
  };
};

export default Alert;
