/**
 * Card Component - Production-ready container component
 *
 * Quantum Trader AI - Institutional Trading Platform
 * Provides consistent container styling for dashboard widgets
 */

import React, { HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Card title */
  title?: string;
  /** Card subtitle/description */
  subtitle?: string;
  /** Header actions (buttons, icons, etc.) */
  headerAction?: React.ReactNode;
  /** Card footer content */
  footer?: React.ReactNode;
  /** Enable hover effect */
  hoverable?: boolean;
  /** Add padding to card body */
  padding?: 'none' | 'sm' | 'md' | 'lg';
  /** Card border variant */
  bordered?: boolean;
  /** Loading state */
  loading?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Child elements */
  children?: React.ReactNode;
}

const paddingStyles = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
};

/**
 * Card component for containing content sections
 *
 * @example
 * ```tsx
 * <Card title="Portfolio Overview" subtitle="Real-time positions">
 *   <PositionList />
 * </Card>
 *
 * <Card
 *   title="Active Orders"
 *   headerAction={<Button variant="ghost" size="sm">View All</Button>}
 *   padding="md"
 * >
 *   <OrderTable />
 * </Card>
 * ```
 */
export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  headerAction,
  footer,
  hoverable = false,
  padding = 'md',
  bordered = true,
  loading = false,
  className = '',
  children,
  ...props
}) => {
  const baseStyles = 'bg-gray-800 rounded-lg';
  const borderStyles = bordered ? 'border border-gray-700' : '';
  const hoverStyles = hoverable ? 'hover:shadow-lg hover:border-gray-600 transition-all duration-200' : '';
  const paddingStyle = paddingStyles[padding];

  const combinedClassName = `${baseStyles} ${borderStyles} ${hoverStyles} ${className}`.trim();

  return (
    <div className={combinedClassName} {...props}>
      {(title || subtitle || headerAction) && (
        <div className="px-6 py-4 border-b border-gray-700">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              {title && (
                <h3 className="text-lg font-semibold text-gray-100">
                  {title}
                </h3>
              )}
              {subtitle && (
                <p className="mt-1 text-sm text-gray-400">
                  {subtitle}
                </p>
              )}
            </div>
            {headerAction && (
              <div className="ml-4 flex-shrink-0">
                {headerAction}
              </div>
            )}
          </div>
        </div>
      )}

      <div className={paddingStyle}>
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="flex flex-col items-center space-y-4">
              <svg
                className="animate-spin h-8 w-8 text-blue-500"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <p className="text-sm text-gray-400">Loading...</p>
            </div>
          </div>
        ) : (
          children
        )}
      </div>

      {footer && (
        <div className="px-6 py-4 border-t border-gray-700 bg-gray-800/50">
          {footer}
        </div>
      )}
    </div>
  );
};

export default Card;
