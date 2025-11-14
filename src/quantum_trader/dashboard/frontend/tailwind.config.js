/**
 * Tailwind CSS Configuration
 *
 * Configuration for the Quantum Trader AI dashboard UI.
 * Defines custom theme, colors, typography, and component styles.
 *
 * @type {import('tailwindcss').Config}
 */

const colors = require('tailwindcss/colors');
const defaultTheme = require('tailwindcss/defaultTheme');

module.exports = {
  content: [
    './src/**/*.{js,jsx,ts,tsx}',
    './public/index.html',
  ],
  darkMode: 'class', // Enable dark mode with class strategy
  theme: {
    extend: {
      colors: {
        // Primary brand colors
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        // Success colors (for profits, positive changes)
        success: {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        // Danger colors (for losses, negative changes)
        danger: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
        },
        // Warning colors
        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        // Info colors
        info: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Dark theme colors
        dark: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        // Chart colors
        chart: {
          bullish: '#22c55e',
          bearish: '#ef4444',
          neutral: '#64748b',
          volume: '#3b82f6',
        },
      },
      fontFamily: {
        sans: ['Inter var', ...defaultTheme.fontFamily.sans],
        mono: ['JetBrains Mono', ...defaultTheme.fontFamily.mono],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
        xs: ['0.75rem', { lineHeight: '1rem' }],
        sm: ['0.875rem', { lineHeight: '1.25rem' }],
        base: ['1rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.75rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
        '5xl': ['3rem', { lineHeight: '1' }],
        '6xl': ['3.75rem', { lineHeight: '1' }],
      },
      spacing: {
        '128': '32rem',
        '144': '36rem',
      },
      borderRadius: {
        '4xl': '2rem',
      },
      boxShadow: {
        'inner-lg': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.1)',
        'glow-sm': '0 0 4px rgba(59, 130, 246, 0.5)',
        'glow': '0 0 8px rgba(59, 130, 246, 0.5)',
        'glow-lg': '0 0 16px rgba(59, 130, 246, 0.5)',
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-slow': 'bounce 3s infinite',
        'fade-in': 'fadeIn 0.3s ease-in',
        'fade-out': 'fadeOut 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeOut: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      gridTemplateColumns: {
        '13': 'repeat(13, minmax(0, 1fr))',
        '14': 'repeat(14, minmax(0, 1fr))',
        '15': 'repeat(15, minmax(0, 1fr))',
        '16': 'repeat(16, minmax(0, 1fr))',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
      transitionProperty: {
        'height': 'height',
        'spacing': 'margin, padding',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({
      strategy: 'class',
    }),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
    // Custom plugin for trading-specific utilities
    function({ addUtilities, addComponents, theme }) {
      // Trading status colors
      const tradingUtilities = {
        '.text-profit': {
          color: theme('colors.success.500'),
        },
        '.text-loss': {
          color: theme('colors.danger.500'),
        },
        '.bg-profit': {
          backgroundColor: theme('colors.success.500'),
        },
        '.bg-loss': {
          backgroundColor: theme('colors.danger.500'),
        },
        '.border-profit': {
          borderColor: theme('colors.success.500'),
        },
        '.border-loss': {
          borderColor: theme('colors.danger.500'),
        },
        // Glow effects
        '.glow-profit': {
          boxShadow: `0 0 8px ${theme('colors.success.500')}`,
        },
        '.glow-loss': {
          boxShadow: `0 0 8px ${theme('colors.danger.500')}`,
        },
      };

      // Card components
      const cardComponents = {
        '.card': {
          backgroundColor: theme('colors.white'),
          borderRadius: theme('borderRadius.lg'),
          padding: theme('spacing.6'),
          boxShadow: theme('boxShadow.md'),
        },
        '.card-dark': {
          backgroundColor: theme('colors.dark.800'),
          borderRadius: theme('borderRadius.lg'),
          padding: theme('spacing.6'),
          boxShadow: theme('boxShadow.md'),
          borderWidth: '1px',
          borderColor: theme('colors.dark.700'),
        },
        '.card-compact': {
          backgroundColor: theme('colors.white'),
          borderRadius: theme('borderRadius.md'),
          padding: theme('spacing.4'),
          boxShadow: theme('boxShadow.sm'),
        },
        '.card-compact-dark': {
          backgroundColor: theme('colors.dark.800'),
          borderRadius: theme('borderRadius.md'),
          padding: theme('spacing.4'),
          boxShadow: theme('boxShadow.sm'),
          borderWidth: '1px',
          borderColor: theme('colors.dark.700'),
        },
      };

      // Metric display components
      const metricComponents = {
        '.metric': {
          display: 'flex',
          flexDirection: 'column',
          gap: theme('spacing.1'),
        },
        '.metric-label': {
          fontSize: theme('fontSize.xs'),
          color: theme('colors.gray.500'),
          textTransform: 'uppercase',
          letterSpacing: theme('letterSpacing.wide'),
        },
        '.metric-value': {
          fontSize: theme('fontSize.2xl'),
          fontWeight: theme('fontWeight.bold'),
          color: theme('colors.gray.900'),
        },
        '.metric-value-dark': {
          fontSize: theme('fontSize.2xl'),
          fontWeight: theme('fontWeight.bold'),
          color: theme('colors.gray.100'),
        },
      };

      addUtilities(tradingUtilities);
      addComponents(cardComponents);
      addComponents(metricComponents);
    },
  ],
  // Safelist classes that might be dynamically generated
  safelist: [
    'text-profit',
    'text-loss',
    'bg-profit',
    'bg-loss',
    'border-profit',
    'border-loss',
    'glow-profit',
    'glow-loss',
    {
      pattern: /^(bg|text|border)-(success|danger|warning|info)-(50|100|200|300|400|500|600|700|800|900)$/,
    },
  ],
  // Variants
  variants: {
    extend: {
      backgroundColor: ['active', 'disabled'],
      textColor: ['active', 'disabled'],
      borderColor: ['active', 'disabled'],
      opacity: ['disabled'],
      cursor: ['disabled'],
    },
  },
};
