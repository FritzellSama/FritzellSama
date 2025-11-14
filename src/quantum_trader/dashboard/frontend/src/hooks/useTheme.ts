/**
 * Theme management hook for Quantum Trader AI Dashboard
 * Handles dark/light mode switching with persistence
 */

import { useState, useEffect, useCallback } from 'react';

export type Theme = 'light' | 'dark' | 'auto';

interface ThemeConfig {
  theme: Theme;
  effectiveTheme: 'light' | 'dark';
}

interface UseThemeReturn {
  theme: Theme;
  effectiveTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const STORAGE_KEY = process.env.VITE_THEME_STORAGE_KEY || 'quantum-trader-theme';
const DEFAULT_THEME: Theme = (process.env.VITE_DEFAULT_THEME as Theme) || 'dark';

/**
 * Get system theme preference
 */
const getSystemTheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

/**
 * Load theme from storage
 */
const loadStoredTheme = (): Theme => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && ['light', 'dark', 'auto'].includes(stored)) {
      return stored as Theme;
    }
  } catch (error) {
    console.error('[useTheme] Failed to load theme from storage:', error);
  }
  return DEFAULT_THEME;
};

/**
 * Save theme to storage
 */
const saveTheme = (theme: Theme): void => {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch (error) {
    console.error('[useTheme] Failed to save theme to storage:', error);
  }
};

/**
 * Calculate effective theme based on user preference and system settings
 */
const calculateEffectiveTheme = (theme: Theme): 'light' | 'dark' => {
  if (theme === 'auto') {
    return getSystemTheme();
  }
  return theme;
};

/**
 * Apply theme to document
 */
const applyTheme = (effectiveTheme: 'light' | 'dark'): void => {
  if (typeof document === 'undefined') return;

  const root = document.documentElement;
  root.classList.remove('light', 'dark');
  root.classList.add(effectiveTheme);
  root.setAttribute('data-theme', effectiveTheme);
};

/**
 * Custom hook for theme management
 *
 * Features:
 * - Persistent theme storage
 * - System theme detection
 * - Auto theme switching based on system preferences
 * - Type-safe theme values
 *
 * @example
 * ```tsx
 * function App() {
 *   const { theme, effectiveTheme, setTheme, toggleTheme } = useTheme();
 *
 *   return (
 *     <div>
 *       <p>Current theme: {theme}</p>
 *       <p>Effective theme: {effectiveTheme}</p>
 *       <button onClick={toggleTheme}>Toggle Theme</button>
 *       <button onClick={() => setTheme('auto')}>Auto Theme</button>
 *     </div>
 *   );
 * }
 * ```
 */
export const useTheme = (): UseThemeReturn => {
  const [theme, setThemeState] = useState<Theme>(loadStoredTheme);
  const [effectiveTheme, setEffectiveTheme] = useState<'light' | 'dark'>(() =>
    calculateEffectiveTheme(loadStoredTheme())
  );

  /**
   * Update theme and persist to storage
   */
  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
    saveTheme(newTheme);
    const effective = calculateEffectiveTheme(newTheme);
    setEffectiveTheme(effective);
    applyTheme(effective);
  }, []);

  /**
   * Toggle between light and dark themes
   */
  const toggleTheme = useCallback(() => {
    const newTheme = effectiveTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
  }, [effectiveTheme, setTheme]);

  /**
   * Initialize theme on mount
   */
  useEffect(() => {
    const effective = calculateEffectiveTheme(theme);
    setEffectiveTheme(effective);
    applyTheme(effective);
  }, [theme]);

  /**
   * Listen for system theme changes when in auto mode
   */
  useEffect(() => {
    if (theme !== 'auto') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    const handleChange = (e: MediaQueryListEvent) => {
      const newEffectiveTheme = e.matches ? 'dark' : 'light';
      setEffectiveTheme(newEffectiveTheme);
      applyTheme(newEffectiveTheme);
    };

    try {
      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    } catch (error) {
      console.error('[useTheme] Failed to add media query listener:', error);
    }
  }, [theme]);

  return {
    theme,
    effectiveTheme,
    setTheme,
    toggleTheme,
  };
};

export default useTheme;
