import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CpuChipIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';

interface LoginFormData {
  username: string;
  password: string;
}

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

interface ErrorResponse {
  detail: string;
  code?: string;
}

const Login: React.FC = () => {
  const [formData, setFormData] = useState<LoginFormData>({
    username: '',
    password: '',
  });
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [rememberMe, setRememberMe] = useState<boolean>(false);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      validateAndRedirect(token);
    }

    const savedUsername = localStorage.getItem('remembered_username');
    if (savedUsername) {
      setFormData((prev) => ({ ...prev, username: savedUsername }));
      setRememberMe(true);
    }
  }, []);

  const validateAndRedirect = async (token: string): Promise<void> => {
    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/api/v1/user/validate`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        navigate('/dashboard');
      } else {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
      }
    } catch (err) {
      console.error('Token validation error:', err);
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    setError('');
  };

  const validateForm = (): boolean => {
    if (!formData.username.trim()) {
      setError('Username is required');
      return false;
    }

    if (!formData.password) {
      setError('Password is required');
      return false;
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setError('');

    try {
      const apiUrl = process.env.REACT_APP_API_URL || window.REACT_APP_API_URL;
      const maxRetries = parseInt(process.env.REACT_APP_MAX_LOGIN_RETRIES || '3', 10);

      let lastError: Error | null = null;

      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          const response = await fetch(`${apiUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              username: formData.username,
              password: formData.password,
            }),
          });

          if (!response.ok) {
            const errorData: ErrorResponse = await response.json();

            if (response.status === 401) {
              setError('Invalid username or password');
              setLoading(false);
              return;
            } else if (response.status === 429) {
              setError('Too many login attempts. Please try again later.');
              setLoading(false);
              return;
            } else if (response.status >= 500) {
              throw new Error(errorData.detail || 'Server error. Please try again.');
            } else {
              setError(errorData.detail || 'Login failed. Please try again.');
              setLoading(false);
              return;
            }
          }

          const data: LoginResponse = await response.json();

          localStorage.setItem('auth_token', data.access_token);
          localStorage.setItem('refresh_token', data.refresh_token);

          if (rememberMe) {
            localStorage.setItem('remembered_username', formData.username);
          } else {
            localStorage.removeItem('remembered_username');
          }

          navigate('/dashboard');
          return;
        } catch (err) {
          lastError = err as Error;

          if (attempt < maxRetries - 1) {
            const delay = Math.pow(2, attempt) * 1000;
            await new Promise((resolve) => setTimeout(resolve, delay));
          }
        }
      }

      throw lastError;
    } catch (err) {
      console.error('Login error:', err);
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Logo and Title */}
        <div className="text-center">
          <div className="flex justify-center">
            <CpuChipIcon className="h-16 w-16 text-blue-500" />
          </div>
          <h2 className="mt-6 text-4xl font-extrabold text-white">
            Quantum Trader AI
          </h2>
          <p className="mt-2 text-sm text-gray-400">
            Institutional Trading Platform
          </p>
        </div>

        {/* Login Form */}
        <div className="bg-gray-800 rounded-lg shadow-2xl p-8">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Error Message */}
            {error && (
              <div className="bg-red-500 bg-opacity-10 border border-red-500 rounded-md p-4 flex items-start">
                <ExclamationCircleIcon className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                <span className="text-sm text-red-400">{error}</span>
              </div>
            )}

            {/* Username */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                required
                value={formData.username}
                onChange={handleChange}
                className="mt-1 block w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your username"
                disabled={loading}
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                value={formData.password}
                onChange={handleChange}
                className="mt-1 block w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
                disabled={loading}
              />
            </div>

            {/* Remember Me */}
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="h-4 w-4 bg-gray-700 border-gray-600 rounded text-blue-500 focus:ring-blue-500"
                disabled={loading}
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-300">
                Remember me
              </label>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className={`w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors ${
                loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? (
                <span className="flex items-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
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
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Signing in...
                </span>
              ) : (
                'Sign in'
              )}
            </button>
          </form>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-400">
          <p>&copy; 2025 Quantum Trader AI. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
