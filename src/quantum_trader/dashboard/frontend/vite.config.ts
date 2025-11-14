/**
 * Vite configuration for Quantum Trader AI Dashboard
 * Production-optimized build configuration
 */

import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '');

  const isDevelopment = mode === 'development';
  const isProduction = mode === 'production';

  return {
    plugins: [
      react({
        // Enable fast refresh in development
        fastRefresh: isDevelopment,
        // Babel configuration
        babel: {
          plugins: isDevelopment ? [] : [],
        },
      }),
    ],

    // Path resolution
    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '@components': resolve(__dirname, './src/components'),
        '@hooks': resolve(__dirname, './src/hooks'),
        '@services': resolve(__dirname, './src/services'),
        '@utils': resolve(__dirname, './src/utils'),
        '@types': resolve(__dirname, './src/types'),
        '@assets': resolve(__dirname, './src/assets'),
        '@styles': resolve(__dirname, './src/styles'),
      },
    },

    // Development server configuration
    server: {
      host: env.VITE_DEV_HOST || 'localhost',
      port: parseInt(env.VITE_DEV_PORT || '3000', 10),
      strictPort: true,
      open: env.VITE_DEV_OPEN === 'true',
      cors: true,
      // Proxy API requests to backend
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          ws: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        '/ws': {
          target: env.VITE_WS_URL || 'ws://localhost:8000',
          changeOrigin: true,
          secure: false,
          ws: true,
        },
      },
      // HMR configuration
      hmr: {
        protocol: 'ws',
        host: env.VITE_HMR_HOST || 'localhost',
        port: parseInt(env.VITE_HMR_PORT || '3000', 10),
        clientPort: parseInt(env.VITE_HMR_CLIENT_PORT || '3000', 10),
      },
    },

    // Preview server configuration (for production builds)
    preview: {
      host: env.VITE_PREVIEW_HOST || 'localhost',
      port: parseInt(env.VITE_PREVIEW_PORT || '4173', 10),
      strictPort: true,
      open: env.VITE_PREVIEW_OPEN === 'true',
    },

    // Build configuration
    build: {
      outDir: env.VITE_OUT_DIR || 'dist',
      assetsDir: 'assets',
      sourcemap: env.VITE_SOURCEMAP === 'true' || isDevelopment,
      minify: isProduction ? 'esbuild' : false,
      target: 'es2020',
      // Chunk size warning limit (in KB)
      chunkSizeWarningLimit: parseInt(env.VITE_CHUNK_SIZE_WARNING_LIMIT || '1000', 10),
      // Rollup options
      rollupOptions: {
        output: {
          // Manual chunk splitting for better caching
          manualChunks: {
            'react-vendor': ['react', 'react-dom', 'react-router-dom'],
            'chart-vendor': ['chart.js', 'react-chartjs-2'],
            'decimal-vendor': ['decimal.js'],
          },
          // Asset file naming
          assetFileNames: (assetInfo) => {
            const info = assetInfo.name?.split('.');
            const ext = info?.[info.length - 1];
            if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext || '')) {
              return `assets/images/[name]-[hash][extname]`;
            } else if (/woff|woff2|eot|ttf|otf/i.test(ext || '')) {
              return `assets/fonts/[name]-[hash][extname]`;
            }
            return `assets/[name]-[hash][extname]`;
          },
          chunkFileNames: 'assets/js/[name]-[hash].js',
          entryFileNames: 'assets/js/[name]-[hash].js',
        },
      },
      // Terser options for production
      terserOptions: isProduction
        ? {
            compress: {
              drop_console: env.VITE_DROP_CONSOLE === 'true',
              drop_debugger: true,
              pure_funcs: env.VITE_DROP_CONSOLE === 'true' ? ['console.log', 'console.debug'] : [],
            },
            format: {
              comments: false,
            },
          }
        : undefined,
      // Enable CSS code splitting
      cssCodeSplit: true,
      // Report compressed size
      reportCompressedSize: isProduction,
      // Disable Brotli size reporting for faster builds
      brotliSize: false,
    },

    // Optimization
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        'decimal.js',
      ],
      exclude: [],
      esbuildOptions: {
        target: 'es2020',
      },
    },

    // CSS configuration
    css: {
      modules: {
        localsConvention: 'camelCase',
        generateScopedName: isDevelopment
          ? '[name]__[local]___[hash:base64:5]'
          : '[hash:base64:8]',
      },
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
      devSourcemap: isDevelopment,
    },

    // Environment variables
    define: {
      __APP_VERSION__: JSON.stringify(env.VITE_APP_VERSION || '1.0.0'),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    },

    // Logging
    logLevel: env.VITE_LOG_LEVEL || (isDevelopment ? 'info' : 'warn'),

    // Clear screen
    clearScreen: env.VITE_CLEAR_SCREEN !== 'false',

    // Base path
    base: env.VITE_BASE_PATH || '/',
  };
});
