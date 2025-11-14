/**
 * WebSocket hook for real-time data streaming
 * Handles connection management, reconnection, and message handling
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { WebSocketService, WebSocketMessage, ConnectionState } from '../services/websocket';

interface UseWebSocketOptions {
  url?: string;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

interface UseWebSocketReturn {
  connectionState: ConnectionState;
  lastMessage: WebSocketMessage | null;
  send: (type: string, payload: any) => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  connect: () => void;
  disconnect: () => void;
  isConnected: boolean;
}

const DEFAULT_URL = process.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const DEFAULT_RECONNECT = true;
const DEFAULT_RECONNECT_INTERVAL = parseInt(process.env.VITE_WS_RECONNECT_INTERVAL || '3000', 10);
const DEFAULT_RECONNECT_ATTEMPTS = parseInt(process.env.VITE_WS_RECONNECT_ATTEMPTS || '10', 10);

/**
 * Custom hook for WebSocket connections
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Connection state management
 * - Message handling and serialization
 * - Channel subscription management
 * - Type-safe message handling
 *
 * @param options - WebSocket configuration options
 *
 * @example
 * ```tsx
 * function TradingDashboard() {
 *   const {
 *     connectionState,
 *     lastMessage,
 *     send,
 *     subscribe,
 *     isConnected
 *   } = useWebSocket({
 *     url: 'ws://localhost:8000/ws',
 *     onMessage: (message) => {
 *       console.log('Received:', message);
 *     },
 *     onError: (error) => {
 *       console.error('WebSocket error:', error);
 *     }
 *   });
 *
 *   useEffect(() => {
 *     if (isConnected) {
 *       subscribe('market_data');
 *       subscribe('orders');
 *     }
 *   }, [isConnected, subscribe]);
 *
 *   return (
 *     <div>
 *       <p>Status: {connectionState}</p>
 *       {lastMessage && <pre>{JSON.stringify(lastMessage, null, 2)}</pre>}
 *     </div>
 *   );
 * }
 * ```
 */
export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    url = DEFAULT_URL,
    reconnect = DEFAULT_RECONNECT,
    reconnectInterval = DEFAULT_RECONNECT_INTERVAL,
    reconnectAttempts = DEFAULT_RECONNECT_ATTEMPTS,
    onOpen,
    onClose,
    onError,
    onMessage,
  } = options;

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  const wsServiceRef = useRef<WebSocketService | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const isMountedRef = useRef<boolean>(true);

  /**
   * Calculate reconnect delay with exponential backoff
   */
  const getReconnectDelay = useCallback((attemptNumber: number): number => {
    const maxDelay = parseInt(process.env.VITE_WS_MAX_RECONNECT_DELAY || '30000', 10);
    const delay = Math.min(reconnectInterval * Math.pow(2, attemptNumber), maxDelay);
    return delay;
  }, [reconnectInterval]);

  /**
   * Handle connection open
   */
  const handleOpen = useCallback(() => {
    if (!isMountedRef.current) return;

    setConnectionState('connected');
    reconnectAttemptsRef.current = 0;

    if (onOpen) {
      try {
        onOpen();
      } catch (error) {
        console.error('[useWebSocket] Error in onOpen callback:', error);
      }
    }
  }, [onOpen]);

  /**
   * Handle connection close
   */
  const handleClose = useCallback(() => {
    if (!isMountedRef.current) return;

    setConnectionState('disconnected');

    if (onClose) {
      try {
        onClose();
      } catch (error) {
        console.error('[useWebSocket] Error in onClose callback:', error);
      }
    }

    // Attempt reconnection
    if (reconnect && reconnectAttemptsRef.current < reconnectAttempts) {
      const delay = getReconnectDelay(reconnectAttemptsRef.current);
      reconnectAttemptsRef.current += 1;

      setConnectionState('reconnecting');

      reconnectTimeoutRef.current = setTimeout(() => {
        if (isMountedRef.current && wsServiceRef.current) {
          wsServiceRef.current.connect();
        }
      }, delay);
    }
  }, [reconnect, reconnectAttempts, getReconnectDelay, onClose]);

  /**
   * Handle connection error
   */
  const handleError = useCallback((error: Event) => {
    if (!isMountedRef.current) return;

    setConnectionState('error');

    if (onError) {
      try {
        onError(error);
      } catch (err) {
        console.error('[useWebSocket] Error in onError callback:', err);
      }
    }
  }, [onError]);

  /**
   * Handle incoming message
   */
  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (!isMountedRef.current) return;

    setLastMessage(message);

    if (onMessage) {
      try {
        onMessage(message);
      } catch (error) {
        console.error('[useWebSocket] Error in onMessage callback:', error);
      }
    }
  }, [onMessage]);

  /**
   * Initialize WebSocket service
   */
  useEffect(() => {
    isMountedRef.current = true;

    try {
      wsServiceRef.current = new WebSocketService(url);

      wsServiceRef.current.onOpen(handleOpen);
      wsServiceRef.current.onClose(handleClose);
      wsServiceRef.current.onError(handleError);
      wsServiceRef.current.onMessage(handleMessage);

      wsServiceRef.current.connect();
    } catch (error) {
      console.error('[useWebSocket] Failed to initialize WebSocket service:', error);
      setConnectionState('error');
    }

    return () => {
      isMountedRef.current = false;

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      if (wsServiceRef.current) {
        wsServiceRef.current.disconnect();
        wsServiceRef.current = null;
      }
    };
  }, [url, handleOpen, handleClose, handleError, handleMessage]);

  /**
   * Send message to server
   */
  const send = useCallback((type: string, payload: any) => {
    if (!wsServiceRef.current) {
      console.error('[useWebSocket] WebSocket service not initialized');
      return;
    }

    try {
      wsServiceRef.current.send(type, payload);
    } catch (error) {
      console.error('[useWebSocket] Failed to send message:', error);
    }
  }, []);

  /**
   * Subscribe to channel
   */
  const subscribe = useCallback((channel: string) => {
    if (!wsServiceRef.current) {
      console.error('[useWebSocket] WebSocket service not initialized');
      return;
    }

    try {
      wsServiceRef.current.subscribe(channel);
    } catch (error) {
      console.error('[useWebSocket] Failed to subscribe to channel:', error);
    }
  }, []);

  /**
   * Unsubscribe from channel
   */
  const unsubscribe = useCallback((channel: string) => {
    if (!wsServiceRef.current) {
      console.error('[useWebSocket] WebSocket service not initialized');
      return;
    }

    try {
      wsServiceRef.current.unsubscribe(channel);
    } catch (error) {
      console.error('[useWebSocket] Failed to unsubscribe from channel:', error);
    }
  }, []);

  /**
   * Manually connect
   */
  const connect = useCallback(() => {
    if (!wsServiceRef.current) {
      console.error('[useWebSocket] WebSocket service not initialized');
      return;
    }

    try {
      reconnectAttemptsRef.current = 0;
      wsServiceRef.current.connect();
    } catch (error) {
      console.error('[useWebSocket] Failed to connect:', error);
    }
  }, []);

  /**
   * Manually disconnect
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    if (!wsServiceRef.current) {
      return;
    }

    try {
      wsServiceRef.current.disconnect();
    } catch (error) {
      console.error('[useWebSocket] Failed to disconnect:', error);
    }
  }, []);

  const isConnected = connectionState === 'connected';

  return {
    connectionState,
    lastMessage,
    send,
    subscribe,
    unsubscribe,
    connect,
    disconnect,
    isConnected,
  };
};

export default useWebSocket;
