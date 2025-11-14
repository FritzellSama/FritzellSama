/**
 * WebSocket service for real-time data streaming
 * Production-grade WebSocket client with reconnection and error handling
 */

export type ConnectionState = 'connecting' | 'connected' | 'disconnecting' | 'disconnected' | 'reconnecting' | 'error';

export interface WebSocketMessage {
  type: string;
  channel?: string;
  payload: any;
  timestamp: number;
}

interface WebSocketConfig {
  url: string;
  protocols?: string | string[];
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
}

type MessageHandler = (message: WebSocketMessage) => void;
type StateHandler = () => void;
type ErrorHandler = (error: Event) => void;

const DEFAULT_RECONNECT_INTERVAL = parseInt(process.env.VITE_WS_RECONNECT_INTERVAL || '3000', 10);
const DEFAULT_RECONNECT_ATTEMPTS = parseInt(process.env.VITE_WS_RECONNECT_ATTEMPTS || '10', 10);
const DEFAULT_HEARTBEAT_INTERVAL = parseInt(process.env.VITE_WS_HEARTBEAT_INTERVAL || '30000', 10);
const DEFAULT_MESSAGE_QUEUE_SIZE = parseInt(process.env.VITE_WS_MESSAGE_QUEUE_SIZE || '100', 10);

/**
 * WebSocket service for managing real-time connections
 *
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Message queuing during disconnection
 * - Heartbeat/ping-pong for connection health
 * - Channel-based subscriptions
 * - Type-safe message handling
 *
 * @example
 * ```typescript
 * const ws = new WebSocketService('ws://localhost:8000/ws');
 *
 * ws.onMessage((message) => {
 *   console.log('Received:', message);
 * });
 *
 * ws.onOpen(() => {
 *   ws.subscribe('market_data');
 *   ws.subscribe('orders');
 * });
 *
 * ws.connect();
 * ```
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts: number = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageQueue: Array<{ type: string; payload: any }> = [];
  private subscriptions: Set<string> = new Set();

  private messageHandlers: Set<MessageHandler> = new Set();
  private openHandlers: Set<StateHandler> = new Set();
  private closeHandlers: Set<StateHandler> = new Set();
  private errorHandlers: Set<ErrorHandler> = new Set();

  constructor(urlOrConfig: string | WebSocketConfig) {
    const config = typeof urlOrConfig === 'string'
      ? { url: urlOrConfig }
      : urlOrConfig;

    this.config = {
      url: config.url,
      protocols: config.protocols,
      reconnect: config.reconnect ?? true,
      reconnectInterval: config.reconnectInterval ?? DEFAULT_RECONNECT_INTERVAL,
      reconnectAttempts: config.reconnectAttempts ?? DEFAULT_RECONNECT_ATTEMPTS,
      heartbeatInterval: config.heartbeatInterval ?? DEFAULT_HEARTBEAT_INTERVAL,
      messageQueueSize: config.messageQueueSize ?? DEFAULT_MESSAGE_QUEUE_SIZE,
    };
  }

  /**
   * Get current connection state
   */
  public getState(): ConnectionState {
    return this.state;
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.state === 'connected' && this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Register message handler
   */
  public onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  /**
   * Register open handler
   */
  public onOpen(handler: StateHandler): () => void {
    this.openHandlers.add(handler);
    return () => this.openHandlers.delete(handler);
  }

  /**
   * Register close handler
   */
  public onClose(handler: StateHandler): () => void {
    this.closeHandlers.add(handler);
    return () => this.closeHandlers.delete(handler);
  }

  /**
   * Register error handler
   */
  public onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);
    return () => this.errorHandlers.delete(handler);
  }

  /**
   * Connect to WebSocket server
   */
  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      console.warn('[WebSocketService] Already connected or connecting');
      return;
    }

    try {
      this.setState('connecting');

      this.ws = new WebSocket(this.config.url, this.config.protocols);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
    } catch (error) {
      console.error('[WebSocketService] Connection error:', error);
      this.setState('error');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    this.clearReconnectTimeout();
    this.clearHeartbeat();
    this.reconnectAttempts = 0;

    if (this.ws) {
      this.setState('disconnecting');
      this.ws.close();
      this.ws = null;
    }

    this.setState('disconnected');
  }

  /**
   * Send message to server
   */
  public send(type: string, payload: any): void {
    const message = { type, payload, timestamp: Date.now() };

    if (!this.isConnected()) {
      this.queueMessage(type, payload);
      console.warn('[WebSocketService] Not connected, message queued');
      return;
    }

    try {
      this.ws!.send(JSON.stringify(message));
    } catch (error) {
      console.error('[WebSocketService] Send error:', error);
      this.queueMessage(type, payload);
    }
  }

  /**
   * Subscribe to channel
   */
  public subscribe(channel: string): void {
    if (this.subscriptions.has(channel)) {
      return;
    }

    this.subscriptions.add(channel);
    this.send('subscribe', { channel });
  }

  /**
   * Unsubscribe from channel
   */
  public unsubscribe(channel: string): void {
    if (!this.subscriptions.has(channel)) {
      return;
    }

    this.subscriptions.delete(channel);
    this.send('unsubscribe', { channel });
  }

  /**
   * Get active subscriptions
   */
  public getSubscriptions(): string[] {
    return Array.from(this.subscriptions);
  }

  /**
   * Handle connection open
   */
  private handleOpen(): void {
    this.setState('connected');
    this.reconnectAttempts = 0;

    this.startHeartbeat();
    this.resubscribe();
    this.flushMessageQueue();

    this.openHandlers.forEach(handler => {
      try {
        handler();
      } catch (error) {
        console.error('[WebSocketService] Error in open handler:', error);
      }
    });
  }

  /**
   * Handle connection close
   */
  private handleClose(): void {
    this.clearHeartbeat();
    this.setState('disconnected');

    this.closeHandlers.forEach(handler => {
      try {
        handler();
      } catch (error) {
        console.error('[WebSocketService] Error in close handler:', error);
      }
    });

    if (this.config.reconnect && this.reconnectAttempts < this.config.reconnectAttempts) {
      this.scheduleReconnect();
    }
  }

  /**
   * Handle connection error
   */
  private handleError(error: Event): void {
    this.setState('error');

    this.errorHandlers.forEach(handler => {
      try {
        handler(error);
      } catch (err) {
        console.error('[WebSocketService] Error in error handler:', err);
      }
    });
  }

  /**
   * Handle incoming message
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      this.messageHandlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error('[WebSocketService] Error in message handler:', error);
        }
      });
    } catch (error) {
      console.error('[WebSocketService] Failed to parse message:', error);
    }
  }

  /**
   * Set connection state
   */
  private setState(state: ConnectionState): void {
    if (this.state !== state) {
      this.state = state;
    }
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.clearReconnectTimeout();

    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts),
      parseInt(process.env.VITE_WS_MAX_RECONNECT_DELAY || '30000', 10)
    );

    this.reconnectAttempts += 1;
    this.setState('reconnecting');

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Clear reconnect timeout
   */
  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    this.clearHeartbeat();

    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.send('ping', { timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Clear heartbeat
   */
  private clearHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Queue message for later delivery
   */
  private queueMessage(type: string, payload: any): void {
    if (this.messageQueue.length >= this.config.messageQueueSize) {
      this.messageQueue.shift();
    }
    this.messageQueue.push({ type, payload });
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.isConnected()) {
      const message = this.messageQueue.shift()!;
      this.send(message.type, message.payload);
    }
  }

  /**
   * Resubscribe to channels after reconnection
   */
  private resubscribe(): void {
    this.subscriptions.forEach(channel => {
      this.send('subscribe', { channel });
    });
  }
}

export default WebSocketService;
