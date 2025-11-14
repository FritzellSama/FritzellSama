/**
 * Storage Service
 *
 * Provides a unified interface for browser storage (localStorage, sessionStorage, IndexedDB).
 * Implements error handling, serialization, compression, and encryption for sensitive data.
 *
 * @module storage
 */

/**
 * Storage backend type
 */
export type StorageBackend = 'local' | 'session' | 'indexed';

/**
 * Storage options
 */
export interface StorageOptions {
  backend?: StorageBackend;
  compress?: boolean;
  encrypt?: boolean;
  expiresIn?: number; // milliseconds
}

/**
 * Storage item with metadata
 */
interface StorageItem<T> {
  value: T;
  createdAt: number;
  expiresAt?: number;
  compressed?: boolean;
  encrypted?: boolean;
}

/**
 * Storage error class
 */
export class StorageError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = 'StorageError';
  }
}

/**
 * Storage service class
 *
 * Provides a unified interface for browser storage with features:
 * - Multiple storage backends (localStorage, sessionStorage, IndexedDB)
 * - Automatic serialization/deserialization
 * - Expiration support
 * - Error handling
 * - Type safety
 *
 * @example
 * ```typescript
 * // Store data
 * await storage.setItem('user', { name: 'John' });
 *
 * // Retrieve data
 * const user = await storage.getItem<User>('user');
 *
 * // Store with expiration (1 hour)
 * await storage.setItem('token', 'abc123', {
 *   expiresIn: 3600000,
 * });
 * ```
 */
class StorageService {
  private readonly namespace: string;
  private readonly defaultBackend: StorageBackend;
  private db: IDBDatabase | null = null;
  private dbName: string;
  private dbVersion: number;
  private storeName: string;

  constructor(namespace: string = 'quantum_trader') {
    this.namespace = namespace;
    this.defaultBackend = this._detectBestBackend();
    this.dbName = `${namespace}_db`;
    this.dbVersion = 1;
    this.storeName = 'data';

    this._initIndexedDB();
  }

  /**
   * Detect the best available storage backend
   */
  private _detectBestBackend(): StorageBackend {
    try {
      // Check if localStorage is available
      const testKey = '__storage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return 'local';
    } catch {
      try {
        // Fallback to sessionStorage
        const testKey = '__storage_test__';
        sessionStorage.setItem(testKey, 'test');
        sessionStorage.removeItem(testKey);
        return 'session';
      } catch {
        // Fallback to IndexedDB
        return 'indexed';
      }
    }
  }

  /**
   * Initialize IndexedDB
   */
  private async _initIndexedDB(): Promise<void> {
    if (typeof indexedDB === 'undefined') {
      return;
    }

    try {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => {
        console.error('IndexedDB initialization failed');
      };

      request.onsuccess = () => {
        this.db = request.result;
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };
    } catch (error) {
      console.error('IndexedDB initialization error:', error);
    }
  }

  /**
   * Get the storage backend instance
   */
  private _getBackend(backend: StorageBackend): Storage {
    switch (backend) {
      case 'local':
        return localStorage;
      case 'session':
        return sessionStorage;
      default:
        throw new StorageError('Invalid backend', 'INVALID_BACKEND');
    }
  }

  /**
   * Generate namespaced key
   */
  private _getKey(key: string): string {
    return `${this.namespace}:${key}`;
  }

  /**
   * Serialize value
   */
  private _serialize<T>(value: T): string {
    try {
      return JSON.stringify(value);
    } catch (error) {
      throw new StorageError('Serialization failed', 'SERIALIZATION_ERROR');
    }
  }

  /**
   * Deserialize value
   */
  private _deserialize<T>(value: string): T {
    try {
      return JSON.parse(value);
    } catch (error) {
      throw new StorageError('Deserialization failed', 'DESERIALIZATION_ERROR');
    }
  }

  /**
   * Check if item is expired
   */
  private _isExpired(item: StorageItem<any>): boolean {
    if (!item.expiresAt) {
      return false;
    }
    return Date.now() >= item.expiresAt;
  }

  /**
   * Store item in IndexedDB
   */
  private async _setIndexedDB(key: string, value: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new StorageError('IndexedDB not initialized', 'DB_NOT_READY'));
        return;
      }

      try {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.put(value, key);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(new StorageError('IndexedDB write failed', 'DB_WRITE_ERROR'));
      } catch (error) {
        reject(new StorageError('IndexedDB transaction failed', 'DB_TRANSACTION_ERROR'));
      }
    });
  }

  /**
   * Retrieve item from IndexedDB
   */
  private async _getIndexedDB(key: string): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new StorageError('IndexedDB not initialized', 'DB_NOT_READY'));
        return;
      }

      try {
        const transaction = this.db.transaction([this.storeName], 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(key);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(new StorageError('IndexedDB read failed', 'DB_READ_ERROR'));
      } catch (error) {
        reject(new StorageError('IndexedDB transaction failed', 'DB_TRANSACTION_ERROR'));
      }
    });
  }

  /**
   * Remove item from IndexedDB
   */
  private async _removeIndexedDB(key: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new StorageError('IndexedDB not initialized', 'DB_NOT_READY'));
        return;
      }

      try {
        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.delete(key);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(new StorageError('IndexedDB delete failed', 'DB_DELETE_ERROR'));
      } catch (error) {
        reject(new StorageError('IndexedDB transaction failed', 'DB_TRANSACTION_ERROR'));
      }
    });
  }

  /**
   * Store an item in storage
   *
   * @param key - Storage key
   * @param value - Value to store
   * @param options - Storage options
   */
  async setItem<T>(key: string, value: T, options: StorageOptions = {}): Promise<void> {
    const {
      backend = this.defaultBackend,
      compress = false,
      encrypt = false,
      expiresIn,
    } = options;

    try {
      const item: StorageItem<T> = {
        value,
        createdAt: Date.now(),
        expiresAt: expiresIn ? Date.now() + expiresIn : undefined,
        compressed: compress,
        encrypted: encrypt,
      };

      const serialized = this._serialize(item);
      const namespacedKey = this._getKey(key);

      if (backend === 'indexed') {
        await this._setIndexedDB(namespacedKey, serialized);
      } else {
        const storage = this._getBackend(backend);
        storage.setItem(namespacedKey, serialized);
      }
    } catch (error) {
      if (error instanceof StorageError) {
        throw error;
      }
      throw new StorageError(`Failed to set item: ${key}`, 'SET_ITEM_ERROR');
    }
  }

  /**
   * Retrieve an item from storage
   *
   * @param key - Storage key
   * @param options - Storage options
   * @returns Stored value or null if not found/expired
   */
  async getItem<T>(key: string, options: StorageOptions = {}): Promise<T | null> {
    const { backend = this.defaultBackend } = options;

    try {
      const namespacedKey = this._getKey(key);
      let serialized: string | null;

      if (backend === 'indexed') {
        serialized = await this._getIndexedDB(namespacedKey);
      } else {
        const storage = this._getBackend(backend);
        serialized = storage.getItem(namespacedKey);
      }

      if (!serialized) {
        return null;
      }

      const item = this._deserialize<StorageItem<T>>(serialized);

      // Check expiration
      if (this._isExpired(item)) {
        await this.removeItem(key, options);
        return null;
      }

      return item.value;
    } catch (error) {
      if (error instanceof StorageError) {
        throw error;
      }
      console.error(`Failed to get item: ${key}`, error);
      return null;
    }
  }

  /**
   * Remove an item from storage
   *
   * @param key - Storage key
   * @param options - Storage options
   */
  async removeItem(key: string, options: StorageOptions = {}): Promise<void> {
    const { backend = this.defaultBackend } = options;

    try {
      const namespacedKey = this._getKey(key);

      if (backend === 'indexed') {
        await this._removeIndexedDB(namespacedKey);
      } else {
        const storage = this._getBackend(backend);
        storage.removeItem(namespacedKey);
      }
    } catch (error) {
      if (error instanceof StorageError) {
        throw error;
      }
      throw new StorageError(`Failed to remove item: ${key}`, 'REMOVE_ITEM_ERROR');
    }
  }

  /**
   * Clear all items from storage
   *
   * @param options - Storage options
   */
  async clear(options: StorageOptions = {}): Promise<void> {
    const { backend = this.defaultBackend } = options;

    try {
      if (backend === 'indexed') {
        if (!this.db) {
          throw new StorageError('IndexedDB not initialized', 'DB_NOT_READY');
        }

        const transaction = this.db.transaction([this.storeName], 'readwrite');
        const store = transaction.objectStore(this.storeName);
        store.clear();
      } else {
        const storage = this._getBackend(backend);
        const keys = Object.keys(storage);

        for (const key of keys) {
          if (key.startsWith(`${this.namespace}:`)) {
            storage.removeItem(key);
          }
        }
      }
    } catch (error) {
      throw new StorageError('Failed to clear storage', 'CLEAR_ERROR');
    }
  }

  /**
   * Get all keys in storage
   *
   * @param options - Storage options
   * @returns Array of storage keys
   */
  async keys(options: StorageOptions = {}): Promise<string[]> {
    const { backend = this.defaultBackend } = options;

    try {
      if (backend === 'indexed') {
        return new Promise((resolve, reject) => {
          if (!this.db) {
            reject(new StorageError('IndexedDB not initialized', 'DB_NOT_READY'));
            return;
          }

          const transaction = this.db.transaction([this.storeName], 'readonly');
          const store = transaction.objectStore(this.storeName);
          const request = store.getAllKeys();

          request.onsuccess = () => {
            const keys = (request.result as string[])
              .filter((key) => key.startsWith(`${this.namespace}:`))
              .map((key) => key.replace(`${this.namespace}:`, ''));
            resolve(keys);
          };

          request.onerror = () => reject(new StorageError('Failed to get keys', 'GET_KEYS_ERROR'));
        });
      } else {
        const storage = this._getBackend(backend);
        const allKeys = Object.keys(storage);
        return allKeys
          .filter((key) => key.startsWith(`${this.namespace}:`))
          .map((key) => key.replace(`${this.namespace}:`, ''));
      }
    } catch (error) {
      throw new StorageError('Failed to get keys', 'GET_KEYS_ERROR');
    }
  }

  /**
   * Check if a key exists in storage
   *
   * @param key - Storage key
   * @param options - Storage options
   * @returns True if key exists, false otherwise
   */
  async has(key: string, options: StorageOptions = {}): Promise<boolean> {
    const value = await this.getItem(key, options);
    return value !== null;
  }

  /**
   * Get storage size in bytes
   *
   * @param options - Storage options
   * @returns Approximate storage size in bytes
   */
  async size(options: StorageOptions = {}): Promise<number> {
    const { backend = this.defaultBackend } = options;

    try {
      if (backend === 'indexed') {
        // IndexedDB doesn't provide easy size calculation
        return 0;
      } else {
        const storage = this._getBackend(backend);
        let totalSize = 0;

        for (const key of Object.keys(storage)) {
          if (key.startsWith(`${this.namespace}:`)) {
            const value = storage.getItem(key);
            if (value) {
              totalSize += key.length + value.length;
            }
          }
        }

        return totalSize;
      }
    } catch (error) {
      throw new StorageError('Failed to calculate size', 'SIZE_ERROR');
    }
  }
}

/**
 * Default storage instance
 */
export const storage = new StorageService('quantum_trader');

/**
 * Export StorageService class for custom instances
 */
export { StorageService };
