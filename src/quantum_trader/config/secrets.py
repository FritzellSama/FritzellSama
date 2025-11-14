"""
Secrets Management - Secure handling of API keys, credentials, and sensitive data.

This module provides encrypted storage and retrieval of sensitive configuration
including exchange API keys, database credentials, and encryption keys.
"""

import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class ExchangeCredentials:
    """Exchange API credentials."""

    exchange: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    testnet: bool = False


@dataclass
class DatabaseCredentials:
    """Database connection credentials."""

    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str = 'require'


class SecretsManager:
    """
    Manages encrypted secrets and sensitive configuration.

    Provides secure storage, retrieval, and rotation of sensitive data
    including API keys, passwords, and encryption keys.

    Attributes:
        encryption_key: Fernet encryption key
        secrets: Loaded secrets dictionary

    Example:
        >>> secrets_mgr = SecretsManager(config)
        >>> creds = secrets_mgr.get_exchange_credentials('binance')
        >>> db_url = secrets_mgr.get_database_url()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize secrets manager.

        Args:
            config: Configuration dict with keys:
                - master_key_path: Path to master encryption key file
                - secrets_path: Path to encrypted secrets file
                - use_env_vars: Whether to load from environment variables

        Raises:
            ValueError: If config invalid or encryption key missing
        """
        self.config = config
        self._validate_config()

        self.master_key_path = Path(config.get('master_key_path', '/etc/quantum_trader/master.key'))
        self.secrets_path = Path(config.get('secrets_path', '/etc/quantum_trader/secrets.enc'))
        self.use_env_vars = bool(config.get('use_env_vars', True))

        # Initialize encryption
        self.encryption_key = self._load_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)

        # Load secrets
        self.secrets: Dict[str, Any] = {}
        self._load_secrets()

        logger.info(
            "SecretsManager initialized",
            use_env_vars=self.use_env_vars,
            secrets_count=len(self.secrets)
        )

    def _validate_config(self) -> None:
        """Validate configuration."""
        pass  # Config keys are optional with defaults

    def _load_or_create_encryption_key(self) -> bytes:
        """
        Load or create encryption key.

        Returns:
            Fernet encryption key bytes

        Raises:
            ValueError: If key generation fails
        """
        try:
            # Try to load from environment first
            env_key = os.getenv('QUANTUM_TRADER_MASTER_KEY')
            if env_key:
                logger.info("Loaded encryption key from environment")
                return base64.urlsafe_b64decode(env_key.encode())

            # Try to load from file
            if self.master_key_path.exists():
                with open(self.master_key_path, 'rb') as f:
                    key = f.read()
                logger.info("Loaded encryption key from file", path=str(self.master_key_path))
                return key

            # Generate new key
            key = Fernet.generate_key()

            # Save to file if path writable
            try:
                self.master_key_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.master_key_path, 'wb') as f:
                    f.write(key)
                os.chmod(self.master_key_path, 0o600)
                logger.info("Generated new encryption key", path=str(self.master_key_path))
            except Exception as e:
                logger.warning(
                    "Could not save encryption key to file",
                    error=str(e),
                    note="Set QUANTUM_TRADER_MASTER_KEY environment variable"
                )

            return key

        except Exception as e:
            logger.error("Failed to load/create encryption key", error=str(e))
            raise ValueError(f"Encryption key initialization failed: {e}")

    def _load_secrets(self) -> None:
        """Load secrets from environment variables or encrypted file."""
        try:
            # Load from environment variables if enabled
            if self.use_env_vars:
                self._load_from_env()

            # Load from encrypted file if exists
            if self.secrets_path.exists():
                self._load_from_file()

        except Exception as e:
            logger.error("Failed to load secrets", error=str(e))
            raise

    def _load_from_env(self) -> None:
        """Load secrets from environment variables."""
        # Exchange credentials
        exchanges = os.getenv('EXCHANGES', '').split(',')
        for exchange in exchanges:
            if not exchange:
                continue

            api_key = os.getenv(f'{exchange.upper()}_API_KEY')
            api_secret = os.getenv(f'{exchange.upper()}_API_SECRET')
            passphrase = os.getenv(f'{exchange.upper()}_PASSPHRASE')
            testnet = os.getenv(f'{exchange.upper()}_TESTNET', 'false').lower() == 'true'

            if api_key and api_secret:
                self.secrets[f'exchange_{exchange}'] = {
                    'api_key': api_key,
                    'api_secret': api_secret,
                    'passphrase': passphrase,
                    'testnet': testnet
                }

        # Database credentials
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            self.secrets['database_url'] = db_url

        # Redis credentials
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            self.secrets['redis_url'] = redis_url

        # Notification credentials
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if telegram_token:
            self.secrets['telegram_bot_token'] = telegram_token

        email_password = os.getenv('EMAIL_PASSWORD')
        if email_password:
            self.secrets['email_password'] = email_password

        logger.debug("Loaded secrets from environment variables")

    def _load_from_file(self) -> None:
        """Load secrets from encrypted file."""
        try:
            with open(self.secrets_path, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)

            import json
            file_secrets = json.loads(decrypted_data.decode('utf-8'))

            # Merge with existing secrets (env vars take precedence)
            for key, value in file_secrets.items():
                if key not in self.secrets:
                    self.secrets[key] = value

            logger.info("Loaded secrets from encrypted file", path=str(self.secrets_path))

        except Exception as e:
            logger.error("Failed to load encrypted secrets file", error=str(e))
            raise

    def get_exchange_credentials(self, exchange: str) -> ExchangeCredentials:
        """
        Get exchange API credentials.

        Args:
            exchange: Exchange name (e.g., 'binance', 'bybit')

        Returns:
            ExchangeCredentials object

        Raises:
            ValueError: If credentials not found
        """
        key = f'exchange_{exchange.lower()}'

        if key not in self.secrets:
            raise ValueError(f"Credentials not found for exchange: {exchange}")

        creds = self.secrets[key]

        return ExchangeCredentials(
            exchange=exchange,
            api_key=creds['api_key'],
            api_secret=creds['api_secret'],
            passphrase=creds.get('passphrase'),
            testnet=bool(creds.get('testnet', False))
        )

    def get_database_url(self) -> str:
        """
        Get database connection URL.

        Returns:
            Database URL string

        Raises:
            ValueError: If database URL not configured
        """
        if 'database_url' not in self.secrets:
            raise ValueError("Database URL not configured")

        return self.secrets['database_url']

    def get_redis_url(self) -> str:
        """
        Get Redis connection URL.

        Returns:
            Redis URL string

        Raises:
            ValueError: If Redis URL not configured
        """
        if 'redis_url' not in self.secrets:
            raise ValueError("Redis URL not configured")

        return self.secrets['redis_url']

    def get_secret(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a secret value by key.

        Args:
            key: Secret key
            default: Default value if key not found

        Returns:
            Secret value or default
        """
        return self.secrets.get(key, default)

    def set_secret(self, key: str, value: Any) -> None:
        """
        Set a secret value.

        Args:
            key: Secret key
            value: Secret value
        """
        self.secrets[key] = value
        logger.debug("Secret updated", key=key)

    def save_secrets(self) -> None:
        """
        Save secrets to encrypted file.

        Raises:
            IOError: If file write fails
        """
        try:
            import json

            # Serialize secrets
            secrets_json = json.dumps(self.secrets, indent=2)

            # Encrypt
            encrypted_data = self.cipher.encrypt(secrets_json.encode('utf-8'))

            # Save to file
            self.secrets_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.secrets_path, 'wb') as f:
                f.write(encrypted_data)

            # Set restrictive permissions
            os.chmod(self.secrets_path, 0o600)

            logger.info("Secrets saved to encrypted file", path=str(self.secrets_path))

        except Exception as e:
            logger.error("Failed to save secrets", error=str(e))
            raise

    def rotate_exchange_key(self, exchange: str, new_api_key: str, new_api_secret: str) -> None:
        """
        Rotate exchange API credentials.

        Args:
            exchange: Exchange name
            new_api_key: New API key
            new_api_secret: New API secret
        """
        key = f'exchange_{exchange.lower()}'

        if key in self.secrets:
            self.secrets[key]['api_key'] = new_api_key
            self.secrets[key]['api_secret'] = new_api_secret

            logger.info("Exchange credentials rotated", exchange=exchange)
            self.save_secrets()
        else:
            raise ValueError(f"Exchange credentials not found: {exchange}")

    def validate_credentials(self, exchange: str) -> bool:
        """
        Validate that exchange credentials exist and are non-empty.

        Args:
            exchange: Exchange name

        Returns:
            True if credentials valid, False otherwise
        """
        try:
            creds = self.get_exchange_credentials(exchange)
            return bool(creds.api_key and creds.api_secret)
        except ValueError:
            return False

    def list_configured_exchanges(self) -> list[str]:
        """
        Get list of exchanges with configured credentials.

        Returns:
            List of exchange names
        """
        exchanges = []
        for key in self.secrets.keys():
            if key.startswith('exchange_'):
                exchange_name = key.replace('exchange_', '')
                exchanges.append(exchange_name)
        return exchanges

    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a value.

        Args:
            value: Value to encrypt

        Returns:
            Base64-encoded encrypted value
        """
        encrypted = self.cipher.encrypt(value.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt a value.

        Args:
            encrypted_value: Base64-encoded encrypted value

        Returns:
            Decrypted value

        Raises:
            ValueError: If decryption fails
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise ValueError(f"Failed to decrypt value: {e}")
