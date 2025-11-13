"""
HashiCorp Vault Integration for Secrets Management
Production-ready vault operations with comprehensive error handling
"""

from decimal import Decimal
from typing import Optional, Dict, List, Any
import logging
import asyncio
import os
from datetime import datetime, timedelta
import aiohttp
import json

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class VaultIntegration:
    """HashiCorp Vault integration for secure secrets management"""

    def __init__(self) -> None:
        """Initialize Vault integration with configuration"""
        self.config = get_config()
        self._load_config()
        self._session: Optional[aiohttp.ClientSession] = None
        self._token_expire_time: Optional[datetime] = None
        logger.info("VaultIntegration initialized")

    def _load_config(self) -> None:
        """Load configuration from security.yaml"""
        vault_url = os.getenv('VAULT_URL')
        if not vault_url:
            raise ValueError("VAULT_URL environment variable not set")

        vault_token = os.getenv('VAULT_TOKEN')
        if not vault_token:
            raise ValueError("VAULT_TOKEN environment variable not set")

        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token
        self.mount_point = self.config.get_string('security', 'vault.mount_point')
        self.secrets_path = self.config.get_string('security', 'vault.secrets_path')
        self.rotation_days = self.config.get_int('security', 'vault.rotation_days')
        self.timeout_seconds = self.config.get_int('security', 'vault.timeout_seconds')

        logger.info(f"Vault configured: url={self.vault_url}, mount={self.mount_point}")

    async def connect_to_vault(self) -> bool:
        """
        Establish connection to HashiCorp Vault

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionError: If unable to connect after retries
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                    self._session = aiohttp.ClientSession(
                        headers={
                            'X-Vault-Token': self.vault_token,
                            'Content-Type': 'application/json'
                        },
                        timeout=timeout
                    )

                # Verify connection with health check
                url = f"{self.vault_url}/v1/sys/health"
                async with self._session.get(url) as response:
                    if response.status in [200, 429, 472, 473]:
                        logger.info("Successfully connected to Vault")
                        self._token_expire_time = datetime.utcnow() + timedelta(hours=24)
                        return True
                    else:
                        error_text = await response.text()
                        raise ConnectionError(f"Vault health check failed: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Vault connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise ConnectionError(f"Failed to connect to Vault after {max_retries} attempts: {e}")

        return False

    async def get_secret(self, secret_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret from Vault

        Args:
            secret_key: Key of the secret to retrieve

        Returns:
            Dictionary containing secret data or None if not found

        Raises:
            ValueError: If secret_key is invalid
            RuntimeError: If retrieval fails after retries
        """
        if not secret_key:
            raise ValueError("secret_key cannot be empty")

        # Ensure we're connected
        if self._session is None or self._session.closed:
            await self.connect_to_vault()

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Construct path: /v1/{mount_point}/data/{secrets_path}/{secret_key}
                url = f"{self.vault_url}/v1/{self.mount_point}/data/{self.secrets_path}/{secret_key}"

                async with self._session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        secret_data = data.get('data', {}).get('data', {})
                        logger.info(f"Successfully retrieved secret: {secret_key}")
                        return secret_data
                    elif response.status == 404:
                        logger.warning(f"Secret not found: {secret_key}")
                        return None
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to get secret: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Get secret attempt {attempt + 1} failed for {secret_key}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to get secret {secret_key} after {max_retries} attempts: {e}")

        return None

    async def set_secret(self, secret_key: str, secret_data: Dict[str, Any]) -> bool:
        """
        Store a secret in Vault

        Args:
            secret_key: Key for the secret
            secret_data: Dictionary containing secret data

        Returns:
            bool: True if secret was stored successfully

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If storage fails after retries
        """
        if not secret_key:
            raise ValueError("secret_key cannot be empty")
        if not secret_data:
            raise ValueError("secret_data cannot be empty")

        # Ensure we're connected
        if self._session is None or self._session.closed:
            await self.connect_to_vault()

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Add metadata
                payload = {
                    'data': secret_data,
                    'options': {
                        'cas': 0  # Check-and-set version
                    }
                }

                # Construct path: /v1/{mount_point}/data/{secrets_path}/{secret_key}
                url = f"{self.vault_url}/v1/{self.mount_point}/data/{self.secrets_path}/{secret_key}"

                async with self._session.post(url, json=payload) as response:
                    if response.status in [200, 204]:
                        logger.info(f"Successfully stored secret: {secret_key}")
                        return True
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to set secret: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Set secret attempt {attempt + 1} failed for {secret_key}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to set secret {secret_key} after {max_retries} attempts: {e}")

        return False

    async def rotate_secret(self, secret_key: str, new_secret_data: Dict[str, Any]) -> bool:
        """
        Rotate a secret in Vault

        Args:
            secret_key: Key of the secret to rotate
            new_secret_data: New secret data

        Returns:
            bool: True if rotation successful

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If rotation fails
        """
        if not secret_key:
            raise ValueError("secret_key cannot be empty")
        if not new_secret_data:
            raise ValueError("new_secret_data cannot be empty")

        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Get current version
                current_secret = await self.get_secret(secret_key)

                # Add rotation metadata
                rotation_data = {
                    **new_secret_data,
                    'rotated_at': datetime.utcnow().isoformat(),
                    'previous_version': current_secret.get('version', 'unknown') if current_secret else 'initial'
                }

                # Store new version
                success = await self.set_secret(secret_key, rotation_data)

                if success:
                    logger.info(f"Successfully rotated secret: {secret_key}")
                    return True
                else:
                    raise RuntimeError(f"Failed to rotate secret: {secret_key}")

            except Exception as e:
                logger.error(f"Rotate secret attempt {attempt + 1} failed for {secret_key}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to rotate secret {secret_key} after {max_retries} attempts: {e}")

        return False

    async def list_secrets(self, path: Optional[str] = None) -> List[str]:
        """
        List all secrets at a given path

        Args:
            path: Optional subpath, defaults to secrets_path

        Returns:
            List of secret keys
        """
        if self._session is None or self._session.closed:
            await self.connect_to_vault()

        target_path = path if path else self.secrets_path
        url = f"{self.vault_url}/v1/{self.mount_point}/metadata/{target_path}"

        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with self._session.request('LIST', url) as response:
                    if response.status == 200:
                        data = await response.json()
                        keys = data.get('data', {}).get('keys', [])
                        logger.info(f"Listed {len(keys)} secrets at path: {target_path}")
                        return keys
                    elif response.status == 404:
                        logger.warning(f"Path not found: {target_path}")
                        return []
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to list secrets: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"List secrets attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to list secrets after {max_retries} attempts: {e}")

        return []

    async def delete_secret(self, secret_key: str) -> bool:
        """
        Delete a secret from Vault (soft delete - marks as deleted)

        Args:
            secret_key: Key of the secret to delete

        Returns:
            bool: True if deletion successful
        """
        if not secret_key:
            raise ValueError("secret_key cannot be empty")

        if self._session is None or self._session.closed:
            await self.connect_to_vault()

        url = f"{self.vault_url}/v1/{self.mount_point}/data/{self.secrets_path}/{secret_key}"

        max_retries = 3

        for attempt in range(max_retries):
            try:
                async with self._session.delete(url) as response:
                    if response.status in [200, 204]:
                        logger.info(f"Successfully deleted secret: {secret_key}")
                        return True
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to delete secret: {response.status} - {error_text}")

            except Exception as e:
                logger.error(f"Delete secret attempt {attempt + 1} failed for {secret_key}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to delete secret {secret_key} after {max_retries} attempts: {e}")

        return False

    async def close(self) -> None:
        """Close the Vault session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("Vault session closed")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect_to_vault()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
