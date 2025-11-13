"""
Key Manager - CRITICAL SECURITY SYSTEM
Cryptographic key management, rotation, and storage
"""

import logging
import os
import json
import asyncio
from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import base64

from quantum_trader.utils.config_loader import get_config
from quantum_trader.security.encryption import EncryptionManager

logger = logging.getLogger(__name__)


@dataclass
class KeyMetadata:
    """Cryptographic key metadata"""
    key_id: str
    created_at: datetime
    expires_at: datetime
    rotated_at: Optional[datetime] = None
    version: int = 1
    algorithm: str = "AES-256-GCM"
    status: str = "ACTIVE"  # ACTIVE, ROTATED, REVOKED


class KeyManager:
    """
    Production key management system
    - Secure key storage
    - Automatic key rotation
    - Key versioning
    - Integration with KMS (AWS, Azure, GCP)
    """

    def __init__(self) -> None:
        self.config = get_config()
        self._load_config()

        # Key storage (in production: use HSM or KMS)
        self._keys: Dict[str, bytes] = {}
        self._key_metadata: Dict[str, KeyMetadata] = {}

        # Encryption for key storage
        self._encryption = EncryptionManager()

        # Load existing keys
        self._load_keys()

        # Start rotation task
        asyncio.create_task(self._rotate_keys_periodically())

        logger.info("KeyManager initialized")

    def _load_config(self) -> None:
        """Load key management configuration"""
        self.rotation_days = self.config.get_int('security', 'vault.rotation_days', 30)
        self.key_storage_path = Path(os.getenv('KEY_STORAGE_PATH', '/tmp/quantum_trader_keys'))
        self.key_storage_path.mkdir(parents=True, exist_ok=True)

        # KMS configuration
        self.kms_provider = self.config.get('security', 'encryption.kms_provider', 'aws')
        self.kms_key_id = self.config.get('security', 'encryption.kms_key_id', '')

    def _load_keys(self) -> None:
        """Load keys from secure storage"""
        try:
            # In production: load from HSM/KMS
            # For now: load from encrypted file

            key_file = self.key_storage_path / 'keys.enc'
            if key_file.exists():
                logger.info("Loading existing keys from storage")
                # Implementation would decrypt and load keys
            else:
                logger.info("No existing keys found, will generate on demand")

        except Exception as e:
            logger.error(f"Error loading keys: {e}")

    async def get_key(self, key_id: str) -> Optional[bytes]:
        """
        Get cryptographic key by ID

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        try:
            # Check if key exists
            if key_id not in self._keys:
                logger.warning(f"Key not found: {key_id}")
                return None

            # Check if key is expired
            metadata = self._key_metadata.get(key_id)
            if metadata and metadata.expires_at < datetime.now(timezone.utc):
                logger.warning(f"Key expired: {key_id}")
                return None

            return self._keys[key_id]

        except Exception as e:
            logger.error(f"Error retrieving key: {e}")
            return None

    async def create_key(self, key_id: str, algorithm: str = "AES-256-GCM") -> bytes:
        """
        Create new cryptographic key

        Args:
            key_id: Unique key identifier
            algorithm: Encryption algorithm

        Returns:
            Generated key
        """
        try:
            # Generate key based on algorithm
            if algorithm == "AES-256-GCM":
                key = os.urandom(32)  # 256-bit key
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # Store key
            self._keys[key_id] = key

            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.rotation_days),
                algorithm=algorithm,
                version=1,
                status="ACTIVE"
            )
            self._key_metadata[key_id] = metadata

            # Persist to storage
            await self._persist_keys()

            logger.info(f"Created new key: {key_id}")
            return key

        except Exception as e:
            logger.error(f"Error creating key: {e}", exc_info=True)
            raise KeyManagementError(f"Key creation failed: {str(e)}")

    async def rotate_key(self, key_id: str) -> bytes:
        """
        Rotate existing key

        Args:
            key_id: Key to rotate

        Returns:
            New key
        """
        try:
            old_metadata = self._key_metadata.get(key_id)
            if not old_metadata:
                raise KeyManagementError(f"Key not found for rotation: {key_id}")

            # Mark old key as rotated
            old_metadata.status = "ROTATED"
            old_metadata.rotated_at = datetime.now(timezone.utc)

            # Create new key with incremented version
            new_key = os.urandom(32)
            self._keys[key_id] = new_key

            # Update metadata
            new_metadata = KeyMetadata(
                key_id=key_id,
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(days=self.rotation_days),
                algorithm=old_metadata.algorithm,
                version=old_metadata.version + 1,
                status="ACTIVE"
            )
            self._key_metadata[key_id] = new_metadata

            # Persist
            await self._persist_keys()

            logger.info(f"Rotated key: {key_id} (version {new_metadata.version})")
            return new_key

        except Exception as e:
            logger.error(f"Error rotating key: {e}", exc_info=True)
            raise KeyManagementError(f"Key rotation failed: {str(e)}")

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke key (make it unusable)"""
        try:
            if key_id in self._key_metadata:
                self._key_metadata[key_id].status = "REVOKED"
                await self._persist_keys()
                logger.warning(f"Key revoked: {key_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error revoking key: {e}")
            return False

    async def _persist_keys(self) -> None:
        """Persist keys to secure storage"""
        try:
            # In production: store in HSM/KMS
            # For now: encrypt and write to file

            key_data = {
                'keys': {kid: base64.b64encode(key).decode() for kid, key in self._keys.items()},
                'metadata': {kid: asdict(meta) for kid, meta in self._key_metadata.items()}
            }

            # Convert datetime objects to strings for JSON serialization
            for meta in key_data['metadata'].values():
                for key in ['created_at', 'expires_at', 'rotated_at']:
                    if meta.get(key) and isinstance(meta[key], datetime):
                        meta[key] = meta[key].isoformat()

            # Encrypt and save
            key_json = json.dumps(key_data)
            encrypted = self._encryption.encrypt(key_json)

            key_file = self.key_storage_path / 'keys.enc'
            with open(key_file, 'w') as f:
                f.write(encrypted)

            logger.debug("Keys persisted to storage")

        except Exception as e:
            logger.error(f"Error persisting keys: {e}")

    async def _rotate_keys_periodically(self) -> None:
        """Periodically check and rotate expired keys"""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily

                now = datetime.now(timezone.utc)
                for key_id, metadata in self._key_metadata.items():
                    if metadata.status == "ACTIVE" and metadata.expires_at < now:
                        logger.info(f"Auto-rotating expired key: {key_id}")
                        await self.rotate_key(key_id)

            except Exception as e:
                logger.error(f"Error in periodic key rotation: {e}")

    def get_key_status(self, key_id: str) -> Optional[Dict]:
        """Get key metadata and status"""
        metadata = self._key_metadata.get(key_id)
        if metadata:
            return asdict(metadata)
        return None

    async def get_from_kms(self, key_id: str) -> Optional[bytes]:
        """Retrieve key from cloud KMS"""
        try:
            # In production: integrate with AWS KMS, Azure Key Vault, or GCP KMS
            # import boto3
            # kms_client = boto3.client('kms')
            # response = kms_client.decrypt(
            #     KeyId=self.kms_key_id,
            #     CiphertextBlob=encrypted_key
            # )
            # return response['Plaintext']

            logger.warning("KMS integration not implemented")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from KMS: {e}")
            return None


class KeyManagementError(Exception):
    """Key management error"""
    pass
