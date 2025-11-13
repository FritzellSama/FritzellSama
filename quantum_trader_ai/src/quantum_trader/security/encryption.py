"""
Encryption Module - CRITICAL SECURITY SYSTEM
AES-256-GCM encryption/decryption for sensitive data
"""

import logging
import os
import base64
from typing import Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Production-grade encryption manager using AES-256-GCM"""

    def __init__(self, master_key: bytes = None) -> None:
        self.config = get_config()
        self._load_config()

        # Master key (from KMS in production)
        self.master_key = master_key or self._get_master_key()

        logger.info("EncryptionManager initialized")

    def _load_config(self) -> None:
        """Load encryption configuration"""
        self.algorithm = self.config.get('security', 'encryption.algorithm', 'AES-256-GCM')
        self.key_size_bits = self.config.get_int('security', 'encryption.key_size_bits', 256)
        self.iv_size_bytes = self.config.get_int('security', 'encryption.iv_size_bytes', 16)

    def _get_master_key(self) -> bytes:
        """Get master encryption key"""
        # In production: retrieve from AWS KMS, Azure Key Vault, or HashiCorp Vault
        key_b64 = os.getenv('MASTER_ENCRYPTION_KEY')
        if key_b64:
            return base64.b64decode(key_b64)

        logger.warning("No master key found, generating temporary key (NOT PRODUCTION-SAFE)")
        return os.urandom(32)  # 256-bit key

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext using AES-256-GCM

        Returns:
            Base64 encoded: IV (16 bytes) + ciphertext + tag (16 bytes)
        """
        try:
            # Generate random IV
            iv = os.urandom(self.iv_size_bytes)

            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()

            # Encrypt
            ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()

            # Combine IV + ciphertext + tag
            encrypted = iv + ciphertext + encryptor.tag

            # Return base64 encoded
            return base64.b64encode(encrypted).decode('utf-8')

        except Exception as e:
            logger.error(f"Encryption error: {e}", exc_info=True)
            raise EncryptionError(f"Encryption failed: {str(e)}")

    def decrypt(self, encrypted_b64: str) -> str:
        """
        Decrypt AES-256-GCM encrypted data

        Args:
            encrypted_b64: Base64 encoded encrypted data

        Returns:
            Decrypted plaintext
        """
        try:
            # Decode base64
            encrypted = base64.b64decode(encrypted_b64)

            # Extract components
            iv = encrypted[:self.iv_size_bytes]
            tag = encrypted[-16:]  # GCM tag is always 16 bytes
            ciphertext = encrypted[self.iv_size_bytes:-16]

            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.master_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()

            # Decrypt
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            return plaintext.decode('utf-8')

        except Exception as e:
            logger.error(f"Decryption error: {e}", exc_info=True)
            raise EncryptionError(f"Decryption failed: {str(e)}")

    def encrypt_field(self, data: dict, field: str) -> dict:
        """Encrypt specific field in dictionary"""
        if field in data and data[field]:
            data[field] = self.encrypt(str(data[field]))
        return data

    def decrypt_field(self, data: dict, field: str) -> dict:
        """Decrypt specific field in dictionary"""
        if field in data and data[field]:
            data[field] = self.decrypt(data[field])
        return data

    def derive_key(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2

        Returns:
            Tuple of (derived_key, salt)
        """
        try:
            if salt is None:
                salt = os.urandom(16)

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256-bit key
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )

            key = kdf.derive(password.encode('utf-8'))
            return (key, salt)

        except Exception as e:
            logger.error(f"Key derivation error: {e}")
            raise EncryptionError(f"Key derivation failed: {str(e)}")


class EncryptionError(Exception):
    """Encryption-related error"""
    pass
