"""
Two-Factor Authentication Module - TOTP-based 2FA
CRITICAL: Production-ready TOTP 2FA with backup codes
"""

import logging
import secrets
import hmac
import hashlib
import struct
import time
import base64
from typing import List, Optional, Set, Dict
from datetime import datetime, timedelta
import asyncio

from quantum_trader.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class TwoFactorAuth:
    """Production TOTP-based two-factor authentication"""

    def __init__(self):
        """Initialize 2FA from configuration"""
        self._config = get_config()
        self._enabled: bool = True
        self._provider: str = "totp"
        self._issuer: str = "QuantumTrader"
        self._token_validity_seconds: int = 30
        self._backup_codes_count: int = 10
        self._time_window: int = 1  # Accept tokens +/- 1 time step
        self._secret_length: int = 32  # bytes for secret
        self._backup_code_length: int = 8  # characters per backup code

        # Storage for user secrets and backup codes (in production, use database)
        self._user_secrets: Dict[str, str] = {}
        self._backup_codes: Dict[str, Set[str]] = {}
        self._used_backup_codes: Dict[str, Set[str]] = {}

        self._lock = asyncio.Lock()

        # Load configuration
        self._load_config()

        logger.info(
            f"Two-factor auth initialized: enabled={self._enabled}, "
            f"provider={self._provider}, validity={self._token_validity_seconds}s"
        )

    def _load_config(self) -> None:
        """Load 2FA configuration"""
        try:
            # Load MFA settings
            self._enabled = self._config.get_bool('security', 'mfa.enabled', True)
            self._provider = self._config.get('security', 'mfa.provider', 'totp')
            self._issuer = self._config.get('security', 'mfa.issuer', 'QuantumTrader')
            self._token_validity_seconds = self._config.get_int(
                'security', 'mfa.token_validity_seconds', 30
            )
            self._backup_codes_count = self._config.get_int(
                'security', 'mfa.backup_codes_count', 10
            )

            logger.info("Two-factor auth configuration loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load 2FA configuration: {e}")
            # Set safe defaults
            self._enabled = True
            self._provider = "totp"
            self._issuer = "QuantumTrader"

    async def generate_secret(self, user_id: str) -> str:
        """
        Generate a new TOTP secret for a user

        Args:
            user_id: User identifier

        Returns:
            Base32-encoded secret string
        """
        async with self._lock:
            try:
                # Generate random secret
                secret_bytes = secrets.token_bytes(self._secret_length)

                # Encode as base32 (TOTP standard)
                secret = base64.b32encode(secret_bytes).decode('utf-8').rstrip('=')

                # Store secret for user
                self._user_secrets[user_id] = secret

                logger.info(f"Generated TOTP secret for user {user_id}")

                return secret

            except Exception as e:
                logger.error(f"Failed to generate secret for user {user_id}: {e}")
                raise

    async def get_provisioning_uri(self, user_id: str, account_name: str, secret: Optional[str] = None) -> str:
        """
        Generate provisioning URI for QR code

        Args:
            user_id: User identifier
            account_name: Account name/email for display
            secret: Optional secret (will use stored secret if not provided)

        Returns:
            Provisioning URI string for QR code generation
        """
        try:
            # Get secret
            if secret is None:
                if user_id not in self._user_secrets:
                    logger.error(f"No secret found for user {user_id}")
                    raise ValueError(f"No secret found for user {user_id}")
                secret = self._user_secrets[user_id]

            # Build provisioning URI
            uri = (
                f"otpauth://totp/{self._issuer}:{account_name}"
                f"?secret={secret}"
                f"&issuer={self._issuer}"
                f"&algorithm=SHA1"
                f"&digits=6"
                f"&period={self._token_validity_seconds}"
            )

            logger.debug(f"Generated provisioning URI for user {user_id}")

            return uri

        except Exception as e:
            logger.error(f"Failed to generate provisioning URI for user {user_id}: {e}")
            raise

    def _generate_totp(self, secret: str, time_step: int) -> str:
        """
        Generate TOTP code for given secret and time step

        Args:
            secret: Base32-encoded secret
            time_step: Time step (current_time // period)

        Returns:
            6-digit TOTP code
        """
        try:
            # Decode secret
            secret_bytes = base64.b32decode(secret + '=' * ((8 - len(secret) % 8) % 8))

            # Convert time step to bytes
            time_bytes = struct.pack('>Q', time_step)

            # Generate HMAC-SHA1
            hmac_hash = hmac.new(secret_bytes, time_bytes, hashlib.sha1).digest()

            # Dynamic truncation
            offset = hmac_hash[-1] & 0x0F
            truncated = struct.unpack('>I', hmac_hash[offset:offset + 4])[0]
            truncated &= 0x7FFFFFFF

            # Generate 6-digit code
            code = str(truncated % 1000000).zfill(6)

            return code

        except Exception as e:
            logger.error(f"Failed to generate TOTP: {e}")
            raise

    async def verify_token(self, user_id: str, token: str) -> bool:
        """
        Verify TOTP token for user

        Args:
            user_id: User identifier
            token: 6-digit TOTP token to verify

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if 2FA is enabled
            if not self._enabled:
                logger.warning("2FA verification skipped - 2FA is disabled")
                return True

            # Get user secret
            if user_id not in self._user_secrets:
                logger.error(f"No secret found for user {user_id}")
                return False

            secret = self._user_secrets[user_id]

            # Get current time step
            current_time = int(time.time())
            current_step = current_time // self._token_validity_seconds

            # Check token against current time step and window
            for offset in range(-self._time_window, self._time_window + 1):
                time_step = current_step + offset
                expected_token = self._generate_totp(secret, time_step)

                # Constant-time comparison to prevent timing attacks
                if hmac.compare_digest(token, expected_token):
                    logger.info(f"TOTP token verified for user {user_id}")
                    return True

            logger.warning(f"TOTP token verification FAILED for user {user_id}")
            return False

        except Exception as e:
            logger.error(f"Error verifying token for user {user_id}: {e}")
            return False

    async def generate_backup_codes(self, user_id: str) -> List[str]:
        """
        Generate backup codes for user

        Args:
            user_id: User identifier

        Returns:
            List of backup code strings
        """
        async with self._lock:
            try:
                # Generate backup codes
                backup_codes = []
                for _ in range(self._backup_codes_count):
                    # Generate random alphanumeric code
                    code = ''.join(
                        secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789')
                        for _ in range(self._backup_code_length)
                    )
                    backup_codes.append(code)

                # Store backup codes for user (hashed)
                self._backup_codes[user_id] = {
                    self._hash_backup_code(code) for code in backup_codes
                }

                # Initialize used codes set if not exists
                if user_id not in self._used_backup_codes:
                    self._used_backup_codes[user_id] = set()

                logger.info(f"Generated {len(backup_codes)} backup codes for user {user_id}")

                return backup_codes

            except Exception as e:
                logger.error(f"Failed to generate backup codes for user {user_id}: {e}")
                raise

    def _hash_backup_code(self, code: str) -> str:
        """
        Hash backup code for secure storage

        Args:
            code: Backup code to hash

        Returns:
            Hashed code
        """
        return hashlib.sha256(code.encode('utf-8')).hexdigest()

    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """
        Verify and consume a backup code

        Args:
            user_id: User identifier
            code: Backup code to verify

        Returns:
            True if valid and not used, False otherwise
        """
        async with self._lock:
            try:
                # Check if 2FA is enabled
                if not self._enabled:
                    logger.warning("2FA verification skipped - 2FA is disabled")
                    return True

                # Check if user has backup codes
                if user_id not in self._backup_codes:
                    logger.error(f"No backup codes found for user {user_id}")
                    return False

                # Hash provided code
                code_hash = self._hash_backup_code(code.upper())

                # Check if code is valid and not used
                if code_hash in self._backup_codes[user_id]:
                    if code_hash not in self._used_backup_codes.get(user_id, set()):
                        # Mark code as used
                        if user_id not in self._used_backup_codes:
                            self._used_backup_codes[user_id] = set()
                        self._used_backup_codes[user_id].add(code_hash)

                        logger.info(f"Backup code verified and consumed for user {user_id}")
                        return True
                    else:
                        logger.warning(f"Backup code already used for user {user_id}")
                        return False
                else:
                    logger.warning(f"Invalid backup code for user {user_id}")
                    return False

            except Exception as e:
                logger.error(f"Error verifying backup code for user {user_id}: {e}")
                return False

    async def get_remaining_backup_codes(self, user_id: str) -> int:
        """
        Get count of remaining (unused) backup codes

        Args:
            user_id: User identifier

        Returns:
            Number of remaining backup codes
        """
        try:
            if user_id not in self._backup_codes:
                return 0

            total_codes = len(self._backup_codes[user_id])
            used_codes = len(self._used_backup_codes.get(user_id, set()))

            return total_codes - used_codes

        except Exception as e:
            logger.error(f"Error getting remaining backup codes for user {user_id}: {e}")
            return 0

    async def regenerate_backup_codes(self, user_id: str) -> List[str]:
        """
        Regenerate backup codes (invalidate old codes)

        Args:
            user_id: User identifier

        Returns:
            List of new backup codes
        """
        async with self._lock:
            try:
                # Clear old backup codes
                if user_id in self._backup_codes:
                    del self._backup_codes[user_id]
                if user_id in self._used_backup_codes:
                    del self._used_backup_codes[user_id]

                # Generate new backup codes
                new_codes = await self.generate_backup_codes(user_id)

                logger.info(f"Regenerated backup codes for user {user_id}")

                return new_codes

            except Exception as e:
                logger.error(f"Failed to regenerate backup codes for user {user_id}: {e}")
                raise

    async def disable_2fa(self, user_id: str) -> bool:
        """
        Disable 2FA for user (remove secret and backup codes)

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Remove user secret
                if user_id in self._user_secrets:
                    del self._user_secrets[user_id]

                # Remove backup codes
                if user_id in self._backup_codes:
                    del self._backup_codes[user_id]
                if user_id in self._used_backup_codes:
                    del self._used_backup_codes[user_id]

                logger.info(f"Disabled 2FA for user {user_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to disable 2FA for user {user_id}: {e}")
                return False

    async def is_2fa_enabled_for_user(self, user_id: str) -> bool:
        """
        Check if 2FA is enabled for user

        Args:
            user_id: User identifier

        Returns:
            True if 2FA is set up, False otherwise
        """
        return user_id in self._user_secrets

    async def get_current_totp(self, user_id: str) -> Optional[str]:
        """
        Get current TOTP token for user (for testing/verification)

        Args:
            user_id: User identifier

        Returns:
            Current TOTP token or None if not found
        """
        try:
            if user_id not in self._user_secrets:
                logger.error(f"No secret found for user {user_id}")
                return None

            secret = self._user_secrets[user_id]

            # Get current time step
            current_time = int(time.time())
            current_step = current_time // self._token_validity_seconds

            # Generate current token
            token = self._generate_totp(secret, current_step)

            return token

        except Exception as e:
            logger.error(f"Failed to get current TOTP for user {user_id}: {e}")
            return None

    async def get_time_remaining(self) -> int:
        """
        Get seconds remaining until next TOTP token

        Returns:
            Seconds remaining in current time window
        """
        try:
            current_time = int(time.time())
            time_remaining = self._token_validity_seconds - (current_time % self._token_validity_seconds)
            return time_remaining

        except Exception as e:
            logger.error(f"Failed to get time remaining: {e}")
            return 0

    async def validate_secret(self, secret: str) -> bool:
        """
        Validate a TOTP secret format

        Args:
            secret: Base32-encoded secret to validate

        Returns:
            True if valid format, False otherwise
        """
        try:
            # Try to decode as base32
            base64.b32decode(secret + '=' * ((8 - len(secret) % 8) % 8))
            return True

        except Exception:
            return False

    async def get_2fa_statistics(self, user_id: str) -> Dict[str, any]:
        """
        Get 2FA statistics for user

        Args:
            user_id: User identifier

        Returns:
            Dictionary with 2FA statistics
        """
        try:
            is_enabled = await self.is_2fa_enabled_for_user(user_id)
            remaining_codes = await self.get_remaining_backup_codes(user_id)
            time_remaining = await self.get_time_remaining()

            return {
                '2fa_enabled': is_enabled,
                'provider': self._provider,
                'backup_codes_remaining': remaining_codes,
                'totp_time_remaining_seconds': time_remaining,
                'token_validity_seconds': self._token_validity_seconds,
            }

        except Exception as e:
            logger.error(f"Error getting 2FA statistics for user {user_id}: {e}")
            return {}

    async def import_secret(self, user_id: str, secret: str) -> bool:
        """
        Import an existing TOTP secret for user

        Args:
            user_id: User identifier
            secret: Base32-encoded secret to import

        Returns:
            True if successful, False otherwise
        """
        async with self._lock:
            try:
                # Validate secret format
                if not await self.validate_secret(secret):
                    logger.error(f"Invalid secret format for user {user_id}")
                    return False

                # Store secret
                self._user_secrets[user_id] = secret.upper().rstrip('=')

                logger.info(f"Imported TOTP secret for user {user_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to import secret for user {user_id}: {e}")
                return False

    async def verify_setup(self, user_id: str, token: str) -> bool:
        """
        Verify 2FA setup by checking first token

        Args:
            user_id: User identifier
            token: TOTP token to verify

        Returns:
            True if setup is valid, False otherwise
        """
        try:
            # Verify token
            is_valid = await self.verify_token(user_id, token)

            if is_valid:
                logger.info(f"2FA setup verified for user {user_id}")
            else:
                logger.warning(f"2FA setup verification failed for user {user_id}")

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying 2FA setup for user {user_id}: {e}")
            return False
