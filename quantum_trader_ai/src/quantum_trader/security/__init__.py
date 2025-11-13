"""
Quantum Trader Security Module
CRITICAL: Authentication, encryption, key management, and security controls
"""

# API Security
from quantum_trader.security.api_security import (
    SecurityManager,
    Credentials,
    SignedRequest,
    SecurityError
)

# Audit Trail
from quantum_trader.security.audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditLevel,
    AuditCategory
)

# Authentication
from quantum_trader.security.authentication import (
    AuthenticationManager,
    AuthStatus,
    Session,
    User
)

# Encryption
from quantum_trader.security.encryption import (
    EncryptionManager,
    EncryptionError
)

# Key Management
from quantum_trader.security.key_manager import (
    KeyManager,
    KeyMetadata,
    KeyManagementError
)

# Vault Integration
from quantum_trader.security.vault_integration import VaultIntegration

__all__ = [
    # API Security
    "SecurityManager",
    "Credentials",
    "SignedRequest",
    "SecurityError",

    # Audit Trail
    "AuditTrail",
    "AuditEvent",
    "AuditLevel",
    "AuditCategory",

    # Authentication
    "AuthenticationManager",
    "AuthStatus",
    "Session",
    "User",

    # Encryption
    "EncryptionManager",
    "EncryptionError",

    # Key Management
    "KeyManager",
    "KeyMetadata",
    "KeyManagementError",

    # Vault Integration
    "VaultIntegration",
]

__version__ = "1.0.0"
__author__ = "Quantum Trader Development Team"
