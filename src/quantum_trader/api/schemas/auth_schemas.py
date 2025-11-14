"""Authentication and authorization schemas.

This module defines Pydantic models for authentication, authorization,
user management, and API key management.

Example:
    ```python
    from quantum_trader.api.schemas.auth_schemas import LoginRequest, TokenResponse

    # Login request
    login_req = LoginRequest(
        username="trader@example.com",
        password="secure_password",
        two_factor_code="123456"
    )

    # Token response
    token_resp = TokenResponse(
        access_token="eyJ...",
        token_type="bearer",
        expires_in=3600
    )
    ```
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, validator


class LoginRequest(BaseModel):
    """Login request schema.

    Attributes:
        username: User email or username
        password: User password
        two_factor_code: Optional 2FA code (required if 2FA enabled)
    """

    username: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    two_factor_code: Optional[str] = Field(None, min_length=6, max_length=6, description="Two-factor authentication code")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "trader@example.com",
                "password": "SecurePassword123!",
                "two_factor_code": "123456"
            }
        }


class TokenResponse(BaseModel):
    """Authentication token response.

    Attributes:
        access_token: JWT access token
        token_type: Token type (always "bearer")
        expires_in: Token expiration time in seconds
        refresh_token: Optional refresh token
        user_id: User identifier
        permissions: List of user permissions
    """

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for obtaining new access token")
    user_id: str = Field(..., description="Unique user identifier")
    permissions: List[str] = Field(default_factory=list, description="List of user permissions")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "user_id": "usr_123456789",
                "permissions": ["trading.read", "trading.write", "analytics.read"]
            }
        }


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema.

    Attributes:
        refresh_token: Refresh token to exchange for new access token
    """

    refresh_token: str = Field(..., description="Refresh token")

    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class RegisterRequest(BaseModel):
    """User registration request.

    Attributes:
        email: User email address
        password: User password
        confirm_password: Password confirmation
        full_name: User's full name
        organization: Optional organization name
    """

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    confirm_password: str = Field(..., min_length=8, description="Password confirmation")
    full_name: str = Field(..., min_length=2, max_length=100, description="User's full name")
    organization: Optional[str] = Field(None, max_length=100, description="Organization name")

    @validator("confirm_password")
    def passwords_match(cls, v: str, values: dict) -> str:
        """Validate that passwords match."""
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("password")
    def password_strength(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "email": "newtrader@example.com",
                "password": "SecurePassword123!",
                "confirm_password": "SecurePassword123!",
                "full_name": "John Trader",
                "organization": "Quantum Trading LLC"
            }
        }


class UserResponse(BaseModel):
    """User information response.

    Attributes:
        user_id: Unique user identifier
        email: User email address
        full_name: User's full name
        organization: Organization name
        permissions: List of user permissions
        is_active: Whether user account is active
        is_verified: Whether email is verified
        created_at: Account creation timestamp
        last_login: Last login timestamp
    """

    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    full_name: str = Field(..., description="User's full name")
    organization: Optional[str] = Field(None, description="Organization name")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="Account active status")
    is_verified: bool = Field(default=False, description="Email verification status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_123456789",
                "email": "trader@example.com",
                "full_name": "John Trader",
                "organization": "Quantum Trading LLC",
                "permissions": ["trading.read", "trading.write", "analytics.read"],
                "is_active": True,
                "is_verified": True,
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-01-15T10:30:00Z"
            }
        }


class APIKeyCreateRequest(BaseModel):
    """API key creation request.

    Attributes:
        name: Descriptive name for the API key
        permissions: List of permissions for the key
        expires_in_days: Optional expiration in days (None = no expiration)
    """

    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(..., min_items=1, description="List of permissions")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production Trading Bot",
                "permissions": ["trading.read", "trading.write"],
                "expires_in_days": 90
            }
        }


class APIKeyResponse(BaseModel):
    """API key response.

    Attributes:
        key_id: Unique key identifier
        name: API key name
        api_key: The actual API key (only shown on creation)
        permissions: List of permissions
        created_at: Creation timestamp
        expires_at: Optional expiration timestamp
        last_used: Optional last usage timestamp
        is_active: Whether key is active
    """

    key_id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="API key name")
    api_key: Optional[str] = Field(None, description="API key value (only on creation)")
    permissions: List[str] = Field(..., description="Key permissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    is_active: bool = Field(default=True, description="Active status")

    class Config:
        json_schema_extra = {
            "example": {
                "key_id": "key_987654321",
                "name": "Production Trading Bot",
                "api_key": "qtai_1234567890abcdef",
                "permissions": ["trading.read", "trading.write"],
                "created_at": "2024-01-01T00:00:00Z",
                "expires_at": "2024-04-01T00:00:00Z",
                "last_used": "2024-01-15T10:30:00Z",
                "is_active": True
            }
        }


class ChangePasswordRequest(BaseModel):
    """Change password request.

    Attributes:
        current_password: Current password
        new_password: New password
        confirm_new_password: New password confirmation
    """

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_new_password: str = Field(..., min_length=8, description="New password confirmation")

    @validator("confirm_new_password")
    def passwords_match(cls, v: str, values: dict) -> str:
        """Validate that new passwords match."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("New passwords do not match")
        return v

    @validator("new_password")
    def password_strength(cls, v: str, values: dict) -> str:
        """Validate password strength and difference from current."""
        if "current_password" in values and v == values["current_password"]:
            raise ValueError("New password must be different from current password")

        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecurePassword456!",
                "confirm_new_password": "NewSecurePassword456!"
            }
        }


class Enable2FARequest(BaseModel):
    """Enable two-factor authentication request.

    Attributes:
        password: User password for verification
    """

    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "password": "SecurePassword123!"
            }
        }


class Enable2FAResponse(BaseModel):
    """Enable 2FA response with QR code.

    Attributes:
        secret: TOTP secret key
        qr_code_url: QR code URL for authenticator apps
        backup_codes: List of backup codes
    """

    secret: str = Field(..., description="TOTP secret key")
    qr_code_url: str = Field(..., description="QR code data URL")
    backup_codes: List[str] = Field(..., description="One-time backup codes")

    class Config:
        json_schema_extra = {
            "example": {
                "secret": "JBSWY3DPEHPK3PXP",
                "qr_code_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...",
                "backup_codes": ["12345678", "87654321", "11111111", "22222222", "33333333"]
            }
        }


class Verify2FARequest(BaseModel):
    """Verify 2FA setup request.

    Attributes:
        code: 6-digit TOTP code from authenticator app
    """

    code: str = Field(..., min_length=6, max_length=6, description="6-digit TOTP code")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "123456"
            }
        }


class PermissionResponse(BaseModel):
    """Permission information.

    Attributes:
        permission: Permission identifier
        description: Human-readable description
        resource: Resource this permission applies to
        actions: List of allowed actions
    """

    permission: str = Field(..., description="Permission identifier")
    description: str = Field(..., description="Permission description")
    resource: str = Field(..., description="Resource type")
    actions: List[str] = Field(..., description="Allowed actions")

    class Config:
        json_schema_extra = {
            "example": {
                "permission": "trading.write",
                "description": "Execute trades and manage orders",
                "resource": "trading",
                "actions": ["create_order", "cancel_order", "modify_order"]
            }
        }
