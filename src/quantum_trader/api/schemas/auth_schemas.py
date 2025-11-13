"""
Authentication schemas for Quantum Trader AI API.

Pydantic models for request/response validation.
"""

from decimal import Decimal
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, validator
from enum import Enum


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "ADMIN"
    TRADER = "TRADER"
    VIEWER = "VIEWER"
    API_USER = "API_USER"


class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "ACCESS"
    REFRESH = "REFRESH"
    API_KEY = "API_KEY"


class RegisterRequest(BaseModel):
    """User registration request.

    Attributes:
        email: User email address
        password: User password (min 8 chars)
        username: Username (optional)
        full_name: Full name (optional)
    """
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")

    @validator("password")
    def validate_password(cls, v: str) -> str:
        """Validate password strength.

        Args:
            v: Password string

        Returns:
            Validated password

        Raises:
            ValueError: If password is weak
        """
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "trader@example.com",
                "password": "SecurePass123",
                "username": "trader1",
                "full_name": "John Trader"
            }
        }


class LoginRequest(BaseModel):
    """User login request.

    Attributes:
        email: User email address
        password: User password
    """
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

    class Config:
        schema_extra = {
            "example": {
                "email": "trader@example.com",
                "password": "SecurePass123"
            }
        }


class TokenResponse(BaseModel):
    """Authentication token response.

    Attributes:
        access_token: JWT access token
        refresh_token: JWT refresh token
        token_type: Token type (Bearer)
        expires_in: Token expiration in seconds
    """
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "Bearer",
                "expires_in": 3600
            }
        }


class RefreshTokenRequest(BaseModel):
    """Refresh token request.

    Attributes:
        refresh_token: JWT refresh token
    """
    refresh_token: str = Field(..., description="JWT refresh token")

    class Config:
        schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class UserResponse(BaseModel):
    """User information response.

    Attributes:
        user_id: User unique identifier
        email: User email address
        username: Username
        full_name: Full name
        role: User role
        is_active: Account active status
        created_at: Account creation timestamp
        last_login: Last login timestamp
    """
    user_id: str = Field(..., description="User unique identifier")
    email: str = Field(..., description="User email address")
    username: Optional[str] = Field(None, description="Username")
    full_name: Optional[str] = Field(None, description="Full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(..., description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp (UTC)")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "usr_1234567890",
                "email": "trader@example.com",
                "username": "trader1",
                "full_name": "John Trader",
                "role": "TRADER",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z",
                "last_login": "2024-01-15T12:30:00Z"
            }
        }


class APIKeyRequest(BaseModel):
    """API key creation request.

    Attributes:
        name: API key name/description
        permissions: List of permissions
        expires_in_days: Expiration in days (optional)
    """
    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(..., description="List of permissions")
    expires_in_days: Optional[int] = Field(None, gt=0, le=365, description="Expiration in days")

    class Config:
        schema_extra = {
            "example": {
                "name": "Trading Bot API Key",
                "permissions": ["trade", "read_market_data"],
                "expires_in_days": 90
            }
        }


class APIKeyResponse(BaseModel):
    """API key response.

    Attributes:
        api_key_id: API key identifier
        api_key: The actual API key (only shown once)
        name: API key name
        permissions: List of permissions
        created_at: Creation timestamp
        expires_at: Expiration timestamp
    """
    api_key_id: str = Field(..., description="API key identifier")
    api_key: str = Field(..., description="The actual API key (only shown once)")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="List of permissions")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp (UTC)")

    class Config:
        schema_extra = {
            "example": {
                "api_key_id": "key_1234567890",
                "api_key": "qtai_1234567890abcdef",
                "name": "Trading Bot API Key",
                "permissions": ["trade", "read_market_data"],
                "created_at": "2024-01-01T00:00:00Z",
                "expires_at": "2024-04-01T00:00:00Z"
            }
        }


class ChangePasswordRequest(BaseModel):
    """Change password request.

    Attributes:
        current_password: Current password
        new_password: New password
    """
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")

    @validator("new_password")
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    class Config:
        schema_extra = {
            "example": {
                "current_password": "OldPass123",
                "new_password": "NewSecurePass456"
            }
        }


class ResetPasswordRequest(BaseModel):
    """Password reset request.

    Attributes:
        email: User email address
    """
    email: EmailStr = Field(..., description="User email address")

    class Config:
        schema_extra = {
            "example": {
                "email": "trader@example.com"
            }
        }


class ResetPasswordConfirm(BaseModel):
    """Password reset confirmation.

    Attributes:
        token: Password reset token
        new_password: New password
    """
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")

    @validator("new_password")
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    class Config:
        schema_extra = {
            "example": {
                "token": "reset_token_1234567890",
                "new_password": "NewSecurePass456"
            }
        }
