"""
Authentication Pydantic Schemas

Production-ready Pydantic models for authentication request/response validation.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field, EmailStr, validator
from structlog import get_logger

logger = get_logger(__name__)


class LoginRequest(BaseModel):
    """Login request schema."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, max_length=100, description="Password")

    class Config:
        schema_extra = {
            "example": {
                "username": "trader123",
                "password": "secure_password_123"
            }
        }


class LoginResponse(BaseModel):
    """Login response schema."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(..., description="User information")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800,
                "user": {
                    "user_id": "123",
                    "username": "trader123",
                    "email": "trader@example.com",
                    "roles": ["trader"]
                }
            }
        }


class TokenRefreshRequest(BaseModel):
    """Token refresh request schema."""

    refresh_token: Optional[str] = Field(None, description="Refresh token (if using refresh tokens)")

    class Config:
        schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenResponse(BaseModel):
    """Token response schema."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class UserInfo(BaseModel):
    """User information schema."""

    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[EmailStr] = Field(None, description="Email address")
    roles: List[str] = Field(default_factory=list, description="User roles")
    created_at: Optional[datetime] = Field(None, description="Account creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "123",
                "username": "trader123",
                "email": "trader@example.com",
                "roles": ["trader", "analyst"],
                "created_at": "2025-01-01T00:00:00Z",
                "last_login": "2025-01-14T12:00:00Z"
            }
        }


class RegisterRequest(BaseModel):
    """User registration request schema."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=100, description="Password")
    confirm_password: str = Field(..., min_length=8, max_length=100, description="Confirm password")

    @validator("confirm_password")
    def passwords_match(cls, v, values, **kwargs):
        """Validate passwords match."""
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v

    class Config:
        schema_extra = {
            "example": {
                "username": "newtrader",
                "email": "newtrader@example.com",
                "password": "secure_password_123",
                "confirm_password": "secure_password_123"
            }
        }


class RegisterResponse(BaseModel):
    """User registration response schema."""

    user_id: str = Field(..., description="Created user ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    created_at: datetime = Field(..., description="Account creation timestamp")
    message: str = Field(..., description="Success message")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "456",
                "username": "newtrader",
                "email": "newtrader@example.com",
                "created_at": "2025-01-14T12:00:00Z",
                "message": "User registered successfully"
            }
        }


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""

    current_password: str = Field(..., min_length=8, max_length=100, description="Current password")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    confirm_new_password: str = Field(..., min_length=8, max_length=100, description="Confirm new password")

    @validator("confirm_new_password")
    def passwords_match(cls, v, values, **kwargs):
        """Validate passwords match."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    @validator("new_password")
    def password_different(cls, v, values, **kwargs):
        """Validate new password is different."""
        if "current_password" in values and v == values["current_password"]:
            raise ValueError("New password must be different from current password")
        return v

    class Config:
        schema_extra = {
            "example": {
                "current_password": "old_password_123",
                "new_password": "new_secure_password_456",
                "confirm_new_password": "new_secure_password_456"
            }
        }


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""

    email: EmailStr = Field(..., description="Email address")

    class Config:
        schema_extra = {
            "example": {
                "email": "trader@example.com"
            }
        }


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    reset_token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=100, description="New password")
    confirm_new_password: str = Field(..., min_length=8, max_length=100, description="Confirm new password")

    @validator("confirm_new_password")
    def passwords_match(cls, v, values, **kwargs):
        """Validate passwords match."""
        if "new_password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v

    class Config:
        schema_extra = {
            "example": {
                "reset_token": "abc123def456",
                "new_password": "new_secure_password_789",
                "confirm_new_password": "new_secure_password_789"
            }
        }


class APIKeyCreate(BaseModel):
    """API key creation request schema."""

    name: str = Field(..., min_length=1, max_length=100, description="API key name")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")

    class Config:
        schema_extra = {
            "example": {
                "name": "Trading Bot API Key",
                "permissions": ["read:markets", "write:orders", "read:portfolio"],
                "expires_in_days": 90
            }
        }


class APIKeyResponse(BaseModel):
    """API key response schema."""

    key_id: str = Field(..., description="API key ID")
    api_key: str = Field(..., description="API key (only shown once)")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="API key permissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    class Config:
        schema_extra = {
            "example": {
                "key_id": "key_789",
                "api_key": "qta_1234567890abcdef",
                "name": "Trading Bot API Key",
                "permissions": ["read:markets", "write:orders"],
                "created_at": "2025-01-14T12:00:00Z",
                "expires_at": "2025-04-14T12:00:00Z"
            }
        }


class APIKeyInfo(BaseModel):
    """API key information schema (without secret)."""

    key_id: str = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="API key permissions")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    is_active: bool = Field(..., description="Whether key is active")

    class Config:
        schema_extra = {
            "example": {
                "key_id": "key_789",
                "name": "Trading Bot API Key",
                "permissions": ["read:markets", "write:orders"],
                "created_at": "2025-01-14T12:00:00Z",
                "expires_at": "2025-04-14T12:00:00Z",
                "last_used": "2025-01-14T13:30:00Z",
                "is_active": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    details: Optional[Any] = Field(None, description="Error details")
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "Authentication failed",
                "details": "Invalid credentials",
                "timestamp": "2025-01-14T12:00:00Z"
            }
        }


class SuccessResponse(BaseModel):
    """Success response schema."""

    message: str = Field(..., description="Success message")
    timestamp: datetime = Field(..., description="Response timestamp")

    class Config:
        schema_extra = {
            "example": {
                "message": "Operation completed successfully",
                "timestamp": "2025-01-14T12:00:00Z"
            }
        }
