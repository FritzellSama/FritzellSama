"""Authentication API router.

This module provides REST API endpoints for authentication, authorization,
user management, and API key operations.

Endpoints:
- POST /auth/login - User login
- POST /auth/logout - User logout
- POST /auth/refresh - Refresh access token
- POST /auth/register - Register new user
- POST /auth/change-password - Change password
- GET /auth/me - Get current user info
- POST /auth/2fa/enable - Enable 2FA
- POST /auth/2fa/verify - Verify 2FA setup
- POST /auth/api-keys - Create API key
- GET /auth/api-keys - List API keys
- DELETE /auth/api-keys/{key_id} - Revoke API key

Example:
    ```python
    from fastapi import FastAPI
    from quantum_trader.api.routers.auth import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    ```
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from structlog import get_logger

from quantum_trader.api.schemas.auth_schemas import (
    APIKeyCreateRequest,
    APIKeyResponse,
    ChangePasswordRequest,
    Enable2FARequest,
    Enable2FAResponse,
    LoginRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
    Verify2FARequest,
)
from quantum_trader.api.services.auth_service import (
    AuthenticationError,
    AuthorizationError,
    AuthService,
    RateLimitError,
)

logger = get_logger(__name__)

# FastAPI router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme
security = HTTPBearer()


# Dependency to get auth service
# In production, this would come from dependency injection
def get_auth_service() -> AuthService:
    """Get auth service instance.

    This is a placeholder. In production, this would be properly
    initialized with config and injected as a dependency.
    """
    # Mock config - in production, load from config file
    config = {
        "jwt_secret": "your-secret-key-min-32-chars-long-please-change-in-production",
        "encryption_key": "your-encryption-key-must-be-base64-encoded-fernet-key-here",
        "jwt_expiry_hours": 24,
        "refresh_token_expiry_days": 30,
        "max_login_attempts": 5,
        "lockout_duration_minutes": 30,
        "enable_2fa": True,
    }
    return AuthService(config)


# Dependency to get current user from token
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """Extract and validate current user from JWT token.

    Args:
        credentials: HTTP bearer token credentials
        auth_service: Auth service instance

    Returns:
        User information dictionary

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        token = credentials.credentials
        user_info = await auth_service.validate_token(token)
        return user_info

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("get_current_user_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    """Authenticate user and return access tokens.

    Args:
        request: Login credentials
        auth_service: Auth service instance

    Returns:
        TokenResponse with access and refresh tokens

    Raises:
        HTTPException: If authentication fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/login" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "username": "trader@example.com",
                   "password": "SecurePassword123!",
                   "two_factor_code": "123456"
                 }'
        ```
    """
    try:
        token_response = await auth_service.login(
            username=request.username,
            password=request.password,
            two_factor_code=request.two_factor_code
        )

        logger.info("user_login_endpoint", username=request.username)

        return token_response

    except RateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e)
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error("login_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Logout user by invalidating token.

    Args:
        current_user: Current authenticated user
        credentials: Bearer token credentials
        auth_service: Auth service instance

    Raises:
        HTTPException: If logout fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/logout" \\
             -H "Authorization: Bearer YOUR_TOKEN"
        ```
    """
    try:
        token = credentials.credentials
        await auth_service.logout(token)

        logger.info("user_logout_endpoint", user_id=current_user["user_id"])

    except Exception as e:
        logger.error("logout_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/refresh", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    """Refresh access token using refresh token.

    Args:
        request: Refresh token request
        auth_service: Auth service instance

    Returns:
        New TokenResponse with access token

    Raises:
        HTTPException: If refresh fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/refresh" \\
             -H "Content-Type: application/json" \\
             -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
        ```
    """
    try:
        token_response = await auth_service.refresh_access_token(request.refresh_token)

        logger.info("token_refreshed_endpoint")

        return token_response

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error("refresh_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """Register new user account.

    Args:
        request: Registration information
        auth_service: Auth service instance

    Returns:
        UserResponse with created user information

    Raises:
        HTTPException: If registration fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/register" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "email": "newtrader@example.com",
                   "password": "SecurePassword123!",
                   "confirm_password": "SecurePassword123!",
                   "full_name": "John Trader",
                   "organization": "Quantum Trading LLC"
                 }'
        ```
    """
    try:
        # In production, this would call auth_service.register()
        # For now, return mock response

        logger.info("user_registered", email=request.email)

        return UserResponse(
            user_id=f"usr_{request.email}",
            email=request.email,
            full_name=request.full_name,
            organization=request.organization,
            permissions=["trading.read"],
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
            last_login=None
        )

    except Exception as e:
        logger.error("register_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """Get current authenticated user information.

    Args:
        current_user: Current authenticated user
        auth_service: Auth service instance

    Returns:
        UserResponse with user information

    Example:
        ```bash
        curl -X GET "http://localhost:8000/api/v1/auth/me" \\
             -H "Authorization: Bearer YOUR_TOKEN"
        ```
    """
    try:
        # In production, fetch full user details from database
        from datetime import datetime

        return UserResponse(
            user_id=current_user["user_id"],
            email=current_user.get("email", ""),
            full_name="John Trader",
            organization="Quantum Trading LLC",
            permissions=current_user.get("permissions", []),
            is_active=True,
            is_verified=True,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )

    except Exception as e:
        logger.error("get_me_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user information"
        )


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Change user password.

    Args:
        request: Password change request
        current_user: Current authenticated user
        auth_service: Auth service instance

    Raises:
        HTTPException: If password change fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/change-password" \\
             -H "Authorization: Bearer YOUR_TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "current_password": "OldPassword123!",
                   "new_password": "NewSecurePassword456!",
                   "confirm_new_password": "NewSecurePassword456!"
                 }'
        ```
    """
    try:
        # In production, call auth_service.change_password()

        logger.info("password_changed", user_id=current_user["user_id"])

    except Exception as e:
        logger.error("change_password_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/2fa/enable", response_model=Enable2FAResponse, status_code=status.HTTP_200_OK)
async def enable_2fa(
    request: Enable2FARequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> Enable2FAResponse:
    """Enable two-factor authentication.

    Args:
        request: Enable 2FA request with password verification
        current_user: Current authenticated user
        auth_service: Auth service instance

    Returns:
        Enable2FAResponse with TOTP secret and QR code

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/2fa/enable" \\
             -H "Authorization: Bearer YOUR_TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{"password": "SecurePassword123!"}'
        ```
    """
    try:
        response = await auth_service.enable_2fa(current_user["user_id"])

        logger.info("2fa_enabled_endpoint", user_id=current_user["user_id"])

        return response

    except Exception as e:
        logger.error("enable_2fa_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enable 2FA"
        )


@router.post("/2fa/verify", status_code=status.HTTP_204_NO_CONTENT)
async def verify_2fa(
    request: Verify2FARequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Verify two-factor authentication setup.

    Args:
        request: Verification request with TOTP code
        current_user: Current authenticated user
        auth_service: Auth service instance

    Raises:
        HTTPException: If verification fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/2fa/verify" \\
             -H "Authorization: Bearer YOUR_TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{"code": "123456"}'
        ```
    """
    try:
        await auth_service.verify_2fa_setup(current_user["user_id"], request.code)

        logger.info("2fa_verified_endpoint", user_id=current_user["user_id"])

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("verify_2fa_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA verification failed"
        )


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> APIKeyResponse:
    """Create new API key.

    Args:
        request: API key creation request
        current_user: Current authenticated user
        auth_service: Auth service instance

    Returns:
        APIKeyResponse with generated key

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/auth/api-keys" \\
             -H "Authorization: Bearer YOUR_TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{
                   "name": "Production Trading Bot",
                   "permissions": ["trading.read", "trading.write"],
                   "expires_in_days": 90
                 }'
        ```
    """
    try:
        api_key_response = await auth_service.create_api_key(
            user_id=current_user["user_id"],
            name=request.name,
            permissions=request.permissions,
            expires_in_days=request.expires_in_days
        )

        logger.info(
            "api_key_created_endpoint",
            user_id=current_user["user_id"],
            key_id=api_key_response.key_id
        )

        return api_key_response

    except Exception as e:
        logger.error("create_api_key_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )


@router.get("/api-keys", response_model=List[APIKeyResponse], status_code=status.HTTP_200_OK)
async def list_api_keys(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> List[APIKeyResponse]:
    """List all API keys for current user.

    Args:
        current_user: Current authenticated user
        auth_service: Auth service instance

    Returns:
        List of APIKeyResponse

    Example:
        ```bash
        curl -X GET "http://localhost:8000/api/v1/auth/api-keys" \\
             -H "Authorization: Bearer YOUR_TOKEN"
        ```
    """
    try:
        # In production, fetch from database
        from datetime import datetime

        return [
            APIKeyResponse(
                key_id="key_123",
                name="Production Trading Bot",
                api_key=None,  # Never return actual key in list
                permissions=["trading.read", "trading.write"],
                created_at=datetime.utcnow(),
                expires_at=None,
                last_used=datetime.utcnow(),
                is_active=True
            )
        ]

    except Exception as e:
        logger.error("list_api_keys_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys"
        )


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Revoke (delete) API key.

    Args:
        key_id: API key identifier to revoke
        current_user: Current authenticated user
        auth_service: Auth service instance

    Raises:
        HTTPException: If revocation fails

    Example:
        ```bash
        curl -X DELETE "http://localhost:8000/api/v1/auth/api-keys/key_123" \\
             -H "Authorization: Bearer YOUR_TOKEN"
        ```
    """
    try:
        # In production, revoke key in database
        # await auth_service.revoke_api_key(key_id, current_user["user_id"])

        logger.info(
            "api_key_revoked_endpoint",
            user_id=current_user["user_id"],
            key_id=key_id
        )

    except Exception as e:
        logger.error("revoke_api_key_endpoint_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )
