"""
Authentication router for Quantum Trader AI API.

Handles user registration, login, token management, and API keys.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from structlog import get_logger

from quantum_trader.api.schemas.auth_schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    RefreshTokenRequest,
    UserResponse,
    APIKeyRequest,
    APIKeyResponse,
    ChangePasswordRequest,
    ResetPasswordRequest,
    ResetPasswordConfirm
)
from quantum_trader.api.services.auth_service import AuthService
from quantum_trader.api.dependencies import (
    get_auth_service,
    get_current_user,
    require_permissions
)
from quantum_trader.api.exceptions import (
    AuthenticationException,
    ValidationException
)

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with email and password"
)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> UserResponse:
    """Register new user.

    Args:
        request: Registration request data
        auth_service: Authentication service dependency

    Returns:
        Created user information

    Raises:
        HTTPException: If registration fails
    """
    try:
        logger.info("User registration attempt", email=request.email)

        user = await auth_service.register_user(
            email=request.email,
            password=request.password,
            username=request.username,
            full_name=request.full_name
        )

        logger.info("User registered successfully", user_id=user["user_id"])

        return UserResponse(**user)

    except ValidationException as e:
        logger.warning("Registration validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and return access tokens"
)
async def login(
    request: LoginRequest,
    http_request: Request,
    auth_service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    """User login.

    Args:
        request: Login request data
        http_request: HTTP request for IP/user-agent logging
        auth_service: Authentication service dependency

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If authentication fails
    """
    try:
        logger.info("Login attempt", email=request.email)

        # Get client IP and user agent for security logging
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent", "unknown")

        tokens = await auth_service.authenticate_user(
            email=request.email,
            password=request.password,
            client_ip=client_ip,
            user_agent=user_agent
        )

        logger.info("Login successful", email=request.email)

        return TokenResponse(**tokens)

    except AuthenticationException as e:
        logger.warning("Authentication failed", email=request.email, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token"
)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> TokenResponse:
    """Refresh access token.

    Args:
        request: Refresh token request
        auth_service: Authentication service dependency

    Returns:
        New access and refresh tokens

    Raises:
        HTTPException: If token refresh fails
    """
    try:
        logger.debug("Token refresh attempt")

        tokens = await auth_service.refresh_access_token(request.refresh_token)

        logger.debug("Token refresh successful")

        return TokenResponse(**tokens)

    except AuthenticationException as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User logout",
    description="Invalidate user tokens"
)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """User logout.

    Args:
        current_user: Current authenticated user
        auth_service: Authentication service dependency

    Raises:
        HTTPException: If logout fails
    """
    try:
        logger.info("Logout request", user_id=current_user["user_id"])

        await auth_service.logout_user(current_user["user_id"])

        logger.info("Logout successful", user_id=current_user["user_id"])

    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get current authenticated user information"
)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> UserResponse:
    """Get current user information.

    Args:
        current_user: Current authenticated user

    Returns:
        User information
    """
    return UserResponse(**current_user)


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change password",
    description="Change user password"
)
async def change_password(
    request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Change user password.

    Args:
        request: Change password request
        current_user: Current authenticated user
        auth_service: Authentication service dependency

    Raises:
        HTTPException: If password change fails
    """
    try:
        logger.info("Password change request", user_id=current_user["user_id"])

        await auth_service.change_password(
            user_id=current_user["user_id"],
            current_password=request.current_password,
            new_password=request.new_password
        )

        logger.info("Password changed successfully", user_id=current_user["user_id"])

    except AuthenticationException as e:
        logger.warning("Password change failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Password change error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post(
    "/reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Request password reset",
    description="Request password reset email"
)
async def reset_password_request(
    request: ResetPasswordRequest,
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Request password reset.

    Args:
        request: Password reset request
        auth_service: Authentication service dependency

    Raises:
        HTTPException: If request fails
    """
    try:
        logger.info("Password reset requested", email=request.email)

        await auth_service.request_password_reset(request.email)

        # Always return success to prevent email enumeration
        logger.info("Password reset email sent", email=request.email)

    except Exception as e:
        logger.error("Password reset request error", error=str(e))
        # Don't raise exception to prevent email enumeration


@router.post(
    "/reset-password/confirm",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Confirm password reset",
    description="Reset password with token"
)
async def reset_password_confirm(
    request: ResetPasswordConfirm,
    auth_service: AuthService = Depends(get_auth_service)
) -> None:
    """Confirm password reset.

    Args:
        request: Password reset confirmation
        auth_service: Authentication service dependency

    Raises:
        HTTPException: If reset fails
    """
    try:
        logger.info("Password reset confirmation")

        await auth_service.confirm_password_reset(
            token=request.token,
            new_password=request.new_password
        )

        logger.info("Password reset successful")

    except AuthenticationException as e:
        logger.warning("Password reset confirmation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    except Exception as e:
        logger.error("Password reset error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create new API key for programmatic access"
)
async def create_api_key(
    request: APIKeyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
    _: None = Depends(require_permissions(["manage_api_keys"]))
) -> APIKeyResponse:
    """Create API key.

    Args:
        request: API key creation request
        current_user: Current authenticated user
        auth_service: Authentication service dependency

    Returns:
        Created API key information

    Raises:
        HTTPException: If creation fails
    """
    try:
        logger.info("API key creation request", user_id=current_user["user_id"])

        api_key_data = await auth_service.create_api_key(
            user_id=current_user["user_id"],
            name=request.name,
            permissions=request.permissions,
            expires_in_days=request.expires_in_days
        )

        logger.info("API key created", api_key_id=api_key_data["api_key_id"])

        return APIKeyResponse(**api_key_data)

    except ValidationException as e:
        logger.warning("API key creation validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("API key creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key creation failed"
        )


@router.delete(
    "/api-keys/{api_key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke/delete API key"
)
async def revoke_api_key(
    api_key_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
    _: None = Depends(require_permissions(["manage_api_keys"]))
) -> None:
    """Revoke API key.

    Args:
        api_key_id: API key identifier
        current_user: Current authenticated user
        auth_service: Authentication service dependency

    Raises:
        HTTPException: If revocation fails
    """
    try:
        logger.info("API key revocation request", api_key_id=api_key_id)

        await auth_service.revoke_api_key(
            user_id=current_user["user_id"],
            api_key_id=api_key_id
        )

        logger.info("API key revoked", api_key_id=api_key_id)

    except Exception as e:
        logger.error("API key revocation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key revocation failed"
        )
