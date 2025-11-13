"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from structlog import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint."""
    # Placeholder authentication
    if form_data.username == "admin" and form_data.password == "admin":
        return {
            "access_token": "dummy_token",
            "token_type": "bearer"
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password"
    )


@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout endpoint."""
    logger.info("User logged out")
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user."""
    return {
        "username": "admin",
        "email": "admin@quantumtrader.ai",
        "role": "admin"
    }
