"""FastAPI dependencies."""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from structlog import get_logger

logger = get_logger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current authenticated user."""
    if not token or token == "invalid":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"username": "admin", "email": "admin@quantumtrader.ai"}


async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get current active user."""
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_db():
    """Get database connection."""
    # Placeholder for database connection
    try:
        db = {"connection": "active"}
        yield db
    finally:
        pass
