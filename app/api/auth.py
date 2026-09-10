import logging

from fastapi import APIRouter, HTTPException, status
from datetime import timedelta
from app.models.user import UserCreate, UserLogin, Token
from app.services.auth import create_user, authenticate_user, create_access_token
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)

@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    """Register a new user"""
    try:
        create_user(user.email, user.password)
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("User registration failed")
        raise HTTPException(status_code=500, detail="Unable to register user")

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """Login and get access token"""
    user_obj = authenticate_user(user.email, user.password)
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
