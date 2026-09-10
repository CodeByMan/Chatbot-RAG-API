from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings
from app.models.user import TokenData, User
from app.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters"
        )

    if len(password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password must be 72 UTF-8 bytes or fewer"
        )
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_user(email: str, password: str) -> User:
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user
        hashed_password = get_password_hash(password)
        cursor.execute(
            "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
            (email, hashed_password)
        )
        conn.commit()

        return User(email=email, hashed_password=hashed_password)


def authenticate_user(email: str, password: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email, hashed_password FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()

        if not row:
            return False

        user = User(email=row["email"], hashed_password=row["hashed_password"])
        if not verify_password(password, user.hashed_password):
            return False

        return user


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email, hashed_password FROM users WHERE email = ?", (token_data.email,))
        row = cursor.fetchone()

        if not row:
            raise credentials_exception

        return User(email=row["email"], hashed_password=row["hashed_password"])
