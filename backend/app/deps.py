"""Auth + rate-limit dependencies shared across routers."""
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_db
from .models import User
from .redis_client import rate_limit_hit
from .security import decode_token

_bearer = HTTPBearer(auto_error=False)


async def _user_from_token(
    creds: HTTPAuthorizationCredentials | None, db: AsyncSession
) -> User | None:
    if not creds:
        return None
    try:
        payload = decode_token(creds.credentials, expected_type="access")
    except jwt.PyJWTError:
        return None
    return await db.get(User, payload.get("sub"))


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    user = await _user_from_token(creds, db)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return user


async def get_optional_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    return await _user_from_token(creds, db)


async def enforce_rate_limit(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    identity = "anon"
    if creds:
        try:
            payload = decode_token(creds.credentials, expected_type="access")
            identity = payload.get("sub", "anon")
        except jwt.PyJWTError:
            pass
    if identity == "anon":
        identity = request.client.host if request.client else "unknown"

    over = await rate_limit_hit(
        identity, settings.rate_limit_optimize, settings.rate_limit_window_seconds
    )
    if over:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
